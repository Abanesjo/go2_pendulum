# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--eval-profile",
    choices=("nominal", "randomized", "stress"),
    default="nominal",
    help="Evaluation conditions: deterministic nominal, in-distribution randomized, or held-out stress.",
)
parser.add_argument(
    "--difficulty",
    type=int,
    choices=range(1, 6),
    default=5,
    help="Force one curriculum difficulty level during evaluation (default: 5).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from rsl_rl_compat import sanitize_rsl_rl_config

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2_pendulum.tasks  # noqa: F401


def _apply_evaluation_profile(env_cfg: DirectRLEnvCfg) -> None:
    """Apply a reproducible, non-curricular evaluation profile."""
    if not hasattr(env_cfg, "difficulty_override"):
        return

    env_cfg.enable_curriculum = False
    env_cfg.difficulty_override = args_cli.difficulty
    # Evaluation should use the canonical deployment coordinate system. The
    # episode-level reflection remains a training augmentation.
    if hasattr(env_cfg, "enable_episode_mirroring"):
        env_cfg.enable_episode_mirroring = False
    # Training chains settled goals to keep locomotion represented. Evaluation
    # keeps one fixed goal for the full horizon to measure convergence and hold.
    if hasattr(env_cfg, "enable_goal_chaining"):
        env_cfg.enable_goal_chaining = False
    if hasattr(env_cfg, "stagger_initial_episode_lengths"):
        env_cfg.stagger_initial_episode_lengths = False
    if args_cli.eval_profile == "nominal":
        env_cfg.enable_domain_randomization = False
        env_cfg.enable_external_wrench_push = False
        env_cfg.evaluation_dr_scale_multiplier = 0.0
        env_cfg.evaluation_push_scale_multiplier = 0.0
    elif args_cli.eval_profile == "randomized":
        env_cfg.enable_domain_randomization = True
        env_cfg.enable_external_wrench_push = True
        env_cfg.evaluation_dr_scale_multiplier = 1.0
        env_cfg.evaluation_push_scale_multiplier = 1.0
    else:
        env_cfg.enable_domain_randomization = True
        env_cfg.enable_external_wrench_push = True
        # Held-out stress intentionally exceeds the maximum training envelope.
        env_cfg.evaluation_dr_scale_multiplier = 1.5
        env_cfg.evaluation_push_scale_multiplier = 1.25

    print(
        f"[INFO] Evaluation profile: {args_cli.eval_profile}; "
        f"forced difficulty: {args_cli.difficulty}."
    )


def _validate_checkpoint_contract(checkpoint_path: str, expected_version: str) -> None:
    """Reject same-shaped checkpoints trained with incompatible observation semantics."""
    env_metadata_path = os.path.join(os.path.dirname(checkpoint_path), "params", "env.yaml")
    if not os.path.isfile(env_metadata_path):
        raise RuntimeError(
            "Cannot verify the checkpoint policy contract because its params/env.yaml is missing: "
            f"{env_metadata_path}. Contract-v1 checkpoints must not be played or exported as v2."
        )

    saved_version = None
    with open(env_metadata_path, encoding="utf-8") as metadata_file:
        for line in metadata_file:
            if line.startswith("policy_contract_version:"):
                saved_version = line.split(":", maxsplit=1)[1].strip().strip("'\"")
                break
    if saved_version != expected_version:
        raise RuntimeError(
            f"Checkpoint policy contract mismatch: expected '{expected_version}', found "
            f"'{saved_version or 'missing'}' in {env_metadata_path}. Use a v2 checkpoint."
        )


def _policy_contract_metadata(env_cfg: DirectRLEnvCfg, policy_nn: torch.nn.Module, step_dt: float, normalizer):
    """Build the machine-readable deployment contract emitted beside ONNX."""
    observation_dim = int(env_cfg.observation_space)
    action_dim = int(env_cfg.action_space)
    is_recurrent = bool(getattr(policy_nn, "is_recurrent", False))
    inputs = [{"name": "obs", "dtype": "float32", "shape": [1, observation_dim]}]
    outputs = [{"name": "actions", "dtype": "float32", "shape": [1, action_dim]}]
    recurrent = {"enabled": is_recurrent}

    if is_recurrent:
        memory = getattr(policy_nn, "memory_a", getattr(policy_nn, "memory_s", None))
        rnn = getattr(memory, "rnn", None)
        if rnn is None:
            raise RuntimeError("Could not resolve the recurrent actor memory needed for ONNX contract metadata.")
        rnn_type = type(rnn).__name__.lower()
        hidden_shape = [int(rnn.num_layers), 1, int(rnn.hidden_size)]
        inputs.append({"name": "h_in", "dtype": "float32", "shape": hidden_shape})
        outputs.append({"name": "h_out", "dtype": "float32", "shape": hidden_shape})
        if rnn_type == "lstm":
            inputs.append({"name": "c_in", "dtype": "float32", "shape": hidden_shape})
            outputs.append({"name": "c_out", "dtype": "float32", "shape": hidden_shape})
        recurrent.update(
            {
                "type": rnn_type,
                "num_layers": int(rnn.num_layers),
                "hidden_size": int(rnn.hidden_size),
                "reset_on_episode_or_controller_reset": True,
            }
        )

    return {
        "contract_version": str(getattr(env_cfg, "policy_contract_version", "unknown")),
        "checkpoint_compatible_with_v1": False,
        "control_period_s": float(step_dt),
        "observation_dim": observation_dim,
        "action_dim": action_dim,
        "joint_order": list(env_cfg.leg_joint_names),
        "default_joint_position_rad": [
            float(env_cfg.robot_cfg.init_state.joint_pos[joint_name]) for joint_name in env_cfg.leg_joint_names
        ],
        "observation_layout": [
            {"range": [0, 3], "actor": "finite_difference_mocap_base_linear_velocity_body"},
            {"range": [3, 6], "actor": "imu_angular_velocity_body"},
            {"range": [6, 9], "actor": "imu_projected_gravity_body"},
            {
                "range": [9, 12],
                "actor": "body_command_forward_lateral_yaw_rate",
                "critic": "body_goal_xy_error_and_final_yaw_error",
            },
            {"range": [12, 24], "actor": "leg_joint_position_offset"},
            {"range": [24, 36], "actor": "leg_joint_velocity"},
            {"range": [36, 38], "actor": "mocap_reconstructed_pendulum_angle"},
            {"range": [38, 40], "actor": "finite_difference_wrapped_pendulum_velocity"},
            {"range": [40, 52], "actor": "previous_delivered_normalized_action"},
            {"range": [52, 56], "actor": "phase_sin_phase_cos_move_gate_stand_gate"},
        ],
        "navigation": {
            "max_forward_speed_m_s": float(env_cfg.command_max_forward_speed_m_s),
            "max_yaw_rate_rad_s": float(env_cfg.command_max_yaw_rate_rad_s),
            "k_rho": float(env_cfg.command_k_rho),
            "k_alpha": float(env_cfg.command_k_alpha),
            "k_beta": float(env_cfg.command_k_beta),
            "k_final_yaw": float(env_cfg.command_k_final_yaw),
            "heading_blend_near_m": float(env_cfg.command_heading_blend_near_m),
            "heading_blend_far_m": float(env_cfg.command_heading_blend_far_m),
            "forward_heading_cutoff_rad": float(env_cfg.command_forward_heading_cutoff_rad),
            "stand_enter_distance_m": float(env_cfg.stand_enter_distance_m),
            "stand_exit_distance_m": float(env_cfg.stand_exit_distance_m),
            "stand_enter_yaw_rad": float(env_cfg.stand_enter_yaw_rad),
            "stand_exit_yaw_rad": float(env_cfg.stand_exit_yaw_rad),
            "stand_correction_position_gain_s": float(env_cfg.stand_correction_position_gain_s),
            "stand_correction_yaw_gain_s": float(env_cfg.stand_correction_yaw_gain_s),
            "stand_correction_max_linear_m_s": float(env_cfg.stand_correction_max_linear_m_s),
            "stand_correction_max_yaw_rate_rad_s": float(env_cfg.stand_correction_max_yaw_rate_rad_s),
            "training_goal_ranges_m": {
                "stand": list(env_cfg.goal_stand_distance_range),
                "short": list(env_cfg.goal_short_distance_range),
                "walk": list(env_cfg.goal_walk_distance_range),
            },
        },
        "gait": {
            "frequency_range_hz": [float(env_cfg.gait_frequency_min_hz), float(env_cfg.gait_frequency_max_hz)],
            "duty_factor_range": [
                float(env_cfg.gait_duty_factor_min_speed),
                float(env_cfg.gait_duty_factor_max_speed),
            ],
        },
        "mocap": {
            "base_linear_velocity": "unfiltered_policy_rate_finite_difference",
            "pendulum_velocity": "unfiltered_policy_rate_wrapped_finite_difference",
            "pendulum_hinge_offset_b": list(env_cfg.pendulum_hinge_offset_b),
        },
        "action_pipeline": {
            "clip": float(env_cfg.action_clip),
            "action_scale_rad": float(env_cfg.action_scale),
            "nominal_lpf_cutoff_hz": float(env_cfg.command_lpf_cutoff_hz),
            "training_lpf_cutoff_range_hz": list(env_cfg.command_lpf_cutoff_range_hz),
            "joint_target_slew_rate_rad_s": float(env_cfg.joint_target_slew_rate_rad_s),
            "joint_limit_margin_rad": float(env_cfg.joint_limit_margin_rad),
            "previous_action_is_post_pipeline": True,
        },
        "onnx": {
            "opset_version": 18,
            "fixed_batch_size": 1,
            "inputs": inputs,
            "outputs": outputs,
            "actor_observation_normalizer_embedded": normalizer is not None,
            "recurrent": recurrent,
        },
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    _apply_evaluation_profile(env_cfg)
    if hasattr(env_cfg, "action_clip") and float(agent_cfg.clip_actions) != float(env_cfg.action_clip):
        raise ValueError(
            f"Runner clip_actions ({agent_cfg.clip_actions}) must match the environment action_clip "
            f"({env_cfg.action_clip}) for policy contract v2."
        )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    if hasattr(env_cfg, "policy_contract_version"):
        _validate_checkpoint_contract(resume_path, str(env_cfg.policy_contract_version))

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner_cfg = sanitize_rsl_rl_config(agent_cfg.to_dict())
    runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    contract = _policy_contract_metadata(env.unwrapped.cfg, policy_nn, dt, normalizer)
    dump_yaml(os.path.join(export_model_dir, "policy_contract.yaml"), contract)
    print(f"[INFO] Exported policy and deployment contract to: {export_model_dir}")

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
