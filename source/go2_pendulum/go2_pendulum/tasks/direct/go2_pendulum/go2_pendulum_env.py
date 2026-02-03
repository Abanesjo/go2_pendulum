# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.utils.math import sample_uniform
import isaaclab.utils.math as math_utils

from .go2_pendulum_env_cfg import Go2PendulumEnvCfg


class Go2PendulumEnv(DirectRLEnv):
    cfg: Go2PendulumEnvCfg

    def __init__(self, cfg: Go2PendulumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # gait shaping
        self._feet_ids = []
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        for name in foot_names:
            id_list, _ = self.robot.find_bodies(name)
            self._feet_ids.append(id_list[0])
        self._feet_ids_sensor = []
        for name in foot_names:
            id_list, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids_sensor.append(id_list[0])
        self._feet_ids_sensor = torch.tensor(self._feet_ids_sensor, device=self.device, dtype=torch.long)

        self.gait_indices = torch.zeros(
            self.num_envs,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.clock_inputs = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.desired_contact_states = torch.zeros(
            self.num_envs,
            4,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Resolve leg and pendulum joints.
        leg_joint_ids = []
        for idx, name in enumerate(self.robot.joint_names):
            if name.endswith("_hip_joint") or name.endswith("_thigh_joint") or name.endswith("_calf_joint"):
                leg_joint_ids.append(idx)
        if len(leg_joint_ids) != self.cfg.action_space:
            raise RuntimeError(
                "Leg joint count does not match action space: "
                f"{len(leg_joint_ids)} vs {self.cfg.action_space}."
            )
        self._leg_dof_ids = torch.tensor(leg_joint_ids, device=self.device, dtype=torch.long)

        self._pendulum_dof_count = len(self.cfg.pendulum_joint_names)
        if self.cfg.use_pendulum:
            self._pendulum_dof_ids = []
            for joint_name in self.cfg.pendulum_joint_names:
                joint_idx, _ = self.robot.find_joints(joint_name)
                if len(joint_idx) != 1:
                    raise RuntimeError(f"Expected exactly one joint for '{joint_name}', got {joint_idx}.")
                self._pendulum_dof_ids.append(joint_idx[0])
            self._pendulum_dof_ids = torch.tensor(self._pendulum_dof_ids, device=self.device, dtype=torch.long)
        else:
            self._pendulum_dof_ids = torch.tensor([], device=self.device, dtype=torch.long)

        # Joint position command (deviation from default joint positions).
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # Target state [x_d, y_d, yaw_d] in environment frame (randomized per reset).
        self.target_state = None
        # Track consecutive failure steps for termination logic.
        self.pendulum_failure_steps = None
        self.position_failure_steps = None

        # Marker visualization buffers.
        self._marker_offset = None
        self._marker_orientations = None
        self._marker_locations = None
        self._marker_up = torch.tensor([0.0, 0.0, 1.0])

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "upright",
                "position",
                "yaw_alignment",
                "pendulum_velocity",
                "angular_velocity",
                "balanced_movement",
                "action_magnitude",
                "action_delta",
                "tilt",
                "alive",
                "terminated",
            ]
        }

        # Effort limit for saturation logging (used by DC motor model).
        base_legs_actuator = self.cfg.robot_cfg.actuators.get("base_legs")
        effort_limit = None if base_legs_actuator is None else base_legs_actuator.effort_limit
        if effort_limit is None:
            effort_limit = float("inf")
        self._leg_effort_limit = torch.tensor(effort_limit, device=self.device, dtype=torch.float32)

        # Track termination causes for accurate logging.
        self._base_contact_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.last_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Get specific body indices.
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        undesired_contact_ids, _ = self._contact_sensor.find_bodies(".*_thigh")
        base_contact_ids, _ = self._contact_sensor.find_bodies("base")
        undesired_contact_ids = list(dict.fromkeys(base_contact_ids + undesired_contact_ids))
        self._undesired_contact_body_ids = (
            torch.tensor(undesired_contact_ids, device=self.device, dtype=torch.long)
            if len(undesired_contact_ids) > 0
            else None
        )

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._pendulum_contact_sensor = None
        if self.cfg.use_pendulum:
            self._pendulum_contact_sensor = ContactSensor(self.cfg.pendulum_contact_sensor)
        # register assets and sensors so they get replicated and updated
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if self._pendulum_contact_sensor is not None:
            self.scene.sensors["pendulum_contact_sensor"] = self._pendulum_contact_sensor
        self._height_scanner = None
        if self.cfg.enable_height_scanner and self.cfg.height_scanner is not None:
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # create target visualizer after scene is set up
        self.target_visualizer = VisualizationMarkers(self.cfg.target_marker_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions

        self.desired_joint_pos = self.robot.data.default_joint_pos.clone()
        self.desired_joint_pos[:, self._leg_dof_ids] = (
            self.robot.data.default_joint_pos[:, self._leg_dof_ids] + self._processed_actions
        )

        # No velocity command generation in position tracking mode.

    def _apply_action(self) -> None:
        # Send position targets; actuator model handles PD and torque limits.
        self.robot.set_joint_position_target(
            self.desired_joint_pos[:, self._leg_dof_ids],
            joint_ids=self._leg_dof_ids,
        )

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if self._height_scanner is not None:
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        leg_joint_pos = self.robot.data.joint_pos[:, self._leg_dof_ids] - self.robot.data.default_joint_pos[:, self._leg_dof_ids]
        leg_joint_vel = self.robot.data.joint_vel[:, self._leg_dof_ids]
        if self.cfg.use_pendulum:
            pendulum_joint_pos = self.robot.data.joint_pos[:, self._pendulum_dof_ids]
            pendulum_joint_vel = self.robot.data.joint_vel[:, self._pendulum_dof_ids]
        else:
            pendulum_joint_pos = torch.zeros(
                self.num_envs,
                self._pendulum_dof_count,
                device=self.device,
                dtype=self.robot.data.joint_pos.dtype,
            )
            pendulum_joint_vel = torch.zeros_like(pendulum_joint_pos)

        position_error_b_xy, yaw_error = self._compute_body_frame_errors()
        state_error = torch.cat([position_error_b_xy, yaw_error.unsqueeze(-1)], dim=-1)

        policy_tensors = [
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            state_error,
            leg_joint_pos,
            leg_joint_vel,
            pendulum_joint_pos,
            pendulum_joint_vel,
        ]
        if self.cfg.use_height_scan:
            if height_data is None:
                raise RuntimeError("Height scan requested but height scanner is disabled.")
            policy_tensors.append(height_data)
        policy_tensors.append(self._actions)
        policy_obs = torch.cat(policy_tensors, dim=-1)
        observations = {"policy": policy_obs}
        if self.cfg.return_teacher_obs:
            if height_data is None:
                raise RuntimeError("Teacher observations require the height scanner to be enabled.")
            teacher_obs = torch.cat(
                [
                    self.robot.data.root_lin_vel_b,
                    self.robot.data.root_ang_vel_b,
                    self.robot.data.projected_gravity_b,
                    state_error,
                    leg_joint_pos,
                    leg_joint_vel,
                    pendulum_joint_pos,
                    pendulum_joint_vel,
                    height_data,
                    self._actions,
                ],
                dim=-1,
            )
            observations["teacher"] = teacher_obs
        return observations

    def _compute_body_frame_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos = self.robot.data.root_pos_w
        root_quat = self.robot.data.root_quat_w
        env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
        base_pos_xy = root_pos[:, :2] - env_origins[:, :2]

        if self.target_state is not None:
            target_xy = self.target_state[:, :2]
        else:
            target_xy = torch.zeros((self.num_envs, 2), device=self.device)

        position_error_xy = target_xy - base_pos_xy
        position_error_w = root_pos.new_zeros((self.num_envs, 3))
        position_error_w[:, :2] = position_error_xy
        position_error_b = math_utils.quat_apply_inverse(root_quat, position_error_w)
        position_error_b_xy = position_error_b[:, :2]

        direction_norm = torch.linalg.norm(position_error_xy, dim=-1, keepdim=True)
        direction_norm = torch.clamp(direction_norm, min=1e-6)
        direction_unit = position_error_xy / direction_norm
        target_heading_w = root_pos.new_zeros((self.num_envs, 3))
        target_heading_w[:, :2] = direction_unit
        target_heading_b = math_utils.quat_apply_inverse(root_quat, target_heading_w)
        yaw_error = torch.atan2(target_heading_b[:, 1], target_heading_b[:, 0])

        return position_error_b_xy, yaw_error

    def _get_rewards(self) -> torch.Tensor:
        # position and yaw alignment tracking (environment frame)
        position_error_b_xy, yaw_error = self._compute_body_frame_errors()
        if self.target_state is not None:
            position_deviation = torch.linalg.norm(position_error_b_xy, dim=-1)
            position_reward = torch.exp(-position_deviation)
        else:
            position_reward = torch.ones(self.num_envs, device=self.device)
        yaw_alignment_reward = torch.exp(-torch.abs(yaw_error))

        # pendulum rewards from joint angles/velocities (target is zero)
        if self.cfg.use_pendulum:
            pendulum_joint_pos = self.robot.data.joint_pos[:, self._pendulum_dof_ids]
            pendulum_joint_vel = self.robot.data.joint_vel[:, self._pendulum_dof_ids]
            pendulum_joint_pos = torch.nan_to_num(pendulum_joint_pos, nan=0.0, posinf=math.pi, neginf=-math.pi)
            pendulum_joint_vel = torch.nan_to_num(pendulum_joint_vel, nan=0.0, posinf=100.0, neginf=-100.0)
            pendulum_norm = torch.linalg.norm(pendulum_joint_pos, dim=1)
            pendulum_vel_norm = torch.linalg.norm(pendulum_joint_vel, dim=1)
        else:
            pendulum_norm = torch.zeros(self.num_envs, device=self.device)
            pendulum_vel_norm = torch.zeros(self.num_envs, device=self.device)

        upright_reward = torch.exp(-pendulum_norm)
        pendulum_velocity_reward = torch.exp(-pendulum_vel_norm)

        # angular velocity reward: reward for not spinning around z-axis
        root_ang_vel = self.robot.data.root_ang_vel_w
        root_ang_vel = torch.nan_to_num(root_ang_vel, nan=0.0, posinf=100.0, neginf=-100.0)
        z_angular_speed = torch.abs(root_ang_vel[:, 2])
        angular_velocity_reward = torch.exp(-z_angular_speed)

        # balanced movement (not moving while unbalanced)
        base_lin_vel = self.robot.data.root_lin_vel_w[:, :2]
        base_lin_vel = torch.nan_to_num(base_lin_vel, nan=0.0, posinf=100.0, neginf=-100.0)
        base_speed = torch.linalg.norm(base_lin_vel, dim=1)
        balanced_movement_reward = torch.exp(-pendulum_norm * base_speed)

        # action penalties
        action_magnitude_reward = torch.sum(self._actions**2, dim=1)
        action_delta_reward = torch.sum((self._actions - self.last_actions[:, :, 0]) ** 2, dim=1)

        # body tilt penalty
        roll, pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        tilt_cos = torch.cos(roll) * torch.cos(pitch)
        tilt_angle = torch.acos(torch.clamp(tilt_cos, -1.0, 1.0))

        rewards = {
            "upright": self.cfg.rew_scale_upright * upright_reward,
            "position": self.cfg.rew_scale_position * position_reward,
            "yaw_alignment": self.cfg.rew_scale_yaw_alignment * yaw_alignment_reward,
            "pendulum_velocity": self.cfg.rew_scale_pendulum_velocity * pendulum_velocity_reward,
            "angular_velocity": self.cfg.rew_scale_angular_velocity * angular_velocity_reward,
            "balanced_movement": self.cfg.rew_scale_balanced_movement * balanced_movement_reward,
            "action_magnitude": self.cfg.rew_scale_action_magnitude * action_magnitude_reward,
            "action_delta": self.cfg.rew_scale_action_delta * action_delta_reward,
            "tilt": self.cfg.rew_scale_tilt * tilt_angle,
            "alive": self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float()),
            "terminated": self.cfg.rew_scale_terminated * self.reset_terminated.float(),
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        cstr_termination_contacts = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )
        # Allow a grace period so brief settling contacts right after reset don't terminate.
        contact_grace = self.episode_length_buf > math.ceil(self.cfg.base_contact_grace_s / self.step_dt)
        cstr_termination_contacts = cstr_termination_contacts & contact_grace

        self._base_contact_terminated = cstr_termination_contacts

        env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins

        # Pendulum balance termination.
        if self.cfg.use_pendulum:
            pendulum_pos = self.robot.data.joint_pos[:, self._pendulum_dof_ids]
            pendulum_norm = torch.linalg.norm(pendulum_pos, dim=-1)
            if self.pendulum_failure_steps is None:
                self.pendulum_failure_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            pendulum_failing = pendulum_norm > (self.cfg.pendulum_failure_angle_deg * math.pi / 180.0)
            self.pendulum_failure_steps = torch.where(
                pendulum_failing,
                self.pendulum_failure_steps + 1,
                torch.zeros_like(self.pendulum_failure_steps),
            )
            failure_threshold = max(1, int(self.cfg.pendulum_failure_timeout_s / self.step_dt))
            pendulum_terminated = self.pendulum_failure_steps >= failure_threshold
        else:
            pendulum_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Position deviation termination.
        position_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        position_tolerance_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.target_state is not None:
            base_pos_xy = self.robot.data.root_pos_w[:, :2] - env_origins[:, :2]
            target_xy = self.target_state[:, :2]
            position_deviation = torch.linalg.norm(base_pos_xy - target_xy, dim=-1)
            position_terminated = position_deviation > self.cfg.max_displacement
            if self.position_failure_steps is None:
                self.position_failure_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            position_failing = position_deviation > self.cfg.position_tolerance
            self.position_failure_steps = torch.where(
                position_failing,
                self.position_failure_steps + 1,
                torch.zeros_like(self.position_failure_steps),
            )
            failure_threshold = max(1, int(self.cfg.position_failure_timeout_s / self.step_dt))
            position_tolerance_terminated = self.position_failure_steps >= failure_threshold

        terminated = (
            cstr_termination_contacts
            | pendulum_terminated
            | position_terminated
            | position_tolerance_terminated
        )

        if self.cfg.use_pendulum and self._pendulum_contact_sensor is not None:
            pendulum_contact_forces = self._pendulum_contact_sensor.data.net_forces_w
            pendulum_contact = torch.any(
                torch.norm(pendulum_contact_forces, dim=-1) > self.cfg.pendulum_contact_force_threshold,
                dim=1,
            )
            terminated = terminated | pendulum_contact

        return terminated, time_out

    def _update_terrain_curriculum(self, env_ids: torch.Tensor) -> None:
        if self._terrain.terrain_origins is None or self._terrain.cfg.terrain_generator is None:
            return
        if env_ids.numel() == 0:
            return
        distance = torch.norm(
            self.robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2],
            dim=1,
        )
        move_up = distance > self._terrain.cfg.terrain_generator.size[0] / 2
        move_down = distance < torch.norm(self.robot.data.root_lin_vel_b[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        move_down &= ~move_up
        self._terrain.update_env_origins(env_ids, move_up, move_down)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._update_terrain_curriculum(env_ids)
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time.
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Reset variables.
        self.last_actions[env_ids] = 0
        self.gait_indices[env_ids] = 0

        # Sample new targets.
        if self.target_state is None:
            self.target_state = torch.zeros(self.num_envs, 3, device=self.device)
        num_reset_envs = env_ids.shape[0]
        goal_range = self.cfg.goal_randomization_range
        goal_noise_x = sample_uniform(-goal_range, goal_range, (num_reset_envs,), self.device)
        goal_noise_y = sample_uniform(-goal_range, goal_range, (num_reset_envs,), self.device)
        self.target_state[env_ids, 0] = goal_noise_x
        self.target_state[env_ids, 1] = goal_noise_y
        self.target_state[env_ids, 2] = 0.0

        # Reset robot state.
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        for joint_idx in self._pendulum_dof_ids:
            joint_pos[:, joint_idx] += sample_uniform(
                self.cfg.pendulum_angle_min,
                self.cfg.pendulum_angle_max,
                joint_pos[:, joint_idx].shape,
                joint_pos.device,
            )

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._visualize_target_markers()

        # Reset failure counters for reset environments.
        if self.pendulum_failure_steps is None:
            self.pendulum_failure_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        else:
            self.pendulum_failure_steps[env_ids] = 0
        if self.position_failure_steps is None:
            self.position_failure_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        else:
            self.position_failure_steps[env_ids] = 0

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        if self._terrain.terrain_origins is not None:
            terrain_level = torch.mean(self._terrain.terrain_levels[env_ids].float()).item()
            extras["Episode_Curriculum/terrain_level"] = terrain_level
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.target_marker_cfg)
            if (
                self._height_scanner is not None
                and self.cfg.height_scan_debug_vis
                and not hasattr(self, "height_scan_visualizer")
            ):
                height_scan_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/HeightScan")
                height_scan_cfg.markers["hit"].radius = 0.01
                self.height_scan_visualizer = VisualizationMarkers(height_scan_cfg)
            # set their visibility to true
            if self.target_visualizer is not None:
                self.target_visualizer.set_visibility(True)
            if self._height_scanner is not None and self.cfg.height_scan_debug_vis:
                self.height_scan_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer") and self.target_visualizer is not None:
                self.target_visualizer.set_visibility(False)
            if hasattr(self, "height_scan_visualizer"):
                self.height_scan_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        if self._height_scanner is not None and self.cfg.height_scan_debug_vis:
            ray_hits_w = self._height_scanner.data.ray_hits_w
            self.height_scan_visualizer.visualize(translations=ray_hits_w.reshape(-1, 3))
        self._visualize_target_markers()

    def _visualize_target_markers(self) -> None:
        if self.target_state is None or self.target_visualizer is None:
            return
        if self._marker_locations is None:
            self._marker_up = self._marker_up.to(device=self.device)
            self._marker_locations = torch.zeros((self.num_envs, 3), device=self.device)
            self._marker_offset = torch.zeros((self.num_envs, 3), device=self.device)
            self._marker_offset[:, -1] = 1.0
            self._marker_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
        target_xy = self.target_state[:, :2]
        target_pos_world = target_xy + env_origins[:, :2]

        robot_pos_3d = self.robot.data.root_pos_w
        robot_pos_2d = robot_pos_3d[:, :2]

        direction_to_target = target_pos_world - robot_pos_2d
        direction_norm = torch.linalg.norm(direction_to_target, dim=-1, keepdim=True)
        direction_norm = torch.clamp(direction_norm, min=1e-6)
        direction_unit = direction_to_target / direction_norm

        yaw_angles = torch.atan2(direction_unit[:, 1], direction_unit[:, 0])
        self._marker_orientations = math_utils.quat_from_angle_axis(yaw_angles, self._marker_up)

        target_pos_3d = torch.zeros_like(robot_pos_3d)
        target_pos_3d[:, :2] = target_pos_world
        target_pos_3d[:, 2] = env_origins[:, 2]
        robot_pos_3d = robot_pos_3d.clone()
        robot_pos_3d[:, 2] = env_origins[:, 2]
        midpoint = (robot_pos_3d + target_pos_3d) / 2.0
        self._marker_locations = midpoint

        loc = self._marker_locations + self._marker_offset
        sphere_locations = torch.zeros_like(self._marker_locations)
        sphere_locations[:, :2] = target_pos_world
        sphere_locations[:, 2] = env_origins[:, 2]
        sphere_loc = sphere_locations + self._marker_offset

        sphere_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        sphere_orientations[:, 3] = 1.0

        all_locations = torch.vstack((loc, sphere_loc))
        all_orientations = torch.vstack((self._marker_orientations, sphere_orientations))

        all_envs = torch.arange(self.num_envs, device=self.device)
        arrow_indices = torch.zeros_like(all_envs)
        sphere_indices = torch.ones_like(all_envs)
        marker_indices = torch.hstack((arrow_indices, sphere_indices))

        self.target_visualizer.visualize(all_locations, all_orientations, marker_indices=marker_indices)

    @property
    def foot_positions_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _step_contact_targets(self):
        frequencies = 3.0
        phases = 0.5
        offsets = 0.0
        bounds = 0.0
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,
            self.gait_indices + offsets,
            self.gait_indices + bounds,
            self.gait_indices + phases,
        ]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                0.5 / (1 - durations[swing_idxs])
            )

        self.clock_inputs[:, 0] = torch.sin(2 * torch.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * torch.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * torch.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * torch.pi * foot_indices[3])

        # von mises distribution
        kappa = 0.07
        smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf

        smoothing_multiplier_FL = smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_FR = smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_RL = smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)
        )
        smoothing_multiplier_RR = smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)
        ) + smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)
        )

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = math_utils.quat_apply_yaw(
                math_utils.quat_conjugate(self.robot.data.root_quat_w), cur_footsteps_translated[:, i, :]
            )

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor(
            [
                desired_stance_width / 2,
                -desired_stance_width / 2,
                desired_stance_width / 2,
                -desired_stance_width / 2,
            ],
            device=self.device,
        ).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor(
            [
                desired_stance_length / 2,
                desired_stance_length / 2,
                -desired_stance_length / 2,
                -desired_stance_length / 2,
            ],
            device=self.device,
        ).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([3.0], device=self.device)
        x_vel_des = self.robot.data.root_lin_vel_b[:, 0:1]
        yaw_vel_des = self.robot.data.root_ang_vel_b[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward
