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
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.utils import DelayBuffer
from isaaclab.utils.math import sample_uniform

from .go2_pendulum_env_cfg import Go2PendulumEnvCfg


class Go2PendulumEnv(DirectRLEnv):
    cfg: Go2PendulumEnvCfg

    # Difficulty presets: values for each curriculum level, applied at runtime.
    _DIFFICULTY_PRESETS = {
        # Stand and balance only. No goals, tiny spawn perturbation, no pushes.
        1: dict(
            goal_randomization_dist_min=0.0,
            goal_randomization_dist_max=0.0,
            goal_yaw_randomization_min=0.0,
            goal_yaw_randomization_max=0.0,
            pendulum_angle_min=0.0,
            pendulum_angle_max=math.radians(5.0),
            pendulum_joint_limit_min_rad=math.radians(-90.0),
            pendulum_joint_limit_max_rad=math.radians(90.0),
            termination_grace_s=0.1,
            base_height_terminate_duration_s=10.0,
            pendulum_terminate_angle_rad=math.radians(60.0),
            pendulum_terminate_duration_s=0.5,
            position_tolerance=5.0,
            enable_external_wrench_push=False,
        ),
        # Learn to take small steps. Small goals, slightly larger spawn angle. No pushes.
        2: dict(
            goal_randomization_dist_min=0.0,
            goal_randomization_dist_max=0.15,
            goal_yaw_randomization_min=math.radians(-15),
            goal_yaw_randomization_max=math.radians(15),
            pendulum_angle_min=0.0,
            pendulum_angle_max=math.radians(5.0),
            pendulum_joint_limit_min_rad=math.radians(-90.0),
            pendulum_joint_limit_max_rad=math.radians(90.0),
            termination_grace_s=0.1,
            base_height_terminate_duration_s=10.0,
            pendulum_terminate_angle_rad=math.radians(60.0),
            pendulum_terminate_duration_s=0.5,
            position_tolerance=0.5,
            enable_external_wrench_push=False,
        ),
        # Walk further, turn more, and accept larger pendulum errors. Still no pushes.
        3: dict(
            goal_randomization_dist_min=0.1,
            goal_randomization_dist_max=0.3,
            goal_yaw_randomization_min=math.radians(-30),
            goal_yaw_randomization_max=math.radians(30),
            pendulum_angle_min=0.0,
            pendulum_angle_max=math.radians(5.0),
            pendulum_joint_limit_min_rad=math.radians(-90.0),
            pendulum_joint_limit_max_rad=math.radians(90.0),
            termination_grace_s=0.1,
            base_height_terminate_duration_s=10.0,
            pendulum_terminate_angle_rad=math.radians(45.0),
            pendulum_terminate_duration_s=0.5,
            position_tolerance=0.3,
            enable_external_wrench_push=False,
        ),
        # Full goal range and full spawn angle. Introduce gentle pushes. Tighter termination.
        4: dict(
            goal_randomization_dist_min=0.2,
            goal_randomization_dist_max=0.5,
            goal_yaw_randomization_min=math.radians(-45),
            goal_yaw_randomization_max=math.radians(45),
            pendulum_angle_min=0.0,
            pendulum_angle_max=math.radians(5.0),
            pendulum_joint_limit_min_rad=math.radians(-90.0),
            pendulum_joint_limit_max_rad=math.radians(90.0),
            termination_grace_s=0.1,
            base_height_terminate_duration_s=10.0,
            pendulum_terminate_angle_rad=math.radians(30.0),
            pendulum_terminate_duration_s=0.5,
            position_tolerance=0.2,
            enable_external_wrench_push=True,
            push_force_x_range=(-5.0, 5.0),
            push_force_y_range=(-5.0, 5.0),
        ),
        # Full difficulty. Stronger pushes and tightest termination.
        5: dict(
            goal_randomization_dist_min=0.3,
            goal_randomization_dist_max=0.5,
            goal_yaw_randomization_min=math.radians(-60),
            goal_yaw_randomization_max=math.radians(60),
            pendulum_angle_min=0.0,
            pendulum_angle_max=math.radians(5.0),
            pendulum_joint_limit_min_rad=math.radians(-90.0),
            pendulum_joint_limit_max_rad=math.radians(90.0),
            termination_grace_s=0.1,
            base_height_terminate_duration_s=10.0,
            pendulum_terminate_angle_rad=math.radians(15.0),
            pendulum_terminate_duration_s=5.0,
            position_tolerance=0.2,
            enable_external_wrench_push=True,
            push_force_x_range=(-10.0, 10.0),
            push_force_y_range=(-10.0, 10.0),
        ),
    }

    def __init__(self, cfg: Go2PendulumEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._current_difficulty_level = 1

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

        # Resolve leg joints.
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
        self._action_dim = gym.spaces.flatdim(self.single_action_space)
        if self.cfg.action_scale <= 0.0:
            raise ValueError(f"action_scale must be > 0. Got {self.cfg.action_scale}.")
        if self.cfg.action_delay_steps_min < 0:
            raise ValueError(f"action_delay_steps_min must be >= 0. Got {self.cfg.action_delay_steps_min}.")
        if self.cfg.action_delay_steps_max < self.cfg.action_delay_steps_min:
            raise ValueError(
                "action_delay_steps_max must be >= action_delay_steps_min. "
                f"Got {self.cfg.action_delay_steps_max} < {self.cfg.action_delay_steps_min}."
            )
        self._validate_domain_randomization_cfg()
        self._max_action_delay_steps = self.cfg.action_delay_steps_max if self.cfg.enable_action_delay else 0
        seed_cfg = getattr(self.cfg, "seed", None)
        seed = 0 if seed_cfg is None else int(seed_cfg)
        self._dr_rng = torch.Generator(device="cpu")
        self._dr_rng.manual_seed(seed + int(self.cfg.dr_seed_offset))

        self._pendulum_dof_count = len(self.cfg.pendulum_joint_names)
        if self.cfg.use_pendulum:
            self._pendulum_dof_ids = []
            for joint_name in self.cfg.pendulum_joint_names:
                joint_idx, _ = self.robot.find_joints(joint_name)
                if len(joint_idx) != 1:
                    raise RuntimeError(f"Expected exactly one joint for '{joint_name}', got {joint_idx}.")
                self._pendulum_dof_ids.append(joint_idx[0])
            self._pendulum_dof_ids = torch.tensor(self._pendulum_dof_ids, device=self.device, dtype=torch.long)
            pendulum_ee_body_ids, _ = self.robot.find_bodies("pendulum_ee")
            if len(pendulum_ee_body_ids) != 1:
                raise RuntimeError(f"Expected exactly one body for 'pendulum_ee', got {pendulum_ee_body_ids}.")
            self._pendulum_ee_body_id = pendulum_ee_body_ids[0]
        else:
            self._pendulum_dof_ids = torch.tensor([], device=self.device, dtype=torch.long)
            self._pendulum_ee_body_id = None

        self._apply_pendulum_joint_limits()

        if self.cfg.enable_curriculum:
            self._apply_difficulty_preset(1)

        if self.cfg.difficulty_override >= 1:
            self.cfg.enable_curriculum = False
            self._current_difficulty_level = self.cfg.difficulty_override
            self._apply_difficulty_preset(self.cfg.difficulty_override)

        leg_default_joint_pos = self.robot.data.default_joint_pos[:, self._leg_dof_ids]

        # Joint position command from delayed policy offsets relative to default joint positions.
        self._actions_raw_policy = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._actions_delayed = torch.zeros_like(self._actions_raw_policy)
        self._actions_executed = torch.zeros_like(self._actions_raw_policy)
        self.desired_joint_pos = leg_default_joint_pos.clone()
        self._action_delay_buffer = torch.zeros(
            self.num_envs,
            self._action_dim,
            self._max_action_delay_steps + 1,
            device=self.device,
        )
        self._action_delay_steps = torch.full(
            (self.num_envs,),
            self.cfg.action_delay_steps_min,
            device=self.device,
            dtype=torch.long,
        )
        if self.cfg.enable_action_delay and self.cfg.action_delay_randomize_per_reset:
            self._action_delay_steps = torch.randint(
                self.cfg.action_delay_steps_min,
                self.cfg.action_delay_steps_max + 1,
                (self.num_envs,),
                device=self.device,
                dtype=torch.long,
            )
        elif self.cfg.enable_action_delay:
            self._action_delay_steps.fill_(self.cfg.action_delay_steps_max)

        # Target state [x_d, y_d, yaw_d] in environment frame.
        # x/y come from target distance + bearing; yaw is the desired robot heading at the target.
        self.target_state = None

        # Marker visualization buffers.
        self._marker_orientations = None
        self._marker_locations = None
        self._marker_up = torch.tensor([0.0, 0.0, 1.0])
        self._world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        # Logging.
        episode_sum_keys = [
            "position_tracking",
            "progress",
            "yaw_alignment",
            "pendulum_upright",
            "pendulum_velocity",
            "balanced_movement",
            "action_magnitude",
            "rew_action_rate",
            "torque",
            "orient",
            "base_height",
            "lin_vel_z",
            "dof_vel",
            "dof_acc",
            "ang_vel_xy",
            "feet_clearance",
            "feet_air_time",
            "tracking_contacts_shaped_force",
            "undesired_contacts",
            "termination_penalty",
        ]
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in episode_sum_keys
        }
        self._episode_base_height_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_base_height_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_base_tilt_deg_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_base_tilt_deg_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_pendulum_angle_deg_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_pendulum_angle_deg_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_pendulum_speed_deg_s_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_pendulum_speed_deg_s_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_position_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.last_actions = torch.zeros(
            self.num_envs,
            self._action_dim,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        # Get specific body indices.
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        undesired_contact_ids, _ = self._contact_sensor.find_bodies(".*_thigh")
        self._undesired_contact_body_ids = (
            torch.tensor(undesired_contact_ids, device=self.device, dtype=torch.long)
            if len(undesired_contact_ids) > 0
            else None
        )
        self._init_domain_randomization_state()

        # Track termination causes for accurate logging.
        self._base_contact_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._base_height_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._base_tilt_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._pendulum_contact_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._pendulum_angle_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._position_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._base_height_failure_steps = None
        self._pendulum_angle_failure_steps = None
        self._position_failure_steps = None
        self._steps_since_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _apply_pendulum_joint_limits(self) -> None:
        """Write difficulty-dependent hard limits for pendulum joints across all envs."""
        if not self.cfg.use_pendulum or self._pendulum_dof_ids.numel() == 0:
            return

        num_pendulum_joints = self._pendulum_dof_ids.numel()
        limits = torch.zeros((self.num_envs, num_pendulum_joints, 2), device=self.device, dtype=torch.float)
        limits[:, :, 0] = float(self.cfg.pendulum_joint_limit_min_rad)
        limits[:, :, 1] = float(self.cfg.pendulum_joint_limit_max_rad)
        self.robot.write_joint_position_limit_to_sim(
            limits,
            joint_ids=self._pendulum_dof_ids,
            warn_limit_violation=False,
        )

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

        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # create target visualizer after scene is set up
        self.target_visualizer = VisualizationMarkers(self.cfg.target_marker_cfg)

    def _validate_domain_randomization_cfg(self) -> None:
        if self.cfg.obs_delay_steps_min < 0:
            raise ValueError(f"obs_delay_steps_min must be >= 0. Got {self.cfg.obs_delay_steps_min}.")
        if self.cfg.obs_delay_steps_max < self.cfg.obs_delay_steps_min:
            raise ValueError(
                "obs_delay_steps_max must be >= obs_delay_steps_min. "
                f"Got {self.cfg.obs_delay_steps_max} < {self.cfg.obs_delay_steps_min}."
            )
        if self.cfg.obs_delay_jitter_extra_max < 0:
            raise ValueError(f"obs_delay_jitter_extra_max must be >= 0. Got {self.cfg.obs_delay_jitter_extra_max}.")
        if not self.cfg.obs_delay_proprio_only:
            raise ValueError("Only proprioceptive observation delay is supported for this environment.")
        if not 0.0 <= self.cfg.obs_delay_jitter_prob <= 1.0:
            raise ValueError(f"obs_delay_jitter_prob must be in [0, 1]. Got {self.cfg.obs_delay_jitter_prob}.")
        if not 0.0 <= self.cfg.material_randomization_prob <= 1.0:
            raise ValueError(
                f"material_randomization_prob must be in [0, 1]. Got {self.cfg.material_randomization_prob}."
            )
        if self.cfg.material_num_buckets <= 0:
            raise ValueError(f"material_num_buckets must be > 0. Got {self.cfg.material_num_buckets}.")
        self._validate_range("material_static_friction_range", self.cfg.material_static_friction_range)
        self._validate_range("material_dynamic_friction_range", self.cfg.material_dynamic_friction_range)
        self._validate_range("material_restitution_range", self.cfg.material_restitution_range)
        self._validate_range("mass_scale_range", self.cfg.mass_scale_range)
        if self.cfg.mass_scale_range[0] <= 0.0:
            raise ValueError(f"mass_scale_range min must be > 0. Got {self.cfg.mass_scale_range[0]}.")
        self._validate_range("com_offset_x_range", self.cfg.com_offset_x_range)
        self._validate_range("com_offset_y_range", self.cfg.com_offset_y_range)
        self._validate_range("com_offset_z_range", self.cfg.com_offset_z_range)
        self._validate_range("motor_stiffness_scale_range", self.cfg.motor_stiffness_scale_range)
        self._validate_range("motor_damping_scale_range", self.cfg.motor_damping_scale_range)
        if self.cfg.motor_stiffness_scale_range[0] <= 0.0:
            raise ValueError(
                f"motor_stiffness_scale_range min must be > 0. Got {self.cfg.motor_stiffness_scale_range[0]}."
            )
        if self.cfg.motor_damping_scale_range[0] <= 0.0:
            raise ValueError(
                f"motor_damping_scale_range min must be > 0. Got {self.cfg.motor_damping_scale_range[0]}."
            )
        self._validate_range("push_interval_s", (self.cfg.push_interval_s_min, self.cfg.push_interval_s_max))
        self._validate_range("push_duration_s", (self.cfg.push_duration_s_min, self.cfg.push_duration_s_max))
        self._validate_range("push_force_x_range", self.cfg.push_force_x_range)
        self._validate_range("push_force_y_range", self.cfg.push_force_y_range)
        self._validate_range("push_force_z_range", self.cfg.push_force_z_range)
        self._validate_range("push_torque_x_range", self.cfg.push_torque_x_range)
        self._validate_range("push_torque_y_range", self.cfg.push_torque_y_range)
        self._validate_range("push_torque_z_range", self.cfg.push_torque_z_range)

    @staticmethod
    def _validate_range(name: str, value_range: tuple[float, float]) -> None:
        if value_range[1] < value_range[0]:
            raise ValueError(f"{name} max must be >= min. Got {value_range[1]} < {value_range[0]}.")

    def _sample_uniform_cpu(self, value_range: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        low, high = value_range
        if high == low:
            return torch.full(shape, float(low), dtype=torch.float, device="cpu")
        return low + (high - low) * torch.rand(shape, generator=self._dr_rng, device="cpu")

    def _sample_uniform_device(self, value_range: tuple[float, float], shape: tuple[int, ...], device: str) -> torch.Tensor:
        return self._sample_uniform_cpu(value_range, shape).to(device=device)

    def _seconds_to_steps(self, seconds: float) -> int:
        return max(1, math.ceil(seconds / self.step_dt))

    def _init_domain_randomization_state(self) -> None:
        self._dr_all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # Material randomization state.
        self._material_buckets = None
        if self.cfg.enable_domain_randomization and self.cfg.enable_material_randomization:
            static_samples = self._sample_uniform_cpu(
                self.cfg.material_static_friction_range, (self.cfg.material_num_buckets,)
            )
            dynamic_samples = self._sample_uniform_cpu(
                self.cfg.material_dynamic_friction_range, (self.cfg.material_num_buckets,)
            )
            restitution_samples = self._sample_uniform_cpu(
                self.cfg.material_restitution_range, (self.cfg.material_num_buckets,)
            )
            if self.cfg.material_make_consistent:
                dynamic_samples = torch.minimum(dynamic_samples, static_samples)
            self._material_buckets = torch.stack((static_samples, dynamic_samples, restitution_samples), dim=-1)

        # Mass / COM randomization state.
        self._mass_body_ids_cpu = torch.tensor([], dtype=torch.long, device="cpu")
        if self.cfg.enable_domain_randomization and (self.cfg.enable_mass_randomization or self.cfg.enable_com_randomization):
            mass_body_ids, _ = self.robot.find_bodies(self.cfg.mass_randomize_body_name)
            if len(mass_body_ids) == 0:
                raise RuntimeError(
                    f"Could not resolve mass/com randomization body '{self.cfg.mass_randomize_body_name}'."
                )
            self._mass_body_ids_cpu = torch.tensor(mass_body_ids, dtype=torch.long, device="cpu")
        self._default_masses_cpu = self.robot.root_physx_view.get_masses().clone()
        self._default_inertias_cpu = self.robot.root_physx_view.get_inertias().clone()
        self._default_coms_cpu = self.robot.root_physx_view.get_coms().clone()

        # Motor gain randomization state.
        self._motor_actuator = None
        self._motor_default_stiffness = None
        self._motor_default_damping = None
        self._motor_num_joints = 0
        if self.cfg.enable_domain_randomization and self.cfg.enable_motor_gain_randomization:
            if self.cfg.motor_gain_actuator_name not in self.robot.actuators:
                raise RuntimeError(
                    f"Motor gain actuator '{self.cfg.motor_gain_actuator_name}' not found. "
                    f"Available: {list(self.robot.actuators.keys())}"
                )
            self._motor_actuator = self.robot.actuators[self.cfg.motor_gain_actuator_name]
            self._motor_default_stiffness = self._motor_actuator.stiffness.clone()
            self._motor_default_damping = self._motor_actuator.damping.clone()
            self._motor_num_joints = self._motor_default_stiffness.shape[1]

        # Observation delay state.
        self._max_obs_delay_steps = (
            self.cfg.obs_delay_steps_max + self.cfg.obs_delay_jitter_extra_max
            if self.cfg.enable_domain_randomization and self.cfg.enable_obs_delay
            else 0
        )
        # body_lin_vel(3) + body_ang_vel(3) + projected_gravity(3) + leg_pos(12) + leg_vel(12)
        # + pendulum_pos(P) + pendulum_vel(P) + actions(12)
        self._proprio_obs_dim = (
            3
            + 3
            + 3
            + self._action_dim
            + self._action_dim
            + self._pendulum_dof_count
            + self._pendulum_dof_count
            + self._action_dim
        )
        self._obs_delay_buffer = DelayBuffer(self._max_obs_delay_steps, self.num_envs, self.device)
        self._obs_delay_steps = torch.full(
            (self.num_envs,),
            self.cfg.obs_delay_steps_min,
            device=self.device,
            dtype=torch.long,
        )
        if self.cfg.enable_domain_randomization and self.cfg.enable_obs_delay:
            if self.cfg.obs_delay_randomize_per_reset:
                self._obs_delay_steps = torch.randint(
                    self.cfg.obs_delay_steps_min,
                    self.cfg.obs_delay_steps_max + 1,
                    (self.num_envs,),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                self._obs_delay_steps.fill_(self.cfg.obs_delay_steps_max)
            self._obs_delay_buffer.set_time_lag(self._obs_delay_steps)

        # Sensor bias / drift state.
        self._bias_body_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._bias_body_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._bias_projected_gravity = torch.zeros((self.num_envs, 3), device=self.device)
        self._bias_leg_joint_pos = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._bias_leg_joint_vel = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._bias_pendulum_joint_pos = torch.zeros((self.num_envs, self._pendulum_dof_count), device=self.device)
        self._bias_pendulum_joint_vel = torch.zeros((self.num_envs, self._pendulum_dof_count), device=self.device)
        self._sample_sensor_biases(self._dr_all_env_ids)

        # External wrench push state.
        self._push_body_ids = torch.tensor([], dtype=torch.long, device=self.device)
        self._push_num_bodies = 0
        if self.cfg.enable_domain_randomization and self.cfg.enable_external_wrench_push:
            push_body_ids, _ = self.robot.find_bodies(self.cfg.push_body_name)
            if len(push_body_ids) == 0:
                raise RuntimeError(f"Could not resolve push body '{self.cfg.push_body_name}'.")
            self._push_body_ids = torch.tensor(push_body_ids, dtype=torch.long, device=self.device)
            self._push_num_bodies = len(push_body_ids)
        self._push_forces = torch.zeros((self.num_envs, max(1, self._push_num_bodies), 3), device=self.device)
        self._push_torques = torch.zeros_like(self._push_forces)
        self._push_next_step = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._push_end_step = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self._schedule_next_push(self._dr_all_env_ids, torch.zeros(self.num_envs, device=self.device, dtype=torch.long))

    def _randomize_materials(self, env_ids: torch.Tensor) -> None:
        if not (self.cfg.enable_domain_randomization and self.cfg.enable_material_randomization):
            return
        if not self.cfg.material_randomize_on_reset or self._material_buckets is None:
            return
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return
        if self.cfg.material_randomization_prob < 1.0:
            mask = torch.rand((env_ids_cpu.numel(),), generator=self._dr_rng, device="cpu") < self.cfg.material_randomization_prob
            env_ids_cpu = env_ids_cpu[mask]
            if env_ids_cpu.numel() == 0:
                return
        total_num_shapes = self.robot.root_physx_view.max_shapes
        bucket_ids = torch.randint(
            0,
            self.cfg.material_num_buckets,
            (env_ids_cpu.numel(), total_num_shapes),
            generator=self._dr_rng,
            device="cpu",
        )
        material_samples = self._material_buckets[bucket_ids]
        materials = self.robot.root_physx_view.get_material_properties()
        materials[env_ids_cpu] = material_samples
        self.robot.root_physx_view.set_material_properties(materials, env_ids_cpu)

    def _randomize_mass_and_com(self, env_ids: torch.Tensor) -> None:
        if not self.cfg.enable_domain_randomization:
            return
        if self._mass_body_ids_cpu.numel() == 0:
            return
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return
        body_ids_cpu = self._mass_body_ids_cpu

        if self.cfg.enable_mass_randomization:
            masses = self.robot.root_physx_view.get_masses()
            masses[env_ids_cpu[:, None], body_ids_cpu] = self._default_masses_cpu[env_ids_cpu[:, None], body_ids_cpu]
            mass_scales = self._sample_uniform_cpu(self.cfg.mass_scale_range, (env_ids_cpu.numel(), 1))
            masses[env_ids_cpu[:, None], body_ids_cpu] *= mass_scales
            self.robot.root_physx_view.set_masses(masses, env_ids_cpu)

            if self.cfg.mass_recompute_inertia:
                inertias = self.robot.root_physx_view.get_inertias()
                inertias[env_ids_cpu[:, None], body_ids_cpu] = (
                    self._default_inertias_cpu[env_ids_cpu[:, None], body_ids_cpu] * mass_scales.unsqueeze(-1)
                )
                self.robot.root_physx_view.set_inertias(inertias, env_ids_cpu)

        if self.cfg.enable_com_randomization:
            coms = self.robot.root_physx_view.get_coms().clone()
            coms[env_ids_cpu[:, None], body_ids_cpu] = self._default_coms_cpu[env_ids_cpu[:, None], body_ids_cpu]
            com_offsets = torch.zeros((env_ids_cpu.numel(), 1, 3), device="cpu")
            com_offsets[:, :, 0] = self._sample_uniform_cpu(self.cfg.com_offset_x_range, (env_ids_cpu.numel(), 1))
            com_offsets[:, :, 1] = self._sample_uniform_cpu(self.cfg.com_offset_y_range, (env_ids_cpu.numel(), 1))
            com_offsets[:, :, 2] = self._sample_uniform_cpu(self.cfg.com_offset_z_range, (env_ids_cpu.numel(), 1))
            coms[env_ids_cpu[:, None], body_ids_cpu, :3] += com_offsets
            self.robot.root_physx_view.set_coms(coms, env_ids_cpu)

    def _randomize_motor_gains(self, env_ids: torch.Tensor) -> None:
        if not (self.cfg.enable_domain_randomization and self.cfg.enable_motor_gain_randomization):
            return
        if self._motor_actuator is None:
            return
        self._motor_actuator.stiffness[env_ids] = self._motor_default_stiffness[env_ids]
        self._motor_actuator.damping[env_ids] = self._motor_default_damping[env_ids]
        num_envs = env_ids.numel()
        if num_envs == 0:
            return
        if self.cfg.motor_gain_per_joint:
            gain_shape = (num_envs, self._motor_num_joints)
        else:
            gain_shape = (num_envs, 1)
        stiffness_scale = self._sample_uniform_device(self.cfg.motor_stiffness_scale_range, gain_shape, self.device)
        damping_scale = self._sample_uniform_device(self.cfg.motor_damping_scale_range, gain_shape, self.device)
        self._motor_actuator.stiffness[env_ids] *= stiffness_scale
        self._motor_actuator.damping[env_ids] *= damping_scale

    def _sample_sensor_biases(self, env_ids: torch.Tensor) -> None:
        if not (self.cfg.enable_domain_randomization and self.cfg.enable_sensor_bias_drift):
            return
        num_envs = env_ids.numel()
        if num_envs == 0:
            return
        self._bias_body_lin_vel[env_ids] = self._sample_uniform_device(
            (-self.cfg.imu_lin_vel_bias_range, self.cfg.imu_lin_vel_bias_range), (num_envs, 3), self.device
        )
        self._bias_body_ang_vel[env_ids] = self._sample_uniform_device(
            (-self.cfg.imu_ang_vel_bias_range, self.cfg.imu_ang_vel_bias_range), (num_envs, 3), self.device
        )
        self._bias_projected_gravity[env_ids] = self._sample_uniform_device(
            (-self.cfg.imu_gravity_bias_range, self.cfg.imu_gravity_bias_range), (num_envs, 3), self.device
        )
        self._bias_leg_joint_pos[env_ids] = self._sample_uniform_device(
            (-self.cfg.encoder_joint_pos_bias_range, self.cfg.encoder_joint_pos_bias_range),
            (num_envs, self._action_dim),
            self.device,
        )
        self._bias_leg_joint_vel[env_ids] = self._sample_uniform_device(
            (-self.cfg.encoder_joint_vel_bias_range, self.cfg.encoder_joint_vel_bias_range),
            (num_envs, self._action_dim),
            self.device,
        )
        if self._pendulum_dof_count > 0:
            self._bias_pendulum_joint_pos[env_ids] = self._sample_uniform_device(
                (-self.cfg.encoder_pendulum_pos_bias_range, self.cfg.encoder_pendulum_pos_bias_range),
                (num_envs, self._pendulum_dof_count),
                self.device,
            )
            self._bias_pendulum_joint_vel[env_ids] = self._sample_uniform_device(
                (-self.cfg.encoder_pendulum_vel_bias_range, self.cfg.encoder_pendulum_vel_bias_range),
                (num_envs, self._pendulum_dof_count),
                self.device,
            )

    def _update_sensor_bias_drift(self) -> None:
        if not (self.cfg.enable_domain_randomization and self.cfg.enable_sensor_bias_drift):
            return
        drift_scale = math.sqrt(self.step_dt)
        self._bias_body_lin_vel += torch.randn_like(self._bias_body_lin_vel) * (
            self.cfg.imu_lin_vel_drift_std_per_s * drift_scale
        )
        self._bias_body_ang_vel += torch.randn_like(self._bias_body_ang_vel) * (
            self.cfg.imu_ang_vel_drift_std_per_s * drift_scale
        )
        self._bias_projected_gravity += torch.randn_like(self._bias_projected_gravity) * (
            self.cfg.imu_gravity_drift_std_per_s * drift_scale
        )
        self._bias_leg_joint_pos += torch.randn_like(self._bias_leg_joint_pos) * (
            self.cfg.encoder_joint_pos_drift_std_per_s * drift_scale
        )
        self._bias_leg_joint_vel += torch.randn_like(self._bias_leg_joint_vel) * (
            self.cfg.encoder_joint_vel_drift_std_per_s * drift_scale
        )
        if self._pendulum_dof_count > 0:
            self._bias_pendulum_joint_pos += torch.randn_like(self._bias_pendulum_joint_pos) * (
                self.cfg.encoder_pendulum_pos_drift_std_per_s * drift_scale
            )
            self._bias_pendulum_joint_vel += torch.randn_like(self._bias_pendulum_joint_vel) * (
                self.cfg.encoder_pendulum_vel_drift_std_per_s * drift_scale
            )

        self._bias_body_lin_vel = torch.clamp(
            self._bias_body_lin_vel, -self.cfg.imu_lin_vel_bias_range, self.cfg.imu_lin_vel_bias_range
        )
        self._bias_body_ang_vel = torch.clamp(
            self._bias_body_ang_vel, -self.cfg.imu_ang_vel_bias_range, self.cfg.imu_ang_vel_bias_range
        )
        self._bias_projected_gravity = torch.clamp(
            self._bias_projected_gravity, -self.cfg.imu_gravity_bias_range, self.cfg.imu_gravity_bias_range
        )
        self._bias_leg_joint_pos = torch.clamp(
            self._bias_leg_joint_pos, -self.cfg.encoder_joint_pos_bias_range, self.cfg.encoder_joint_pos_bias_range
        )
        self._bias_leg_joint_vel = torch.clamp(
            self._bias_leg_joint_vel, -self.cfg.encoder_joint_vel_bias_range, self.cfg.encoder_joint_vel_bias_range
        )
        if self._pendulum_dof_count > 0:
            self._bias_pendulum_joint_pos = torch.clamp(
                self._bias_pendulum_joint_pos,
                -self.cfg.encoder_pendulum_pos_bias_range,
                self.cfg.encoder_pendulum_pos_bias_range,
            )
            self._bias_pendulum_joint_vel = torch.clamp(
                self._bias_pendulum_joint_vel,
                -self.cfg.encoder_pendulum_vel_bias_range,
                self.cfg.encoder_pendulum_vel_bias_range,
            )

    def _apply_obs_delay_and_jitter(self, proprio_obs: torch.Tensor) -> torch.Tensor:
        if not (self.cfg.enable_domain_randomization and self.cfg.enable_obs_delay):
            return proprio_obs
        effective_delay = self._obs_delay_steps.clone()
        if self.cfg.obs_delay_jitter_extra_max > 0 and self.cfg.obs_delay_jitter_prob > 0.0:
            jitter_mask = torch.rand((self.num_envs,), device=self.device) < self.cfg.obs_delay_jitter_prob
            jitter_extra = torch.randint(
                0,
                self.cfg.obs_delay_jitter_extra_max + 1,
                (self.num_envs,),
                device=self.device,
                dtype=torch.long,
            )
            effective_delay = effective_delay + jitter_mask.long() * jitter_extra
        effective_delay = torch.clamp(effective_delay, 0, self._max_obs_delay_steps)
        self._obs_delay_buffer.set_time_lag(effective_delay)
        return self._obs_delay_buffer.compute(proprio_obs)

    def _schedule_next_push(self, env_ids: torch.Tensor, now_step: torch.Tensor) -> None:
        if not (self.cfg.enable_domain_randomization and self.cfg.enable_external_wrench_push):
            return
        if env_ids.numel() == 0:
            return
        min_interval_steps = self._seconds_to_steps(self.cfg.push_interval_s_min)
        max_interval_steps = self._seconds_to_steps(self.cfg.push_interval_s_max)
        interval_steps = torch.randint(
            min_interval_steps,
            max_interval_steps + 1,
            (env_ids.numel(),),
            device=self.device,
            dtype=torch.long,
        )
        self._push_next_step[env_ids] = now_step + interval_steps

    def _update_external_wrench_pushes(self) -> None:
        if not (self.cfg.enable_domain_randomization and self.cfg.enable_external_wrench_push):
            return
        if self._push_num_bodies == 0:
            return

        now_step = self._steps_since_reset
        push_finished = (self._push_end_step > 0) & (now_step >= self._push_end_step)
        if torch.any(push_finished):
            self._push_forces[push_finished] = 0.0
            self._push_torques[push_finished] = 0.0
            self._push_end_step[push_finished] = 0

        start_push = (self._push_end_step == 0) & (now_step >= self._push_next_step)
        if torch.any(start_push):
            env_ids = torch.nonzero(start_push, as_tuple=False).squeeze(-1)
            min_duration_steps = self._seconds_to_steps(self.cfg.push_duration_s_min)
            max_duration_steps = self._seconds_to_steps(self.cfg.push_duration_s_max)
            duration_steps = torch.randint(
                min_duration_steps,
                max_duration_steps + 1,
                (env_ids.numel(),),
                device=self.device,
                dtype=torch.long,
            )
            push_forces = torch.zeros((env_ids.numel(), self._push_num_bodies, 3), device=self.device)
            push_torques = torch.zeros_like(push_forces)
            push_forces[:, :, 0] = self._sample_uniform_device(
                self.cfg.push_force_x_range, (env_ids.numel(), self._push_num_bodies), self.device
            )
            push_forces[:, :, 1] = self._sample_uniform_device(
                self.cfg.push_force_y_range, (env_ids.numel(), self._push_num_bodies), self.device
            )
            push_forces[:, :, 2] = self._sample_uniform_device(
                self.cfg.push_force_z_range, (env_ids.numel(), self._push_num_bodies), self.device
            )
            push_torques[:, :, 0] = self._sample_uniform_device(
                self.cfg.push_torque_x_range, (env_ids.numel(), self._push_num_bodies), self.device
            )
            push_torques[:, :, 1] = self._sample_uniform_device(
                self.cfg.push_torque_y_range, (env_ids.numel(), self._push_num_bodies), self.device
            )
            push_torques[:, :, 2] = self._sample_uniform_device(
                self.cfg.push_torque_z_range, (env_ids.numel(), self._push_num_bodies), self.device
            )
            self._push_forces[env_ids] = push_forces
            self._push_torques[env_ids] = push_torques
            self._push_end_step[env_ids] = now_step[env_ids] + duration_steps
            self._schedule_next_push(env_ids, self._push_end_step[env_ids])

        self.robot.set_external_force_and_torque(
            forces=self._push_forces[:, : self._push_num_bodies, :],
            torques=self._push_torques[:, : self._push_num_bodies, :],
            body_ids=self._push_body_ids,
            is_global=self.cfg.push_is_global,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._update_external_wrench_pushes()
        self._actions_raw_policy = actions.clone()

        if self.cfg.enable_action_delay:
            self._action_delay_buffer = torch.roll(self._action_delay_buffer, shifts=1, dims=2)
            self._action_delay_buffer[:, :, 0] = self._actions_raw_policy
            delay_idx = self._action_delay_steps.clamp(max=self._max_action_delay_steps).view(self.num_envs, 1, 1)
            delay_idx = delay_idx.expand(-1, self._action_dim, 1)
            self._actions_delayed = torch.gather(self._action_delay_buffer, dim=2, index=delay_idx).squeeze(-1)
        else:
            self._actions_delayed = self._actions_raw_policy.clone()

        self._actions_executed = self._actions_delayed.clone()
        self.desired_joint_pos = self.robot.data.default_joint_pos[:, self._leg_dof_ids] + (
            self.cfg.action_scale * self._actions_executed
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.desired_joint_pos,
            joint_ids=self._leg_dof_ids,
        )

    def _update_curriculum(self) -> None:
        """Update difficulty level based on training progress."""
        if not self.cfg.enable_curriculum or self.cfg.curriculum_total_steps <= 0:
            return
        progress = min(1.0, max(0.0, self.common_step_counter / self.cfg.curriculum_total_steps))

        # Difficulty curriculum: evenly split into 5 levels.
        if progress < 1 / 5:
            new_level = 1
        elif progress < 2 / 5:
            new_level = 2
        elif progress < 3 / 5:
            new_level = 3
        elif progress < 4 / 5:
            new_level = 4
        else:
            new_level = 5

        if new_level != self._current_difficulty_level:
            self._current_difficulty_level = new_level
            self._apply_difficulty_preset(new_level)

    def _apply_difficulty_preset(self, level: int) -> None:
        """Apply difficulty preset values to the config and update physics sim."""
        preset = self._DIFFICULTY_PRESETS[level]
        for key, value in preset.items():
            setattr(self.cfg, key, value)
        # Pendulum joint limits must be written to the physics sim.
        self._apply_pendulum_joint_limits()
        print(f"[Curriculum] Switched to difficulty level {level} at step {self.common_step_counter}")

    def _get_observations(self) -> dict:
        self._update_curriculum()

        leg_joint_pos_raw = (
            self.robot.data.joint_pos[:, self._leg_dof_ids] - self.robot.data.default_joint_pos[:, self._leg_dof_ids]
        )
        leg_joint_vel_raw = self.robot.data.joint_vel[:, self._leg_dof_ids]
        if self.cfg.use_pendulum and self._pendulum_dof_ids.numel() > 0:
            pendulum_joint_pos_raw = self.robot.data.joint_pos[:, self._pendulum_dof_ids]
            pendulum_joint_vel_raw = self.robot.data.joint_vel[:, self._pendulum_dof_ids]
        else:
            pendulum_joint_pos_raw = torch.zeros(
                self.num_envs,
                self._pendulum_dof_count,
                device=self.device,
                dtype=leg_joint_pos_raw.dtype,
            )
            pendulum_joint_vel_raw = torch.zeros_like(pendulum_joint_pos_raw)

        env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
        base_pos_xy = self.robot.data.root_pos_w[:, :2] - env_origins[:, :2]
        _, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)

        if self.target_state is not None:
            target_xy = self.target_state[:, :2]
            target_yaw = self.target_state[:, 2]
        else:
            target_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=leg_joint_pos_raw.dtype)
            target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=leg_joint_pos_raw.dtype)

        position_error_xy_world = target_xy - base_pos_xy
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        position_error_xy = torch.stack(
            (
                cos_yaw * position_error_xy_world[:, 0] + sin_yaw * position_error_xy_world[:, 1],
                -sin_yaw * position_error_xy_world[:, 0] + cos_yaw * position_error_xy_world[:, 1],
            ),
            dim=-1,
        )
        yaw_error = math_utils.wrap_to_pi(target_yaw - yaw)

        # Ground-truth (clean) quantities for critic.
        critic_body_lin_vel_b = self.robot.data.root_lin_vel_b.clone()
        critic_body_ang_vel_b = self.robot.data.root_ang_vel_b.clone()
        critic_projected_gravity_b = self.robot.data.projected_gravity_b.clone()
        critic_leg_joint_pos = leg_joint_pos_raw.clone()
        critic_leg_joint_vel = leg_joint_vel_raw.clone()
        critic_pendulum_joint_pos = pendulum_joint_pos_raw.clone()
        critic_pendulum_joint_vel = pendulum_joint_vel_raw.clone()

        # Actor (noisy) quantities — start from ground truth, apply noise.
        actor_body_lin_vel_b = critic_body_lin_vel_b.clone()
        actor_body_ang_vel_b = critic_body_ang_vel_b.clone()
        actor_projected_gravity_b = critic_projected_gravity_b.clone()
        actor_leg_joint_pos = leg_joint_pos_raw.clone()
        actor_leg_joint_vel = leg_joint_vel_raw.clone()
        actor_pendulum_joint_pos = pendulum_joint_pos_raw.clone()
        actor_pendulum_joint_vel = pendulum_joint_vel_raw.clone()

        # Apply sensor bias/drift (if DR enabled).
        self._update_sensor_bias_drift()
        if self.cfg.enable_domain_randomization and self.cfg.enable_sensor_bias_drift:
            actor_body_lin_vel_b = actor_body_lin_vel_b + self._bias_body_lin_vel
            actor_body_ang_vel_b = actor_body_ang_vel_b + self._bias_body_ang_vel
            actor_projected_gravity_b = actor_projected_gravity_b + self._bias_projected_gravity
            actor_leg_joint_pos = actor_leg_joint_pos + self._bias_leg_joint_pos
            actor_leg_joint_vel = actor_leg_joint_vel + self._bias_leg_joint_vel
            if self._pendulum_dof_count > 0:
                actor_pendulum_joint_pos = actor_pendulum_joint_pos + self._bias_pendulum_joint_pos
                actor_pendulum_joint_vel = actor_pendulum_joint_vel + self._bias_pendulum_joint_vel

        # Apply fixed-magnitude observation noise.
        body_lin_vel_noise = self.cfg.body_lin_vel_noise
        body_ang_vel_noise = self.cfg.body_ang_vel_noise
        position_noise = self.cfg.position_noise
        orientation_noise = self.cfg.orientation_noise
        pendulum_joint_pos_noise = self.cfg.pendulum_joint_pos_noise
        pendulum_joint_vel_noise = self.cfg.pendulum_joint_vel_noise

        if body_lin_vel_noise > 0.0:
            actor_body_lin_vel_b = actor_body_lin_vel_b + sample_uniform(
                -body_lin_vel_noise,
                body_lin_vel_noise,
                actor_body_lin_vel_b.shape,
                actor_body_lin_vel_b.device,
            )
        if body_ang_vel_noise > 0.0:
            actor_body_ang_vel_b = actor_body_ang_vel_b + sample_uniform(
                -body_ang_vel_noise,
                body_ang_vel_noise,
                actor_body_ang_vel_b.shape,
                actor_body_ang_vel_b.device,
            )
        if position_noise > 0.0:
            position_error_xy = position_error_xy + sample_uniform(
                -position_noise,
                position_noise,
                position_error_xy.shape,
                position_error_xy.device,
            )
        if orientation_noise > 0.0:
            yaw_error = yaw_error + sample_uniform(
                -orientation_noise,
                orientation_noise,
                yaw_error.shape,
                yaw_error.device,
            )
        if self.cfg.use_pendulum and self._pendulum_dof_ids.numel() > 0:
            if pendulum_joint_pos_noise > 0.0:
                actor_pendulum_joint_pos = actor_pendulum_joint_pos + sample_uniform(
                    -pendulum_joint_pos_noise,
                    pendulum_joint_pos_noise,
                    actor_pendulum_joint_pos.shape,
                    actor_pendulum_joint_pos.device,
                )
            if pendulum_joint_vel_noise > 0.0:
                actor_pendulum_joint_vel = actor_pendulum_joint_vel + sample_uniform(
                    -pendulum_joint_vel_noise,
                    pendulum_joint_vel_noise,
                    actor_pendulum_joint_vel.shape,
                    actor_pendulum_joint_vel.device,
                )

        state_error = torch.cat([position_error_xy, yaw_error.unsqueeze(-1)], dim=-1)

        # Apply observation delay to actor proprioceptive block.
        # Action history uses the actually executed delayed command, not the newest policy output.
        proprio_block = torch.cat(
            [
                actor_body_lin_vel_b,
                actor_body_ang_vel_b,
                actor_projected_gravity_b,
                actor_leg_joint_pos,
                actor_leg_joint_vel,
                actor_pendulum_joint_pos,
                actor_pendulum_joint_vel,
                self._actions_executed,
            ],
            dim=-1,
        )
        proprio_block = self._apply_obs_delay_and_jitter(proprio_block)
        idx0 = 3
        idx1 = idx0 + 3
        idx2 = idx1 + 3
        idx3 = idx2 + self._action_dim
        idx4 = idx3 + self._action_dim
        idx5 = idx4 + self._pendulum_dof_count
        idx6 = idx5 + self._pendulum_dof_count
        delayed_body_lin_vel = proprio_block[:, :idx0]
        delayed_body_ang_vel = proprio_block[:, idx0:idx1]
        delayed_projected_gravity = proprio_block[:, idx1:idx2]
        delayed_leg_joint_pos = proprio_block[:, idx2:idx3]
        delayed_leg_joint_vel = proprio_block[:, idx3:idx4]
        delayed_pendulum_joint_pos = proprio_block[:, idx4:idx5]
        delayed_pendulum_joint_vel = proprio_block[:, idx5:idx6]
        delayed_actions = proprio_block[:, idx6:]

        # Policy obs (actor — potentially noisy/delayed).
        policy_obs = torch.cat(
            [
                delayed_body_lin_vel,
                delayed_body_ang_vel,
                delayed_projected_gravity,
                state_error,
                delayed_leg_joint_pos,
                delayed_leg_joint_vel,
                delayed_pendulum_joint_pos,
                delayed_pendulum_joint_vel,
                delayed_actions,
                self.clock_inputs,
            ],
            dim=-1,
        )

        # Critic obs (always clean ground truth).
        critic_obs = torch.cat(
            [
                critic_body_lin_vel_b,
                critic_body_ang_vel_b,
                critic_projected_gravity_b,
                state_error,
                critic_leg_joint_pos,
                critic_leg_joint_vel,
                critic_pendulum_joint_pos,
                critic_pendulum_joint_vel,
                self._actions_executed,
                self.clock_inputs,
            ],
            dim=-1,
        )

        return {"policy": policy_obs, "critic": critic_obs}

    def _compute_base_tilt_rad(self) -> torch.Tensor:
        projected_gravity_b = self.robot.data.projected_gravity_b
        return torch.atan2(torch.linalg.norm(projected_gravity_b[:, :2], dim=1), -projected_gravity_b[:, 2])

    def _get_rewards(self) -> torch.Tensor:
        env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
        if self.target_state is not None:
            base_pos_xy = self.robot.data.root_pos_w[:, :2] - env_origins[:, :2]
            position_error = torch.linalg.norm(self.target_state[:, :2] - base_pos_xy, dim=1)
            _, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
            yaw_error = math_utils.wrap_to_pi(self.target_state[:, 2] - yaw)
            position_sigma = max(self.cfg.position_reward_sigma, 1e-6)
            position_tracking_reward = torch.exp(-(position_error / position_sigma))
            rew_progress = self._prev_position_error - position_error
            self._prev_position_error = position_error.clone()
            yaw_sigma = max(self.cfg.yaw_alignment_reward_sigma, 1e-6)
            yaw_alignment_reward = torch.exp(-torch.square(yaw_error) / (yaw_sigma * yaw_sigma))
        else:
            position_error = torch.zeros(self.num_envs, device=self.device)
            position_tracking_reward = torch.zeros(self.num_envs, device=self.device)
            rew_progress = torch.zeros(self.num_envs, device=self.device)
            yaw_alignment_reward = torch.zeros(self.num_envs, device=self.device)

        base_height = self.robot.data.root_pos_w[:, 2] - env_origins[:, 2]
        base_height_error = self.cfg.base_height_target - base_height
        sigma = max(self.cfg.base_height_reward_sigma, 1e-6)
        base_height_reward = torch.exp(-torch.square(base_height_error) / (sigma * sigma))
        self._episode_base_height_sum += base_height
        self._episode_base_height_count += 1
        base_tilt_deg = torch.rad2deg(self._compute_base_tilt_rad())
        self._episode_base_tilt_deg_sum += base_tilt_deg
        self._episode_base_tilt_deg_count += 1

        rew_action_magnitude = torch.sum(
            torch.square(self._actions_executed), dim=1
        ) * (self.cfg.action_scale**2)

        rew_action_rate = torch.sum(
            torch.square(self._actions_executed - self.last_actions[:, :, 0]),
            dim=1,
        ) * (self.cfg.action_scale**2)

        rew_action_rate += torch.sum(
            torch.square(self._actions_executed - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]),
            dim=1,
        ) * (self.cfg.action_scale**2)

        # penalize non-vertical orientation (projected gravity on xy plane)
        orient_error = torch.sum(
            torch.square(self.robot.data.projected_gravity_b[:, :2]),
            dim=1,
        )
        orient_sigma = max(self.cfg.orient_reward_sigma, 1e-6)
        rew_orient = torch.exp(-(orient_error / orient_sigma))

        # penalize vertical velocity (z-component of base linear velocity)
        rew_lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])

        # penalize high joint velocities
        rew_dof_vel = torch.sum(
            torch.square(self.robot.data.joint_vel[:, self._leg_dof_ids]),
            dim=1,
        )

        # penalize high joint accelerations
        rew_dof_acc = torch.sum(
            torch.square(self.robot.data.joint_acc[:, self._leg_dof_ids]),
            dim=1,
        )

        # penalize angular velocity in xy plane
        rew_ang_vel_xy = torch.sum(
            torch.square(self.robot.data.root_ang_vel_b[:, :2]),
            dim=1,
        )

        if self.cfg.use_pendulum and self._pendulum_dof_ids.numel() > 0:
            pendulum_joint_pos = self.robot.data.joint_pos[:, self._pendulum_dof_ids]
            pendulum_joint_vel = self.robot.data.joint_vel[:, self._pendulum_dof_ids]
            # Relative-to-base upright error: sum of squared joint angles. Decouples
            # from world frame so the policy can't exploit base tilt to fake uprightness.
            pendulum_upright_error = torch.sum(torch.square(pendulum_joint_pos), dim=1)
            pendulum_upright_sigma = max(self.cfg.pendulum_upright_reward_sigma, 1e-6)
            pendulum_upright_reward = torch.exp(-(pendulum_upright_error / pendulum_upright_sigma))

            pendulum_vel_norm = torch.linalg.norm(pendulum_joint_vel, dim=1)
            pendulum_velocity_reward = torch.sum(torch.square(pendulum_joint_vel), dim=1)
            pendulum_angle_deg = torch.rad2deg(torch.linalg.norm(pendulum_joint_pos, dim=1))
            pendulum_speed_deg_s = torch.rad2deg(pendulum_vel_norm)

            # Reward moving while balanced: speed only helps when the pendulum stays upright.
            base_speed = torch.linalg.norm(self.robot.data.root_lin_vel_w[:, :2], dim=1)
            balanced_movement_reward = torch.exp(-pendulum_upright_error) * base_speed
        else:
            pendulum_upright_reward = torch.zeros(self.num_envs, device=self.device)
            pendulum_velocity_reward = torch.zeros(self.num_envs, device=self.device)
            balanced_movement_reward = torch.zeros(self.num_envs, device=self.device)
            pendulum_angle_deg = torch.zeros(self.num_envs, device=self.device)
            pendulum_speed_deg_s = torch.zeros(self.num_envs, device=self.device)

        self._episode_pendulum_angle_deg_sum += pendulum_angle_deg
        self._episode_pendulum_angle_deg_count += 1
        self._episode_pendulum_speed_deg_s_sum += pendulum_speed_deg_s
        self._episode_pendulum_speed_deg_s_count += 1

        # penalize high torques from the actuator model
        rew_torque = torch.sum(torch.square(self.robot.data.applied_torque[:, self._leg_dof_ids]), dim=1)

        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions_executed[:]

        # gait shaping
        self._step_contact_targets()

        phases = 1 - torch.abs(1.0 - torch.clamp((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = self.foot_positions_w[:, :, 2]
        target_height = 0.08 * phases + 0.02
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        rew_feet_clearance = torch.sum(rew_foot_clearance, dim=1)

        foot_forces = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :], dim=-1)
        desired_contact = self.desired_contact_states
        rew_tracking_contacts_shaped_force = torch.zeros(self.num_envs, device=self.device)
        for i in range(4):
            rew_tracking_contacts_shaped_force += -(1 - desired_contact[:, i]) * (
                1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.0)
            )
        rew_tracking_contacts_shaped_force = rew_tracking_contacts_shaped_force / 4

        # feet air time (same structure as go2_isaaclab, gated by active position command)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids_sensor]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids_sensor]
        rew_feet_air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (position_error > 0.1).float()

        # undesired contacts (e.g. thighs)
        if self._undesired_contact_body_ids is not None:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            is_contact = (
                torch.max(
                    torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1
                )[0]
                > 1.0
            )
            rew_undesired_contacts = torch.sum(is_contact, dim=1)
        else:
            rew_undesired_contacts = torch.zeros(self.num_envs, device=self.device)

        early_terminated = self.reset_terminated & ~self.reset_time_outs
        rew_termination_penalty = early_terminated.float() * self.cfg.termination_penalty

        rewards = {
            "position_tracking": position_tracking_reward * self.cfg.position_reward_scale,
            "progress": rew_progress * self.cfg.progress_reward_scale,
            "yaw_alignment": yaw_alignment_reward * self.cfg.yaw_alignment_reward_scale,
            "pendulum_upright": pendulum_upright_reward * self.cfg.pendulum_upright_reward_scale,
            "pendulum_velocity": pendulum_velocity_reward * self.cfg.pendulum_vel_reward_scale,
            "balanced_movement": balanced_movement_reward * self.cfg.balanced_movement_reward_scale,
            "action_magnitude": rew_action_magnitude * self.cfg.action_magnitude_reward_scale,
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "torque": rew_torque * self.cfg.torque_reward_scale,
            "orient": rew_orient * self.cfg.orient_reward_scale,
            "base_height": base_height_reward * self.cfg.base_height_reward_scale,
            "lin_vel_z": rew_lin_vel_z * self.cfg.lin_vel_z_reward_scale,
            "dof_vel": rew_dof_vel * self.cfg.dof_vel_reward_scale,
            "dof_acc": rew_dof_acc * self.cfg.dof_acc_reward_scale,
            "ang_vel_xy": rew_ang_vel_xy * self.cfg.ang_vel_xy_reward_scale,
            "feet_clearance": rew_feet_clearance * self.cfg.feet_clearance_reward_scale,
            "feet_air_time": rew_feet_air_time * self.cfg.feet_air_time_reward_scale,
            "tracking_contacts_shaped_force": (
                rew_tracking_contacts_shaped_force * self.cfg.tracking_contacts_shaped_force_reward_scale
            ),
            "undesired_contacts": rew_undesired_contacts * self.cfg.undesired_contact_reward_scale,
            "termination_penalty": rew_termination_penalty,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        steps_since_reset = self._steps_since_reset

        base_contact_grace_steps = max(0, math.ceil(self.cfg.base_contact_grace_s / self.step_dt))
        termination_grace_steps = max(0, math.ceil(self.cfg.termination_grace_s / self.step_dt))
        in_termination_grace = steps_since_reset < termination_grace_steps
        termination_allowed = ~in_termination_grace

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        cstr_termination_contacts = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )

        # Allow a grace period so brief settling contacts right after reset don't terminate.
        contact_grace_elapsed = steps_since_reset >= base_contact_grace_steps
        cstr_termination_contacts = cstr_termination_contacts & contact_grace_elapsed & termination_allowed

        terminated = cstr_termination_contacts

        base_height = self.robot.data.root_pos_w[:, 2]
        if self._base_height_failure_steps is None:
            self._base_height_failure_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        base_height_failing = (base_height < self.cfg.base_height_min) & termination_allowed
        self._base_height_failure_steps = torch.where(
            base_height_failing,
            self._base_height_failure_steps + 1,
            torch.zeros_like(self._base_height_failure_steps),
        )
        base_height_failure_threshold = max(1, math.ceil(self.cfg.base_height_terminate_duration_s / self.step_dt))
        cstr_base_height_min = self._base_height_failure_steps >= base_height_failure_threshold
        terminated = terminated | cstr_base_height_min

        base_tilt_rad = self._compute_base_tilt_rad()
        base_tilt_terminated = base_tilt_rad > self.cfg.base_tilt_terminate_angle_rad
        terminated = terminated | base_tilt_terminated

        pendulum_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.use_pendulum and self._pendulum_contact_sensor is not None:
            pendulum_contact_forces = self._pendulum_contact_sensor.data.net_forces_w
            pendulum_contact = torch.any(
                torch.norm(pendulum_contact_forces, dim=-1) > self.cfg.pendulum_contact_force_threshold,
                dim=1,
            )
            pendulum_contact = pendulum_contact & termination_allowed
            terminated = terminated | pendulum_contact

        pendulum_angle_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.use_pendulum and self._pendulum_dof_ids.numel() > 0:
            pendulum_joint_pos = self.robot.data.joint_pos[:, self._pendulum_dof_ids]
            pendulum_angle_norm = torch.linalg.norm(pendulum_joint_pos, dim=1)
            if self._pendulum_angle_failure_steps is None:
                self._pendulum_angle_failure_steps = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.long
                )
            pendulum_failing = (pendulum_angle_norm > self.cfg.pendulum_terminate_angle_rad) & termination_allowed
            self._pendulum_angle_failure_steps = torch.where(
                pendulum_failing,
                self._pendulum_angle_failure_steps + 1,
                torch.zeros_like(self._pendulum_angle_failure_steps),
            )
            failure_steps_threshold = max(1, math.ceil(self.cfg.pendulum_terminate_duration_s / self.step_dt))
            pendulum_angle_terminated = self._pendulum_angle_failure_steps >= failure_steps_threshold
            terminated = terminated | pendulum_angle_terminated

        position_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.target_state is not None:
            env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
            base_pos_xy = self.robot.data.root_pos_w[:, :2] - env_origins[:, :2]
            position_error = torch.linalg.norm(self.target_state[:, :2] - base_pos_xy, dim=1)
            if self._position_failure_steps is None:
                self._position_failure_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            position_failing = (position_error > self.cfg.position_tolerance) & termination_allowed
            self._position_failure_steps = torch.where(
                position_failing,
                self._position_failure_steps + 1,
                torch.zeros_like(self._position_failure_steps),
            )
            position_failure_threshold = max(1, math.ceil(self.cfg.position_terminate_duration_s / self.step_dt))
            position_terminated = self._position_failure_steps >= position_failure_threshold
            terminated = terminated | position_terminated

        self._base_contact_terminated = cstr_termination_contacts
        self._base_height_terminated = cstr_base_height_min
        self._base_tilt_terminated = base_tilt_terminated
        self._pendulum_contact_terminated = pendulum_contact
        self._pendulum_angle_terminated = pendulum_angle_terminated
        self._position_terminated = position_terminated
        self._steps_since_reset += 1

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        if not torch.is_tensor(env_ids):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time.
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions_raw_policy[env_ids] = 0.0
        self._actions_delayed[env_ids] = 0.0
        self._actions_executed[env_ids] = 0.0
        self._action_delay_buffer[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0
        if self.cfg.enable_action_delay:
            if self.cfg.action_delay_randomize_per_reset:
                self._action_delay_steps[env_ids] = torch.randint(
                    self.cfg.action_delay_steps_min,
                    self.cfg.action_delay_steps_max + 1,
                    (env_ids.shape[0],),
                    device=self.device,
                    dtype=torch.long,
                )
            else:
                self._action_delay_steps[env_ids] = self.cfg.action_delay_steps_max

        if self.cfg.enable_domain_randomization:
            self._randomize_materials(env_ids)
            self._randomize_mass_and_com(env_ids)
            self._randomize_motor_gains(env_ids)
            self._sample_sensor_biases(env_ids)
            if self.cfg.enable_obs_delay:
                self._obs_delay_buffer.reset(env_ids)
                if self.cfg.obs_delay_randomize_per_reset:
                    self._obs_delay_steps[env_ids] = torch.randint(
                        self.cfg.obs_delay_steps_min,
                        self.cfg.obs_delay_steps_max + 1,
                        (env_ids.numel(),),
                        device=self.device,
                        dtype=torch.long,
                    )
                else:
                    self._obs_delay_steps[env_ids] = self.cfg.obs_delay_steps_max
            if self.cfg.enable_external_wrench_push:
                self._push_forces[env_ids] = 0.0
                self._push_torques[env_ids] = 0.0
                self._push_end_step[env_ids] = 0
                self._schedule_next_push(env_ids, self._steps_since_reset[env_ids])

        # Reset variables.
        self.last_actions[env_ids] = 0
        self.gait_indices[env_ids] = 0
        if self._base_height_failure_steps is not None:
            self._base_height_failure_steps[env_ids] = 0
        if self._pendulum_angle_failure_steps is not None:
            self._pendulum_angle_failure_steps[env_ids] = 0
        if self._position_failure_steps is not None:
            self._position_failure_steps[env_ids] = 0

        # Sample new targets.
        if self.target_state is None:
            self.target_state = torch.zeros(self.num_envs, 3, device=self.device)

        num_reset_envs = env_ids.shape[0]
        dist_min = min(self.cfg.goal_randomization_dist_min, self.cfg.goal_randomization_dist_max)
        dist_max = max(self.cfg.goal_randomization_dist_min, self.cfg.goal_randomization_dist_max)
        bearing_min = min(self.cfg.goal_randomization_angle_min, self.cfg.goal_randomization_angle_max)
        bearing_max = max(self.cfg.goal_randomization_angle_min, self.cfg.goal_randomization_angle_max)
        yaw_min = min(self.cfg.goal_yaw_randomization_min, self.cfg.goal_yaw_randomization_max)
        yaw_max = max(self.cfg.goal_yaw_randomization_min, self.cfg.goal_yaw_randomization_max)

        goal_distance = sample_uniform(dist_min, dist_max, (num_reset_envs,), self.device)
        # Target position bearing and desired final heading are sampled independently.
        goal_bearing = sample_uniform(bearing_min, bearing_max, (num_reset_envs,), self.device)
        goal_noise_x = goal_distance * torch.cos(goal_bearing)
        goal_noise_y = goal_distance * torch.sin(goal_bearing)
        goal_yaw = sample_uniform(
            yaw_min,
            yaw_max,
            (num_reset_envs,),
            self.device,
        )
        self.target_state[env_ids, 0] = goal_noise_x
        self.target_state[env_ids, 1] = goal_noise_y
        self.target_state[env_ids, 2] = goal_yaw

        # Reset robot state.
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.desired_joint_pos[env_ids] = joint_pos[:, self._leg_dof_ids]

        if self.cfg.use_pendulum and self._pendulum_dof_ids.numel() > 0:
            for joint_idx in self._pendulum_dof_ids.tolist():
                signs = torch.randint(0, 2, joint_pos[:, joint_idx].shape, device=joint_pos.device) * 2 - 1
                magnitudes = sample_uniform(
                    self.cfg.pendulum_angle_min,
                    self.cfg.pendulum_angle_max,
                    joint_pos[:, joint_idx].shape,
                    joint_pos.device,
                )
                joint_pos[:, joint_idx] += signs.float() * magnitudes

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Initialize progress baseline so the first step in an episode has well-defined progress.
        initial_base_pos_xy = default_root_state[:, :2] - self._terrain.env_origins[env_ids, :2]
        initial_position_error = torch.linalg.norm(self.target_state[env_ids, :2] - initial_base_pos_xy, dim=1)
        self._prev_position_error[env_ids] = initial_position_error

        self._visualize_target_markers()

        # Logging
        extras = dict()
        for key in self._episode_sums:
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        extras = dict()
        base_contact_resets = self._base_contact_terminated[env_ids] & self.reset_terminated[env_ids]
        base_height_resets = self._base_height_terminated[env_ids] & self.reset_terminated[env_ids]
        base_tilt_resets = self._base_tilt_terminated[env_ids] & self.reset_terminated[env_ids]
        pendulum_contact_resets = self._pendulum_contact_terminated[env_ids] & self.reset_terminated[env_ids]
        pendulum_angle_resets = self._pendulum_angle_terminated[env_ids] & self.reset_terminated[env_ids]
        position_resets = self._position_terminated[env_ids] & self.reset_terminated[env_ids]
        any_labeled_reset = (
            base_contact_resets
            | base_height_resets
            | base_tilt_resets
            | pendulum_contact_resets
            | pendulum_angle_resets
            | position_resets
        )
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(base_contact_resets).item()
        extras["Episode_Termination/base_height"] = torch.count_nonzero(base_height_resets).item()
        extras["Episode_Termination/base_tilt"] = torch.count_nonzero(base_tilt_resets).item()
        extras["Episode_Termination/pendulum_contact"] = torch.count_nonzero(pendulum_contact_resets).item()
        extras["Episode_Termination/pendulum_angle"] = torch.count_nonzero(pendulum_angle_resets).item()
        extras["Episode_Termination/position_error"] = torch.count_nonzero(position_resets).item()
        extras["Episode_Termination/other"] = torch.count_nonzero(
            self.reset_terminated[env_ids] & ~any_labeled_reset
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        steps = torch.clamp(self._episode_base_height_count[env_ids], min=1).to(dtype=torch.float)
        mean_base_height = torch.mean(self._episode_base_height_sum[env_ids] / steps)
        extras["Episode_Metric/mean_base_height"] = mean_base_height.item()
        self._episode_base_height_sum[env_ids] = 0.0
        self._episode_base_height_count[env_ids] = 0
        base_tilt_steps = torch.clamp(self._episode_base_tilt_deg_count[env_ids], min=1).to(dtype=torch.float)
        mean_base_tilt_deg = torch.mean(self._episode_base_tilt_deg_sum[env_ids] / base_tilt_steps)
        extras["Episode_Metric/mean_base_tilt_deg"] = mean_base_tilt_deg.item()
        self._episode_base_tilt_deg_sum[env_ids] = 0.0
        self._episode_base_tilt_deg_count[env_ids] = 0
        pendulum_steps = torch.clamp(self._episode_pendulum_angle_deg_count[env_ids], min=1).to(dtype=torch.float)
        mean_pendulum_angle_deg = torch.mean(self._episode_pendulum_angle_deg_sum[env_ids] / pendulum_steps)
        extras["Episode_Metric/mean_pendulum_angle_deg"] = mean_pendulum_angle_deg.item()
        self._episode_pendulum_angle_deg_sum[env_ids] = 0.0
        self._episode_pendulum_angle_deg_count[env_ids] = 0
        pendulum_speed_steps = torch.clamp(self._episode_pendulum_speed_deg_s_count[env_ids], min=1).to(dtype=torch.float)
        mean_pendulum_speed_deg_s = torch.mean(self._episode_pendulum_speed_deg_s_sum[env_ids] / pendulum_speed_steps)
        extras["Episode_Metric/mean_pendulum_speed_deg_s"] = mean_pendulum_speed_deg_s.item()
        self._episode_pendulum_speed_deg_s_sum[env_ids] = 0.0
        self._episode_pendulum_speed_deg_s_count[env_ids] = 0
        if self.cfg.enable_curriculum and self.cfg.curriculum_total_steps > 0:
            extras["Episode_Metric/curriculum_progress"] = min(1.0, max(0.0, self.common_step_counter / self.cfg.curriculum_total_steps))
        else:
            extras["Episode_Metric/curriculum_progress"] = 0.0
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if self.target_visualizer is None:
                self.target_visualizer = VisualizationMarkers(self.cfg.target_marker_cfg)

            if self.target_visualizer is not None:
                self.target_visualizer.set_visibility(True)
        else:
            if self.target_visualizer is not None:
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return

        self._visualize_target_markers()

    def _visualize_target_markers(self) -> None:
        if self.target_state is None or self.target_visualizer is None:
            return

        if self._marker_locations is None:
            self._marker_up = self._marker_up.to(device=self.device)
            self._marker_locations = torch.zeros((self.num_envs, 3), device=self.device)
            self._marker_orientations = torch.zeros((self.num_envs, 4), device=self.device)

        env_origins = (
            self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
        )

        # Arrow at goal XY, 1m above ground.
        self._marker_locations[:, :2] = self.target_state[:, :2] + env_origins[:, :2]
        self._marker_locations[:, 2] = env_origins[:, 2] + 1.0

        # Arrow oriented by goal_yaw around Z.
        self._marker_orientations = math_utils.quat_from_angle_axis(
            self.target_state[:, 2], self._marker_up
        )

        self.target_visualizer.visualize(self._marker_locations, self._marker_orientations)

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
