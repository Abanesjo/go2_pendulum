# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.sensors.imu import Imu
from isaaclab.utils.math import sample_uniform

from .go2_pendulum_env_cfg import Go2PendulumEnvCfg


class Go2PendulumEnv(DirectRLEnv):
    cfg: Go2PendulumEnvCfg

    # Difficulty presets: values for each curriculum level, applied at runtime.
    # Physical joint limits deliberately do not appear here: curriculum changes
    # the task and disturbance distributions, not the mechanism being learned.
    _DIFFICULTY_PRESETS = {
        1: dict(
            goal_distance_mixture=(0.70, 0.30, 0.00),
            goal_stand_distance_range=(0.0, 0.05),
            goal_short_distance_range=(0.05, 0.20),
            goal_walk_distance_range=(0.20, 0.20),
            goal_randomization_angle_min=math.radians(-180),
            goal_randomization_angle_max=math.radians(180),
            goal_yaw_randomization_min=math.radians(-5),
            goal_yaw_randomization_max=math.radians(5),
            pendulum_angle_min=math.radians(0.0),
            pendulum_angle_max=math.radians(3.0),
            pendulum_recovery_reset_fraction=0.0,
            domain_randomization_scale=0.25,
            termination_grace_s=0.5,
            pendulum_termination_grace_s=0.5,
            base_height_terminate_duration_s=0.25,
            pendulum_terminate_angle_rad=math.radians(20.0),
            pendulum_terminate_duration_s=0.25,
            position_tolerance=2.5,
            enable_external_wrench_push=False,
            push_force_x_range=(0.0, 0.0),
            push_force_y_range=(0.0, 0.0),
            push_torque_z_range=(0.0, 0.0),
        ),
        2: dict(
            goal_distance_mixture=(0.50, 0.40, 0.10),
            goal_stand_distance_range=(0.0, 0.05),
            goal_short_distance_range=(0.05, 0.30),
            goal_walk_distance_range=(0.30, 0.80),
            goal_randomization_angle_min=math.radians(-180),
            goal_randomization_angle_max=math.radians(180),
            goal_yaw_randomization_min=math.radians(-15),
            goal_yaw_randomization_max=math.radians(15),
            pendulum_angle_min=math.radians(0.0),
            pendulum_angle_max=math.radians(5.0),
            pendulum_recovery_reset_fraction=0.0,
            domain_randomization_scale=0.50,
            termination_grace_s=0.5,
            pendulum_termination_grace_s=0.5,
            base_height_terminate_duration_s=0.25,
            pendulum_terminate_angle_rad=math.radians(20.0),
            pendulum_terminate_duration_s=0.25,
            position_tolerance=2.5,
            enable_external_wrench_push=True,
            push_force_x_range=(-10.0, 10.0),
            push_force_y_range=(-10.0, 10.0),
            push_torque_z_range=(0.0, 0.0),
        ),
        3: dict(
            goal_distance_mixture=(0.35, 0.40, 0.25),
            goal_stand_distance_range=(0.0, 0.05),
            goal_short_distance_range=(0.05, 0.35),
            goal_walk_distance_range=(0.35, 1.00),
            goal_randomization_angle_min=math.radians(-180),
            goal_randomization_angle_max=math.radians(180),
            goal_yaw_randomization_min=math.radians(-30),
            goal_yaw_randomization_max=math.radians(30),
            pendulum_angle_min=math.radians(0.0),
            pendulum_angle_max=math.radians(7.0),
            pendulum_recovery_reset_fraction=0.0,
            domain_randomization_scale=0.75,
            termination_grace_s=0.5,
            pendulum_termination_grace_s=0.5,
            base_height_terminate_duration_s=0.25,
            pendulum_terminate_angle_rad=math.radians(15.0),
            pendulum_terminate_duration_s=0.25,
            position_tolerance=2.5,
            enable_external_wrench_push=True,
            push_force_x_range=(-20.0, 20.0),
            push_force_y_range=(-20.0, 20.0),
            push_torque_z_range=(0.0, 0.0),
        ),
        4: dict(
            goal_distance_mixture=(0.25, 0.35, 0.40),
            goal_stand_distance_range=(0.0, 0.05),
            goal_short_distance_range=(0.05, 0.40),
            goal_walk_distance_range=(0.40, 1.30),
            goal_randomization_angle_min=math.radians(-180),
            goal_randomization_angle_max=math.radians(180),
            goal_yaw_randomization_min=math.radians(-60),
            goal_yaw_randomization_max=math.radians(60),
            pendulum_angle_min=math.radians(0.0),
            pendulum_angle_max=math.radians(9.0),
            pendulum_recovery_reset_fraction=0.0,
            domain_randomization_scale=1.00,
            termination_grace_s=0.5,
            pendulum_termination_grace_s=0.5,
            base_height_terminate_duration_s=0.25,
            pendulum_terminate_angle_rad=math.radians(12.0),
            pendulum_terminate_duration_s=0.25,
            position_tolerance=2.5,
            enable_external_wrench_push=True,
            push_force_x_range=(-30.0, 30.0),
            push_force_y_range=(-30.0, 30.0),
            push_torque_z_range=(-2.0, 2.0),
        ),
        5: dict(
            goal_distance_mixture=(0.25, 0.25, 0.50),
            goal_stand_distance_range=(0.0, 0.05),
            goal_short_distance_range=(0.05, 0.45),
            goal_walk_distance_range=(0.45, 1.50),
            goal_randomization_angle_min=math.radians(-180),
            goal_randomization_angle_max=math.radians(180),
            goal_yaw_randomization_min=math.radians(-90),
            goal_yaw_randomization_max=math.radians(90),
            pendulum_angle_min=math.radians(0.0),
            pendulum_angle_max=math.radians(10.0),
            pendulum_recovery_reset_fraction=0.10,
            domain_randomization_scale=1.00,
            termination_grace_s=0.5,
            pendulum_termination_grace_s=0.5,
            base_height_terminate_duration_s=0.25,
            pendulum_terminate_angle_rad=math.radians(12.0),
            pendulum_terminate_duration_s=0.25,
            position_tolerance=2.5,
            enable_external_wrench_push=True,
            push_force_x_range=(-40.0, 40.0),
            push_force_y_range=(-40.0, 40.0),
            push_torque_z_range=(-3.0, 3.0),
        ),
    }

    def __init__(self, cfg: Go2PendulumEnvCfg, render_mode: str | None = None, **kwargs):
        self._prev_base_pos_w = None
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

        # Resolve leg joints in the exact configured policy/action order.
        leg_joint_ids = []
        for joint_name in self.cfg.leg_joint_names:
            joint_idx, _ = self.robot.find_joints(joint_name)
            if len(joint_idx) != 1:
                raise RuntimeError(f"Expected exactly one joint for '{joint_name}', got {joint_idx}.")
            leg_joint_ids.append(joint_idx[0])
        if len(leg_joint_ids) != self.cfg.action_space:
            raise RuntimeError(
                "Leg joint count does not match action space: "
                f"{len(leg_joint_ids)} vs {self.cfg.action_space}."
            )
        self._leg_dof_ids = torch.tensor(leg_joint_ids, device=self.device, dtype=torch.long)
        self._action_dim = gym.spaces.flatdim(self.single_action_space)
        if self.cfg.action_scale <= 0.0:
            raise ValueError(f"action_scale must be > 0. Got {self.cfg.action_scale}.")

        # Left/right coordinate relabeling uses the configured grouped joint
        # order: hips, thighs, calves; FL, FR, RL, RR within each group.
        expected_mirror_joint_order = [
            "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
            "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
            "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
        ]
        if list(self.cfg.leg_joint_names) != expected_mirror_joint_order:
            raise RuntimeError(
                "Episode mirroring requires the documented Go2 joint order; got "
                f"{list(self.cfg.leg_joint_names)}."
            )
        self._lr_joint_permutation = torch.tensor(
            [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], device=self.device, dtype=torch.long
        )
        self._episode_mirrored = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self._validate_domain_randomization_cfg()
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
            pendulum_hinge_body_ids, _ = self.robot.find_bodies("pendulum_dof2")
            if len(pendulum_hinge_body_ids) != 1:
                raise RuntimeError(
                    "Expected exactly one simulator hinge body for 'pendulum_dof2', "
                    f"got {pendulum_hinge_body_ids}."
                )
            self._pendulum_hinge_body_id = pendulum_hinge_body_ids[0]
        else:
            self._pendulum_dof_ids = torch.tensor([], device=self.device, dtype=torch.long)
            self._pendulum_ee_body_id = None
            self._pendulum_hinge_body_id = None

        # The hardware hinge offset is expressed in the deployed mocap marker
        # frame.  The simulator USD uses a different base/marker mounting
        # geometry, so calibrate one fixed simulator-only angle offset at the
        # nominal joint pose.  Actor values remain pose-derived at runtime;
        # simulator joint packets are not used by the observation estimator.
        self._sim_mocap_pendulum_angle_offset = torch.zeros(
            self.num_envs, self._pendulum_dof_count, device=self.device
        )
        self._sim_mocap_hinge_offset_b = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg.use_pendulum:
            if self._pendulum_dof_ids.numel() != 2:
                raise RuntimeError(
                    "Mocap pendulum reconstruction requires exactly two joints, "
                    f"got {self._pendulum_dof_ids.numel()}."
                )
            self._sim_mocap_hinge_offset_b.copy_(
                math_utils.quat_apply_inverse(
                    self.robot.data.root_quat_w,
                    self.robot.data.body_pos_w[:, self._pendulum_hinge_body_id]
                    - self.robot.data.root_pos_w,
                )
            )
            nominal_angles = self._reconstruct_pendulum_angles_from_pose(
                self.robot.data.root_pos_w,
                self.robot.data.root_quat_w,
                self.robot.data.body_pos_w[:, self._pendulum_ee_body_id],
                self._sim_mocap_hinge_offset_b,
            )
            nominal_joint_pos = self.robot.data.default_joint_pos[:, self._pendulum_dof_ids]
            self._sim_mocap_pendulum_angle_offset.copy_(
                math_utils.wrap_to_pi(nominal_angles - nominal_joint_pos)
            )

        self._apply_pendulum_joint_limits()

        if self.cfg.enable_curriculum:
            curriculum_start = int(getattr(self.cfg, "curriculum_start_step", 0))
            curriculum_total = max(int(self.cfg.curriculum_total_steps), 1)
            initial_progress = min(max(curriculum_start / curriculum_total, 0.0), 1.0)
            initial_level = min(5, int(initial_progress * 5.0) + 1)
            self._current_difficulty_level = initial_level
            self._apply_difficulty_preset(initial_level)

        if self.cfg.difficulty_override >= 1:
            self.cfg.enable_curriculum = False
            self._current_difficulty_level = self.cfg.difficulty_override
            self._apply_difficulty_preset(self.cfg.difficulty_override)

        leg_default_joint_pos = self.robot.data.default_joint_pos[:, self._leg_dof_ids]

        # Joint position command from the latest delayed policy action offsets relative to default joint positions.
        self.last_action = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._physical_delivered_action = torch.zeros_like(self.last_action)
        self._filtered_action = torch.zeros_like(self.last_action)
        self._command_lpf_cutoff_hz = torch.full(
            (self.num_envs,), float(getattr(self.cfg, "command_lpf_cutoff_hz", 4.0)), device=self.device
        )
        self.desired_joint_pos = leg_default_joint_pos.clone()

        # Target state [x_d, y_d, yaw_d] in environment frame.
        # x/y come from target distance + bearing; yaw is the desired robot heading at the target.
        self.target_state = None
        self._stand_mode = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._actor_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._move_gate = torch.zeros(self.num_envs, device=self.device)
        self._stand_gate = torch.ones(self.num_envs, device=self.device)
        self._gait_frequency = torch.full(
            (self.num_envs,), float(getattr(self.cfg, "gait_frequency_min_hz", 1.5)), device=self.device
        )
        self._gait_duty_factor = torch.full(
            (self.num_envs,), float(getattr(self.cfg, "gait_duty_factor_min_speed", 0.62)), device=self.device
        )

        # Marker visualization buffers.
        self._marker_orientations = None
        self._marker_locations = None
        self._marker_up = torch.tensor([0.0, 0.0, 1.0])
        self._world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self._world_gravity_dir = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self._prev_base_pos_w = self.robot.data.root_pos_w.clone()

        # Logging.
        episode_sum_keys = [
            "command_lin_vel",
            "command_yaw_rate",
            "goal_position_hold",
            "goal_yaw_hold",
            "stand_settling",
            "stand_contacts",
            "stand_foot_lift",
            "stand_symmetry",
            "foot_slip",
            "gait_contact_mismatch",
            "pendulum_upright",
            "pendulum_velocity",
            "arrival_dwell",
            "action_magnitude",
            "action_rate_l2",
            "action_acc_l2",
            "torque_l2",
            "torque_rate_l2",
            "orient",
            "base_height",
            "lin_vel_z",
            "dof_vel",
            "joint_acc_l2",
            "ang_vel_xy",
            "feet_clearance",
            "feet_air_time",
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
        self._arrival_dwell_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_arrival_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._current_goal_success_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_goal_success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_four_contact_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_stand_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_foot_lift_events = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._previous_stand_foot_lift = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._action_history = torch.zeros(
            self.num_envs,
            self._action_dim,
            3,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self._prev_torque = torch.zeros(self.num_envs, self._action_dim, dtype=torch.float, device=self.device)

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

        # Observation construction advances policy-rate estimator state.  RSL-RL
        # may query observations multiple times without a physics step (notably
        # during runner construction), so cache by global policy step and reset
        # generation to make those queries strictly idempotent.
        self._observation_reset_generation = 0
        self._observation_cache_step = -1
        self._observation_cache_reset_generation = -1
        self._observation_cache: dict[str, torch.Tensor] | None = None

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

    def _compute_goal_error_terms(
        self,
        base_pos_xy: torch.Tensor,
        base_yaw: torch.Tensor,
        target_xy: torch.Tensor,
        target_yaw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_error_xy_world = target_xy - base_pos_xy
        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        position_error_xy = torch.stack(
            (
                cos_yaw * position_error_xy_world[:, 0] + sin_yaw * position_error_xy_world[:, 1],
                -sin_yaw * position_error_xy_world[:, 0] + cos_yaw * position_error_xy_world[:, 1],
            ),
            dim=-1,
        )
        yaw_error = math_utils.wrap_to_pi(target_yaw - base_yaw)
        return position_error_xy, yaw_error

    def _sample_goal_targets(
        self,
        env_ids: torch.Tensor,
        anchor_pos_xy: torch.Tensor,
        anchor_yaw: torch.Tensor,
        distance_mixture: Sequence[float],
        *,
        is_chain: bool,
    ) -> torch.Tensor:
        """Sample pose goals around a per-environment anchor without resetting any dynamics."""
        if env_ids.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=torch.long)
        if self.target_state is None:
            self.target_state = torch.zeros(self.num_envs, 3, device=self.device)

        num_envs = env_ids.numel()
        mixture = torch.as_tensor(distance_mixture, device=self.device, dtype=torch.float)
        mixture_cdf = torch.cumsum(mixture, dim=0)
        mixture_draw = torch.rand(num_envs, device=self.device)
        goal_class = torch.sum(mixture_draw.unsqueeze(-1) > mixture_cdf.unsqueeze(0), dim=-1).clamp(max=2)

        goal_distance = torch.zeros(num_envs, device=self.device)
        distance_ranges = (
            self.cfg.goal_stand_distance_range,
            self.cfg.goal_short_distance_range,
            self.cfg.goal_walk_distance_range,
        )
        for class_index, distance_range in enumerate(distance_ranges):
            class_mask = goal_class == class_index
            if not torch.any(class_mask):
                continue
            distance_min = float(distance_range[0])
            distance_max = float(distance_range[1])
            # A chained non-stand waypoint must cross the hysteresis exit
            # threshold; a planted class-0 target deliberately remains inside.
            if is_chain and class_index > 0:
                distance_min = max(distance_min, float(self.cfg.stand_exit_distance_m))
                distance_max = max(distance_max, distance_min)
            samples = sample_uniform(distance_min, distance_max, (num_envs,), self.device)
            goal_distance[class_mask] = samples[class_mask]

        bearing_min = min(self.cfg.goal_randomization_angle_min, self.cfg.goal_randomization_angle_max)
        bearing_max = max(self.cfg.goal_randomization_angle_min, self.cfg.goal_randomization_angle_max)
        goal_bearing = sample_uniform(bearing_min, bearing_max, (num_envs,), self.device)
        goal_offset_xy = goal_distance.unsqueeze(-1) * torch.stack(
            (torch.cos(goal_bearing), torch.sin(goal_bearing)), dim=-1
        )

        yaw_offset_min = min(self.cfg.goal_yaw_randomization_min, self.cfg.goal_yaw_randomization_max)
        yaw_offset_max = max(self.cfg.goal_yaw_randomization_min, self.cfg.goal_yaw_randomization_max)
        goal_yaw_offset = sample_uniform(yaw_offset_min, yaw_offset_max, (num_envs,), self.device)
        stand_yaw_range = getattr(
            self.cfg,
            "goal_stand_yaw_offset_range",
            (-math.radians(2.0), math.radians(2.0)),
        )
        stand_yaw_offset = sample_uniform(
            float(stand_yaw_range[0]), float(stand_yaw_range[1]), (num_envs,), self.device
        )
        goal_yaw_offset = torch.where(goal_class == 0, stand_yaw_offset, goal_yaw_offset)

        self.target_state[env_ids, :2] = anchor_pos_xy + goal_offset_xy
        self.target_state[env_ids, 2] = math_utils.wrap_to_pi(anchor_yaw + goal_yaw_offset)

        planted_mask = goal_class == 0
        if torch.any(planted_mask):
            planted_rho = torch.linalg.norm(goal_offset_xy[planted_mask], dim=-1)
            planted_yaw_error = torch.abs(goal_yaw_offset[planted_mask])
            if torch.any(planted_rho > float(self.cfg.stand_enter_distance_m) + 1e-6) or torch.any(
                planted_yaw_error > float(self.cfg.stand_enter_yaw_rad) + 1e-6
            ):
                raise RuntimeError("A class-0 planted goal was sampled outside the stand-entry tolerances.")
        return goal_class

    def _compute_navigation_command(
        self,
        base_pos_xy: torch.Tensor,
        base_yaw: torch.Tensor,
        update_stand_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a world-frame pose goal into a bounded forward/yaw command."""
        if self.target_state is None:
            zeros = torch.zeros((self.num_envs, 3), device=self.device, dtype=base_pos_xy.dtype)
            return zeros, zeros, zeros[:, 0], torch.ones_like(zeros[:, 0])

        position_error_b, final_yaw_error = self._compute_goal_error_terms(
            base_pos_xy, base_yaw, self.target_state[:, :2], self.target_state[:, 2]
        )
        rho = torch.linalg.norm(position_error_b, dim=-1)
        alpha = torch.atan2(position_error_b[:, 1], position_error_b[:, 0])
        beta = math_utils.wrap_to_pi(final_yaw_error - alpha)

        if update_stand_state:
            enter_stand = (
                (rho <= float(getattr(self.cfg, "stand_enter_distance_m", 0.05)))
                & (torch.abs(final_yaw_error) <= float(getattr(self.cfg, "stand_enter_yaw_rad", math.radians(5))))
            )
            exit_stand = (
                (rho >= float(getattr(self.cfg, "stand_exit_distance_m", 0.08)))
                | (torch.abs(final_yaw_error) >= float(getattr(self.cfg, "stand_exit_yaw_rad", math.radians(8))))
            )
            self._stand_mode = torch.where(self._stand_mode, ~exit_stand, enter_stand)

        blend_near = float(getattr(self.cfg, "command_heading_blend_near_m", 0.05))
        blend_far = float(getattr(self.cfg, "command_heading_blend_far_m", 0.20))
        blend_t = torch.clamp((rho - blend_near) / max(blend_far - blend_near, 1e-6), 0.0, 1.0)
        heading_blend = blend_t * blend_t * (3.0 - 2.0 * blend_t)
        yaw_far = (
            float(getattr(self.cfg, "command_k_alpha", 2.5)) * alpha
            + float(getattr(self.cfg, "command_k_beta", -0.8)) * beta
        )
        yaw_near = float(getattr(self.cfg, "command_k_final_yaw", 2.0)) * final_yaw_error
        max_yaw_rate = float(getattr(self.cfg, "command_max_yaw_rate_rad_s", 1.2))
        yaw_far = torch.clamp(yaw_far, -max_yaw_rate, max_yaw_rate)
        yaw_near = torch.clamp(yaw_near, -max_yaw_rate, max_yaw_rate)
        yaw_rate = heading_blend * yaw_far + (1.0 - heading_blend) * yaw_near

        max_forward_speed = float(getattr(self.cfg, "command_max_forward_speed_m_s", 0.6))
        forward_speed = torch.clamp(
            float(getattr(self.cfg, "command_k_rho", 1.5)) * rho, 0.0, max_forward_speed
        )
        forward_speed *= heading_blend * torch.clamp(torch.cos(alpha), min=0.0, max=1.0)
        heading_cutoff = float(getattr(self.cfg, "command_forward_heading_cutoff_rad", math.pi / 2.0))
        forward_speed = torch.where(torch.abs(alpha) < heading_cutoff, forward_speed, torch.zeros_like(forward_speed))

        locomotion_command = torch.stack((forward_speed, torch.zeros_like(forward_speed), yaw_rate), dim=-1)
        move_gate = torch.maximum(
            forward_speed / max(max_forward_speed, 1e-6), torch.abs(yaw_rate) / max(max_yaw_rate, 1e-6)
        ).clamp(0.0, 1.0)
        move_gate = torch.where(self._stand_mode, torch.zeros_like(move_gate), move_gate)
        stand_gate = 1.0 - move_gate

        # A latched stand still requests planted feet and freezes the gait, but
        # the actor must retain signed pose-residual observability to correct
        # small drift with stance/base motion.  These bounded correction cues
        # intentionally do not reopen the locomotion gate.
        stand_xy = float(getattr(self.cfg, "stand_correction_position_gain_s", 1.0)) * position_error_b
        stand_xy_limit = float(getattr(self.cfg, "stand_correction_max_linear_m_s", 0.08))
        stand_xy = torch.clamp(stand_xy, -stand_xy_limit, stand_xy_limit)
        stand_yaw = float(getattr(self.cfg, "stand_correction_yaw_gain_s", 1.5)) * final_yaw_error
        stand_yaw_limit = float(getattr(self.cfg, "stand_correction_max_yaw_rate_rad_s", 0.15))
        stand_yaw = torch.clamp(stand_yaw, -stand_yaw_limit, stand_yaw_limit)
        stand_command = torch.cat((stand_xy, stand_yaw.unsqueeze(-1)), dim=-1)
        command = torch.where(self._stand_mode.unsqueeze(-1), stand_command, locomotion_command)
        state_error = torch.cat((position_error_b, final_yaw_error.unsqueeze(-1)), dim=-1)
        return state_error, command, move_gate, stand_gate

    def _reconstruct_pendulum_angles_from_pose(
        self,
        base_pos_w: torch.Tensor,
        base_quat_w: torch.Tensor,
        ee_pos_w: torch.Tensor,
        hinge_offset_b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reconstruct the deployment-convention hinge angles from synchronized poses."""
        if hinge_offset_b is None:
            hinge_offset = torch.as_tensor(
                getattr(self.cfg, "pendulum_hinge_offset_b", (-0.05, 0.0, 0.06)),
                device=base_pos_w.device,
                dtype=base_pos_w.dtype,
            )
        else:
            hinge_offset = hinge_offset_b.to(device=base_pos_w.device, dtype=base_pos_w.dtype)
        pendulum_vector_b = math_utils.quat_apply_inverse(base_quat_w, ee_pos_w - base_pos_w) - hinge_offset
        return torch.stack(
            (
                torch.atan2(-pendulum_vector_b[:, 1], pendulum_vector_b[:, 2]),
                torch.atan2(
                    pendulum_vector_b[:, 0],
                    torch.hypot(pendulum_vector_b[:, 1], pendulum_vector_b[:, 2]),
                ),
            ),
            dim=-1,
        )

    def _mirror_joint_coordinates(self, values: torch.Tensor) -> torch.Tensor:
        mirrored = values[:, self._lr_joint_permutation].clone()
        mirrored[:, :4] *= -1.0
        return mirrored

    def _mirror_actions_for_selected_envs(self, actions: torch.Tensor) -> torch.Tensor:
        if not bool(getattr(self.cfg, "enable_episode_mirroring", True)):
            return actions
        mirrored = self._mirror_joint_coordinates(actions)
        return torch.where(self._episode_mirrored.unsqueeze(-1), mirrored, actions)

    def _mirror_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Express physical observations in each episode's relabeled policy frame."""
        if not bool(getattr(self.cfg, "enable_episode_mirroring", True)):
            return observations
        mirrored = observations.clone()
        # Polar vectors: lateral sign.  Axial vectors: x/z signs.
        mirrored[:, 1] *= -1.0
        mirrored[:, 3] *= -1.0
        mirrored[:, 5] *= -1.0
        mirrored[:, 7] *= -1.0
        mirrored[:, 10] *= -1.0
        mirrored[:, 11] *= -1.0
        mirrored[:, 12:24] = self._mirror_joint_coordinates(mirrored[:, 12:24])
        mirrored[:, 24:36] = self._mirror_joint_coordinates(mirrored[:, 24:36])
        # pendulum joint 1 is the sagittal-axis hinge and changes sign.
        mirrored[:, 36] *= -1.0
        mirrored[:, 38] *= -1.0
        mirrored[:, 40:52] = self._mirror_joint_coordinates(mirrored[:, 40:52])
        # Swapping trot diagonals is a half-cycle phase shift.
        mirrored[:, 52:54] *= -1.0
        return torch.where(self._episode_mirrored.unsqueeze(-1), mirrored, observations)

    def _update_mocap_estimate(
        self, base_pos_w: torch.Tensor, base_quat_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Deliver one coherent pose packet and finite-difference it at policy rate."""
        if self.cfg.use_pendulum and self._pendulum_ee_body_id is not None:
            ee_pos_w = self.robot.data.body_pos_w[:, self._pendulum_ee_body_id].clone()
        else:
            ee_pos_w = base_pos_w.clone()

        strength = self._dr_strength.unsqueeze(-1) if self.cfg.enable_domain_randomization else 0.0
        position_noise_std = float(getattr(self.cfg, "mocap_position_noise_std_m", 0.0005))
        orientation_noise_std = float(
            getattr(self.cfg, "mocap_orientation_noise_std_rad", math.radians(0.1))
        )
        measured_base_pos = base_pos_w + self._mocap_base_position_bias
        measured_ee_pos = ee_pos_w + self._mocap_ee_position_bias
        if self.cfg.enable_domain_randomization:
            measured_base_pos += torch.randn_like(measured_base_pos) * position_noise_std * strength
            measured_ee_pos += torch.randn_like(measured_ee_pos) * position_noise_std * strength

        roll, pitch, yaw = math_utils.euler_xyz_from_quat(base_quat_w)
        measured_euler = torch.stack((roll, pitch, yaw), dim=-1) + self._mocap_orientation_bias
        if self.cfg.enable_domain_randomization:
            measured_euler += torch.randn_like(measured_euler) * orientation_noise_std * strength
        measured_base_quat = math_utils.quat_from_euler_xyz(
            measured_euler[:, 0], measured_euler[:, 1], measured_euler[:, 2]
        )

        self._insert_delay_sample(self._mocap_base_pos_history, measured_base_pos)
        self._insert_delay_sample(self._mocap_base_quat_history, measured_base_quat)
        self._insert_delay_sample(self._mocap_ee_pos_history, measured_ee_pos)
        first_packet = ~self._mocap_initialized
        self._mocap_base_pos_history[first_packet] = measured_base_pos[first_packet].unsqueeze(-1)
        self._mocap_base_quat_history[first_packet] = measured_base_quat[first_packet].unsqueeze(-1)
        self._mocap_ee_pos_history[first_packet] = measured_ee_pos[first_packet].unsqueeze(-1)

        delivered_base_pos = self._read_delay_sample(
            self._mocap_base_pos_history, self._mocap_delay_steps
        ).clone()
        delivered_base_quat = self._read_delay_sample(
            self._mocap_base_quat_history, self._mocap_delay_steps
        ).clone()
        delivered_ee_pos = self._read_delay_sample(self._mocap_ee_pos_history, self._mocap_delay_steps).clone()
        hold_probability = float(getattr(self.cfg, "mocap_packet_hold_prob", 0.01))
        hold_probability_tensor = torch.clamp(hold_probability * self._dr_strength, 0.0, 1.0)
        hold_packet = (
            (torch.rand(self.num_envs, device=self.device) < hold_probability_tensor)
            & self._mocap_initialized
            & self.cfg.enable_domain_randomization
        )
        delivered_base_pos = torch.where(
            hold_packet.unsqueeze(-1), self._delivered_mocap_base_pos, delivered_base_pos
        )
        delivered_base_quat = torch.where(
            hold_packet.unsqueeze(-1), self._delivered_mocap_base_quat, delivered_base_quat
        )
        delivered_ee_pos = torch.where(
            hold_packet.unsqueeze(-1), self._delivered_mocap_ee_pos, delivered_ee_pos
        )

        base_lin_vel_w = (delivered_base_pos - self._previous_mocap_base_pos) / self.step_dt
        base_lin_vel_w[first_packet] = 0.0
        body_lin_vel_b = math_utils.quat_apply_inverse(delivered_base_quat, base_lin_vel_w)

        if self.cfg.use_pendulum and self._pendulum_dof_count > 0:
            raw_pendulum_pos = self._reconstruct_pendulum_angles_from_pose(
                delivered_base_pos,
                delivered_base_quat,
                delivered_ee_pos,
                self._sim_mocap_hinge_offset_b,
            )
            pendulum_pos = math_utils.wrap_to_pi(
                raw_pendulum_pos - self._sim_mocap_pendulum_angle_offset
            )
            pendulum_vel = math_utils.wrap_to_pi(pendulum_pos - self._previous_mocap_pendulum_pos) / self.step_dt
            pendulum_vel[first_packet] = 0.0
        else:
            pendulum_pos = torch.zeros(
                self.num_envs, self._pendulum_dof_count, device=self.device, dtype=delivered_base_pos.dtype
            )
            pendulum_vel = torch.zeros_like(pendulum_pos)

        self._delivered_mocap_base_pos.copy_(delivered_base_pos)
        self._delivered_mocap_base_quat.copy_(delivered_base_quat)
        self._delivered_mocap_ee_pos.copy_(delivered_ee_pos)
        self._previous_mocap_base_pos.copy_(delivered_base_pos)
        if self.cfg.use_pendulum and self._pendulum_dof_count > 0:
            self._previous_mocap_pendulum_pos.copy_(pendulum_pos)
        self._mocap_initialized[:] = True
        return delivered_base_pos, delivered_base_quat, body_lin_vel_b, pendulum_pos, pendulum_vel

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._imu_sensor = Imu(self.cfg.imu_sensor)
        self._pendulum_contact_sensor = None
        if self.cfg.use_pendulum:
            self._pendulum_contact_sensor = ContactSensor(self.cfg.pendulum_contact_sensor)

        # register assets and sensors so they get replicated and updated
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.sensors["imu_sensor"] = self._imu_sensor
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
        self._validate_range("mass_scale_range", self.cfg.mass_scale_range)
        if self.cfg.mass_scale_range[0] <= 0.0:
            raise ValueError(f"mass_scale_range min must be > 0. Got {self.cfg.mass_scale_range[0]}.")
        pendulum_mass_scale_range = getattr(self.cfg, "pendulum_mass_scale_range", (0.9, 1.1))
        self._validate_range("pendulum_mass_scale_range", pendulum_mass_scale_range)
        if pendulum_mass_scale_range[0] <= 0.0:
            raise ValueError(
                "pendulum_mass_scale_range min must be > 0. "
                f"Got {pendulum_mass_scale_range[0]}."
            )
        self._validate_range(
            "pendulum_effective_length_offset_range_m",
            getattr(self.cfg, "pendulum_effective_length_offset_range_m", (-0.02, 0.02)),
        )
        self._validate_range("com_offset_x_range", self.cfg.com_offset_x_range)
        self._validate_range("com_offset_y_range", self.cfg.com_offset_y_range)
        self._validate_range("com_offset_z_range", self.cfg.com_offset_z_range)
        self._validate_range("foot_friction_range", self.cfg.foot_friction_range)
        self._validate_range("motor_strength_range", self.cfg.motor_strength_range)
        self._validate_range("kp_scale_range", self.cfg.kp_scale_range)
        self._validate_range("kd_scale_range", self.cfg.kd_scale_range)
        self._validate_range("effort_limit_scale_range", self.cfg.effort_limit_scale_range)
        self._validate_range("pendulum_damping_range", self.cfg.pendulum_damping_range)
        if self.cfg.foot_friction_range[0] < 0.0:
            raise ValueError(f"foot_friction_range min must be >= 0. Got {self.cfg.foot_friction_range[0]}.")
        for name in ("motor_strength_range", "kp_scale_range", "kd_scale_range", "effort_limit_scale_range"):
            if getattr(self.cfg, name)[0] <= 0.0:
                raise ValueError(f"{name} min must be > 0. Got {getattr(self.cfg, name)[0]}.")
        if self.cfg.pendulum_damping_range[0] < 0.0:
            raise ValueError(
                "pendulum_damping_range min must be >= 0. "
                f"Got {self.cfg.pendulum_damping_range[0]}."
            )
        self._validate_step_range("action_delay_steps_range", self.cfg.action_delay_steps_range)
        self._validate_step_range("proprio_delay_steps_range", self.cfg.proprio_delay_steps_range)
        self._validate_step_range("base_lin_vel_delay_steps_range", self.cfg.base_lin_vel_delay_steps_range)
        self._validate_step_range("pendulum_delay_steps_range", self.cfg.pendulum_delay_steps_range)
        self._validate_probability("action_hold_prob", self.cfg.action_hold_prob)
        self._validate_probability("proprio_obs_hold_prob", self.cfg.proprio_obs_hold_prob)
        self._validate_probability("pendulum_obs_hold_prob", self.cfg.pendulum_obs_hold_prob)
        self._validate_step_range(
            "mocap_delay_steps_range", getattr(self.cfg, "mocap_delay_steps_range", (0, 1))
        )
        self._validate_probability(
            "mocap_packet_hold_prob", float(getattr(self.cfg, "mocap_packet_hold_prob", 0.01))
        )
        self._validate_probability(
            "episode_mirror_probability", float(getattr(self.cfg, "episode_mirror_probability", 0.5))
        )
        self._validate_range(
            "command_lpf_cutoff_range_hz", getattr(self.cfg, "command_lpf_cutoff_range_hz", (3.0, 5.0))
        )
        if float(getattr(self.cfg, "stand_exit_distance_m", 0.08)) <= float(
            getattr(self.cfg, "stand_enter_distance_m", 0.05)
        ):
            raise ValueError("stand_exit_distance_m must exceed stand_enter_distance_m for hysteresis.")
        if float(getattr(self.cfg, "stand_exit_yaw_rad", math.radians(8.0))) <= float(
            getattr(self.cfg, "stand_enter_yaw_rad", math.radians(5.0))
        ):
            raise ValueError("stand_exit_yaw_rad must exceed stand_enter_yaw_rad for hysteresis.")
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

    @staticmethod
    def _validate_step_range(name: str, value_range: tuple[int, int]) -> None:
        if value_range[0] < 0 or value_range[1] < 0:
            raise ValueError(f"{name} values must be >= 0. Got {value_range}.")
        if value_range[1] < value_range[0]:
            raise ValueError(f"{name} max must be >= min. Got {value_range[1]} < {value_range[0]}.")

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1]. Got {value}.")

    def _sample_uniform_cpu(self, value_range: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        low, high = value_range
        if high == low:
            return torch.full(shape, float(low), dtype=torch.float, device="cpu")
        return low + (high - low) * torch.rand(shape, generator=self._dr_rng, device="cpu")

    def _sample_uniform_device(
        self, value_range: tuple[float, float], shape: tuple[int, ...], device: str
    ) -> torch.Tensor:
        return self._sample_uniform_cpu(value_range, shape).to(device=device)

    def _sample_uniform_noise(self, value: float, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        if value <= 0.0:
            return torch.zeros(shape, device=self.device, dtype=dtype)
        return sample_uniform(-value, value, shape, self.device).to(dtype=dtype)

    def _sample_delay_steps(self, value_range: tuple[int, int], shape: tuple[int, ...], device: str) -> torch.Tensor:
        low, high = value_range
        if high == low:
            return torch.full(shape, int(low), dtype=torch.long, device=device)
        return torch.randint(int(low), int(high) + 1, shape, dtype=torch.long, device=device)

    def _sample_episode_dr_strength(self, env_ids: torch.Tensor) -> None:
        """Sample nominal/uniform/max DR examples for the current curriculum level."""
        if env_ids.numel() == 0:
            return
        if not self.cfg.enable_domain_randomization:
            self._dr_strength[env_ids] = 0.0
            return
        nominal_fraction = float(getattr(self.cfg, "dr_nominal_fraction", 0.20))
        uniform_fraction = float(getattr(self.cfg, "dr_uniform_fraction", 0.70))
        level_scale = float(getattr(self.cfg, "domain_randomization_scale", 1.0))
        level_scale *= float(getattr(self.cfg, "evaluation_dr_scale_multiplier", 1.0))
        level_scale = max(0.0, level_scale)
        selector = torch.rand(env_ids.numel(), device=self.device)
        strength = torch.full((env_ids.numel(),), level_scale, device=self.device)
        uniform_mask = (selector >= nominal_fraction) & (selector < nominal_fraction + uniform_fraction)
        strength[uniform_mask] *= torch.rand(env_ids.numel(), device=self.device)[uniform_mask]
        strength[selector < nominal_fraction] = 0.0
        self._dr_strength[env_ids] = strength

    def _seconds_to_steps(self, seconds: float) -> int:
        return max(1, math.ceil(seconds / self.step_dt))

    def _material_shape_ids_for_bodies(self, body_ids: Sequence[int]) -> torch.Tensor:
        num_shapes_per_body = []
        for link_path in self.robot.root_physx_view.link_paths[0]:
            link_physx_view = self.robot._physics_sim_view.create_rigid_body_view(link_path)
            num_shapes_per_body.append(link_physx_view.max_shapes)

        total_num_shapes = sum(num_shapes_per_body)
        expected_num_shapes = self.robot.root_physx_view.max_shapes
        if total_num_shapes != expected_num_shapes:
            raise RuntimeError(
                "Failed to map body material shapes. "
                f"Expected {expected_num_shapes} shapes, resolved {total_num_shapes}."
            )

        shape_ids = []
        for body_id in body_ids:
            start_idx = sum(num_shapes_per_body[:body_id])
            shape_ids.extend(range(start_idx, start_idx + num_shapes_per_body[body_id]))
        return torch.tensor(shape_ids, dtype=torch.long, device="cpu")

    def _init_domain_randomization_state(self) -> None:
        self._dr_all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        initial_dr_scale = float(getattr(self.cfg, "domain_randomization_scale", 1.0))
        initial_dr_scale *= float(getattr(self.cfg, "evaluation_dr_scale_multiplier", 1.0))
        self._dr_strength = torch.full(
            (self.num_envs,), min(max(initial_dr_scale, 0.0), 1.5), device=self.device
        )

        # Mass / COM randomization state.
        self._mass_body_ids_cpu = torch.tensor([], dtype=torch.long, device="cpu")
        if self.cfg.enable_domain_randomization and (
            self.cfg.enable_mass_randomization or self.cfg.enable_com_randomization
        ):
            mass_body_ids, _ = self.robot.find_bodies(self.cfg.mass_randomize_body_name)
            if len(mass_body_ids) == 0:
                raise RuntimeError(
                    f"Could not resolve mass/com randomization body '{self.cfg.mass_randomize_body_name}'."
                )
            self._mass_body_ids_cpu = torch.tensor(mass_body_ids, dtype=torch.long, device="cpu")
        self._pendulum_payload_body_ids_cpu = torch.tensor([], dtype=torch.long, device="cpu")
        if (
            self.cfg.enable_domain_randomization
            and self.cfg.use_pendulum
            and (
                bool(getattr(self.cfg, "enable_pendulum_mass_randomization", True))
                or bool(getattr(self.cfg, "enable_pendulum_effective_length_randomization", True))
            )
        ):
            # The USD's concentrated removable payload is the pendulum_ee body;
            # keep the mount and encoder bodies fixed while varying payload mass.
            self._pendulum_payload_body_ids_cpu = torch.tensor(
                [self._pendulum_ee_body_id], dtype=torch.long, device="cpu"
            )
        self._default_masses_cpu = self.robot.root_physx_view.get_masses().clone()
        self._default_inertias_cpu = self.robot.root_physx_view.get_inertias().clone()
        self._default_coms_cpu = self.robot.root_physx_view.get_coms().clone()

        # Foot contact material randomization state.
        self._foot_material_shape_ids_cpu = torch.tensor([], dtype=torch.long, device="cpu")
        self._default_materials_cpu = self.robot.root_physx_view.get_material_properties().clone()
        if self.cfg.enable_domain_randomization and self.cfg.enable_foot_friction_randomization:
            foot_body_ids = []
            for body_name in self.cfg.foot_friction_body_names:
                body_ids, _ = self.robot.find_bodies(body_name)
                if len(body_ids) != 1:
                    raise RuntimeError(f"Expected exactly one foot body for '{body_name}', got {body_ids}.")
                foot_body_ids.append(body_ids[0])
            self._foot_material_shape_ids_cpu = self._material_shape_ids_for_bodies(foot_body_ids)
            if self._foot_material_shape_ids_cpu.numel() == 0:
                raise RuntimeError("Foot friction randomization resolved no material shapes.")
            self._randomize_foot_friction(self._dr_all_env_ids)

        # Motor gain randomization state.
        self._motor_actuator = None
        self._motor_default_stiffness = None
        self._motor_default_damping = None
        self._motor_default_effort_limit = None
        self._motor_default_saturation_effort = None
        self._motor_num_joints = 0
        self._pd_stiffness = torch.full((self.num_envs, self._action_dim), 25.0, device=self.device)
        self._pd_damping = torch.full((self.num_envs, self._action_dim), 0.6, device=self.device)
        self._motor_strength = torch.ones((self.num_envs, self._action_dim), device=self.device)
        self._randomized_effort_limit = torch.full((self.num_envs, self._action_dim), 23.5, device=self.device)
        # The task always applies its own explicit PD torque.  Resolve and
        # disable the actuator's internal gains even in nominal evaluation so
        # turning DR off cannot silently add a second controller.
        if self.cfg.motor_gain_actuator_name:
            if self.cfg.motor_gain_actuator_name not in self.robot.actuators:
                raise RuntimeError(
                    f"Motor gain actuator '{self.cfg.motor_gain_actuator_name}' not found. "
                    f"Available: {list(self.robot.actuators.keys())}"
                )
            self._motor_actuator = self.robot.actuators[self.cfg.motor_gain_actuator_name]
            self._motor_default_stiffness = self._motor_actuator.stiffness.clone()
            self._motor_default_damping = self._motor_actuator.damping.clone()
            self._motor_default_effort_limit = self._motor_actuator.effort_limit.clone()
            if hasattr(self._motor_actuator, "_saturation_effort"):
                saturation_effort = self._motor_actuator._saturation_effort
                if torch.is_tensor(saturation_effort):
                    self._motor_default_saturation_effort = saturation_effort.clone()
                else:
                    self._motor_default_saturation_effort = torch.full_like(
                        self._motor_default_effort_limit,
                        float(saturation_effort),
                    )
            self._motor_num_joints = self._motor_default_stiffness.shape[1]
            self._pd_stiffness[:] = self._motor_default_stiffness
            self._pd_damping[:] = self._motor_default_damping
            self._randomized_effort_limit[:] = self._motor_default_effort_limit
            self._motor_actuator.stiffness[:] = 0.0
            self._motor_actuator.damping[:] = 0.0

        # Per-episode delay and packet-hold state.
        max_action_delay = int(self.cfg.action_delay_steps_range[1])
        max_proprio_delay = int(self.cfg.proprio_delay_steps_range[1])
        max_base_lin_vel_delay = int(self.cfg.base_lin_vel_delay_steps_range[1])
        max_pendulum_delay = int(self.cfg.pendulum_delay_steps_range[1])
        proprio_dim = 2 * self._action_dim
        pendulum_dim = 2 * self._pendulum_dof_count
        imu_dim = 6
        self._action_delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._proprio_delay_steps = torch.zeros_like(self._action_delay_steps)
        self._base_lin_vel_delay_steps = torch.zeros_like(self._action_delay_steps)
        self._pendulum_delay_steps = torch.zeros_like(self._action_delay_steps)
        self._action_delay_history = torch.zeros(
            self.num_envs,
            self._action_dim,
            max_action_delay + 1,
            device=self.device,
        )
        self._proprio_delay_history = torch.zeros(
            self.num_envs,
            proprio_dim,
            max_proprio_delay + 1,
            device=self.device,
        )
        self._base_lin_vel_delay_history = torch.zeros(
            self.num_envs,
            3,
            max_base_lin_vel_delay + 1,
            device=self.device,
        )
        self._pendulum_delay_history = torch.zeros(
            self.num_envs,
            pendulum_dim,
            max_pendulum_delay + 1,
            device=self.device,
        )
        self._imu_delay_history = torch.zeros(
            self.num_envs,
            imu_dim,
            max_proprio_delay + 1,
            device=self.device,
        )
        self._held_action_packet = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._delivered_proprio_obs = torch.zeros(self.num_envs, proprio_dim, device=self.device)
        self._delivered_pendulum_obs = torch.zeros(self.num_envs, pendulum_dim, device=self.device)
        self._delivered_imu_obs = torch.zeros(self.num_envs, imu_dim, device=self.device)
        self._proprio_imu_initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Sensor bias / drift state.
        self._bias_body_lin_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._bias_body_ang_vel = torch.zeros((self.num_envs, 3), device=self.device)
        self._bias_projected_gravity = torch.zeros((self.num_envs, 3), device=self.device)
        self._bias_leg_joint_pos = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._bias_leg_joint_vel = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._bias_pendulum_joint_pos = torch.zeros((self.num_envs, self._pendulum_dof_count), device=self.device)
        self._bias_pendulum_joint_vel = torch.zeros((self.num_envs, self._pendulum_dof_count), device=self.device)
        self._sample_sensor_biases(self._dr_all_env_ids)

        # Coherent motion-capture packet state.  Base and pendulum-end-effector
        # poses share delay/hold decisions; velocities are computed only after
        # delivery, exactly once per policy tick.
        max_mocap_delay = int(getattr(self.cfg, "mocap_delay_steps_range", (0, 1))[1])
        self._mocap_delay_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._mocap_base_pos_history = torch.zeros(
            self.num_envs, 3, max_mocap_delay + 1, device=self.device
        )
        self._mocap_base_quat_history = torch.zeros(
            self.num_envs, 4, max_mocap_delay + 1, device=self.device
        )
        self._mocap_ee_pos_history = torch.zeros(
            self.num_envs, 3, max_mocap_delay + 1, device=self.device
        )
        self._delivered_mocap_base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._delivered_mocap_base_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._delivered_mocap_base_quat[:, 0] = 1.0
        self._delivered_mocap_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_mocap_base_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_mocap_pendulum_pos = torch.zeros(
            self.num_envs, self._pendulum_dof_count, device=self.device
        )
        self._mocap_initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._mocap_base_position_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._mocap_ee_position_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._mocap_orientation_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._sample_mocap_biases(self._dr_all_env_ids)

        # External wrench push state.
        self._push_body_ids = torch.tensor([], dtype=torch.long, device=self.device)
        self._push_num_bodies = 0
        # Resolve the push body whenever DR is available, regardless of whether
        # the current curriculum level has pushes active.  Later levels can then
        # enable forces without leaving an empty body-id list.
        if self.cfg.enable_domain_randomization:
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

    def _randomize_mass_and_com(self, env_ids: torch.Tensor) -> None:
        if not self.cfg.enable_domain_randomization:
            return
        if self._mass_body_ids_cpu.numel() == 0 and self._pendulum_payload_body_ids_cpu.numel() == 0:
            return
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return
        body_ids_cpu = self._mass_body_ids_cpu

        if self.cfg.enable_mass_randomization:
            masses = self.robot.root_physx_view.get_masses()
            masses[env_ids_cpu[:, None], body_ids_cpu] = self._default_masses_cpu[env_ids_cpu[:, None], body_ids_cpu]
            mass_scales = self._sample_uniform_cpu(self.cfg.mass_scale_range, (env_ids_cpu.numel(), 1))
            strength = self._dr_strength[env_ids].to(device="cpu").unsqueeze(-1)
            mass_scales = 1.0 + strength * (mass_scales - 1.0)
            masses[env_ids_cpu[:, None], body_ids_cpu] *= mass_scales
            self.robot.root_physx_view.set_masses(masses, env_ids_cpu)

            if self.cfg.mass_recompute_inertia:
                inertias = self.robot.root_physx_view.get_inertias()
                inertias[env_ids_cpu[:, None], body_ids_cpu] = (
                    self._default_inertias_cpu[env_ids_cpu[:, None], body_ids_cpu] * mass_scales.unsqueeze(-1)
                )
                self.robot.root_physx_view.set_inertias(inertias, env_ids_cpu)

        if self._pendulum_payload_body_ids_cpu.numel() > 0:
            payload_body_ids_cpu = self._pendulum_payload_body_ids_cpu
            masses = self.robot.root_physx_view.get_masses()
            inertias = self.robot.root_physx_view.get_inertias()
            coms = self.robot.root_physx_view.get_coms().clone()
            masses[env_ids_cpu[:, None], payload_body_ids_cpu] = self._default_masses_cpu[
                env_ids_cpu[:, None], payload_body_ids_cpu
            ]
            inertias[env_ids_cpu[:, None], payload_body_ids_cpu] = self._default_inertias_cpu[
                env_ids_cpu[:, None], payload_body_ids_cpu
            ]
            coms[env_ids_cpu[:, None], payload_body_ids_cpu] = self._default_coms_cpu[
                env_ids_cpu[:, None], payload_body_ids_cpu
            ]
            strength = self._dr_strength[env_ids].to(device="cpu").unsqueeze(-1)
            if bool(getattr(self.cfg, "enable_pendulum_mass_randomization", True)):
                payload_mass_scales = self._sample_uniform_cpu(
                    getattr(self.cfg, "pendulum_mass_scale_range", (0.9, 1.1)),
                    (env_ids_cpu.numel(), 1),
                )
                payload_mass_scales = 1.0 + strength * (payload_mass_scales - 1.0)
                masses[env_ids_cpu[:, None], payload_body_ids_cpu] *= payload_mass_scales
                if self.cfg.mass_recompute_inertia:
                    inertias[env_ids_cpu[:, None], payload_body_ids_cpu] *= payload_mass_scales.unsqueeze(-1)

            if bool(getattr(self.cfg, "enable_pendulum_effective_length_randomization", True)):
                length_offsets = self._sample_uniform_cpu(
                    getattr(self.cfg, "pendulum_effective_length_offset_range_m", (-0.02, 0.02)),
                    (env_ids_cpu.numel(), 1),
                )
                coms[env_ids_cpu[:, None], payload_body_ids_cpu, 2] += strength * length_offsets

            self.robot.root_physx_view.set_masses(masses, env_ids_cpu)
            self.robot.root_physx_view.set_inertias(inertias, env_ids_cpu)
            self.robot.root_physx_view.set_coms(coms, env_ids_cpu)

        if self.cfg.enable_com_randomization:
            coms = self.robot.root_physx_view.get_coms().clone()
            coms[env_ids_cpu[:, None], body_ids_cpu] = self._default_coms_cpu[env_ids_cpu[:, None], body_ids_cpu]
            com_offsets = torch.zeros((env_ids_cpu.numel(), 1, 3), device="cpu")
            com_offsets[:, :, 0] = self._sample_uniform_cpu(self.cfg.com_offset_x_range, (env_ids_cpu.numel(), 1))
            com_offsets[:, :, 1] = self._sample_uniform_cpu(self.cfg.com_offset_y_range, (env_ids_cpu.numel(), 1))
            com_offsets[:, :, 2] = self._sample_uniform_cpu(self.cfg.com_offset_z_range, (env_ids_cpu.numel(), 1))
            strength = self._dr_strength[env_ids].to(device="cpu").view(-1, 1, 1)
            coms[env_ids_cpu[:, None], body_ids_cpu, :3] += com_offsets * strength
            self.robot.root_physx_view.set_coms(coms, env_ids_cpu)

    def _randomize_foot_friction(self, env_ids: torch.Tensor) -> None:
        if not (self.cfg.enable_domain_randomization and self.cfg.enable_foot_friction_randomization):
            return
        if self._foot_material_shape_ids_cpu.numel() == 0:
            return
        env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
        if env_ids_cpu.numel() == 0:
            return

        shape_ids_cpu = self._foot_material_shape_ids_cpu
        materials = self.robot.root_physx_view.get_material_properties()
        materials[env_ids_cpu[:, None], shape_ids_cpu] = self._default_materials_cpu[
            env_ids_cpu[:, None], shape_ids_cpu
        ]
        friction = self._sample_uniform_cpu(self.cfg.foot_friction_range, (env_ids_cpu.numel(), 1, 1))
        nominal = self._default_materials_cpu[env_ids_cpu[:, None], shape_ids_cpu, 0:2]
        strength = self._dr_strength[env_ids].to(device="cpu").view(-1, 1, 1)
        materials[env_ids_cpu[:, None], shape_ids_cpu, 0:2] = nominal + strength * (friction - nominal)
        self.robot.root_physx_view.set_material_properties(materials, env_ids_cpu)

    def _randomize_motor_gains(self, env_ids: torch.Tensor) -> None:
        if not (self.cfg.enable_domain_randomization and self.cfg.enable_motor_gain_randomization):
            return
        if self._motor_actuator is None:
            return
        num_envs = env_ids.numel()
        if num_envs == 0:
            return
        gain_shape = (num_envs, self._motor_num_joints) if self.cfg.motor_gain_per_joint else (num_envs, 1)
        stiffness_scale = self._sample_uniform_device(self.cfg.kp_scale_range, gain_shape, self.device)
        damping_scale = self._sample_uniform_device(self.cfg.kd_scale_range, gain_shape, self.device)
        motor_strength = self._sample_uniform_device(self.cfg.motor_strength_range, gain_shape, self.device)
        effort_limit_scale = self._sample_uniform_device(self.cfg.effort_limit_scale_range, gain_shape, self.device)
        strength = self._dr_strength[env_ids].unsqueeze(-1)
        stiffness_scale = 1.0 + strength * (stiffness_scale - 1.0)
        damping_scale = 1.0 + strength * (damping_scale - 1.0)
        motor_strength = 1.0 + strength * (motor_strength - 1.0)
        effort_limit_scale = 1.0 + strength * (effort_limit_scale - 1.0)
        self._pd_stiffness[env_ids] = self._motor_default_stiffness[env_ids] * stiffness_scale
        self._pd_damping[env_ids] = self._motor_default_damping[env_ids] * damping_scale
        self._motor_strength[env_ids] = torch.ones_like(self._motor_strength[env_ids]) * motor_strength
        self._randomized_effort_limit[env_ids] = self._motor_default_effort_limit[env_ids] * effort_limit_scale
        self._motor_actuator.stiffness[env_ids] = 0.0
        self._motor_actuator.damping[env_ids] = 0.0
        self._motor_actuator.effort_limit[env_ids] = self._randomized_effort_limit[env_ids]
        if self._motor_default_saturation_effort is not None and hasattr(self._motor_actuator, "_saturation_effort"):
            current_saturation_effort = self._motor_actuator._saturation_effort
            if torch.is_tensor(current_saturation_effort):
                saturation_effort = current_saturation_effort.clone()
            else:
                saturation_effort = self._motor_default_saturation_effort.clone()
            saturation_effort[env_ids] = self._motor_default_saturation_effort[env_ids] * effort_limit_scale
            self._motor_actuator._saturation_effort = saturation_effort
            if hasattr(self._motor_actuator, "_vel_at_effort_lim"):
                self._motor_actuator._vel_at_effort_lim = self._motor_actuator.velocity_limit * (
                    1.0 + self._motor_actuator.effort_limit / self._motor_actuator._saturation_effort
                )

    def _randomize_pendulum_damping(self, env_ids: torch.Tensor) -> None:
        if not (self.cfg.enable_domain_randomization and self.cfg.enable_pendulum_damping_randomization):
            return
        num_envs = env_ids.numel()
        if num_envs == 0 or self._pendulum_dof_ids.numel() == 0:
            return
        damping = self._sample_uniform_device(self.cfg.pendulum_damping_range, (num_envs, 1), self.device)
        damping *= self._dr_strength[env_ids].unsqueeze(-1)
        damping = damping.expand(num_envs, self._pendulum_dof_ids.numel())
        self.robot.write_joint_damping_to_sim(damping, joint_ids=self._pendulum_dof_ids, env_ids=env_ids)

    def _sample_transport_randomization(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if not self.cfg.enable_domain_randomization:
            self._action_delay_steps[env_ids] = 0
            self._proprio_delay_steps[env_ids] = 0
            self._base_lin_vel_delay_steps[env_ids] = 0
            self._pendulum_delay_steps[env_ids] = 0
            self._mocap_delay_steps[env_ids] = 0
            return
        num_envs = env_ids.numel()
        strength = torch.clamp(self._dr_strength[env_ids], 0.0, 1.0)
        self._action_delay_steps[env_ids] = torch.round(
            self._sample_delay_steps(self.cfg.action_delay_steps_range, (num_envs,), self.device) * strength
        ).long()
        self._proprio_delay_steps[env_ids] = torch.round(
            self._sample_delay_steps(self.cfg.proprio_delay_steps_range, (num_envs,), self.device) * strength
        ).long()
        self._base_lin_vel_delay_steps[env_ids] = 0
        self._pendulum_delay_steps[env_ids] = 0
        mocap_delay_range = getattr(self.cfg, "mocap_delay_steps_range", (0, 1))
        self._mocap_delay_steps[env_ids] = torch.round(
            self._sample_delay_steps(mocap_delay_range, (num_envs,), self.device) * strength
        ).long()

    def _sample_sensor_biases(self, env_ids: torch.Tensor) -> None:
        if not (self.cfg.enable_domain_randomization and self.cfg.enable_sensor_bias_drift):
            return
        num_envs = env_ids.numel()
        if num_envs == 0:
            return
        strength = self._dr_strength[env_ids].unsqueeze(-1)
        self._bias_body_lin_vel[env_ids] = strength * self._sample_uniform_device(
            (-self.cfg.base_lin_vel_bias_m_s, self.cfg.base_lin_vel_bias_m_s), (num_envs, 3), self.device
        )
        self._bias_body_ang_vel[env_ids] = strength * self._sample_uniform_device(
            (-self.cfg.base_ang_vel_bias_rad_s, self.cfg.base_ang_vel_bias_rad_s), (num_envs, 3), self.device
        )
        self._bias_projected_gravity[env_ids] = 0.0
        self._bias_leg_joint_pos[env_ids] = strength * self._sample_uniform_device(
            (-self.cfg.joint_pos_bias_rad, self.cfg.joint_pos_bias_rad),
            (num_envs, self._action_dim),
            self.device,
        )
        self._bias_leg_joint_vel[env_ids] = strength * self._sample_uniform_device(
            (-self.cfg.joint_vel_bias_rad_s, self.cfg.joint_vel_bias_rad_s),
            (num_envs, self._action_dim),
            self.device,
        )
        if self._pendulum_dof_count > 0:
            self._bias_pendulum_joint_pos[env_ids] = strength * self._sample_uniform_device(
                (-self.cfg.pendulum_pos_bias_rad, self.cfg.pendulum_pos_bias_rad),
                (num_envs, self._pendulum_dof_count),
                self.device,
            )
            self._bias_pendulum_joint_vel[env_ids] = strength * self._sample_uniform_device(
                (-self.cfg.pendulum_vel_bias_rad_s, self.cfg.pendulum_vel_bias_rad_s),
                (num_envs, self._pendulum_dof_count),
                self.device,
            )

    def _sample_mocap_biases(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        if not self.cfg.enable_domain_randomization:
            self._mocap_base_position_bias[env_ids] = 0.0
            self._mocap_ee_position_bias[env_ids] = 0.0
            self._mocap_orientation_bias[env_ids] = 0.0
            return
        strength = self._dr_strength[env_ids].unsqueeze(-1)
        position_bias = float(getattr(self.cfg, "mocap_position_bias_range_m", 0.002))
        orientation_bias = float(getattr(self.cfg, "mocap_orientation_bias_range_rad", math.radians(0.25)))
        self._mocap_base_position_bias[env_ids] = strength * self._sample_uniform_device(
            (-position_bias, position_bias), (env_ids.numel(), 3), self.device
        )
        self._mocap_ee_position_bias[env_ids] = strength * self._sample_uniform_device(
            (-position_bias, position_bias), (env_ids.numel(), 3), self.device
        )
        self._mocap_orientation_bias[env_ids] = strength * self._sample_uniform_device(
            (-orientation_bias, orientation_bias), (env_ids.numel(), 3), self.device
        )

    def _update_sensor_bias_drift(self) -> None:
        if not (self.cfg.enable_domain_randomization and self.cfg.enable_sensor_bias_drift):
            return
        drift_scale = math.sqrt(self.step_dt)
        strength = self._dr_strength.unsqueeze(-1)
        self._bias_body_ang_vel += strength * torch.randn_like(self._bias_body_ang_vel) * (
            self.cfg.imu_ang_vel_drift_std_per_s * drift_scale
        )
        self._bias_leg_joint_pos += strength * torch.randn_like(self._bias_leg_joint_pos) * (
            self.cfg.encoder_joint_pos_drift_std_per_s * drift_scale
        )
        self._bias_leg_joint_vel += strength * torch.randn_like(self._bias_leg_joint_vel) * (
            self.cfg.encoder_joint_vel_drift_std_per_s * drift_scale
        )
        if self._pendulum_dof_count > 0:
            self._bias_pendulum_joint_pos += strength * torch.randn_like(self._bias_pendulum_joint_pos) * (
                self.cfg.encoder_pendulum_pos_drift_std_per_s * drift_scale
            )
            self._bias_pendulum_joint_vel += strength * torch.randn_like(self._bias_pendulum_joint_vel) * (
                self.cfg.encoder_pendulum_vel_drift_std_per_s * drift_scale
            )

        self._bias_body_ang_vel = torch.clamp(
            self._bias_body_ang_vel, -self.cfg.imu_ang_vel_bias_range, self.cfg.imu_ang_vel_bias_range
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

    def _insert_delay_sample(self, history: torch.Tensor, sample: torch.Tensor) -> None:
        if history.shape[-1] > 1:
            history[:, :, 1:] = history[:, :, :-1].clone()
        history[:, :, 0] = sample

    def _read_delay_sample(self, history: torch.Tensor, delay_steps: torch.Tensor) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return history[env_ids, :, delay_steps]

    def _maybe_hold_packet(self, sample: torch.Tensor, previous_sample: torch.Tensor, hold_prob: float) -> torch.Tensor:
        if not self.cfg.enable_domain_randomization or hold_prob <= 0.0:
            return sample
        hold_probability = torch.clamp(hold_prob * self._dr_strength, 0.0, 1.0).unsqueeze(-1)
        hold_mask = torch.rand((self.num_envs, 1), device=self.device) < hold_probability
        return torch.where(hold_mask, previous_sample, sample)

    def _reset_transport_buffers(
        self,
        env_ids: torch.Tensor,
        leg_joint_pos: torch.Tensor,
        leg_joint_vel: torch.Tensor,
        pendulum_joint_pos: torch.Tensor,
        pendulum_joint_vel: torch.Tensor,
        imu_packet: torch.Tensor,
    ) -> None:
        self._sample_transport_randomization(env_ids)
        self._action_delay_history[env_ids] = 0.0
        self._held_action_packet[env_ids] = 0.0
        proprio_sample = torch.cat([leg_joint_pos, leg_joint_vel], dim=-1)
        self._proprio_delay_history[env_ids] = proprio_sample.unsqueeze(-1)
        self._delivered_proprio_obs[env_ids] = proprio_sample
        if self._pendulum_dof_count > 0:
            pendulum_sample = torch.cat([pendulum_joint_pos, pendulum_joint_vel], dim=-1)
            self._pendulum_delay_history[env_ids] = pendulum_sample.unsqueeze(-1)
            self._delivered_pendulum_obs[env_ids] = pendulum_sample
        self._imu_delay_history[env_ids] = imu_packet.unsqueeze(-1)
        self._delivered_imu_obs[env_ids] = imu_packet
        self._base_lin_vel_delay_history[env_ids] = 0.0

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

        push_start_delay_steps = self._seconds_to_steps(float(getattr(self.cfg, "push_start_delay_s", 2.0)))
        start_push = (
            (self._push_end_step == 0)
            & (now_step >= self._push_next_step)
            & (now_step >= push_start_delay_steps)
        )
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
            max_force_x = max(abs(self.cfg.push_force_x_range[0]), abs(self.cfg.push_force_x_range[1]))
            max_force_y = max(abs(self.cfg.push_force_y_range[0]), abs(self.cfg.push_force_y_range[1]))
            if max_force_x > 0.0 and math.isclose(max_force_x, max_force_y):
                direction = self._sample_uniform_device(
                    (0.0, 2.0 * math.pi), (env_ids.numel(), self._push_num_bodies), self.device
                )
                magnitude = self._sample_uniform_device(
                    (0.5 * max_force_x, max_force_x),
                    (env_ids.numel(), self._push_num_bodies),
                    self.device,
                )
                push_forces[:, :, 0] = magnitude * torch.cos(direction)
                push_forces[:, :, 1] = magnitude * torch.sin(direction)
            else:
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
            level_dr_scale = float(getattr(self.cfg, "domain_randomization_scale", 1.0))
            level_dr_scale *= float(getattr(self.cfg, "evaluation_dr_scale_multiplier", 1.0))
            normalized_episode_strength = self._dr_strength[env_ids] / max(level_dr_scale, 1e-6)
            push_scale = normalized_episode_strength.clamp(0.0, 1.0).view(-1, 1, 1)
            push_scale *= float(getattr(self.cfg, "evaluation_push_scale_multiplier", 1.0))
            push_forces *= push_scale
            push_torques *= push_scale
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
        # Policies in relabeled episodes act in mirrored coordinates.  Convert
        # back to the physical robot before all transport and safety dynamics.
        action_packet = self._mirror_actions_for_selected_envs(actions.clone())
        action_clip = float(getattr(self.cfg, "action_clip", 2.0))
        action_packet = torch.clamp(action_packet, -action_clip, action_clip)
        action_packet = self._maybe_hold_packet(action_packet, self._held_action_packet, self.cfg.action_hold_prob)
        self._held_action_packet = action_packet.clone()
        self._insert_delay_sample(self._action_delay_history, action_packet)
        delayed_action = self._read_delay_sample(self._action_delay_history, self._action_delay_steps).clone()

        lpf_alpha = 1.0 - torch.exp(-2.0 * math.pi * self._command_lpf_cutoff_hz * self.step_dt)
        self._filtered_action += lpf_alpha.unsqueeze(-1) * (delayed_action - self._filtered_action)
        default_joint_pos = self.robot.data.default_joint_pos[:, self._leg_dof_ids]
        requested_joint_pos = default_joint_pos + self.cfg.action_scale * self._filtered_action

        hard_limits = self.robot.data.joint_pos_limits[:, self._leg_dof_ids, :]
        limit_margin = float(getattr(self.cfg, "joint_limit_margin_rad", 0.1))
        lower = hard_limits[:, :, 0] + limit_margin
        upper = hard_limits[:, :, 1] - limit_margin
        invalid_margin = lower > upper
        midpoint = 0.5 * (hard_limits[:, :, 0] + hard_limits[:, :, 1])
        lower = torch.where(invalid_margin, midpoint, lower)
        upper = torch.where(invalid_margin, midpoint, upper)
        requested_joint_pos = torch.maximum(torch.minimum(requested_joint_pos, upper), lower)

        slew_per_step = float(getattr(self.cfg, "joint_target_slew_rate_rad_s", 6.0)) * self.step_dt
        joint_delta = torch.clamp(requested_joint_pos - self.desired_joint_pos, -slew_per_step, slew_per_step)
        self.desired_joint_pos = self.desired_joint_pos + joint_delta
        self._physical_delivered_action = (self.desired_joint_pos - default_joint_pos) / self.cfg.action_scale
        # Observation 40:52 is the actually delivered action expressed back in
        # the policy's (possibly mirrored) coordinate system.
        self.last_action = self._mirror_actions_for_selected_envs(self._physical_delivered_action.clone())

    def _apply_action(self) -> None:
        q = self.robot.data.joint_pos[:, self._leg_dof_ids]
        dq = self.robot.data.joint_vel[:, self._leg_dof_ids]
        desired_torque = self._motor_strength * (
            self._pd_stiffness * (self.desired_joint_pos - q) - self._pd_damping * dq
        )
        torque = torch.clamp(desired_torque, -self._randomized_effort_limit, self._randomized_effort_limit)
        self.robot.set_joint_position_target(q, joint_ids=self._leg_dof_ids)
        self.robot.set_joint_velocity_target(dq, joint_ids=self._leg_dof_ids)
        self.robot.set_joint_effort_target(torque, joint_ids=self._leg_dof_ids)

    def _update_curriculum(self) -> None:
        """Update difficulty level based on training progress."""
        if not self.cfg.enable_curriculum or self.cfg.curriculum_total_steps <= 0:
            return
        curriculum_step = int(getattr(self.cfg, "curriculum_start_step", 0)) + self.common_step_counter
        progress = min(1.0, max(0.0, curriculum_step / self.cfg.curriculum_total_steps))

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
        """Apply a validated task/distribution preset without changing mechanics."""
        preset = self._DIFFICULTY_PRESETS[level]
        for key, value in preset.items():
            if key.endswith("_range") and isinstance(value, tuple) and len(value) == 2:
                self._validate_range(key, value)
        mixture = preset["goal_distance_mixture"]
        if any(value < 0.0 for value in mixture) or not math.isclose(sum(mixture), 1.0, abs_tol=1e-6):
            raise ValueError(f"Difficulty {level} goal_distance_mixture must be non-negative and sum to one.")
        for key, value in preset.items():
            setattr(self.cfg, key, value)

        # A level may toggle pushes on after initialization.  Clear stale
        # wrenches and schedule from the current per-episode time in that case.
        if hasattr(self, "_push_forces"):
            self._push_forces.zero_()
            self._push_torques.zero_()
            self._push_end_step.zero_()
            self._schedule_next_push(self._dr_all_env_ids, self._steps_since_reset)
        print(f"[Curriculum] Switched to difficulty level {level} at step {self.common_step_counter}")

    def _get_observations(self) -> dict:
        """Build the v2 asymmetric observation contract (56 dimensions)."""
        if (
            self._observation_cache is not None
            and self._observation_cache_step == self.common_step_counter
            and self._observation_cache_reset_generation == self._observation_reset_generation
        ):
            # Return a fresh mapping so framework-side replacement of one group
            # (for example an optional noise model) cannot mutate our cache.
            return dict(self._observation_cache)

        self._update_curriculum()

        leg_joint_pos = (
            self.robot.data.joint_pos[:, self._leg_dof_ids]
            - self.robot.data.default_joint_pos[:, self._leg_dof_ids]
        )
        leg_joint_vel = self.robot.data.joint_vel[:, self._leg_dof_ids]
        if self.cfg.use_pendulum and self._pendulum_dof_ids.numel() > 0:
            critic_pendulum_pos = self.robot.data.joint_pos[:, self._pendulum_dof_ids]
            critic_pendulum_vel = self.robot.data.joint_vel[:, self._pendulum_dof_ids]
        else:
            critic_pendulum_pos = torch.zeros(
                self.num_envs, self._pendulum_dof_count, device=self.device, dtype=leg_joint_pos.dtype
            )
            critic_pendulum_vel = torch.zeros_like(critic_pendulum_pos)

        env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_quat_w = self.robot.data.root_quat_w.clone()
        mocap_pos_w, mocap_quat_w, actor_body_lin_vel, actor_pendulum_pos, actor_pendulum_vel = (
            self._update_mocap_estimate(base_pos_w, base_quat_w)
        )

        _, _, mocap_yaw = math_utils.euler_xyz_from_quat(mocap_quat_w)
        _, actor_command, move_gate, stand_gate = self._compute_navigation_command(
            mocap_pos_w[:, :2] - env_origins[:, :2], mocap_yaw, update_stand_state=True
        )
        self._actor_command.copy_(actor_command)
        self._move_gate.copy_(move_gate)
        self._stand_gate.copy_(stand_gate)
        self._update_gait_targets()

        _, _, base_yaw = math_utils.euler_xyz_from_quat(base_quat_w)
        if self.target_state is not None:
            critic_position_error, critic_yaw_error = self._compute_goal_error_terms(
                base_pos_w[:, :2] - env_origins[:, :2],
                base_yaw,
                self.target_state[:, :2],
                self.target_state[:, 2],
            )
            critic_state_error = torch.cat((critic_position_error, critic_yaw_error.unsqueeze(-1)), dim=-1)
        else:
            critic_state_error = torch.zeros(
                self.num_envs, 3, device=self.device, dtype=leg_joint_pos.dtype
            )

        critic_body_lin_vel = self.robot.data.root_lin_vel_b.clone()
        critic_body_ang_vel = self._imu_sensor.data.ang_vel_b.clone()
        critic_projected_gravity = math_utils.quat_apply_inverse(
            self._imu_sensor.data.quat_w, self._world_gravity_dir
        )

        # Encoder and IMU packets retain their own transport model.  Mocap pose
        # delay/holds occur before finite differences and are not randomized a
        # second time as independent velocity/pendulum signals.
        self._update_sensor_bias_drift()
        proprio_clean = torch.cat((leg_joint_pos, leg_joint_vel), dim=-1)
        self._insert_delay_sample(self._proprio_delay_history, proprio_clean)
        actor_proprio = self._read_delay_sample(self._proprio_delay_history, self._proprio_delay_steps).clone()

        imu_clean = torch.cat((critic_body_ang_vel, critic_projected_gravity), dim=-1)
        self._insert_delay_sample(self._imu_delay_history, imu_clean)
        actor_imu = self._read_delay_sample(self._imu_delay_history, self._proprio_delay_steps).clone()

        if self.cfg.enable_domain_randomization and self.cfg.enable_sensor_bias_drift:
            actor_imu[:, :3] += self._bias_body_ang_vel
            actor_imu[:, 3:] += self._bias_projected_gravity
            actor_proprio[:, : self._action_dim] += self._bias_leg_joint_pos
            actor_proprio[:, self._action_dim :] += self._bias_leg_joint_vel
        if self.cfg.enable_domain_randomization:
            noise_strength = self._dr_strength.unsqueeze(-1)
            actor_imu[:, :3] += noise_strength * self._sample_uniform_noise(
                self.cfg.base_ang_vel_noise_rad_s, (self.num_envs, 3), actor_imu.dtype
            )
            actor_imu[:, 3:] += noise_strength * self._sample_uniform_noise(
                self.cfg.projected_gravity_component_noise, (self.num_envs, 3), actor_imu.dtype
            )
            actor_proprio[:, : self._action_dim] += noise_strength * self._sample_uniform_noise(
                self.cfg.joint_pos_noise_rad, (self.num_envs, self._action_dim), actor_proprio.dtype
            )
            actor_proprio[:, self._action_dim :] += noise_strength * self._sample_uniform_noise(
                self.cfg.joint_vel_noise_rad_s, (self.num_envs, self._action_dim), actor_proprio.dtype
            )

        # Packet holds repeat the exact previously delivered packet, including
        # its bias/noise realization.  Cache only after the full sensor model.
        modeled_actor_proprio = actor_proprio
        modeled_actor_imu = actor_imu
        actor_proprio = self._maybe_hold_packet(
            actor_proprio, self._delivered_proprio_obs, self.cfg.proprio_obs_hold_prob
        )
        actor_imu = self._maybe_hold_packet(actor_imu, self._delivered_imu_obs, self.cfg.proprio_obs_hold_prob)
        first_proprio_imu_packet = ~self._proprio_imu_initialized
        actor_proprio = torch.where(
            first_proprio_imu_packet.unsqueeze(-1),
            modeled_actor_proprio,
            actor_proprio,
        )
        actor_imu = torch.where(
            first_proprio_imu_packet.unsqueeze(-1),
            modeled_actor_imu,
            actor_imu,
        )
        self._delivered_proprio_obs.copy_(actor_proprio)
        self._delivered_imu_obs.copy_(actor_imu)
        self._proprio_imu_initialized[:] = True

        # Build in the physical frame first, then relabel the complete vector so
        # actor/critic/action/clock transforms cannot drift apart.
        policy_obs = torch.cat(
            (
                actor_body_lin_vel,
                actor_imu[:, :3],
                actor_imu[:, 3:],
                actor_command,
                actor_proprio[:, : self._action_dim],
                actor_proprio[:, self._action_dim :],
                actor_pendulum_pos,
                actor_pendulum_vel,
                self._physical_delivered_action,
                self.clock_inputs,
            ),
            dim=-1,
        )
        critic_obs = torch.cat(
            (
                critic_body_lin_vel,
                critic_body_ang_vel,
                critic_projected_gravity,
                critic_state_error,
                leg_joint_pos,
                leg_joint_vel,
                critic_pendulum_pos,
                critic_pendulum_vel,
                self._physical_delivered_action,
                self.clock_inputs,
            ),
            dim=-1,
        )
        policy_obs = self._mirror_observations(policy_obs)
        critic_obs = self._mirror_observations(critic_obs)
        if policy_obs.shape[1] != 56 or critic_obs.shape[1] != 56:
            raise RuntimeError(
                f"V2 observation contract must be 56D, got policy={policy_obs.shape[1]}, critic={critic_obs.shape[1]}."
            )
        self._observation_cache = {"policy": policy_obs, "critic": critic_obs}
        self._observation_cache_step = self.common_step_counter
        self._observation_cache_reset_generation = self._observation_reset_generation
        return dict(self._observation_cache)

    def _compute_base_tilt_rad(self) -> torch.Tensor:
        projected_gravity_b = self.robot.data.projected_gravity_b
        return torch.atan2(torch.linalg.norm(projected_gravity_b[:, :2], dim=1), -projected_gravity_b[:, 2])

    def _get_rewards(self) -> torch.Tensor:
        """Compute command-gated locomotion and planted-stance rewards."""
        env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
        base_pos_xy = self.robot.data.root_pos_w[:, :2] - env_origins[:, :2]
        _, _, base_yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        if self.target_state is not None:
            position_error_b, yaw_error = self._compute_goal_error_terms(
                base_pos_xy, base_yaw, self.target_state[:, :2], self.target_state[:, 2]
            )
            position_error = torch.linalg.norm(position_error_b, dim=-1)
        else:
            position_error = torch.zeros(self.num_envs, device=self.device)
            yaw_error = torch.zeros_like(position_error)

        body_lin_vel = self.robot.data.root_lin_vel_b
        body_ang_vel = self.robot.data.root_ang_vel_b
        command_lin_error = body_lin_vel[:, :2] - self._actor_command[:, :2]
        command_lin_reward = torch.exp(
            -torch.sum(torch.square(command_lin_error), dim=-1)
            / max(float(self.cfg.command_lin_vel_reward_sigma) ** 2, 1e-6)
        )
        command_yaw_error = body_ang_vel[:, 2] - self._actor_command[:, 2]
        command_yaw_reward = torch.exp(
            -torch.square(command_yaw_error) / max(float(self.cfg.command_yaw_rate_reward_sigma) ** 2, 1e-6)
        )
        goal_position_hold = torch.exp(
            -torch.square(position_error) / max(float(self.cfg.goal_position_hold_reward_sigma) ** 2, 1e-6)
        ) * self._stand_gate
        goal_yaw_hold = torch.exp(
            -torch.square(yaw_error) / max(float(self.cfg.goal_yaw_hold_reward_sigma) ** 2, 1e-6)
        ) * self._stand_gate
        stand_lin_settling = torch.exp(
            -torch.sum(torch.square(command_lin_error), dim=-1)
            / max(float(self.cfg.stand_lin_vel_reward_sigma) ** 2, 1e-6)
        )
        stand_yaw_settling = torch.exp(
            -torch.square(command_yaw_error) / max(float(self.cfg.stand_yaw_rate_reward_sigma) ** 2, 1e-6)
        )
        stand_settling = 0.5 * (stand_lin_settling + stand_yaw_settling) * self._stand_gate

        if self.cfg.use_pendulum and self._pendulum_dof_ids.numel() > 0:
            pendulum_pos = self.robot.data.joint_pos[:, self._pendulum_dof_ids]
            pendulum_vel = self.robot.data.joint_vel[:, self._pendulum_dof_ids]
            pendulum_angle = torch.linalg.norm(pendulum_pos, dim=-1)
            pendulum_speed = torch.linalg.norm(pendulum_vel, dim=-1)
            pendulum_upright = torch.exp(
                -torch.sum(torch.square(pendulum_pos), dim=-1)
                / max(float(self.cfg.pendulum_upright_reward_sigma) ** 2, 1e-6)
            )
            pendulum_velocity = torch.sum(torch.square(pendulum_vel), dim=-1)
        else:
            pendulum_angle = torch.zeros(self.num_envs, device=self.device)
            pendulum_speed = torch.zeros_like(pendulum_angle)
            pendulum_upright = torch.zeros_like(pendulum_angle)
            pendulum_velocity = torch.zeros_like(pendulum_angle)

        angle_style = 1.0 - torch.clamp(
            (pendulum_angle - float(self.cfg.style_pendulum_angle_full_rad))
            / max(
                float(self.cfg.style_pendulum_angle_zero_rad)
                - float(self.cfg.style_pendulum_angle_full_rad),
                1e-6,
            ),
            0.0,
            1.0,
        )
        speed_style = 1.0 - torch.clamp(
            (pendulum_speed - float(self.cfg.style_pendulum_speed_full_rad_s))
            / max(
                float(self.cfg.style_pendulum_speed_zero_rad_s)
                - float(self.cfg.style_pendulum_speed_full_rad_s),
                1e-6,
            ),
            0.0,
            1.0,
        )
        recovery_style_gate = angle_style * speed_style

        foot_forces = torch.linalg.norm(
            self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :], dim=-1
        )
        contact_threshold = float(self.cfg.foot_contact_force_threshold_n)
        contact_probability = torch.sigmoid((foot_forces - contact_threshold) / max(0.25 * contact_threshold, 1e-3))
        strict_contact = foot_forces > contact_threshold
        stand_style_gate = self._stand_gate * recovery_style_gate
        stand_contacts = torch.mean(contact_probability, dim=-1) * stand_style_gate

        foot_height = self.foot_positions_w[:, :, 2] - env_origins[:, 2].unsqueeze(-1)
        lift_threshold = max(float(self.cfg.stand_foot_lift_threshold_m), 1e-4)
        normalized_lift = torch.relu(foot_height - lift_threshold) / lift_threshold
        stand_foot_lift = torch.mean(torch.square(normalized_lift), dim=-1) * stand_style_gate

        foot_velocity_xy = self.robot.data.body_lin_vel_w[:, self._feet_ids, :2]
        foot_slip = torch.mean(contact_probability * torch.sum(torch.square(foot_velocity_xy), dim=-1), dim=-1)
        gait_contact_mismatch = torch.mean(
            torch.abs(contact_probability - self.desired_contact_states), dim=-1
        ) * self._move_gate

        duty = self._gait_duty_factor.unsqueeze(-1)
        swing_phase = torch.clamp(
            (self.foot_indices - duty) / torch.clamp(1.0 - duty, min=1e-4), 0.0, 1.0
        )
        target_foot_height = float(self.cfg.gait_stance_height_m) + float(
            self.cfg.gait_swing_height_m
        ) * torch.sin(math.pi * swing_phase)
        swing_weight = (1.0 - self.desired_contact_states) * self._move_gate.unsqueeze(-1)
        feet_clearance = torch.mean(
            swing_weight
            * torch.square(
                (foot_height - target_foot_height) / max(float(self.cfg.gait_swing_height_m), 1e-4)
            ),
            dim=-1,
        )

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids_sensor]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids_sensor]
        target_air_time = (1.0 - self._gait_duty_factor) / torch.clamp(self._gait_frequency, min=1e-4)
        air_time_sigma = torch.clamp(0.25 * target_air_time, min=0.02)
        feet_air_time = torch.mean(
            torch.exp(-torch.square((last_air_time - target_air_time.unsqueeze(-1)) / air_time_sigma.unsqueeze(-1)))
            * first_contact.float(),
            dim=-1,
        ) * self._move_gate

        q = self.robot.data.joint_pos[:, self._leg_dof_ids] - self.robot.data.default_joint_pos[:, self._leg_dof_ids]
        joint_symmetry_terms = torch.stack(
            (q[:, 0] + q[:, 1], q[:, 2] + q[:, 3], q[:, 4] - q[:, 5], q[:, 6] - q[:, 7],
             q[:, 8] - q[:, 9], q[:, 10] - q[:, 11]),
            dim=-1,
        )
        foot_relative_w = self.foot_positions_w - self.robot.data.root_pos_w.unsqueeze(1)
        base_quat_repeated = self.robot.data.root_quat_w.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4)
        foot_relative_b = math_utils.quat_apply_inverse(
            base_quat_repeated, foot_relative_w.reshape(-1, 3)
        ).reshape(self.num_envs, 4, 3)
        foot_symmetry_terms = torch.stack(
            (
                foot_relative_b[:, 0, 0] - foot_relative_b[:, 1, 0],
                foot_relative_b[:, 0, 1] + foot_relative_b[:, 1, 1],
                foot_relative_b[:, 0, 2] - foot_relative_b[:, 1, 2],
                foot_relative_b[:, 2, 0] - foot_relative_b[:, 3, 0],
                foot_relative_b[:, 2, 1] + foot_relative_b[:, 3, 1],
                foot_relative_b[:, 2, 2] - foot_relative_b[:, 3, 2],
            ),
            dim=-1,
        )
        stand_symmetry = (
            torch.mean(torch.square(joint_symmetry_terms / 0.10), dim=-1)
            + torch.mean(torch.square(foot_symmetry_terms / 0.05), dim=-1)
        ).clamp(max=20.0) * stand_style_gate

        # Arrival requires sustained settled balance.  Once achieved it pays a
        # continuous dwell reward without terminating the episode.
        arrived_now = (
            (position_error <= float(self.cfg.arrival_position_tolerance_m))
            & (torch.abs(yaw_error) <= float(self.cfg.arrival_yaw_tolerance_rad))
            & (torch.linalg.norm(body_lin_vel[:, :2], dim=-1) <= float(self.cfg.arrival_base_speed_tolerance_m_s))
            & (torch.abs(body_ang_vel[:, 2]) <= float(self.cfg.arrival_yaw_rate_tolerance_rad_s))
            & (pendulum_angle <= float(self.cfg.arrival_pendulum_angle_tolerance_rad))
            & (pendulum_speed <= float(self.cfg.arrival_pendulum_speed_tolerance_rad_s))
        )
        self._arrival_dwell_steps = torch.where(
            arrived_now, self._arrival_dwell_steps + 1, torch.zeros_like(self._arrival_dwell_steps)
        )
        required_dwell_steps = self._seconds_to_steps(float(self.cfg.arrival_dwell_time_s))
        dwell_achieved = self._arrival_dwell_steps >= required_dwell_steps
        newly_achieved = dwell_achieved & ~self._current_goal_success_recorded
        self._episode_goal_success_count += newly_achieved.long()
        self._current_goal_success_recorded |= dwell_achieved
        self._episode_arrival_success |= dwell_achieved
        arrival_dwell = dwell_achieved.float()

        stand_step = self._stand_mode
        all_four_contact = torch.all(strict_contact, dim=-1)
        self._episode_four_contact_sum += (all_four_contact & stand_step).float()
        self._episode_stand_steps += stand_step.long()
        foot_lift_now = torch.any(foot_height > lift_threshold, dim=-1) & stand_step
        self._episode_foot_lift_events += (foot_lift_now & ~self._previous_stand_foot_lift).long()
        self._previous_stand_foot_lift = foot_lift_now

        base_height = self.robot.data.root_pos_w[:, 2] - env_origins[:, 2]
        base_height_reward = torch.exp(
            -torch.square(self.cfg.base_height_target - base_height)
            / max(float(self.cfg.base_height_reward_sigma) ** 2, 1e-6)
        )
        self._episode_base_height_sum += base_height
        self._episode_base_height_count += 1
        base_tilt_deg = torch.rad2deg(self._compute_base_tilt_rad())
        self._episode_base_tilt_deg_sum += base_tilt_deg
        self._episode_base_tilt_deg_count += 1
        self._episode_pendulum_angle_deg_sum += torch.rad2deg(pendulum_angle)
        self._episode_pendulum_angle_deg_count += 1
        self._episode_pendulum_speed_deg_s_sum += torch.rad2deg(pendulum_speed)
        self._episode_pendulum_speed_deg_s_count += 1

        action_magnitude = torch.sum(torch.square(self.last_action), dim=-1) * self.cfg.action_scale**2
        action_rate = torch.sum(torch.square(self.last_action - self._action_history[:, :, 0]), dim=-1) * (
            self.cfg.action_scale**2
        )
        action_acc = torch.sum(
            torch.square(self.last_action - 2.0 * self._action_history[:, :, 0] + self._action_history[:, :, 1]),
            dim=-1,
        ) * self.cfg.action_scale**2
        current_torque = self.robot.data.applied_torque[:, self._leg_dof_ids]
        torque = torch.sum(torch.square(current_torque), dim=-1)
        torque_rate = torch.sum(torch.square(current_torque - self._prev_torque), dim=-1)
        self._prev_torque = current_torque.clone()
        self._action_history = torch.roll(self._action_history, 1, 2)
        self._action_history[:, :, 0] = self.last_action

        orientation_error = torch.sum(torch.square(self.robot.data.projected_gravity_b[:, :2]), dim=-1)
        orient = torch.exp(-orientation_error / max(float(self.cfg.orient_reward_sigma), 1e-6))
        lin_vel_z = torch.square(self.robot.data.root_lin_vel_b[:, 2])
        dof_vel = torch.sum(torch.square(self.robot.data.joint_vel[:, self._leg_dof_ids]), dim=-1)
        dof_acc = torch.sum(torch.square(self.robot.data.joint_acc[:, self._leg_dof_ids]), dim=-1)
        ang_vel_xy = torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=-1)
        if self._undesired_contact_body_ids is not None:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            undesired = (
                torch.max(
                    torch.linalg.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1
                )[0]
                > 1.0
            )
            undesired_contacts = torch.sum(undesired, dim=-1)
        else:
            undesired_contacts = torch.zeros(self.num_envs, device=self.device)

        dt = self.step_dt
        rewards = {
            "command_lin_vel": command_lin_reward * self.cfg.command_lin_vel_reward_scale * dt,
            "command_yaw_rate": command_yaw_reward * self.cfg.command_yaw_rate_reward_scale * dt,
            "goal_position_hold": goal_position_hold * self.cfg.goal_position_hold_reward_scale * dt,
            "goal_yaw_hold": goal_yaw_hold * self.cfg.goal_yaw_hold_reward_scale * dt,
            "stand_settling": stand_settling * self.cfg.stand_settling_reward_scale * dt,
            "stand_contacts": stand_contacts * self.cfg.stand_contact_reward_scale * dt,
            "stand_foot_lift": stand_foot_lift * self.cfg.stand_foot_lift_reward_scale * dt,
            "stand_symmetry": stand_symmetry * self.cfg.stand_symmetry_reward_scale * dt,
            "foot_slip": foot_slip * self.cfg.foot_slip_reward_scale * dt,
            "gait_contact_mismatch": gait_contact_mismatch * self.cfg.gait_contact_mismatch_reward_scale * dt,
            "pendulum_upright": pendulum_upright * self.cfg.pendulum_upright_reward_scale * dt,
            "pendulum_velocity": pendulum_velocity * self.cfg.pendulum_vel_reward_scale * dt,
            "arrival_dwell": arrival_dwell * self.cfg.arrival_dwell_reward_scale * dt,
            "action_magnitude": action_magnitude * self.cfg.action_magnitude_reward_scale * dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * dt,
            "action_acc_l2": action_acc * self.cfg.action_acc_reward_scale * dt,
            "torque_l2": torque * self.cfg.torque_reward_scale * dt,
            "torque_rate_l2": torque_rate * self.cfg.torque_rate_reward_scale * dt,
            "orient": orient * self.cfg.orient_reward_scale * dt,
            "base_height": base_height_reward * self.cfg.base_height_reward_scale * dt,
            "lin_vel_z": lin_vel_z * self.cfg.lin_vel_z_reward_scale * dt,
            "dof_vel": dof_vel * self.cfg.dof_vel_reward_scale * dt,
            "joint_acc_l2": dof_acc * self.cfg.dof_acc_reward_scale * dt,
            "ang_vel_xy": ang_vel_xy * self.cfg.ang_vel_xy_reward_scale * dt,
            "feet_clearance": feet_clearance * self.cfg.feet_clearance_reward_scale * dt,
            # Touchdown is an event rather than a continuous cost.
            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale,
            "undesired_contacts": undesired_contacts * self.cfg.undesired_contact_reward_scale * dt,
        }
        early_terminated = self.reset_terminated & ~self.reset_time_outs
        rewards["termination_penalty"] = early_terminated.float() * self.cfg.termination_penalty
        for key, value in rewards.items():
            self._episode_sums[key] += value
        total_reward = torch.sum(torch.stack(tuple(rewards.values())), dim=0)

        # DirectRLEnv computes done -> reward -> observation. Retarget only
        # after every old-goal reward term is finalized, so the next returned
        # observation is the first state associated with the new goal.
        if bool(getattr(self.cfg, "enable_goal_chaining", True)):
            post_hold_steps = max(
                0,
                math.ceil(float(getattr(self.cfg, "goal_chain_post_arrival_hold_s", 1.0)) / self.step_dt),
            )
            chain_ready = self._arrival_dwell_steps >= required_dwell_steps + post_hold_steps
            chain_ready &= ~self.reset_buf
            if torch.any(chain_ready):
                chain_env_ids = torch.nonzero(chain_ready, as_tuple=False).squeeze(-1)
                self._sample_goal_targets(
                    chain_env_ids,
                    base_pos_xy[chain_env_ids],
                    base_yaw[chain_env_ids],
                    getattr(self.cfg, "goal_chain_distance_mixture", (0.10, 0.20, 0.70)),
                    is_chain=True,
                )
                self._arrival_dwell_steps[chain_env_ids] = 0
                self._current_goal_success_recorded[chain_env_ids] = False
                self._visualize_target_markers()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        steps_since_reset = self._steps_since_reset

        base_contact_grace_steps = max(0, math.ceil(self.cfg.base_contact_grace_s / self.step_dt))
        termination_grace_steps = max(0, math.ceil(self.cfg.termination_grace_s / self.step_dt))
        pendulum_termination_grace_steps = max(0, math.ceil(self.cfg.pendulum_termination_grace_s / self.step_dt))
        in_termination_grace = steps_since_reset < termination_grace_steps
        in_pendulum_termination_grace = steps_since_reset < pendulum_termination_grace_steps
        termination_allowed = ~in_termination_grace
        pendulum_termination_allowed = ~in_pendulum_termination_grace

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        cstr_termination_contacts = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )

        # Allow a grace period so brief settling contacts right after reset don't terminate.
        contact_grace_elapsed = steps_since_reset >= base_contact_grace_steps
        cstr_termination_contacts = cstr_termination_contacts & contact_grace_elapsed & termination_allowed

        terminated = cstr_termination_contacts

        env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
        base_height = self.robot.data.root_pos_w[:, 2] - env_origins[:, 2]
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
        base_tilt_terminated = (base_tilt_rad > self.cfg.base_tilt_terminate_angle_rad) & termination_allowed
        terminated = terminated | base_tilt_terminated

        pendulum_contact = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.use_pendulum and self._pendulum_contact_sensor is not None:
            pendulum_contact_forces = self._pendulum_contact_sensor.data.net_forces_w
            pendulum_contact = torch.any(
                torch.norm(pendulum_contact_forces, dim=-1) > self.cfg.pendulum_contact_force_threshold,
                dim=1,
            )
            pendulum_contact &= pendulum_termination_allowed
            terminated = terminated | pendulum_contact

        pendulum_angle_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.cfg.use_pendulum and self._pendulum_dof_ids.numel() > 0:
            pendulum_joint_pos = self.robot.data.joint_pos[:, self._pendulum_dof_ids]
            pendulum_angle_norm = torch.linalg.norm(pendulum_joint_pos, dim=1)
            if self._pendulum_angle_failure_steps is None:
                self._pendulum_angle_failure_steps = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.long
                )
            pendulum_failing = (
                (pendulum_angle_norm > self.cfg.pendulum_terminate_angle_rad) & pendulum_termination_allowed
            )
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
            env_origins = (
                self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
            )
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

        # A reset changes physical/sensor/task state without necessarily
        # advancing common_step_counter (for example an explicit Gym reset).
        # Bump a CPU-side generation so the next observation is rebuilt once.
        self._observation_reset_generation += 1
        self._observation_cache = None
        self._observation_cache_step = -1
        self._observation_cache_reset_generation = -1

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs and bool(
            getattr(self.cfg, "stagger_initial_episode_lengths", self.cfg.enable_curriculum)
        ):
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time.
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.last_action[env_ids] = 0.0
        self._physical_delivered_action[env_ids] = 0.0
        self._filtered_action[env_ids] = 0.0
        self._prev_torque[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0
        self._stand_mode[env_ids] = False
        self._actor_command[env_ids] = 0.0
        self._move_gate[env_ids] = 0.0
        self._stand_gate[env_ids] = 1.0
        self._mocap_initialized[env_ids] = False
        self._proprio_imu_initialized[env_ids] = False
        self._arrival_dwell_steps[env_ids] = 0
        self._current_goal_success_recorded[env_ids] = False
        self._previous_stand_foot_lift[env_ids] = False

        if bool(getattr(self.cfg, "enable_episode_mirroring", True)):
            self._episode_mirrored[env_ids] = (
                torch.rand(env_ids.numel(), device=self.device)
                < float(getattr(self.cfg, "episode_mirror_probability", 0.5))
            )
        else:
            self._episode_mirrored[env_ids] = False

        if self.cfg.enable_domain_randomization:
            self._sample_episode_dr_strength(env_ids)
            self._randomize_mass_and_com(env_ids)
            self._randomize_foot_friction(env_ids)
            self._randomize_motor_gains(env_ids)
            self._randomize_pendulum_damping(env_ids)
            self._sample_sensor_biases(env_ids)
            self._sample_mocap_biases(env_ids)
            if self.cfg.enable_external_wrench_push:
                self._push_forces[env_ids] = 0.0
                self._push_torques[env_ids] = 0.0
                self._push_end_step[env_ids] = 0
                self._schedule_next_push(env_ids, self._steps_since_reset[env_ids])
        else:
            self._dr_strength[env_ids] = 0.0
            self._sample_mocap_biases(env_ids)

        nominal_cutoff = float(getattr(self.cfg, "command_lpf_cutoff_hz", 4.0))
        cutoff_range = getattr(self.cfg, "command_lpf_cutoff_range_hz", (3.0, 5.0))
        sampled_cutoff = self._sample_uniform_device(cutoff_range, (env_ids.numel(),), self.device)
        cutoff_strength = torch.clamp(self._dr_strength[env_ids], 0.0, 1.0)
        self._command_lpf_cutoff_hz[env_ids] = nominal_cutoff + cutoff_strength * (
            sampled_cutoff - nominal_cutoff
        )

        # Reset variables.
        self._action_history[env_ids] = 0
        self.gait_indices[env_ids] = torch.rand(env_ids.numel(), device=self.device)
        self.clock_inputs[env_ids] = 0.0
        self.desired_contact_states[env_ids] = 1.0
        if self._base_height_failure_steps is not None:
            self._base_height_failure_steps[env_ids] = 0
        if self._pendulum_angle_failure_steps is not None:
            self._pendulum_angle_failure_steps[env_ids] = 0
        if self._position_failure_steps is not None:
            self._position_failure_steps[env_ids] = 0

        num_reset_envs = env_ids.shape[0]
        # Reset robot state.
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.desired_joint_pos[env_ids] = joint_pos[:, self._leg_dof_ids]

        leg_pos_noise = float(getattr(self.cfg, "reset_leg_joint_pos_noise_rad", 0.03))
        leg_vel_noise = float(getattr(self.cfg, "reset_leg_joint_vel_noise_rad_s", 0.10))
        joint_pos[:, self._leg_dof_ids] += sample_uniform(
            -leg_pos_noise, leg_pos_noise, (num_reset_envs, self._action_dim), self.device
        )
        leg_limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self._leg_dof_ids, :]
        joint_pos[:, self._leg_dof_ids] = torch.maximum(
            torch.minimum(joint_pos[:, self._leg_dof_ids], leg_limits[:, :, 1]), leg_limits[:, :, 0]
        )
        joint_vel[:, self._leg_dof_ids] += sample_uniform(
            -leg_vel_noise, leg_vel_noise, (num_reset_envs, self._action_dim), self.device
        )

        if self.cfg.use_pendulum and self._pendulum_dof_ids.numel() > 0:
            if self._pendulum_dof_ids.numel() != 2:
                raise RuntimeError(
                    f"Radial pendulum reset expects exactly two joints, got {self._pendulum_dof_ids.numel()}."
                )
            radius = sample_uniform(
                self.cfg.pendulum_angle_min,
                self.cfg.pendulum_angle_max,
                (num_reset_envs,),
                joint_pos.device,
            )
            recovery_fraction = float(getattr(self.cfg, "pendulum_recovery_reset_fraction", 0.0))
            recovery_mask = torch.rand(num_reset_envs, device=self.device) < recovery_fraction
            if torch.any(recovery_mask):
                recovery_range = getattr(
                    self.cfg,
                    "pendulum_recovery_angle_range",
                    (math.radians(10.0), math.radians(11.0)),
                )
                recovery_radius = sample_uniform(
                    float(recovery_range[0]),
                    float(recovery_range[1]),
                    (num_reset_envs,),
                    joint_pos.device,
                )
                radius = torch.where(recovery_mask, recovery_radius, radius)
            theta = sample_uniform(0.0, 2.0 * math.pi, (num_reset_envs,), joint_pos.device)
            offsets = torch.stack((radius * torch.cos(theta), radius * torch.sin(theta)), dim=-1)
            joint_pos[:, self._pendulum_dof_ids] += offsets

        reset_leg_joint_pos = joint_pos[:, self._leg_dof_ids] - self.robot.data.default_joint_pos[
            env_ids[:, None], self._leg_dof_ids
        ]
        reset_leg_joint_vel = joint_vel[:, self._leg_dof_ids]
        if self.cfg.use_pendulum and self._pendulum_dof_ids.numel() > 0:
            reset_pendulum_joint_pos = joint_pos[:, self._pendulum_dof_ids]
            reset_pendulum_joint_vel = joint_vel[:, self._pendulum_dof_ids]
        else:
            reset_pendulum_joint_pos = torch.zeros(
                num_reset_envs,
                self._pendulum_dof_count,
                device=self.device,
                dtype=joint_pos.dtype,
            )
            reset_pendulum_joint_vel = torch.zeros_like(reset_pendulum_joint_pos)
        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        base_xy_noise = float(getattr(self.cfg, "reset_base_xy_noise_m", 0.02))
        default_root_state[:, :2] += sample_uniform(
            -base_xy_noise, base_xy_noise, (num_reset_envs, 2), self.device
        )
        default_roll, default_pitch, default_yaw = math_utils.euler_xyz_from_quat(default_root_state[:, 3:7])
        roll_pitch_noise = float(getattr(self.cfg, "reset_base_roll_pitch_noise_rad", math.radians(2.0)))
        yaw_noise = float(getattr(self.cfg, "reset_base_yaw_noise_rad", math.radians(5.0)))
        default_roll += sample_uniform(-roll_pitch_noise, roll_pitch_noise, (num_reset_envs,), self.device)
        default_pitch += sample_uniform(-roll_pitch_noise, roll_pitch_noise, (num_reset_envs,), self.device)
        default_yaw += sample_uniform(-yaw_noise, yaw_noise, (num_reset_envs,), self.device)
        default_root_state[:, 3:7] = math_utils.quat_from_euler_xyz(default_roll, default_pitch, default_yaw)
        # Anchor every reset goal to the final perturbed base pose. Desired yaw
        # ranges are offsets from this heading, and class 0 uses a guaranteed
        # in-tolerance planted yaw offset.
        reset_anchor_xy = default_root_state[:, :2] - self._terrain.env_origins[env_ids, :2]
        reset_goal_class = self._sample_goal_targets(
            env_ids,
            reset_anchor_xy,
            default_yaw,
            self.cfg.goal_distance_mixture,
            is_chain=False,
        )
        # Class-0 is explicitly a planted reset. Seed the hysteresis latch so
        # bounded mocap error near the enter threshold cannot relabel it as a
        # low-speed locomotion sample on the first policy tick.
        self._stand_mode[env_ids] = reset_goal_class == 0
        base_lin_vel_noise = float(getattr(self.cfg, "reset_base_lin_vel_noise_m_s", 0.05))
        base_ang_vel_noise = float(getattr(self.cfg, "reset_base_ang_vel_noise_rad_s", 0.10))
        default_root_state[:, 7:10] += sample_uniform(
            -base_lin_vel_noise, base_lin_vel_noise, (num_reset_envs, 3), self.device
        )
        default_root_state[:, 10:13] += sample_uniform(
            -base_ang_vel_noise, base_ang_vel_noise, (num_reset_envs, 3), self.device
        )

        # Seed delayed IMU packets from the final perturbed reset state.  Root
        # angular velocity is world-frame here; both actor IMU channels are
        # delivered in the perturbed base frame.
        reset_body_ang_vel = math_utils.quat_apply_inverse(
            default_root_state[:, 3:7], default_root_state[:, 10:13]
        )
        reset_projected_gravity = math_utils.quat_apply_inverse(
            default_root_state[:, 3:7], self._world_gravity_dir[env_ids]
        )
        reset_imu_packet = torch.cat((reset_body_ang_vel, reset_projected_gravity), dim=-1)
        self._reset_transport_buffers(
            env_ids,
            reset_leg_joint_pos,
            reset_leg_joint_vel,
            reset_pendulum_joint_pos,
            reset_pendulum_joint_vel,
            reset_imu_packet,
        )

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        if self._prev_base_pos_w is not None:
            self._prev_base_pos_w[env_ids] = default_root_state[:, :3]
        self._imu_sensor.reset(env_ids)

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
        pendulum_speed_steps = torch.clamp(self._episode_pendulum_speed_deg_s_count[env_ids], min=1).to(
            dtype=torch.float
        )
        mean_pendulum_speed_deg_s = torch.mean(self._episode_pendulum_speed_deg_s_sum[env_ids] / pendulum_speed_steps)
        extras["Episode_Metric/mean_pendulum_speed_deg_s"] = mean_pendulum_speed_deg_s.item()
        self._episode_pendulum_speed_deg_s_sum[env_ids] = 0.0
        self._episode_pendulum_speed_deg_s_count[env_ids] = 0
        extras["Episode_Metric/arrival_success_rate"] = torch.mean(
            self._episode_arrival_success[env_ids].float()
        ).item()
        extras["Episode_Metric/success_count"] = torch.mean(
            self._episode_goal_success_count[env_ids].float()
        ).item()
        stand_steps = torch.clamp(self._episode_stand_steps[env_ids], min=1).float()
        extras["Episode_Metric/four_contact_fraction_in_stand"] = torch.mean(
            self._episode_four_contact_sum[env_ids] / stand_steps
        ).item()
        extras["Episode_Metric/stand_foot_lift_events"] = torch.mean(
            self._episode_foot_lift_events[env_ids].float()
        ).item()
        self._episode_arrival_success[env_ids] = False
        self._episode_goal_success_count[env_ids] = 0
        self._episode_four_contact_sum[env_ids] = 0.0
        self._episode_stand_steps[env_ids] = 0
        self._episode_foot_lift_events[env_ids] = 0
        if self.cfg.enable_curriculum and self.cfg.curriculum_total_steps > 0:
            curriculum_step = int(getattr(self.cfg, "curriculum_start_step", 0)) + self.common_step_counter
            extras["Episode_Metric/curriculum_progress"] = min(
                1.0, max(0.0, curriculum_step / self.cfg.curriculum_total_steps)
            )
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

    def _update_gait_targets(self) -> None:
        """Advance a command-dependent diagonal trot or request planted stance."""
        frequency_min = float(getattr(self.cfg, "gait_frequency_min_hz", 1.5))
        frequency_max = float(getattr(self.cfg, "gait_frequency_max_hz", 2.5))
        duty_min_speed = float(getattr(self.cfg, "gait_duty_factor_min_speed", 0.62))
        duty_max_speed = float(getattr(self.cfg, "gait_duty_factor_max_speed", 0.52))
        self._gait_frequency = frequency_min + (frequency_max - frequency_min) * self._move_gate
        self._gait_duty_factor = duty_min_speed + (duty_max_speed - duty_min_speed) * self._move_gate

        phase_increment = self.step_dt * self._gait_frequency * (~self._stand_mode).float()
        self.gait_indices = torch.remainder(self.gait_indices + phase_increment, 1.0)
        foot_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)
        self.foot_indices = torch.remainder(self.gait_indices.unsqueeze(-1) + foot_offsets, 1.0)

        duty = self._gait_duty_factor.unsqueeze(-1)
        smoothing = max(float(getattr(self.cfg, "gait_contact_smoothing_sigma", 0.07)), 1e-4)
        periodic_contact_score = torch.cos(2.0 * math.pi * (self.foot_indices - 0.5 * duty))
        contact_boundary = torch.cos(math.pi * duty)
        gait_contacts = torch.sigmoid((periodic_contact_score - contact_boundary) / smoothing)
        self.desired_contact_states = (
            self._move_gate.unsqueeze(-1) * gait_contacts + (1.0 - self._move_gate).unsqueeze(-1)
        )

        # Phase freezes in stand, but its sin/cos values remain observable so
        # leaving stand resumes a continuous, fully specified gait state.
        self.clock_inputs[:, 0] = torch.sin(2.0 * math.pi * self.gait_indices)
        self.clock_inputs[:, 1] = torch.cos(2.0 * math.pi * self.gait_indices)
        self.clock_inputs[:, 2] = self._move_gate
        self.clock_inputs[:, 3] = self._stand_gate
