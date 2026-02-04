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
        self._prev_actions = torch.zeros_like(self._actions)

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
                "feet_air_time",
                "dof_torques_l2",
                "dof_acc_l2",
                "undesired_contacts",
                "tilt",
                "action_delta",
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
        # Bound actions to expected [-1, 1] range before scaling.
        self._prev_actions.copy_(self._actions)
        self._actions = torch.tanh(actions)
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

        pendulum_joint_pos_raw = pendulum_joint_pos
        pendulum_joint_vel_raw = pendulum_joint_vel
        if self.cfg.use_pendulum:
            if self.cfg.pendulum_position_noise > 0.0:
                pendulum_joint_pos = pendulum_joint_pos + sample_uniform(
                    -self.cfg.pendulum_position_noise,
                    self.cfg.pendulum_position_noise,
                    pendulum_joint_pos.shape,
                    pendulum_joint_pos.device,
                )
            if self.cfg.pendulum_velocity_noise > 0.0:
                pendulum_joint_vel = pendulum_joint_vel + sample_uniform(
                    -self.cfg.pendulum_velocity_noise,
                    self.cfg.pendulum_velocity_noise,
                    pendulum_joint_vel.shape,
                    pendulum_joint_vel.device,
                )

        position_error_b_xy_raw, yaw_error_raw = self._compute_body_frame_errors()
        position_error_b_xy = position_error_b_xy_raw
        yaw_error = yaw_error_raw
        if self.cfg.state_position_noise > 0.0:
            position_error_b_xy = position_error_b_xy + sample_uniform(
                -self.cfg.state_position_noise,
                self.cfg.state_position_noise,
                position_error_b_xy.shape,
                position_error_b_xy.device,
            )
        if self.cfg.state_orientation_noise > 0.0:
            yaw_error = yaw_error + sample_uniform(
                -self.cfg.state_orientation_noise,
                self.cfg.state_orientation_noise,
                yaw_error.shape,
                yaw_error.device,
            )
        state_error = torch.cat([position_error_b_xy, yaw_error.unsqueeze(-1)], dim=-1)
        state_error_raw = torch.cat([position_error_b_xy_raw, yaw_error_raw.unsqueeze(-1)], dim=-1)

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        foot_contacts = (
            torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids_sensor], dim=-1), dim=1)[0]
            > self.cfg.foot_contact_force_threshold
        ).float()

        policy_tensors = [
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            state_error,
            leg_joint_pos,
            leg_joint_vel,
            pendulum_joint_pos,
            pendulum_joint_vel,
            foot_contacts,
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
                    state_error_raw,
                    leg_joint_pos,
                    leg_joint_vel,
                    pendulum_joint_pos_raw,
                    pendulum_joint_vel_raw,
                    foot_contacts,
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
            target_yaw = self.target_state[:, 2]
        else:
            target_xy = torch.zeros((self.num_envs, 2), device=self.device)
            target_yaw = torch.zeros(self.num_envs, device=self.device)

        position_error_xy = target_xy - base_pos_xy
        position_error_w = root_pos.new_zeros((self.num_envs, 3))
        position_error_w[:, :2] = position_error_xy
        position_error_b = math_utils.quat_apply_inverse(root_quat, position_error_w)
        position_error_b_xy = position_error_b[:, :2]

        _, _, yaw = math_utils.euler_xyz_from_quat(root_quat)
        yaw_error = math_utils.wrap_to_pi(target_yaw - yaw)

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

        # action delta penalty (smoothness)
        action_delta = torch.sum(torch.square(self._actions - self._prev_actions), dim=1)

        # quadruped-specific terms
        joint_torques = torch.nan_to_num(
            self.robot.data.applied_torque[:, self._leg_dof_ids], nan=0.0, posinf=0.0, neginf=0.0
        )
        joint_torques = torch.sum(torch.square(joint_torques), dim=1)
        joint_accel = torch.nan_to_num(
            self.robot.data.joint_acc[:, self._leg_dof_ids], nan=0.0, posinf=0.0, neginf=0.0
        )
        joint_accel = torch.sum(torch.square(joint_accel), dim=1)

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids_sensor]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids_sensor]
        air_time = torch.sum((last_air_time - self.cfg.feet_air_time_threshold_s) * first_contact, dim=1)
        feet_air_time_reward = air_time * (base_speed > self.cfg.feet_air_time_speed_threshold)

        if self._undesired_contact_body_ids is not None:
            net_contact_forces = self._contact_sensor.data.net_forces_w_history
            is_contact = (
                torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0]
                > self.cfg.undesired_contact_force_threshold
            )
            undesired_contacts = torch.sum(is_contact, dim=1)
        else:
            undesired_contacts = torch.zeros(self.num_envs, device=self.device)

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
            "action_delta": self.cfg.rew_scale_action_delta * action_delta,
            "feet_air_time": self.cfg.rew_scale_feet_air_time * feet_air_time_reward,
            "dof_torques_l2": self.cfg.rew_scale_dof_torques * joint_torques,
            "dof_acc_l2": self.cfg.rew_scale_dof_acc * joint_accel,
            "undesired_contacts": self.cfg.rew_scale_undesired_contacts * undesired_contacts,
            "tilt": self.cfg.rew_scale_tilt * tilt_angle,
            "alive": self.cfg.rew_scale_alive * (1.0 - self.reset_terminated.float()),
            "terminated": self.cfg.rew_scale_terminated * self.reset_terminated.float(),
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

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
        curriculum_level = self._get_curriculum_level()
        if self.cfg.use_pendulum and curriculum_level >= self.cfg.pendulum_termination_start_level:
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
            if self.pendulum_failure_steps is not None:
                self.pendulum_failure_steps.zero_()

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

        # Tilt termination.
        roll, pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        tilt_cos = torch.cos(roll) * torch.cos(pitch)
        tilt_angle = torch.acos(torch.clamp(tilt_cos, -1.0, 1.0))
        tilt_terminated = tilt_angle > (self.cfg.tilt_terminate_angle_deg * math.pi / 180.0)

        terminated = (
            cstr_termination_contacts
            | pendulum_terminated
            | position_terminated
            | position_tolerance_terminated
            | tilt_terminated
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

    def _get_curriculum_scale(self) -> float:
        """Returns a [0, 1] curriculum scale based on global step count."""
        if not self.cfg.curriculum_enabled:
            return 1.0
        levels = max(1, int(self.cfg.curriculum_levels))
        if levels == 1:
            return 1.0
        steps_per_level = max(1, int(self.cfg.curriculum_steps_per_level))
        step_count = int(self.common_step_counter)
        level = min(levels - 1, step_count // steps_per_level)
        return level / (levels - 1)

    def _get_curriculum_level(self) -> int:
        if not self.cfg.curriculum_enabled:
            return int(self.cfg.curriculum_levels) - 1
        levels = max(1, int(self.cfg.curriculum_levels))
        steps_per_level = max(1, int(self.cfg.curriculum_steps_per_level))
        step_count = int(self.common_step_counter)
        return min(levels - 1, step_count // steps_per_level)

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
        self._prev_actions[env_ids] = 0.0

        # Reset variables.
        self.gait_indices[env_ids] = 0

        # Sample new targets.
        if self.target_state is None:
            self.target_state = torch.zeros(self.num_envs, 3, device=self.device)
        num_reset_envs = env_ids.shape[0]
        curriculum_scale = self._get_curriculum_scale()
        goal_range = self.cfg.goal_randomization_range_min + (
            self.cfg.goal_randomization_range - self.cfg.goal_randomization_range_min
        ) * curriculum_scale
        goal_noise_x = sample_uniform(-goal_range, goal_range, (num_reset_envs,), self.device)
        goal_noise_y = sample_uniform(-goal_range, goal_range, (num_reset_envs,), self.device)
        goal_yaw = sample_uniform(
            -self.cfg.goal_randomization_angle,
            self.cfg.goal_randomization_angle,
            (num_reset_envs,),
            self.device,
        )
        self.target_state[env_ids, 0] = goal_noise_x
        self.target_state[env_ids, 1] = goal_noise_y
        self.target_state[env_ids, 2] = goal_yaw

        # Reset robot state.
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        pendulum_angle_min = self.cfg.pendulum_angle_min * curriculum_scale
        pendulum_angle_max = self.cfg.pendulum_angle_max * curriculum_scale
        for joint_idx in self._pendulum_dof_ids:
            signs = torch.randint(0, 2, joint_pos[:, joint_idx].shape, device=joint_pos.device) * 2 - 1
            magnitudes = sample_uniform(
                pendulum_angle_min,
                pendulum_angle_max,
                joint_pos[:, joint_idx].shape,
                joint_pos.device,
            )
            joint_pos[:, joint_idx] += signs.float() * magnitudes

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
