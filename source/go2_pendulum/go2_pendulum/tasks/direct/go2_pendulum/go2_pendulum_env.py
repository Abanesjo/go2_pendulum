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
from isaaclab.utils.math import sample_uniform

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

        # Joint position command (deviation from default joint positions).
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._actions_clipped = torch.zeros_like(self._actions)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # Target state [x_d, y_d, yaw_d] in environment frame (randomized per reset).
        self.target_state = None

        # Marker visualization buffers.
        self._marker_offset = None
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
            "rew_action_rate",
            "action_over_limit",
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
        self._episode_pendulum_angle_deg_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_pendulum_angle_deg_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_pendulum_speed_deg_s_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_pendulum_speed_deg_s_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_mean_action_abs_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._episode_action_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_position_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

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
        self._undesired_contact_body_ids = (
            torch.tensor(undesired_contact_ids, device=self.device, dtype=torch.long)
            if len(undesired_contact_ids) > 0
            else None
        )

        # Track termination causes for accurate logging.
        self._base_contact_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._base_height_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._pendulum_contact_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._pendulum_angle_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._position_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._base_height_failure_steps = None
        self._pendulum_angle_failure_steps = None
        self._position_failure_steps = None
        self._steps_since_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

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

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        if self.cfg.enable_action_clipping:
            self._actions_clipped = torch.clamp(self._actions, -self.cfg.action_clip, self.cfg.action_clip)
        else:
            self._actions_clipped = self._actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions_clipped

        self.desired_joint_pos = (
            self.robot.data.default_joint_pos[:, self._leg_dof_ids] + self._processed_actions
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.desired_joint_pos,
            joint_ids=self._leg_dof_ids,
        )

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        leg_joint_pos = (
            self.robot.data.joint_pos[:, self._leg_dof_ids] - self.robot.data.default_joint_pos[:, self._leg_dof_ids]
        )
        leg_joint_vel = self.robot.data.joint_vel[:, self._leg_dof_ids]
        if self.cfg.use_pendulum and self._pendulum_dof_ids.numel() > 0:
            pendulum_joint_pos = self.robot.data.joint_pos[:, self._pendulum_dof_ids]
            pendulum_joint_vel = self.robot.data.joint_vel[:, self._pendulum_dof_ids]
        else:
            pendulum_joint_pos = torch.zeros(
                self.num_envs,
                self._pendulum_dof_count,
                device=self.device,
                dtype=leg_joint_pos.dtype,
            )
            pendulum_joint_vel = torch.zeros_like(pendulum_joint_pos)

        env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
        base_pos_xy = self.robot.data.root_pos_w[:, :2] - env_origins[:, :2]
        _, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)

        if self.target_state is not None:
            target_xy = self.target_state[:, :2]
            target_yaw = self.target_state[:, 2]
        else:
            target_xy = torch.zeros((self.num_envs, 2), device=self.device, dtype=leg_joint_pos.dtype)
            target_yaw = torch.zeros(self.num_envs, device=self.device, dtype=leg_joint_pos.dtype)

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
        if self.cfg.track_goal:
            goal_heading = torch.atan2(position_error_xy[:, 1], position_error_xy[:, 0])
            yaw_error = math_utils.wrap_to_pi(goal_heading)
            near_goal = torch.linalg.norm(position_error_xy, dim=1) < 1e-6
            yaw_error = torch.where(near_goal, torch.zeros_like(yaw_error), yaw_error)
        else:
            yaw_error = math_utils.wrap_to_pi(target_yaw - yaw)

        position_noise = self.cfg.position_noise * self.cfg.observation_noise_scale
        orientation_noise = self.cfg.orientation_noise * self.cfg.observation_noise_scale
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
        state_error = torch.cat([position_error_xy, yaw_error.unsqueeze(-1)], dim=-1)

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

        policy_tensors.extend([self._actions_clipped, self.clock_inputs])

        observations = {"policy": torch.cat(policy_tensors, dim=-1)}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
        if self.target_state is not None:
            base_pos_xy = self.robot.data.root_pos_w[:, :2] - env_origins[:, :2]
            position_error = torch.linalg.norm(self.target_state[:, :2] - base_pos_xy, dim=1)
            _, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
            if self.cfg.track_goal:
                goal_dir_xy = self.target_state[:, :2] - base_pos_xy
                goal_heading = torch.atan2(goal_dir_xy[:, 1], goal_dir_xy[:, 0])
                yaw_error = math_utils.wrap_to_pi(goal_heading - yaw)
                yaw_error = torch.where(position_error < 1e-6, torch.zeros_like(yaw_error), yaw_error)
            else:
                yaw_error = math_utils.wrap_to_pi(self.target_state[:, 2] - yaw)
            position_sigma = max(self.cfg.position_reward_sigma, 1e-6)
            position_tracking_reward = 1.0 - (position_error / position_sigma)
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

        rew_action_rate = torch.sum(
            torch.square(self._actions - self.last_actions[:, :, 0]),
            dim=1,
        ) * (self.cfg.action_scale**2)

        rew_action_rate += torch.sum(
            torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]),
            dim=1,
        ) * (self.cfg.action_scale**2)
        # Penalize actions that exceed a soft limit with a steep quartic cost.
        action_overshoot = torch.relu(torch.abs(self._actions) - self.cfg.action_soft_limit)
        rew_action_over_limit = torch.sum(torch.pow(action_overshoot, 4), dim=1)

        self._episode_mean_action_abs_sum += torch.mean(torch.abs(self._actions), dim=1)
        self._episode_action_count += 1

        # penalize non-vertical orientation (projected gravity on xy plane)
        rew_orient = torch.sum(
            torch.square(self.robot.data.projected_gravity_b[:, :2]),
            dim=1,
        )
        rew_orient = torch.exp(-rew_orient)

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
            pendulum_ee_quat_w = self.robot.data.body_quat_w[:, self._pendulum_ee_body_id]
            pendulum_ee_z_w = math_utils.quat_apply(
                pendulum_ee_quat_w, self._world_up.unsqueeze(0).expand(self.num_envs, -1)
            )
            # Same spirit as base orient reward: penalize tilt of the frame z-axis in xy without angle conversions.
            pendulum_upright_error = torch.sum(torch.square(pendulum_ee_z_w[:, :2]), dim=1)
            pendulum_upright_sigma = max(self.cfg.pendulum_upright_reward_sigma, 1e-6)
            pendulum_upright_reward = 1.0 - (pendulum_upright_error / pendulum_upright_sigma)

            pendulum_joint_vel = self.robot.data.joint_vel[:, self._pendulum_dof_ids]
            pendulum_vel_norm = torch.linalg.norm(pendulum_joint_vel, dim=1)
            pendulum_velocity_reward = torch.sum(torch.square(pendulum_joint_vel), dim=1)
            pendulum_angle_deg = torch.rad2deg(torch.linalg.norm(self.robot.data.joint_pos[:, self._pendulum_dof_ids], dim=1))
            pendulum_speed_deg_s = torch.rad2deg(pendulum_vel_norm)

            # Match omniwheel: reward high when pendulum is balanced and/or base speed is low.
            base_speed = torch.linalg.norm(self.robot.data.root_lin_vel_w[:, :2], dim=1)
            balanced_movement_reward = torch.exp(-pendulum_upright_error * base_speed)
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
        self.last_actions[:, :, 0] = self._actions[:]

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
            "rew_action_rate": rew_action_rate * self.cfg.action_rate_reward_scale,
            "action_over_limit": rew_action_over_limit * self.cfg.action_over_limit_reward_scale,
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

        self._actions[env_ids] = 0.0
        self._actions_clipped[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Reset variables.
        self._steps_since_reset[env_ids] = 0
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
        pendulum_contact_resets = self._pendulum_contact_terminated[env_ids] & self.reset_terminated[env_ids]
        pendulum_angle_resets = self._pendulum_angle_terminated[env_ids] & self.reset_terminated[env_ids]
        position_resets = self._position_terminated[env_ids] & self.reset_terminated[env_ids]
        any_labeled_reset = (
            base_contact_resets
            | base_height_resets
            | pendulum_contact_resets
            | pendulum_angle_resets
            | position_resets
        )
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(base_contact_resets).item()
        extras["Episode_Termination/base_height"] = torch.count_nonzero(base_height_resets).item()
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
        action_steps = torch.clamp(self._episode_action_count[env_ids], min=1).to(dtype=torch.float)
        mean_action_abs = torch.mean(self._episode_mean_action_abs_sum[env_ids] / action_steps)
        extras["Episode_Metric/mean_action_abs"] = mean_action_abs.item()
        self._episode_mean_action_abs_sum[env_ids] = 0.0
        self._episode_action_count[env_ids] = 0
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
        self._marker_locations = (robot_pos_3d + target_pos_3d) / 2.0

        sphere_locations = torch.zeros_like(self._marker_locations)
        sphere_locations[:, :2] = target_pos_world
        sphere_locations[:, 2] = env_origins[:, 2]
        sphere_loc = sphere_locations + self._marker_offset

        sphere_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        sphere_orientations[:, 3] = 1.0

        self.target_visualizer.visualize(sphere_loc, sphere_orientations)

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
