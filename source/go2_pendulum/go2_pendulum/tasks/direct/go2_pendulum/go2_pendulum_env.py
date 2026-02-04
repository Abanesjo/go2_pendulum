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

        # X/Y linear velocity and yaw angular velocity commands.
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Target state [x_d, y_d, yaw_d] in environment frame (randomized per reset).
        self.target_state = None

        # Marker visualization buffers.
        self._marker_offset = None
        self._marker_orientations = None
        self._marker_locations = None
        self._marker_up = torch.tensor([0.0, 0.0, 1.0])

        # Logging
        reward_keys = [
            "track_lin_vel_xy_exp",
            "track_ang_vel_z_exp",
            "pendulum_upright",
            "pendulum_velocity",
            "balanced_movement",
        ]
        if self.cfg.raibert_heuristic_reward_scale != 0.0:
            reward_keys.append("raibert_heuristic")
        reward_keys.append("feet_air_time")
        penalty_keys = [
            "action_rate_penalty",
            "orient_penalty",
            "lin_vel_z_penalty",
            "dof_vel_penalty",
            "ang_vel_xy_penalty",
        ]
        if self.cfg.feet_clearance_penalty_scale != 0.0:
            penalty_keys.append("feet_clearance_penalty")
        if self.cfg.tracking_contacts_shaped_force_penalty_scale != 0.0:
            penalty_keys.append("tracking_contacts_shaped_force_penalty")
        penalty_keys.extend(
            [
                "undesired_contacts_penalty",
                "termination_penalty",
                "dof_torques_l2_penalty",
                "dof_acc_l2_penalty",
            ]
        )
        episode_sum_keys = reward_keys + penalty_keys
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in episode_sum_keys
        }
        self._penalty_keys = set(penalty_keys)

        # Effort limit for saturation logging (used by DC motor model).
        base_legs_actuator = self.cfg.robot_cfg.actuators.get("base_legs")
        effort_limit = None if base_legs_actuator is None else base_legs_actuator.effort_limit
        if effort_limit is None:
            effort_limit = float("inf")
        self._leg_effort_limit = torch.tensor(effort_limit, device=self.device, dtype=torch.float32)

        # Track termination causes for accurate logging.
        self._base_contact_terminated = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._pendulum_angle_failure_steps = None
        self._position_failure_steps = None

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
        self.target_visualizer = None
        if self.cfg.tracking_mode:
            self.target_visualizer = VisualizationMarkers(self.cfg.target_marker_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions

        self.desired_joint_pos = self.robot.data.default_joint_pos.clone()
        self.desired_joint_pos[:, self._leg_dof_ids] = (
            self.robot.data.default_joint_pos[:, self._leg_dof_ids] + self._processed_actions
        )

        self._update_commands()

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

        policy_tensors = [
            self.robot.data.root_lin_vel_b,
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            self._commands,
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
                    self._commands,
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

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        rew_action_rate = torch.sum(
            torch.square(self._actions - self.last_actions[:, :, 0]),
            dim=1,
        ) * (self.cfg.action_scale**2)

        rew_action_rate += torch.sum(
            torch.square(self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1
        ) * (self.cfg.action_scale**2)

        # penalize non-vertical orientation (projected gravity on xy plane)
        rew_orient = torch.sum(
            torch.square(self.robot.data.projected_gravity_b[:, :2]),
            dim=1,
        )

        # penalize vertical velocity (z-component of base linear velocity)
        lin_vel_z = torch.nan_to_num(self.robot.data.root_lin_vel_w[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
        lin_vel_z = torch.clamp(lin_vel_z, -self.cfg.lin_vel_z_penalty_clip, self.cfg.lin_vel_z_penalty_clip)
        rew_lin_vel_z = lin_vel_z * lin_vel_z

        # penalize high joint velocities
        rew_dof_vel = torch.sum(
            torch.square(self.robot.data.joint_vel[:, self._leg_dof_ids]),
            dim=1,
        )

        # penalize angular velocity in xy plane
        rew_ang_vel_xy = torch.sum(
            torch.square(self.robot.data.root_ang_vel_b[:, :2]),
            dim=1,
        )

        # pendulum rewards from joint angles/velocities (target is zero)
        if self.cfg.use_pendulum:
            pendulum_joint_pos = self.robot.data.joint_pos[:, self._pendulum_dof_ids]
            pendulum_joint_vel = self.robot.data.joint_vel[:, self._pendulum_dof_ids]
            pendulum_angle_norm = torch.linalg.norm(pendulum_joint_pos, dim=1)
            pendulum_vel_norm = torch.linalg.norm(pendulum_joint_vel, dim=1)
            pendulum_upright_reward = torch.exp(-pendulum_angle_norm)
            pendulum_velocity_reward = torch.exp(-pendulum_vel_norm)
            base_speed = torch.linalg.norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)
            balanced_movement_reward = torch.exp(-pendulum_angle_norm * base_speed)
            upright_scaled = pendulum_upright_reward * self.cfg.pendulum_upright_reward_scale
            balanced_movement_scale = self.cfg.balanced_movement_reward_scale * upright_scaled / 16.0
        else:
            pendulum_upright_reward = torch.zeros(self.num_envs, device=self.device)
            pendulum_velocity_reward = torch.zeros(self.num_envs, device=self.device)
            balanced_movement_reward = torch.zeros(self.num_envs, device=self.device)
            balanced_movement_scale = torch.zeros(self.num_envs, device=self.device)

        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]

        # gait shaping
        self._step_contact_targets()

        phases = 1 - torch.abs(1.0 - torch.clamp((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = self.foot_positions_w[:, :, 2]
        target_height = 0.10 * phases + 0.03
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

        rew_raibert_heuristic = self._reward_raibert_heuristic()

        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids_sensor]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids_sensor]
        rew_feet_air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )

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

        # joint torques and accelerations
        rew_dof_torques_l2 = torch.sum(torch.square(self.robot.data.applied_torque[:, self._leg_dof_ids]), dim=1)
        rew_dof_acc_l2 = torch.sum(torch.square(self.robot.data.joint_acc[:, self._leg_dof_ids]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale,
            "pendulum_upright": pendulum_upright_reward * self.cfg.pendulum_upright_reward_scale,
            "pendulum_velocity": pendulum_velocity_reward * self.cfg.pendulum_vel_reward_scale,
            "balanced_movement": balanced_movement_reward * balanced_movement_scale,
            "action_rate_penalty": rew_action_rate * self.cfg.action_rate_penalty_scale,
            "raibert_heuristic": rew_raibert_heuristic * self.cfg.raibert_heuristic_reward_scale,
            "orient_penalty": rew_orient * self.cfg.orient_penalty_scale,
            "lin_vel_z_penalty": rew_lin_vel_z * self.cfg.lin_vel_z_penalty_scale,
            "dof_vel_penalty": rew_dof_vel * self.cfg.dof_vel_penalty_scale,
            "ang_vel_xy_penalty": rew_ang_vel_xy * self.cfg.ang_vel_xy_penalty_scale,
            "feet_clearance_penalty": rew_feet_clearance * self.cfg.feet_clearance_penalty_scale,
            "tracking_contacts_shaped_force_penalty": (
                rew_tracking_contacts_shaped_force * self.cfg.tracking_contacts_shaped_force_penalty_scale
            ),
            "feet_air_time": rew_feet_air_time * self.cfg.feet_air_time_reward_scale,
            "undesired_contacts_penalty": rew_undesired_contacts * self.cfg.undesired_contact_penalty_scale,
            "termination_penalty": rew_termination_penalty,
            "dof_torques_l2_penalty": rew_dof_torques_l2 * self.cfg.dof_torques_penalty_scale,
            "dof_acc_l2_penalty": rew_dof_acc_l2 * self.cfg.dof_accel_penalty_scale,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            if key in self._episode_sums:
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
        terminated = cstr_termination_contacts
        if self.cfg.use_pendulum and self._pendulum_contact_sensor is not None:
            pendulum_contact_forces = self._pendulum_contact_sensor.data.net_forces_w
            pendulum_contact = torch.any(
                torch.norm(pendulum_contact_forces, dim=-1) > self.cfg.pendulum_contact_force_threshold,
                dim=1,
            )
            terminated = terminated | pendulum_contact
        if self.cfg.use_pendulum and self._pendulum_dof_ids.numel() > 0:
            pendulum_joint_pos = self.robot.data.joint_pos[:, self._pendulum_dof_ids]
            pendulum_angle_norm = torch.linalg.norm(pendulum_joint_pos, dim=1)
            if self._pendulum_angle_failure_steps is None:
                self._pendulum_angle_failure_steps = torch.zeros(
                    self.num_envs, device=self.device, dtype=torch.long
                )
            pendulum_failing = pendulum_angle_norm > self.cfg.pendulum_terminate_angle_rad
            self._pendulum_angle_failure_steps = torch.where(
                pendulum_failing,
                self._pendulum_angle_failure_steps + 1,
                torch.zeros_like(self._pendulum_angle_failure_steps),
            )
            failure_steps_threshold = max(1, math.ceil(self.cfg.pendulum_terminate_duration_s / self.step_dt))
            pendulum_angle_terminated = self._pendulum_angle_failure_steps >= failure_steps_threshold
            terminated = terminated | pendulum_angle_terminated
        if self.cfg.tracking_mode and self.target_state is not None:
            env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
            base_pos_xy = self.robot.data.root_pos_w[:, :2] - env_origins[:, :2]
            position_error = torch.linalg.norm(self.target_state[:, :2] - base_pos_xy, dim=1)
            if self._position_failure_steps is None:
                self._position_failure_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            position_failing = position_error > self.cfg.position_tolerance
            self._position_failure_steps = torch.where(
                position_failing,
                self._position_failure_steps + 1,
                torch.zeros_like(self._position_failure_steps),
            )
            position_failure_threshold = max(1, math.ceil(self.cfg.position_terminate_duration_s / self.step_dt))
            position_terminated = self._position_failure_steps >= position_failure_threshold
            terminated = terminated | position_terminated
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
        move_down = distance < torch.norm(self._commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
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
        if self._pendulum_angle_failure_steps is not None:
            self._pendulum_angle_failure_steps[env_ids] = 0
        if self._position_failure_steps is not None:
            self._position_failure_steps[env_ids] = 0

        if self.cfg.tracking_mode:
            # Sample new targets.
            if self.target_state is None:
                self.target_state = torch.zeros(self.num_envs, 3, device=self.device)
            num_reset_envs = env_ids.shape[0]
            goal_range = self.cfg.goal_randomization_range
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
        else:
            self.target_state = None
            self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)

        # Reset robot state.
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        for joint_idx in self._pendulum_dof_ids:
            signs = torch.randint(0, 2, joint_pos[:, joint_idx].shape, device=joint_pos.device) * 2 - 1
            magnitudes = sample_uniform(
                self.cfg.pendulum_angle_min,
                self.cfg.pendulum_angle_max,
                joint_pos[:, joint_idx].shape,
                joint_pos.device,
            )
            joint_pos[:, joint_idx] += signs.float() * magnitudes

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        if self.cfg.tracking_mode:
            self._update_commands()
            self._visualize_target_markers()

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            prefix = "Episode_Penalty/" if key in self._penalty_keys else "Episode_Reward/"
            extras[prefix + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        if self._terrain.terrain_origins is not None:
            terrain_level = torch.mean(self._terrain.terrain_levels[env_ids].float()).item()
            extras["Episode_Curriculum/terrain_level"] = terrain_level
        self.extras["log"].update(extras)
        extras = dict()
        base_contact_resets = self._base_contact_terminated[env_ids] & self.reset_terminated[env_ids]
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(base_contact_resets).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            if self.cfg.tracking_mode and not hasattr(self, "target_visualizer"):
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
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            if self.cfg.tracking_mode and self.target_visualizer is not None:
                self.target_visualizer.set_visibility(True)
            if self._height_scanner is not None and self.cfg.height_scan_debug_vis:
                self.height_scan_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)
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
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        # Place velocity arrows at the base frame height.
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._commands[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        if self._height_scanner is not None and self.cfg.height_scan_debug_vis:
            ray_hits_w = self._height_scanner.data.ray_hits_w
            self.height_scan_visualizer.visualize(translations=ray_hits_w.reshape(-1, 3))
        self._visualize_target_markers()

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

    def _visualize_target_markers(self) -> None:
        if not self.cfg.tracking_mode or self.target_state is None or self.target_visualizer is None:
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

        sphere_locations = torch.zeros_like(self._marker_locations)
        sphere_locations[:, :2] = target_pos_world
        sphere_locations[:, 2] = env_origins[:, 2]
        sphere_loc = sphere_locations + self._marker_offset

        sphere_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        sphere_orientations[:, 3] = 1.0

        self.target_visualizer.visualize(sphere_loc, sphere_orientations)

    def _update_commands(self) -> None:
        if not self.cfg.tracking_mode:
            return
        if self.target_state is None:
            self._commands[:] = 0.0
            return
        env_origins = self._terrain.env_origins if self._terrain.terrain_origins is not None else self.scene.env_origins
        base_pos_xy = self.robot.data.root_pos_w[:, :2] - env_origins[:, :2]
        target_xy = self.target_state[:, :2]
        delta_xy = target_xy - base_pos_xy
        dist = torch.linalg.norm(delta_xy, dim=1, keepdim=True)
        dist_safe = torch.clamp(dist, min=1e-6)
        direction_unit = delta_xy / dist_safe

        desired_speed = torch.full_like(dist, self.cfg.max_linear_speed)
        close_to_goal = dist.squeeze(-1) <= self.cfg.position_tolerance
        if torch.any(close_to_goal):
            close_dist = dist[close_to_goal]
            close_ratio = torch.clamp(close_dist / self.cfg.position_tolerance, 0.0, 1.0)
            desired_speed[close_to_goal] = self.cfg.max_linear_speed * (
                torch.expm1(close_ratio) / math.expm1(1.0)
            )
        desired_vel_world_xy = direction_unit * desired_speed

        _, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w)
        yaw_error = math_utils.wrap_to_pi(self.target_state[:, 2] - yaw)
        yaw_rate_cmd = torch.clamp(
            yaw_error * self.cfg.yaw_kp,
            -self.cfg.max_yaw_rate,
            self.cfg.max_yaw_rate,
        )

        vel_world = torch.zeros(self.num_envs, 3, device=self.device)
        vel_world[:, :2] = desired_vel_world_xy
        vel_body = math_utils.quat_apply_yaw(math_utils.quat_conjugate(self.robot.data.root_quat_w), vel_world)

        self._commands[:, :2] = vel_body[:, :2]
        self._commands[:, 2] = yaw_rate_cmd

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
        x_vel_des = self._commands[:, 0:1]
        yaw_vel_des = self._commands[:, 2:3]
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
