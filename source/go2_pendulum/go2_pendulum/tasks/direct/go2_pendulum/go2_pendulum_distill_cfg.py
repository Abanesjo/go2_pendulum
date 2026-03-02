# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python scripts/rsl_rl/train.py \
#   --task Template-Go2-Pendulum-Distill-Direct-v0 \
#   --agent rsl_rl_distillation_cfg_entry_point \
#   --experiment_name go2_pendulum_direct \
#   --load_run <teacher_run_folder> \
#   --checkpoint model_1500.pt \
#   --headless

import math

from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass

from .go2_pendulum_env_cfg import Go2PendulumEnvCfg


@configclass
class Go2PendulumDistillEnvCfg(Go2PendulumEnvCfg):
    # Distill task uses LPF to mirror the deployment/control pipeline.
    enable_action_lpf = True
    action_lpf_cutoff_hz = 8.0

    # Enable selected domain randomization for student-policy robustness.
    enable_domain_randomization = False
    dr_seed_offset = 0

    # Material randomization.
    enable_material_randomization = True
    material_randomize_on_reset = True
    material_randomization_prob = 1.0
    material_num_buckets = 64
    material_static_friction_range = (0.5, 1.25)
    material_dynamic_friction_range = (0.4, 1.1)
    material_restitution_range = (0.0, 0.05)
    material_make_consistent = True

    # Base mass / COM randomization.
    enable_mass_randomization = True
    mass_randomize_body_name = "base"
    mass_scale_range = (0.85, 1.15)
    mass_recompute_inertia = True
    enable_com_randomization = True
    com_offset_x_range = (-0.015, 0.015)
    com_offset_y_range = (-0.015, 0.015)
    com_offset_z_range = (-0.01, 0.01)

    # Motor gain randomization.
    enable_motor_gain_randomization = True
    motor_gain_actuator_name = "base_legs"
    motor_stiffness_scale_range = (0.8, 1.2)
    motor_damping_scale_range = (0.8, 1.2)
    motor_gain_per_joint = True

    # Sensor bias and drift randomization.
    enable_sensor_bias_drift = True
    imu_lin_vel_bias_range = 0.05
    imu_ang_vel_bias_range = math.radians(3.0)
    imu_gravity_bias_range = 0.03
    imu_lin_vel_drift_std_per_s = 0.01
    imu_ang_vel_drift_std_per_s = math.radians(0.5)
    imu_gravity_drift_std_per_s = 0.0
    encoder_joint_pos_bias_range = math.radians(1.0)
    encoder_joint_vel_bias_range = math.radians(5.0)
    encoder_pendulum_pos_bias_range = math.radians(0.1)
    encoder_pendulum_vel_bias_range = math.radians(1.0)
    encoder_joint_pos_drift_std_per_s = math.radians(0.0)
    encoder_joint_vel_drift_std_per_s = math.radians(0.0)
    encoder_pendulum_pos_drift_std_per_s = math.radians(0.0)
    encoder_pendulum_vel_drift_std_per_s = math.radians(0.0)

    # External wrench pushes.
    enable_external_wrench_push = True
    push_body_name = "base"
    push_is_global = True
    push_interval_s_min = 2.0
    push_interval_s_max = 5.0
    push_duration_s_min = 0.05
    push_duration_s_max = 0.15
    push_force_x_range = (-25.0, 25.0)
    push_force_y_range = (-25.0, 25.0)
    push_force_z_range = (0.0, 0.0)
    push_torque_x_range = (0.0, 0.0)
    push_torque_y_range = (0.0, 0.0)
    push_torque_z_range = (-3.0, 3.0)

    # Keep observation delay and additive noise disabled for distillation.
    enable_obs_delay = False
    obs_delay_steps_min = 0
    obs_delay_steps_max = 0
    obs_delay_randomize_per_reset = False
    obs_delay_jitter_prob = 0.0
    obs_delay_jitter_extra_max = 0
    obs_delay_proprio_only = True
    observation_noise_scale = 0.0
    position_noise = 0.0
    body_lin_vel_noise = 0.0
    orientation_noise = 0.0
    body_ang_vel_noise = 0.0
    pendulum_joint_pos_noise = 0.0
    pendulum_joint_vel_noise = 0.0

    # IMU attached to base frame.
    imu_sensor: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=0.005,
        gravity_bias=(0.0, 0.0, 9.81),
    )

    def __post_init__(self):
        super().__post_init__()
        # Distill observation: imu_acc(3) + imu_gyro(3) + state_error(3) + leg_pos(12) + leg_vel(12)
        # + pendulum_pos(p) + pendulum_vel(p) + prev_action(12)
        self.observation_space = 45 + 2 * len(self.pendulum_joint_names)
