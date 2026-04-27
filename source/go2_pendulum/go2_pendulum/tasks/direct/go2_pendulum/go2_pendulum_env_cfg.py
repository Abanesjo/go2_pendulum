# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------
# User Guide: deploying a trained policy from this task
#
# This environment exposes a 56-D actor observation and expects a 12-D action.
# The critic uses the same layout, but with clean ground-truth values instead of
# noisy actor measurements.
#
# Observation layout (policy and critic):
#   0:3    body linear velocity in base frame                      [vx, vy, vz]
#          computed from finite-differenced base pose
#   3:6    body angular velocity in base frame                     [wx, wy, wz]
#          from the simulated IMU gyroscope
#   6:9    projected gravity in base frame                        [gx, gy, gz]
#          reconstructed from the simulated IMU quaternion
#   9:12   target-state error in base frame                        [
#             target_x_error_body,
#             target_y_error_body,
#             wrap_to_pi(target_yaw - base_yaw),
#          ]
#   12:24  leg joint position offsets from the default pose       [
#             FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint,
#             FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint,
#             FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint,
#          ]
#   24:36  leg joint velocities                                   [
#             FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint,
#             FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint,
#             FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint,
#          ]
#   36:38  pendulum joint positions                                [
#             pendulum_joint1, pendulum_joint2,
#          ]
#   38:40  pendulum joint velocities                               [
#             pendulum_joint1, pendulum_joint2,
#          ]
#   40:52  last applied action command                             [
#             FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint,
#             FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint,
#             FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint,
#          ]
#   52:56  gait clock inputs                                       [
#             sin(phase_FL), sin(phase_FR), sin(phase_RL), sin(phase_RR),
#          ]
#
# Notes:
#   - The actor observation is the one used by the policy at inference time.
#   - The actor observation includes the same kinds of effects used in training:
#     sensor bias/drift, observation noise, packet holds, and transport delay.
#   - The "last action" term is the latest delayed action command applied by the task.
#
# Action layout:
#   The policy outputs 12 floating-point values ordered as
#     [
#       FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint,
#       FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint,
#       FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint,
#     ]
#
# Action semantics:
#   - These 12 values are joint-position offsets, not absolute joint targets.
#   - With domain randomization enabled, the task may delay/hold action packets
#     and lag the resulting torque command. There is no hard action clipping.
#   - The delivered action is converted to desired joint positions:
#
#       q_des = q_default + action_scale * action
#
#     with
#       action_scale = 0.25  # radians per unit action
#
#   - The current default joint-position offsets used by this task are
#       [
#          0.1, -0.1,  0.1, -0.1,   # FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint
#          0.8,  0.8,  1.0,  1.0,   # FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint
#         -1.5, -1.5, -1.5, -1.5,   # FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint
#       ]
#
#   Example:
#     if the policy outputs a_FR_thigh = 0.4, then the desired FR thigh target
#     sent to the low-level joint position controller is
#
#       q_des_FR_thigh = 0.8 + 0.25 * 0.4 = 0.9 rad
#
# Inference recipe:
#   1. Build the 56-D observation in the exact order above.
#   2. Run the policy to obtain the 12-D joint-offset action.
#   3. Convert the action to desired joint positions with the formula above.
#   4. Send those desired joint positions to the robot's joint position
#      controller in the same joint order.
# -----------------------------------------------------------------------------

import math
import os

from isaaclab.actuators import DCMotorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.imu import ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

GO2_PENDULUM_USD_PATH = os.path.join(os.path.dirname(__file__), "go2_model", "go2_pendulum.usd")
GO2_USD_PATH = os.path.join(os.path.dirname(__file__), "go2_model", "go2.usd")

GO2_LEG_JOINT_NAMES = [
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
]

GO2_PENDULUM_JOINT_NAMES = ["pendulum_joint1", "pendulum_joint2"]

GO2_DEFAULT_JOINT_POS = {
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
    "pendulum_joint1": 0.0,
    "pendulum_joint2": 0.0,
}

GO2_DEFAULT_JOINT_VEL = {joint_name: 0.0 for joint_name in GO2_DEFAULT_JOINT_POS}

LEG_ARMATURE = 0.01
LEG_FRICTION = 0.05
LEG_DYNAMIC_FRICTION = 0.05
LEG_VISCOUS_FRICTION = 0.02

GO2_PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2_PENDULUM_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos=GO2_DEFAULT_JOINT_POS,
        joint_vel=GO2_DEFAULT_JOINT_VEL,
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=GO2_LEG_JOINT_NAMES,
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.6,
            armature=LEG_ARMATURE,
            friction=LEG_FRICTION,
            dynamic_friction=LEG_DYNAMIC_FRICTION,
            viscous_friction=LEG_VISCOUS_FRICTION,
        ),
    },
)


@configclass
class Go2PendulumEnvCfg(DirectRLEnvCfg):
    # Core environment interface.
    decimation = 4
    episode_length_s = 20
    leg_joint_names = GO2_LEG_JOINT_NAMES
    action_space = len(leg_joint_names)
    action_scale = 0.25
    observation_space = 48 + 4 + 4
    state_space = 48 + 4 + 4  # Asymmetric actor-critic: critic obs dimension (set in __post_init__)
    debug_vis = True
    use_pendulum = True

    # --- Curriculum (difficulty progression only) ---
    # Curriculum completes over 75000 * 32 environment steps.
    # Beyond that, training continues at the highest difficulty level.
    enable_curriculum = True
    curriculum_total_steps = 25000 * 32
    difficulty_override: int = -1  # -1 = use curriculum, 1-5 = force that difficulty level

    # --- Difficulty-dependent defaults (level 1 initial values) ---
    # These are updated at runtime by the difficulty curriculum.
    # See _DIFFICULTY_PRESETS in go2_pendulum_env.py for all five levels.

    # Initial conditions (reset sampling).
    goal_randomization_dist_min = 0.0
    goal_randomization_dist_max = 0.0

    # Bearing of the target position in the environment frame.
    # A full-circle range keeps the target position free to spawn anywhere around the robot.
    goal_randomization_angle_min = math.radians(-180)
    goal_randomization_angle_max = math.radians(180)

    # Desired robot heading at the target location.
    goal_yaw_randomization_min = math.radians(0)
    goal_yaw_randomization_max = math.radians(0)

    # Pendulum reset angle sampling.
    pendulum_joint_names = GO2_PENDULUM_JOINT_NAMES
    pendulum_angle_min = math.radians(0.0)
    pendulum_angle_max = math.radians(0.0)

    # Pendulum hard joint limits (applied at runtime, no USD edits needed).
    pendulum_joint_limit_min_rad = math.radians(-90.0)
    pendulum_joint_limit_max_rad = math.radians(90.0)

    # Termination conditions.
    termination_grace_s = 0.1
    pendulum_termination_grace_s = 3.0
    base_contact_grace_s = 0.5
    base_height_min = 0.28
    base_height_terminate_duration_s = 10.0
    base_tilt_terminate_angle_rad = math.radians(60.0)
    pendulum_contact_force_threshold = 1.0
    pendulum_terminate_angle_rad = math.radians(60.0)
    pendulum_terminate_duration_s = 0.1
    position_tolerance = 5.0
    position_terminate_duration_s = 15.0
    termination_penalty = -5.0

    # --- Actor observation/action transport randomization ---
    # Delay ranges are sampled per environment at reset and are counted at the
    # 50 Hz policy/environment step rate.
    action_delay_steps_range = (0, 2)
    proprio_delay_steps_range = (0, 1)
    base_lin_vel_delay_steps_range = (0, 2)
    pendulum_delay_steps_range = (0, 3)

    # Packet holds repeat the previously delivered action/observation packet.
    action_hold_prob = 0.01
    proprio_obs_hold_prob = 0.005
    pendulum_obs_hold_prob = 0.01

    # Observation noise and reset-sampled bias magnitudes in raw units. Noise is
    # uniform in [-value, value]; bias is sampled once per reset in the same range.
    joint_pos_noise_rad = 0.017
    joint_pos_bias_rad = 0.010
    joint_vel_noise_rad_s = 0.50
    joint_vel_bias_rad_s = 0.05
    base_lin_vel_noise_m_s = 0.10
    base_lin_vel_bias_m_s = 0.03
    base_ang_vel_noise_rad_s = 0.25
    base_ang_vel_bias_rad_s = 0.02
    projected_gravity_component_noise = 0.025
    pendulum_pos_noise_rad = 0.008
    pendulum_pos_bias_rad = 0.005
    pendulum_vel_noise_rad_s = 0.40
    pendulum_vel_bias_rad_s = 0.05

    # Deprecated/unused compatibility fields from earlier estimator-aligned DR.
    imu_orientation_noise_std_rad = math.radians(1.0)
    vicon_pos_noise_std_m = 0.001
    vicon_orientation_noise_std_rad = math.radians(0.5)
    body_lin_vel_noise = 0.1
    body_ang_vel_noise = 0.2
    orientation_noise = 0.05
    position_noise = 0.01
    pendulum_joint_pos_noise = math.radians(0.06)
    pendulum_joint_vel_noise = math.radians(3.0)

    # Position tracking and heading alignment.
    position_reward_scale = 0.4
    position_reward_sigma = 0.3
    progress_reward_scale = 10.0
    yaw_alignment_reward_scale = 0.3
    yaw_alignment_reward_sigma = 0.2

    # Pendulum/balance rewards.
    pendulum_upright_reward_scale = 0.45
    pendulum_upright_reward_sigma = 0.15
    pendulum_vel_reward_scale = -0.1
    pendulum_vel_reward_sigma = 0.05  # unused with squared-velocity penalty
    balanced_movement_reward_scale = 0.1

    # Quadruped motion regularization and gait shaping.
    feet_clearance_reward_scale = -20.0
    tracking_contacts_shaped_force_reward_scale = 1.0
    feet_air_time_reward_scale = 0.1
    action_magnitude_reward_scale = -0.1
    action_rate_reward_scale = -0.02
    action_acc_reward_scale = -0.02
    torque_reward_scale = -0.0002
    torque_rate_reward_scale = -1.0e-4
    orient_reward_scale = 0.8
    orient_reward_sigma = 0.05
    lin_vel_z_reward_scale = -2.0
    dof_vel_reward_scale = -0.003
    dof_acc_reward_scale = -5.0e-7
    ang_vel_xy_reward_scale = -0.01
    undesired_contact_reward_scale = -1.0
    # Base-height shaping reward (separate from base-height termination above).
    base_height_target = 0.33
    base_height_reward_sigma = 0.1
    base_height_reward_scale = 0.2

    # --- Domain randomization (disabled by default, enable for sim-to-real) ---
    enable_domain_randomization = True
    dr_seed_offset = 0

    # Base mass / COM randomization.
    enable_mass_randomization = True
    mass_randomize_body_name = "base"
    mass_scale_range = (0.9, 1.2)
    mass_recompute_inertia = True
    enable_com_randomization = False
    com_offset_x_range = (-0.03, 0.03)
    com_offset_y_range = (-0.03, 0.03)
    com_offset_z_range = (-0.02, 0.05)

    # Motor/actuator randomization.
    enable_motor_gain_randomization = True
    motor_gain_actuator_name = "base_legs"
    motor_strength_range = (0.8, 1.2)
    kp_scale_range = (0.8, 1.3)
    kd_scale_range = (0.5, 1.5)
    effort_limit_scale_range = (0.7, 1.0)
    torque_response_tau_s_range = (0.005, 0.020)
    motor_stiffness_scale_range = kp_scale_range
    motor_damping_scale_range = kd_scale_range
    motor_gain_per_joint = False

    # Sensor bias and drift randomization.
    enable_sensor_bias_drift = True
    imu_lin_vel_bias_range = 0.05  # deprecated/unused after the estimator-aligned IMU/Vicon refactor
    imu_ang_vel_bias_range = math.radians(3.0)
    imu_gravity_bias_range = 0.03  # deprecated/unused after the estimator-aligned IMU/Vicon refactor
    imu_lin_vel_drift_std_per_s = 0.0  # deprecated/unused after the estimator-aligned IMU/Vicon refactor
    imu_ang_vel_drift_std_per_s = math.radians(0.0)
    imu_gravity_drift_std_per_s = 0.0  # deprecated/unused after the estimator-aligned IMU/Vicon refactor
    encoder_joint_pos_bias_range = math.radians(1.0)
    encoder_joint_vel_bias_range = math.radians(5.0)
    encoder_pendulum_pos_bias_range = math.radians(1.0)
    encoder_pendulum_vel_bias_range = math.radians(3.0)
    encoder_joint_pos_drift_std_per_s = math.radians(0.0)
    encoder_joint_vel_drift_std_per_s = math.radians(0.0)
    encoder_pendulum_pos_drift_std_per_s = math.radians(0.0)
    encoder_pendulum_vel_drift_std_per_s = math.radians(0.0)

    # External wrench pushes.
    enable_external_wrench_push = False
    push_body_name = "base"
    push_is_global = True
    push_interval_s_min = 5.0
    push_interval_s_max = 10.0
    push_duration_s_min = 0.05
    push_duration_s_max = 0.15
    push_force_x_range = (-10.0, 10.0)
    push_force_y_range = (-10.0, 10.0)
    push_force_z_range = (0.0, 0.0)
    push_torque_x_range = (0.0, 0.0)
    push_torque_y_range = (0.0, 0.0)
    push_torque_z_range = (0.0, 0.0)

    # Simulation and scene.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # robot(s)
    robot_cfg: ArticulationCfg = GO2_PENDULUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.articulation_root_prim_path = "/base"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    pendulum_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/pendulum_ee", history_length=1, update_period=0.005, track_air_time=False
    )
    imu_sensor: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=0.0,
        debug_vis=False,
    )

    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetMarkers",
        markers={
            "target_arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.5, 0.5, 0.2),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if len(self.leg_joint_names) != self.action_space:
            raise ValueError(
                f"action_space must match the number of configured leg joints. Got {self.action_space} and"
                f" {len(self.leg_joint_names)} leg joints."
            )
        if len(set(self.leg_joint_names)) != len(self.leg_joint_names):
            raise ValueError("leg_joint_names contains duplicates. It must define a unique canonical joint order.")
        if len(set(self.pendulum_joint_names)) != len(self.pendulum_joint_names):
            raise ValueError(
                "pendulum_joint_names contains duplicates. It must define a unique canonical pendulum joint order."
            )

        # Keep pendulum toggle behavior available, but disable by default.
        if not self.use_pendulum:
            self.robot_cfg = self.robot_cfg.replace(
                spawn=self.robot_cfg.spawn.replace(usd_path=GO2_USD_PATH),
            )
            self.pendulum_contact_sensor = self.pendulum_contact_sensor.replace(
                prim_path="/World/envs/env_.*/Robot/base"
            )

        # Keep observation dims fixed so policies are compatible across pendulum modes.
        self.observation_space = 48 + 4 + 2 * len(self.pendulum_joint_names)
        # Critic gets same structure as actor, just ground-truth (no noise).
        self.state_space = self.observation_space

        # Increase GPU rigid patch buffer to avoid PhysX patch overflow.
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
