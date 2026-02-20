# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

from isaaclab.actuators import DCMotorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

GO2_PENDULUM_USD_PATH = os.path.join(os.path.dirname(__file__), "go2_model", "go2_pendulum.usd")
GO2_USD_PATH = os.path.join(os.path.dirname(__file__), "go2_model", "go2.usd")

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
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.6,
            friction=0.0,
        ),
    },
)


@configclass
class Go2PendulumEnvCfg(DirectRLEnvCfg):
    # Core environment interface.
    decimation = 4
    episode_length_s = 20
    action_space = 12
    action_scale = 0.25
    enable_action_clipping = False
    action_clip = 1.0
    enable_action_lpf = True
    action_lpf_cutoff_hz = 8.0
    enable_action_delay = True
    action_delay_steps_min = 0
    action_delay_steps_max = 2
    action_delay_randomize_per_reset = True
    enable_per_joint_action_bounds = True
    action_bound_margin = 1.0
    enable_desired_joint_pos_hard_clamp = True
    action_over_limit_power = 2.0
    observation_space = 48 + 4 + 4
    state_space = 0
    debug_vis = True
    use_pendulum = True
    track_goal = True

    # Domain randomization.
    enable_domain_randomization = False
    dr_seed_offset = 0

    # Material randomization (robot only).
    enable_material_randomization = True
    material_randomize_on_reset = True
    material_randomization_prob = 1.0
    material_num_buckets = 64
    material_static_friction_range = (0.5, 1.25)
    material_dynamic_friction_range = (0.4, 1.1)
    material_restitution_range = (0.0, 0.05)
    material_make_consistent = True

    # Base mass / center-of-mass randomization.
    enable_mass_randomization = True
    mass_randomize_body_name = "base"
    mass_scale_range = (0.85, 1.15)
    mass_recompute_inertia = True
    enable_com_randomization = True
    com_offset_x_range = (-0.015, 0.015)
    com_offset_y_range = (-0.015, 0.015)
    com_offset_z_range = (-0.01, 0.01)

    # Motor gain randomization (PD gain scaling).
    enable_motor_gain_randomization = True
    motor_gain_actuator_name = "base_legs"
    motor_stiffness_scale_range = (0.8, 1.2)
    motor_damping_scale_range = (0.8, 1.2)
    motor_gain_per_joint = False

    # Proprioceptive observation delay / jitter.
    enable_obs_delay = True
    obs_delay_steps_min = 0
    obs_delay_steps_max = 1
    obs_delay_randomize_per_reset = True
    obs_delay_jitter_prob = 0.1
    obs_delay_jitter_extra_max = 1
    obs_delay_proprio_only = True

    # Sensor bias and drift randomization.
    enable_sensor_bias_drift = True
    imu_lin_vel_bias_range = 0.05
    imu_ang_vel_bias_range = math.radians(3.0)
    imu_gravity_bias_range = 0.03
    imu_lin_vel_drift_std_per_s = 0.01
    imu_ang_vel_drift_std_per_s = math.radians(0.5)
    imu_gravity_drift_std_per_s = 0.005
    encoder_joint_pos_bias_range = math.radians(1.5)
    encoder_joint_vel_bias_range = math.radians(6.0)
    encoder_pendulum_pos_bias_range = math.radians(0.8)
    encoder_pendulum_vel_bias_range = math.radians(4.0)
    encoder_joint_pos_drift_std_per_s = math.radians(0.2)
    encoder_joint_vel_drift_std_per_s = math.radians(1.0)
    encoder_pendulum_pos_drift_std_per_s = math.radians(0.1)
    encoder_pendulum_vel_drift_std_per_s = math.radians(0.5)

    # External wrench pushes.
    enable_external_wrench_push = False
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

    # Initial conditions (reset sampling).
    # - Goal target sampling in the environment frame.
    goal_randomization_dist_min = 0.5
    goal_randomization_dist_max = 0.6
    goal_randomization_angle_min = math.radians(0)
    goal_randomization_angle_max = math.radians(360)
    if enable_domain_randomization:
        goal_yaw_randomization_min = math.radians(-30)
        goal_yaw_randomization_max = math.radians(30)
    else:
        goal_yaw_randomization_min = math.radians(0)
        goal_yaw_randomization_max = math.radians(0)

    # - Pendulum reset angle sampling.
    pendulum_joint_names = ["pendulum_joint1", "pendulum_joint2"]
    pendulum_angle_min = math.radians(0.0)
    if enable_domain_randomization:
        pendulum_angle_max = math.radians(0.5)
    else:
        pendulum_angle_max = math.radians(9.9)
    

    # Termination conditions.
    termination_grace_s = 0.1
    base_contact_grace_s = 0.5
    base_height_min = 0.15
    base_height_terminate_duration_s = 25.0

    pendulum_contact_force_threshold = 1.0

    if enable_domain_randomization:
        pendulum_terminate_angle_rad = math.radians(5.0)
    else:
        pendulum_terminate_angle_rad = math.radians(15)
    
    pendulum_terminate_duration_s = 5.0
    

    position_tolerance = 0.1
    position_terminate_duration_s = 25.0
    termination_penalty = -500.0

    # Position tracking and heading alignment.
    position_reward_scale = 0.2
    position_reward_sigma = 0.3
    progress_reward_scale = 100.0
    yaw_alignment_reward_scale = 0.3
    yaw_alignment_reward_sigma = 0.2

    # Pendulum/balance rewards.
    pendulum_upright_reward_scale = 0.4
    pendulum_upright_reward_sigma = 0.052
    pendulum_vel_reward_scale = -0.05
    pendulum_vel_reward_sigma = 0.05  # unused with squared-velocity penalty
    balanced_movement_reward_scale = 0.1

    # Quadruped motion regularization and gait shaping.
    feet_clearance_reward_scale = -20.0
    tracking_contacts_shaped_force_reward_scale = 1.0
    feet_air_time_reward_scale = 0.1
    action_magnitude_reward_scale = -0.001
    action_rate_reward_scale = -0.0001
    action_soft_limit = 2.0
    action_over_limit_reward_scale = -0.01
    torque_reward_scale = -0.0001
    orient_reward_scale = 0.1
    lin_vel_z_reward_scale = -2.0
    dof_vel_reward_scale = -0.003
    dof_acc_reward_scale = -2.5e-7
    ang_vel_xy_reward_scale = -0.01
    undesired_contact_reward_scale = -1.0
    # Base-height shaping reward (separate from base-height termination above).
    base_height_target = 0.35
    base_height_reward_sigma = 0.1
    base_height_reward_scale = 0.5

    # Observation noise.
    observation_noise_scale = 1.0
    position_noise = 0.02  # meters, applied to x/y position-error observation
    body_lin_vel_noise = 0.1  # m/s, applied to root_lin_vel_b

    orientation_noise = math.radians(1.0)  # radians, applied to yaw-error observation
    body_ang_vel_noise = math.radians(5.0)  # rad/s, applied to root_ang_vel_b
   
    pendulum_joint_pos_noise = math.radians(1.0)  # rad, applied to pendulum joint-angle observation
    pendulum_joint_vel_noise = math.radians(5.0)  # rad/s, applied to pendulum joint-velocity observation

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

    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetMarkers",
        markers={
            "target_sphere": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()

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

        # Increase GPU rigid patch buffer to avoid PhysX patch overflow.
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
