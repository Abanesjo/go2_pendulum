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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

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
    enable_action_delay = True
    action_delay_steps_min = 0
    action_delay_steps_max = 2
    action_delay_randomize_per_reset = True
    enable_per_joint_action_bounds = True
    action_bound_margin = 1.0
    enable_desired_joint_pos_hard_clamp = True
    action_over_limit_power = 2.0
    observation_space = 48 + 4 + 4
    state_space = 48 + 4 + 4  # Asymmetric actor-critic: critic obs dimension (set in __post_init__)
    debug_vis = True
    use_pendulum = True

    # Action low-pass filter.
    enable_action_lpf = False
    action_lpf_cutoff_hz = 8.0

    # --- Curriculum (noise ramp + difficulty progression) ---
    # Curriculum completes over 40000 iterations (40000 * 32 = 480000 steps).
    # Beyond that, training continues at the highest difficulty level.
    enable_curriculum = True
    curriculum_total_steps = 15000 * 32
    noise_curriculum_start_scale = 0.0
    noise_curriculum_end_scale = 1.0
    difficulty_override: int = -1  # -1 = use curriculum, 1-4 = force that difficulty level

    # --- Difficulty-dependent defaults (level 1 initial values) ---
    # These are updated at runtime by the difficulty curriculum.
    # See _DIFFICULTY_PRESETS in go2_pendulum_env.py for all four levels.

    # Initial conditions (reset sampling).
    goal_randomization_dist_min = 0.0
    goal_randomization_dist_max = 0.0

    goal_randomization_angle_min = math.radians(0)
    goal_randomization_angle_max = math.radians(360)

    goal_yaw_randomization_min = math.radians(0)
    goal_yaw_randomization_max = math.radians(0)

    # Pendulum reset angle sampling.
    pendulum_joint_names = ["pendulum_joint1", "pendulum_joint2"]
    pendulum_angle_min = math.radians(0.0)
    pendulum_angle_max = math.radians(0.0)

    # Pendulum hard joint limits (applied at runtime, no USD edits needed).
    pendulum_joint_limit_min_rad = math.radians(-90.0)
    pendulum_joint_limit_max_rad = math.radians(90.0)

    # Termination conditions.
    termination_grace_s = 0.1
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

    # --- Observation noise (applied to actor obs only, scaled by observation_noise_scale) ---
    # TODO(human): Tune these noise magnitudes based on real sensor characteristics.
    # These define the max noise at observation_noise_scale=1.0 (uniform ±value).
    observation_noise_scale = 0.0
    body_lin_vel_noise = 0.1
    body_ang_vel_noise = 0.2
    orientation_noise = 0.05
    position_noise = 0.01
    pendulum_joint_pos_noise = 0.02
    pendulum_joint_vel_noise = 1.0

    # Position tracking and heading alignment.
    position_reward_scale = 0.4
    position_reward_sigma = 0.3
    progress_reward_scale = 10.0
    yaw_alignment_reward_scale = 0.3
    yaw_alignment_reward_sigma = 0.2

    # Pendulum/balance rewards.
    pendulum_upright_reward_scale = 0.45
    pendulum_upright_reward_sigma = 0.01
    pendulum_vel_reward_scale = -0.1
    pendulum_vel_reward_sigma = 0.05  # unused with squared-velocity penalty
    balanced_movement_reward_scale = 0.1

    # Quadruped motion regularization and gait shaping.
    feet_clearance_reward_scale = -20.0
    tracking_contacts_shaped_force_reward_scale = 1.0
    feet_air_time_reward_scale = 0.1
    action_magnitude_reward_scale = -0.1
    action_rate_reward_scale = -0.01
    action_soft_limit = 2.0
    action_over_limit_reward_scale = -0.01
    torque_reward_scale = -0.0001
    orient_reward_scale = 0.8
    orient_reward_sigma = 0.05
    lin_vel_z_reward_scale = -2.0
    dof_vel_reward_scale = -0.003
    dof_acc_reward_scale = -2.5e-7
    ang_vel_xy_reward_scale = -0.01
    undesired_contact_reward_scale = -1.0
    # Base-height shaping reward (separate from base-height termination above).
    base_height_target = 0.33
    base_height_reward_sigma = 0.1
    base_height_reward_scale = 0.2

    # --- Domain randomization (disabled by default, enable for sim-to-real) ---
    enable_domain_randomization = True
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
    mass_scale_range = (0.9, 1.55)
    mass_recompute_inertia = True
    enable_com_randomization = True
    com_offset_x_range = (-0.03, 0.03)
    com_offset_y_range = (-0.03, 0.03)
    com_offset_z_range = (-0.02, 0.05)

    # Motor gain randomization.
    enable_motor_gain_randomization = True
    motor_gain_actuator_name = "base_legs"
    motor_stiffness_scale_range = (0.8, 1.2)
    motor_damping_scale_range = (0.8, 1.2)
    motor_gain_per_joint = True

    # Observation delay.
    enable_obs_delay = True
    obs_delay_steps_min = 1
    obs_delay_steps_max = 4
    obs_delay_randomize_per_reset = False
    obs_delay_jitter_prob = 0.0
    obs_delay_jitter_extra_max = 0
    obs_delay_proprio_only = True

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
    encoder_pendulum_pos_bias_range = math.radians(2.5)
    encoder_pendulum_vel_bias_range = math.radians(3.0)
    encoder_joint_pos_drift_std_per_s = math.radians(0.0)
    encoder_joint_vel_drift_std_per_s = math.radians(0.0)
    encoder_pendulum_pos_drift_std_per_s = math.radians(0.03)
    encoder_pendulum_vel_drift_std_per_s = math.radians(0.1)

    # External wrench pushes.
    enable_external_wrench_push = True
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
