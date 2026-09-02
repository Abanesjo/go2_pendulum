# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# -----------------------------------------------------------------------------
# Policy contract v3 (old checkpoints are intentionally incompatible)
#
# Actor observations (56-D):
#   0:3    body linear velocity from 50 Hz finite-differenced mocap position
#   3:6    IMU body angular velocity
#   6:9    IMU projected gravity
#   9:12   [forward velocity command, lateral velocity command, yaw-rate command]
#   12:24  leg joint position offsets in GO2_LEG_JOINT_NAMES order
#   24:36  leg joint velocities in GO2_LEG_JOINT_NAMES order
#   36:38  mocap-reconstructed pendulum joint angles
#   38:40  50 Hz wrapped finite differences of the pendulum angles
#   40:52  previous normalized action actually delivered by the action pipeline
#   52:56  [sin(gait phase), cos(gait phase), move gate, stand gate]
#
# The critic keeps the same dimension and clean simulator values, but its 9:12
# slice is [body-frame goal x error, body-frame goal y error, final yaw error].
# The policy output uses GO2_LEG_JOINT_NAMES order and is processed as:
#   clip -> packet hold/delay -> randomized low-pass filter -> add the scaled
#   offset to q_default -> joint-limit clamp -> target slew limit.
# See README.md and the policy_contract.yaml emitted by play.py for deployment.
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

GO2_PENDULUM_USD_PATH = os.path.join(os.path.dirname(__file__), "go2_model", "go2_pendulum_realistic.usd")
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

POLICY_CONTRACT_VERSION = "go2_pendulum_policy_v3"
POLICY_OBSERVATION_DIM = 56
POLICY_ACTION_DIM = 12

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
            solver_velocity_iteration_count=1,
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
    policy_contract_version = POLICY_CONTRACT_VERSION
    decimation = 4
    policy_control_hz = 50.0
    episode_length_s = 20
    leg_joint_names = GO2_LEG_JOINT_NAMES
    action_space = len(leg_joint_names)
    action_scale = 0.25
    observation_space = POLICY_OBSERVATION_DIM
    state_space = POLICY_OBSERVATION_DIM  # Asymmetric critic semantics; same tensor dimension.
    debug_vis = True
    use_pendulum = True

    # --- Curriculum (difficulty progression only) ---
    # train.py always derives this from the final max_iterations and
    # num_steps_per_env after Hydra/CLI overrides. The value here is a safe
    # standalone default for scripts that instantiate the task directly.
    enable_curriculum = True
    curriculum_total_steps = 25000 * 32
    # Populated from checkpoint iteration metadata before constructing a
    # resumed environment, so curriculum difficulty does not restart at level 1.
    curriculum_start_step = 0
    difficulty_override: int = -1  # -1 = use curriculum, 1-5 = force that difficulty level

    # Each reset is either nominal, uniformly randomized up to the active
    # curriculum scale, or pinned at the active maximum. These placeholders
    # should be calibrated from real bags when they become available.
    domain_randomization_scale = 0.25
    dr_nominal_fraction = 0.20
    dr_uniform_fraction = 0.70
    dr_max_fraction = 0.10
    evaluation_dr_scale_multiplier = 1.0
    evaluation_push_scale_multiplier = 1.0

    # --- Difficulty-dependent defaults (first curriculum anchor) ---
    # These are interpolated at runtime between the five anchors in
    # go2_pendulum_env.py. A difficulty override pins one exact anchor.

    # Per-reset level-1 goal mixture: planted/nearby/walking tasks. Curriculum
    # presets update these without ever removing planted examples.
    goal_distance_mixture = (0.50, 0.40, 0.10)
    goal_stand_distance_range = (0.0, 0.05)
    goal_short_distance_range = (0.10, 0.35)
    goal_walk_distance_range = (0.35, 1.50)
    # Goal chains are biased toward locomotion because true planted starts are
    # already supplied by the reset mixture above.
    enable_goal_chaining = True
    goal_chain_distance_mixture = (0.10, 0.55, 0.35)
    goal_chain_post_arrival_hold_s = 1.0
    # Evaluation may force a single class without changing training mixtures.
    goal_profile_override: str = "mixed"  # mixed | stand | short | walk

    # Bearing offset from the reset/current base yaw. The range grows from a
    # forward-facing cone to the full circle during the curriculum.
    goal_randomization_angle_min = math.radians(-45)
    goal_randomization_angle_max = math.radians(45)

    # Desired robot heading offset from the final perturbed reset/current base
    # heading. Planted goals use the narrower dedicated range.
    goal_yaw_randomization_min = math.radians(-10.0)
    goal_yaw_randomization_max = math.radians(10.0)
    goal_stand_yaw_offset_range = (math.radians(-2.0), math.radians(2.0))

    # Pendulum reset angle sampling.
    pendulum_joint_names = GO2_PENDULUM_JOINT_NAMES
    pendulum_angle_min = math.radians(0.0)
    pendulum_angle_max = math.radians(3.0)
    pendulum_recovery_reset_fraction = 0.0
    pendulum_recovery_angle_range = (math.radians(10.0), math.radians(11.0))
    # Hinge origin in base coordinates used to reconstruct the two pendulum
    # angles from the base and pendulum-end-effector mocap poses.
    pendulum_hinge_offset_b = (-0.05, 0.0, 0.06)

    # Small reset perturbations prevent a single deterministic startup state.
    reset_leg_joint_pos_noise_rad = 0.03
    reset_leg_joint_vel_noise_rad_s = 0.10
    reset_base_xy_noise_m = 0.02
    reset_base_roll_pitch_noise_rad = math.radians(2.0)
    reset_base_yaw_noise_rad = math.radians(5.0)
    reset_base_lin_vel_noise_m_s = 0.05
    reset_base_ang_vel_noise_rad_s = 0.10
    # Randomizing the initial episode phase is useful for desynchronizing
    # large training batches, but evaluation disables it for full episodes.
    stagger_initial_episode_lengths: bool = True

    # Pendulum hard joint limits (applied at runtime, no USD edits needed).
    pendulum_joint_limit_min_rad = math.radians(-90.0)
    pendulum_joint_limit_max_rad = math.radians(90.0)

    # Termination conditions.
    termination_grace_s = 0.5
    pendulum_termination_grace_s = 0.5
    base_contact_grace_s = 0.5
    base_height_min = 0.28
    base_height_terminate_duration_s = 0.25
    base_tilt_terminate_angle_rad = math.radians(60.0)
    pendulum_contact_force_threshold = 1.0
    pendulum_terminate_angle_rad = math.radians(20.0)
    pendulum_terminate_duration_s = 0.25
    # Goal-relative locomotion guards. Planted-class goals are exempt from the
    # progress watchdog and relative divergence; the absolute guard remains a
    # final arena-safety bound for every goal class.
    absolute_position_divergence_m = 2.5
    absolute_position_divergence_duration_s = 0.25
    relative_position_divergence_margin_m = 0.35
    relative_position_divergence_duration_s = 0.50
    goal_watchdog_initial_window_s = 4.0
    goal_watchdog_progress_window_s = 3.0
    goal_watchdog_exempt_distance_m = 0.08
    goal_watchdog_push_cooldown_s = 0.50
    termination_penalty = -20.0

    # --- Analytic world-goal to body-command navigation layer ---
    command_max_forward_speed_m_s = 0.6
    command_max_yaw_rate_rad_s = 1.2
    command_k_rho = 1.5
    command_k_alpha = 2.5
    command_k_beta = -0.8
    command_k_final_yaw = 2.0
    command_heading_blend_near_m = 0.05
    command_heading_blend_far_m = 0.20
    command_forward_heading_cutoff_rad = math.pi / 2
    stand_enter_distance_m = 0.05
    stand_exit_distance_m = 0.08
    stand_enter_yaw_rad = math.radians(5.0)
    stand_exit_yaw_rad = math.radians(8.0)
    # While the gait remains planted in stand mode, retain a small observable
    # correction command so the policy can remove residual pose drift.
    stand_correction_position_gain_s: float = 1.0
    stand_correction_yaw_gain_s: float = 1.5
    stand_correction_max_linear_m_s: float = 0.08
    stand_correction_max_yaw_rate_rad_s: float = 0.15

    # Arrival is a metric/dwell bonus, not an episode termination. Hysteresis
    # above prevents repeated walk/stand switching at its boundary.
    arrival_position_tolerance_m = 0.05
    arrival_yaw_tolerance_rad = math.radians(5.0)
    arrival_base_speed_tolerance_m_s = 0.08
    arrival_yaw_rate_tolerance_rad_s = 0.15
    arrival_pendulum_angle_tolerance_rad = math.radians(3.0)
    arrival_pendulum_speed_tolerance_rad_s = 0.25
    arrival_dwell_time_s = 1.0
    arrival_dwell_reward_scale = 1.0

    # --- Delivered action pipeline and safety envelope ---
    action_clip = 2.0
    command_lpf_cutoff_hz = 4.0
    command_lpf_cutoff_range_hz = (3.0, 5.0)
    joint_target_slew_rate_rad_s = 6.0
    joint_limit_margin_rad = 0.1

    # --- Actor observation/action transport randomization ---
    # Delay ranges are sampled per environment at reset and are counted at the
    # 50 Hz policy/environment step rate.
    action_delay_steps_range = (0, 2)
    proprio_delay_steps_range = (0, 1)
    # Pose delay is applied coherently before finite differentiation, so the
    # legacy independently delayed velocity/pendulum channels stay disabled.
    base_lin_vel_delay_steps_range = (0, 0)
    pendulum_delay_steps_range = (0, 0)
    mocap_delay_steps_range = (0, 1)

    # Packet holds repeat the previously delivered action/observation packet.
    action_hold_prob = 0.01
    proprio_obs_hold_prob = 0.005
    pendulum_obs_hold_prob = 0.0
    mocap_packet_hold_prob = 0.01

    # Mocap pose errors are applied before computing world-goal commands and
    # the 50 Hz finite differences used by the actor. Values are placeholders
    # until hardware bags are available.
    mocap_position_noise_std_m = 0.0005
    mocap_position_bias_range_m = 0.002
    mocap_orientation_noise_std_rad = math.radians(0.1)
    mocap_orientation_bias_range_rad = math.radians(0.25)

    # Observation noise and reset-sampled bias magnitudes in raw units. Noise is
    # uniform in [-value, value]; bias is sampled once per reset in the same range.
    joint_pos_noise_rad = 0.003
    joint_pos_bias_rad = 0.005
    joint_vel_noise_rad_s = 0.10
    joint_vel_bias_rad_s = 0.05
    base_lin_vel_noise_m_s = 0.0
    base_lin_vel_bias_m_s = 0.0
    base_ang_vel_noise_rad_s = 0.02
    base_ang_vel_bias_rad_s = 0.03
    projected_gravity_component_noise = 0.01
    pendulum_pos_noise_rad = 0.0
    pendulum_pos_bias_rad = 0.0
    pendulum_vel_noise_rad_s = 0.0
    pendulum_vel_bias_rad_s = 0.0

    # Position tracking and heading alignment.
    # Legacy direct goal terms are disabled. Contract v3 adds signed progress,
    # residual-distance, and locomotion-time terms below.
    position_reward_scale = 0.0
    position_reward_sigma = 0.3
    progress_reward_scale = 4.0
    progress_reward_clip_m = 0.02
    goal_distance_cost_scale = -1.0
    locomotion_time_cost_scale = -0.25
    yaw_alignment_reward_scale = 0.0
    yaw_alignment_reward_sigma = 0.2

    # Bounded normalized command-tracking costs apply only while locomoting.
    # Zero error has zero cost, preventing unused command components from
    # paying a positive reward to a stationary policy.
    command_lin_vel_cost_normalizer_m_s = 0.6
    command_lin_vel_reward_scale = -1.5
    command_yaw_rate_cost_normalizer_rad_s = 1.2
    command_yaw_rate_reward_scale = -0.75
    goal_position_hold_reward_sigma = 0.10
    goal_position_hold_reward_scale = 0.5
    goal_yaw_hold_reward_sigma = math.radians(10.0)
    goal_yaw_hold_reward_scale = 0.25
    stand_lin_vel_reward_sigma = 0.08
    stand_yaw_rate_reward_sigma = 0.15
    stand_settling_reward_scale = 1.0
    arrival_bonus_reward_scale = 5.0

    # Pendulum/balance rewards.
    pendulum_upright_reward_scale = 2.0
    pendulum_upright_reward_sigma = math.radians(6.0)
    pendulum_vel_reward_scale = -0.05
    balanced_movement_reward_scale = 0.0

    # Stand styling is relaxed smoothly during large pendulum recovery errors.
    style_pendulum_angle_full_rad = math.radians(5.0)
    style_pendulum_angle_zero_rad = math.radians(10.0)
    style_pendulum_speed_full_rad_s = 0.5
    style_pendulum_speed_zero_rad_s = 1.0

    # Command-dependent diagonal trot. In stand the phase is frozen and all
    # four feet are requested in contact.
    gait_frequency_min_hz = 1.5
    gait_frequency_max_hz = 2.5
    gait_duty_factor_min_speed = 0.62
    gait_duty_factor_max_speed = 0.52
    gait_contact_smoothing_sigma = 0.07
    gait_swing_height_m = 0.08
    gait_stance_height_m = 0.02
    foot_contact_force_threshold_n = 5.0
    stand_foot_lift_threshold_m = 0.025

    # Quadruped motion regularization and gait shaping.
    feet_clearance_reward_scale = -0.25
    tracking_contacts_shaped_force_reward_scale = 0.0
    feet_air_time_reward_scale = 0.2
    stand_contact_reward_scale = 0.75
    stand_foot_lift_reward_scale = -0.75
    foot_slip_reward_scale = -0.25
    gait_contact_mismatch_reward_scale = -0.5
    stand_symmetry_reward_scale = -0.1
    action_magnitude_reward_scale = -0.01
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

    # Episode-level coordinate relabeling preserves recurrent sequences. It is
    # deliberately not the feed-forward-only RSL-RL augmentation callback.
    enable_episode_mirroring = True
    episode_mirror_probability = 0.5
    # Base-height shaping reward (separate from base-height termination above).
    base_height_target = 0.33
    base_height_reward_sigma = 0.1
    base_height_reward_scale = 0.2

    # --- Domain randomization ---
    enable_domain_randomization = True
    dr_seed_offset = 0

    # Base mass / COM randomization.
    enable_mass_randomization = True
    mass_randomize_body_name = "base"
    mass_scale_range = (0.9, 1.1)
    mass_recompute_inertia = True
    enable_pendulum_mass_randomization: bool = True
    pendulum_mass_scale_range: tuple[float, float] = (0.9, 1.1)
    enable_pendulum_effective_length_randomization: bool = True
    pendulum_effective_length_offset_range_m: tuple[float, float] = (-0.02, 0.02)
    enable_com_randomization = True
    com_offset_x_range = (-0.015, 0.015)
    com_offset_y_range = (-0.015, 0.015)
    com_offset_z_range = (-0.010, 0.010)

    # Foot contact material randomization.
    enable_foot_friction_randomization = True
    foot_friction_body_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    foot_friction_range = (0.5, 1.25)

    # Motor/actuator randomization.
    enable_motor_gain_randomization = True
    motor_gain_actuator_name = "base_legs"
    motor_strength_range = (0.9, 1.1)
    kp_scale_range = (0.9, 1.1)
    kd_scale_range = (0.8, 1.2)
    effort_limit_scale_range = (0.85, 1.0)
    motor_gain_per_joint = True
    enable_pendulum_damping_randomization = True
    pendulum_damping_range = (0.0, 0.2)

    # Sensor bias and drift randomization.
    enable_sensor_bias_drift = True
    imu_ang_vel_bias_range = 0.03
    imu_ang_vel_drift_std_per_s = math.radians(0.0)
    encoder_joint_pos_bias_range = 0.005
    encoder_joint_vel_bias_range = 0.05
    encoder_pendulum_pos_bias_range = 0.0
    encoder_pendulum_vel_bias_range = 0.0
    encoder_joint_pos_drift_std_per_s = math.radians(0.0)
    encoder_joint_vel_drift_std_per_s = math.radians(0.0)
    encoder_pendulum_pos_drift_std_per_s = math.radians(0.0)
    encoder_pendulum_vel_drift_std_per_s = math.radians(0.0)

    # External wrench pushes.
    enable_external_wrench_push = True
    push_body_name = "base"
    push_is_global = True
    push_start_delay_s = 2.0
    push_interval_s_min = 4.0
    push_interval_s_max = 8.0
    push_duration_s_min = 0.08
    push_duration_s_max = 0.15
    # Level 1 is force-free; later presets progressively reactivate pushes.
    push_force_x_range = (0.0, 0.0)
    push_force_y_range = (0.0, 0.0)
    push_force_z_range = (0.0, 0.0)
    push_torque_x_range = (0.0, 0.0)
    push_torque_y_range = (0.0, 0.0)
    push_torque_z_range = (0.0, 0.0)

    # Simulation and scene.
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
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

        def _validate_probability(name: str, value: float) -> None:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]. Got {value}.")

        def _validate_ordered_range(name: str, value: tuple[float, float]) -> None:
            if len(value) != 2 or value[0] > value[1]:
                raise ValueError(f"{name} must be an ordered (min, max) pair. Got {value}.")

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
        if self.action_space != POLICY_ACTION_DIM:
            raise ValueError(f"Policy contract v3 requires {POLICY_ACTION_DIM} actions. Got {self.action_space}.")
        if self.difficulty_override not in (-1, 1, 2, 3, 4, 5):
            raise ValueError(f"difficulty_override must be -1 or a level in [1, 5]. Got {self.difficulty_override}.")
        if self.goal_profile_override not in ("mixed", "stand", "short", "walk"):
            raise ValueError(
                "goal_profile_override must be one of mixed, stand, short, or walk. "
                f"Got {self.goal_profile_override!r}."
            )

        if len(self.goal_distance_mixture) != 3:
            raise ValueError("goal_distance_mixture must contain (stand, short, walk) probabilities.")
        for index, probability in enumerate(self.goal_distance_mixture):
            _validate_probability(f"goal_distance_mixture[{index}]", probability)
        if not math.isclose(sum(self.goal_distance_mixture), 1.0, abs_tol=1.0e-6):
            raise ValueError(f"goal_distance_mixture must sum to 1. Got {self.goal_distance_mixture}.")
        _validate_ordered_range("goal_stand_distance_range", self.goal_stand_distance_range)
        _validate_ordered_range("goal_short_distance_range", self.goal_short_distance_range)
        _validate_ordered_range("goal_walk_distance_range", self.goal_walk_distance_range)
        if len(self.goal_chain_distance_mixture) != 3:
            raise ValueError("goal_chain_distance_mixture must contain (stand, short, walk) probabilities.")
        for index, probability in enumerate(self.goal_chain_distance_mixture):
            _validate_probability(f"goal_chain_distance_mixture[{index}]", probability)
        if not math.isclose(sum(self.goal_chain_distance_mixture), 1.0, abs_tol=1.0e-6):
            raise ValueError(
                f"goal_chain_distance_mixture must sum to 1. Got {self.goal_chain_distance_mixture}."
            )
        if self.goal_chain_post_arrival_hold_s < 0.0:
            raise ValueError("goal_chain_post_arrival_hold_s must be non-negative.")
        _validate_ordered_range("goal_stand_yaw_offset_range", self.goal_stand_yaw_offset_range)

        dr_fractions = (self.dr_nominal_fraction, self.dr_uniform_fraction, self.dr_max_fraction)
        for name, probability in zip(("dr_nominal_fraction", "dr_uniform_fraction", "dr_max_fraction"), dr_fractions):
            _validate_probability(name, probability)
        if not math.isclose(sum(dr_fractions), 1.0, abs_tol=1.0e-6):
            raise ValueError(f"DR reset fractions must sum to 1. Got {dr_fractions}.")
        if not 0.0 <= self.domain_randomization_scale <= 1.0:
            raise ValueError(f"domain_randomization_scale must be in [0, 1]. Got {self.domain_randomization_scale}.")
        if self.evaluation_dr_scale_multiplier < 0.0 or self.evaluation_push_scale_multiplier < 0.0:
            raise ValueError("Evaluation randomization/push multipliers must be non-negative.")

        if not 0.0 <= self.command_heading_blend_near_m < self.command_heading_blend_far_m:
            raise ValueError("Command heading blend distances must satisfy 0 <= near < far.")
        if not 0.0 <= self.stand_enter_distance_m < self.stand_exit_distance_m:
            raise ValueError("Stand distance hysteresis must satisfy 0 <= enter < exit.")
        if not 0.0 <= self.stand_enter_yaw_rad < self.stand_exit_yaw_rad:
            raise ValueError("Stand yaw hysteresis must satisfy 0 <= enter < exit.")
        if self.goal_stand_distance_range[0] < 0.0 or self.goal_stand_distance_range[1] > self.stand_enter_distance_m:
            raise ValueError("Planted goal distances must stay inside stand_enter_distance_m.")
        if (
            self.goal_stand_yaw_offset_range[0] < -self.stand_enter_yaw_rad
            or self.goal_stand_yaw_offset_range[1] > self.stand_enter_yaw_rad
        ):
            raise ValueError("Planted goal yaw offsets must stay inside stand_enter_yaw_rad.")
        if self.command_max_forward_speed_m_s <= 0.0 or self.command_max_yaw_rate_rad_s <= 0.0:
            raise ValueError("Command speed limits must be positive.")
        if self.stand_correction_position_gain_s < 0.0 or self.stand_correction_yaw_gain_s < 0.0:
            raise ValueError("Stand correction gains must be non-negative.")
        if self.stand_correction_max_linear_m_s <= 0.0 or self.stand_correction_max_yaw_rate_rad_s <= 0.0:
            raise ValueError("Stand correction speed limits must be positive.")
        if self.absolute_position_divergence_m <= 0.0:
            raise ValueError("absolute_position_divergence_m must be positive.")
        if self.relative_position_divergence_margin_m <= 0.0:
            raise ValueError("relative_position_divergence_margin_m must be positive.")
        for name in (
            "absolute_position_divergence_duration_s",
            "relative_position_divergence_duration_s",
            "goal_watchdog_initial_window_s",
            "goal_watchdog_progress_window_s",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.goal_watchdog_exempt_distance_m < self.stand_exit_distance_m:
            raise ValueError("goal_watchdog_exempt_distance_m must be at least stand_exit_distance_m.")
        if self.goal_watchdog_push_cooldown_s < 0.0:
            raise ValueError("goal_watchdog_push_cooldown_s must be non-negative.")
        if self.progress_reward_clip_m <= 0.0:
            raise ValueError("progress_reward_clip_m must be positive.")
        if self.command_lin_vel_cost_normalizer_m_s <= 0.0:
            raise ValueError("command_lin_vel_cost_normalizer_m_s must be positive.")
        if self.command_yaw_rate_cost_normalizer_rad_s <= 0.0:
            raise ValueError("command_yaw_rate_cost_normalizer_rad_s must be positive.")

        _validate_ordered_range("pendulum_mass_scale_range", self.pendulum_mass_scale_range)
        if self.pendulum_mass_scale_range[0] <= 0.0:
            raise ValueError("pendulum_mass_scale_range values must be positive.")
        _validate_ordered_range(
            "pendulum_effective_length_offset_range_m",
            self.pendulum_effective_length_offset_range_m,
        )

        _validate_ordered_range("command_lpf_cutoff_range_hz", self.command_lpf_cutoff_range_hz)
        if not self.command_lpf_cutoff_range_hz[0] <= self.command_lpf_cutoff_hz <= self.command_lpf_cutoff_range_hz[1]:
            raise ValueError("command_lpf_cutoff_hz must be inside command_lpf_cutoff_range_hz.")
        if self.action_clip <= 0.0 or self.joint_target_slew_rate_rad_s <= 0.0 or self.joint_limit_margin_rad < 0.0:
            raise ValueError("Action clip/slew must be positive and the joint-limit margin must be non-negative.")

        if not 0.0 < self.gait_duty_factor_min_speed < 1.0 or not 0.0 < self.gait_duty_factor_max_speed < 1.0:
            raise ValueError("Gait duty factors must be in (0, 1).")
        if not 0.0 < self.gait_frequency_min_hz <= self.gait_frequency_max_hz:
            raise ValueError("Gait frequencies must satisfy 0 < min <= max.")
        if not 0.0 <= self.style_pendulum_angle_full_rad < self.style_pendulum_angle_zero_rad:
            raise ValueError("Pendulum style angle gates must satisfy 0 <= full < zero.")
        if not 0.0 <= self.style_pendulum_speed_full_rad_s < self.style_pendulum_speed_zero_rad_s:
            raise ValueError("Pendulum style speed gates must satisfy 0 <= full < zero.")

        _validate_probability("episode_mirror_probability", self.episode_mirror_probability)
        _validate_probability("pendulum_recovery_reset_fraction", self.pendulum_recovery_reset_fraction)
        _validate_ordered_range("pendulum_recovery_angle_range", self.pendulum_recovery_angle_range)
        if len(self.pendulum_hinge_offset_b) != 3:
            raise ValueError("pendulum_hinge_offset_b must contain three base-frame coordinates.")

        # Keep pendulum-free evaluation available by swapping to the base Go2 USD.
        if not self.use_pendulum:
            self.robot_cfg = self.robot_cfg.replace(
                spawn=self.robot_cfg.spawn.replace(usd_path=GO2_USD_PATH),
            )
            self.pendulum_contact_sensor = self.pendulum_contact_sensor.replace(
                prim_path="/World/envs/env_.*/Robot/base"
            )

        # Keep observation dims fixed so policies are compatible across pendulum modes.
        self.observation_space = 48 + 4 + 2 * len(self.pendulum_joint_names)
        if self.observation_space != POLICY_OBSERVATION_DIM:
            raise ValueError(
                f"Policy contract v3 requires {POLICY_OBSERVATION_DIM} observations. Got {self.observation_space}."
            )
        # Critic gets same structure as actor, just ground-truth (no noise).
        self.state_space = self.observation_space

        control_period_s = self.decimation * self.sim.dt
        if not math.isclose(control_period_s, 1.0 / self.policy_control_hz, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"decimation * sim.dt ({control_period_s}) must match the v3 policy period "
                f"({1.0 / self.policy_control_hz})."
            )

        # Increase GPU rigid patch buffer to avoid PhysX patch overflow.
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
