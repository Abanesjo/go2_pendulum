# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.markers import VisualizationMarkersCfg

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
            damping=0.5,
            friction=0.0,
        ),
        # Keep pendulum joints passive with light damping (matches omniwheel defaults).
        "pendulum_acts": ImplicitActuatorCfg(
            joint_names_expr=["^pendulum_joint[12]$"], damping=0.0001, stiffness=None
        ),
    },
)


@configclass
class Go2PendulumEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    # Action mapping (per leg joint) derived from URDF joint limits.
    # Order matches leg_joint_names.
    leg_joint_names = [
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ]
    action_scale = [
        1.0472,
        2.53075,
        0.94247,
        1.0472,
        2.53075,
        0.94247,
        1.0472,
        2.53075,
        0.94247,
        1.0472,
        2.53075,
        0.94247,
    ]
    joint_offset = [
        0.0,
        0.95995,
        -1.78023,
        0.0,
        0.95995,
        -1.78023,
        0.0,
        2.00715,
        -1.78023,
        0.0,
        2.00715,
        -1.78023,
    ]
    action_space = len(leg_joint_names)
    observation_space = 243
    state_space = 0
    debug_vis = True
    use_height_scan = True
    enable_height_scanner = True
    height_scan_debug_vis = False
    return_teacher_obs = False
    use_pendulum = True
    rough_terrain = True

    # observation noise (uniform in [-scale, scale])
    state_position_noise = 0.01  # meters
    state_orientation_noise = 2.0 * math.pi / 180.0  # radians
    pendulum_position_noise = 1.0 * math.pi / 180.0  # radians
    pendulum_velocity_noise = 1.0 * math.pi / 180.0  # radians/sec

    # curriculum (independent of terrain curriculum)
    curriculum_enabled = True
    curriculum_levels = 5
    curriculum_steps_per_level = 250_000

    # early stopping
    base_contact_grace_s = 0.0
    pendulum_contact_force_threshold = 1.0
    tilt_terminate_angle_deg = 60.0
    pendulum_termination_start_level = 1

    # termination conditions
    position_tolerance = 0.8  # meters
    max_displacement = 5.0  # meters
    pendulum_failure_angle_deg = 8.0  # degrees
    pendulum_failure_timeout_s = 5.0  # seconds
    position_failure_timeout_s = 10.0  # seconds

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -500.0
    rew_scale_upright = 1.6
    rew_scale_position = 5.0
    rew_scale_yaw_alignment = 4.0
    rew_scale_pendulum_velocity = 5.0
    rew_scale_angular_velocity = 5.0
    rew_scale_balanced_movement = 2.0
    rew_scale_tilt = -2.0
    rew_scale_action_delta = -0.1
    # quadruped-specific reward terms (aligned with Unitree Go2 rough locomotion defaults)
    rew_scale_feet_air_time = 1.0
    rew_scale_dof_torques = -0.0002
    rew_scale_dof_acc = -2.5e-7
    rew_scale_undesired_contacts = -1.0

    # contact/air-time thresholds
    feet_air_time_threshold_s = 0.5
    feet_air_time_speed_threshold = 0.1
    foot_contact_force_threshold = 1.0
    undesired_contact_force_threshold = 1.0

    # goal generation
    goal_randomization_range = 3.0
    goal_randomization_range_min = 0.5
    goal_randomization_angle = math.pi
    position_tolerance = 0.01

    # pendulum setup
    pendulum_joint_names = ["pendulum_joint1", "pendulum_joint2"]
    pendulum_angle_min = 0.0 * math.pi / 180.0
    pendulum_angle_max = 8.0 * math.pi / 180.0
    pendulum_terminate_angle_rad = 60.0 * math.pi / 180.0

    # terrain scaling
    terrain_scale = 0.5

    # simulation
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
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
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
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetMarkers",
        markers={
            "target_arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.33, 0.33, 0.33),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
            "target_sphere": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.rough_terrain:
            self.terrain = self.terrain.replace(terrain_type="plane", terrain_generator=None)
        if not self.use_pendulum:
            self.robot_cfg = self.robot_cfg.replace(
                spawn=self.robot_cfg.spawn.replace(usd_path=GO2_USD_PATH),
            )
            actuators = dict(self.robot_cfg.actuators)
            actuators.pop("pendulum_acts", None)
            self.robot_cfg = self.robot_cfg.replace(actuators=actuators)
            self.rew_scale_upright = 0.0
            self.rew_scale_pendulum_velocity = 0.0
            self.rew_scale_balanced_movement = 0.0
            self.pendulum_contact_sensor = self.pendulum_contact_sensor.replace(
                prim_path="/World/envs/env_.*/Robot/base"
            )
        # Increase GPU rigid patch buffer to avoid PhysX patch overflow.
        self.sim.physx.gpu_max_rigid_patch_count = 2**18
        if self.terrain.terrain_generator is not None:
            terrain_gen = self.terrain.terrain_generator
            terrain_gen.curriculum = True
            scale = self.terrain_scale
            terrain_gen.size = tuple(dim * scale for dim in terrain_gen.size)
            terrain_gen.border_width *= scale
            terrain_gen.horizontal_scale *= scale
            terrain_gen.vertical_scale *= scale

            def _scale_range(value_range: tuple[float, float]) -> tuple[float, float]:
                return (value_range[0] * scale, value_range[1] * scale)

            sub_terrains = terrain_gen.sub_terrains
            if "pyramid_stairs" in sub_terrains:
                sub_terrains["pyramid_stairs"].step_height_range = _scale_range(
                    sub_terrains["pyramid_stairs"].step_height_range
                )
                sub_terrains["pyramid_stairs"].step_width *= scale
                sub_terrains["pyramid_stairs"].platform_width *= scale
                sub_terrains["pyramid_stairs"].border_width *= scale
            if "pyramid_stairs_inv" in sub_terrains:
                sub_terrains["pyramid_stairs_inv"].step_height_range = _scale_range(
                    sub_terrains["pyramid_stairs_inv"].step_height_range
                )
                sub_terrains["pyramid_stairs_inv"].step_width *= scale
                sub_terrains["pyramid_stairs_inv"].platform_width *= scale
                sub_terrains["pyramid_stairs_inv"].border_width *= scale
            if "boxes" in sub_terrains:
                sub_terrains["boxes"].grid_width *= scale
                sub_terrains["boxes"].grid_height_range = _scale_range(sub_terrains["boxes"].grid_height_range)
                sub_terrains["boxes"].platform_width *= scale
            if "random_rough" in sub_terrains:
                sub_terrains["random_rough"].noise_range = _scale_range(sub_terrains["random_rough"].noise_range)
                sub_terrains["random_rough"].noise_step *= scale
                sub_terrains["random_rough"].border_width *= scale
            if "hf_pyramid_slope" in sub_terrains:
                sub_terrains["hf_pyramid_slope"].platform_width *= scale
                sub_terrains["hf_pyramid_slope"].border_width *= scale
            if "hf_pyramid_slope_inv" in sub_terrains:
                sub_terrains["hf_pyramid_slope_inv"].platform_width *= scale
                sub_terrains["hf_pyramid_slope_inv"].border_width *= scale

            sub_terrains["boxes"].grid_height_range = (0.025 * scale, 0.1 * scale)
            sub_terrains["random_rough"].noise_range = (0.01 * scale, 0.06 * scale)
            sub_terrains["random_rough"].noise_step = 0.01 * scale
