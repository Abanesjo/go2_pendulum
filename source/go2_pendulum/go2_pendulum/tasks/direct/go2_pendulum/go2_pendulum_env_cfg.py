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
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

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
    },
)


@configclass
class Go2PendulumEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 0.25
    action_space = 12
    observation_space = 241
    state_space = 0
    debug_vis = True
    use_height_scan = True
    enable_height_scanner = True
    height_scan_debug_vis = False
    return_teacher_obs = False
    use_pendulum = True
    tracking_mode = False

    # gait shaping
    raibert_heuristic_reward_scale = 0.0
    feet_clearance_reward_scale = 0.0
    tracking_contacts_shaped_force_reward_scale = 0.0

    # early stopping
    base_contact_grace_s = 0.0
    termination_penalty = -100.0
    pendulum_contact_force_threshold = 1.0

    # reward scales
    lin_vel_reward_scale = 1.5
    yaw_rate_reward_scale = 0.75
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.01
    undesired_contact_reward_scale = -1.0
    dof_torques_reward_scale = -0.0002
    dof_accel_reward_scale = -2.5e-7
    orient_reward_scale = -1.0
    lin_vel_z_reward_scale = -2.0
    dof_vel_reward_scale = -0.0001
    ang_vel_xy_reward_scale = -0.05
    pendulum_upright_reward_scale = 1.0
    pendulum_vel_reward_scale = 0.5

    # command generation
    yaw_kp = 1.0
    max_yaw_rate = 1.0
    goal_randomization_range = 3.0
    goal_randomization_angle = math.pi
    position_tolerance = 0.1

    # pendulum setup
    pendulum_joint_names = ["pendulum_joint1", "pendulum_joint2"]
    pendulum_angle_min = -9.0 * math.pi / 180.0
    pendulum_angle_max = 9.0 * math.pi / 180.0
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
        max_init_terrain_level=5,
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

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 0.0, 0.0)
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
        if not self.use_pendulum:
            self.robot_cfg = self.robot_cfg.replace(
                spawn=self.robot_cfg.spawn.replace(usd_path=GO2_USD_PATH),
            )
            self.pendulum_upright_reward_scale = 0.0
            self.pendulum_vel_reward_scale = 0.0
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
