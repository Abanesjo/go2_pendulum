# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import os

from numpy import True_

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
            damping=0.5,
            friction=0.0,
        ),
    },
)


@configclass
class Go2PendulumEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 12
    # - spaces definition
    action_scale = 0.25
    action_space = 12
    observation_space = 48 + 4 + 4
    state_space = 0
    debug_vis = True
    use_pendulum = True
    track_goal = False

    # gait shaping
    feet_clearance_reward_scale = -2.0
    tracking_contacts_shaped_force_reward_scale = 0.5

    # torque regularization
    torque_reward_scale = -0.0002

    # early stopping / termination
    termination_grace_s = 2.0
    base_contact_grace_s = 0.5
    base_height_min = 0.25

    base_height_target = 0.35
    base_height_reward_sigma = 0.06
    
    termination_penalty = -100.0
    pendulum_contact_force_threshold = 1.0

    # 20 reward -> 0.4
    position_reward_scale = 0.6
    position_reward_sigma = 0.6
    progress_reward_scale = 100.0

    yaw_alignment_reward_scale = 0.3
    yaw_alignment_reward_sigma = 0.2

    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.01
    orient_reward_scale = 0.1
    lin_vel_z_reward_scale = -2.0
    dof_vel_reward_scale = -0.003
    dof_acc_reward_scale = -2.5e-7
    ang_vel_xy_reward_scale = -0.1
    undesired_contact_reward_scale = -1.0
    base_height_reward_scale = 0.1

    pendulum_upright_reward_scale = 0.6
    pendulum_upright_reward_sigma = math.radians(10)

    pendulum_vel_reward_scale = -2.0
    pendulum_vel_reward_sigma = 0.05  # unused with squared-velocity penalty
    
    balanced_movement_reward_scale = 0.2

    # target randomization
    goal_randomization_dist_min = 0.6
    goal_randomization_dist_max = 0.8
    goal_randomization_angle_min = math.radians(0)
    goal_randomization_angle_max = math.radians(360)
    goal_yaw_randomization_min = math.radians(0)
    goal_yaw_randomization_max = math.radians(360)
    position_tolerance = 0.1
    position_terminate_duration_s = 15.0

    # observation noise (applied to x/y/yaw target error terms only)
    observation_noise_scale = 1.0
    position_noise = 0.02  # meters
    orientation_noise = math.radians(1.0)  # radians

    # pendulum setup
    pendulum_joint_names = ["pendulum_joint1", "pendulum_joint2"]
    pendulum_angle_min = math.radians(0.0)
    pendulum_angle_max = math.radians(9.9)
    pendulum_terminate_angle_rad = math.radians(9.0)
    pendulum_terminate_duration_s = 2.0

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
