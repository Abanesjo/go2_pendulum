# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .go2_pendulum_env_cfg import Go2PendulumEnvCfg


@configclass
class Go2PendulumFlatEnvCfg(Go2PendulumEnvCfg):
    """Flat terrain variant without height scan observations."""

    rough_terrain = False
    use_height_scan = False
    enable_height_scanner = False
    height_scan_debug_vis = False

    def __post_init__(self):
        super().__post_init__()
        # Ensure height scan stays disabled even if base defaults change.
        self.use_height_scan = False
        self.enable_height_scanner = False
        self.height_scan_debug_vis = False
        # Update observation space to exclude height scan.
        self.observation_space = self._compute_policy_obs_size()

    def _compute_policy_obs_size(self) -> int:
        # Components mirrored from Go2PendulumEnv._get_observations.
        base = 3 + 3 + 3 + 3  # lin vel, ang vel, gravity, state error
        base += int(self.action_space) * 3  # joint pos, joint vel, actions
        base += 2 * len(self.pendulum_joint_names)  # pendulum pos/vel (always included)
        base += 4  # foot contacts
        return base
