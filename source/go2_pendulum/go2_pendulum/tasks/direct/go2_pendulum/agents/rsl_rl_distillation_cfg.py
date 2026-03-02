# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class Go2PendulumDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 50
    experiment_name = "go2_pendulum_distill"
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}

    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[256, 256, 64],
        teacher_hidden_dims=[256, 256, 64],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=16,
    )
