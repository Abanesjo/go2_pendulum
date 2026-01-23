# Go2 Pendulum Project (Isaac Lab)

## Overview

This project trains a Unitree Go2 with an inverted pendulum to track target points on rough terrain.
It is set up as an Isaac Lab extension so you can iterate outside the core Isaac Lab repo.

**Key Features:**

- `Go2 + pendulum` USD at `source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/go2_pendulum/go2_pendulum.usd`.
- `Target tracking` Random XY target with fixed speed (0.25 m/s) and yaw alignment near the goal.
- `Rough terrain` Height scanner enabled with curriculum-driven terrain levels.
- `Pendulum alignment` Observations/rewards use `pendulum_ee` world alignment.
- `Curriculum assist` Optional PD assist on pendulum joints before a configurable terrain level.

**Task ID:** `Template-Go2-Pendulum-Direct-v0`

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/go2_pendulum

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task (RSL-RL):

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/rsl_rl/train.py --task=Template-Go2-Pendulum-Direct-v0
        ```

## Task configuration

The Go2 pendulum task config lives at:

`source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/go2_pendulum_env_cfg.py`

Useful knobs:

- `command_speed`: Fixed speed toward the target (m/s).
- `position_tolerance`: Distance threshold for yaw alignment at the goal (m).
- `pendulum_terminate_angle_rad`: Termination tilt angle (rad).
- `pendulum_assist_level_threshold`: Terrain curriculum level where pendulum assist turns off.
- `pendulum_assist_kp`, `pendulum_assist_kd`, `pendulum_assist_torque_limit`: Assist gains and limits.

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```
