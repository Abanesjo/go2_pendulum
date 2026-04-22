# Go2 Pendulum Project

Isaac Lab extension for training a Unitree Go2 quadruped with a 2-DoF inverted pendulum. The task combines balance, locomotion, and target-reaching with an asymmetric actor-critic observation setup.

The environment is registered as `Template-Go2-Pendulum-Direct-v0`.

## What This Repo Contains

- A standalone Isaac Lab task under `source/go2_pendulum/.../go2_pendulum/`
- A direct RL environment split into config and implementation files
- RSL-RL and SKRL training/play entry points
- Utility scripts for env listing and zero/random-action smoke tests

## Current Task Behavior

- The robot starts from a fixed nominal standing pose.
- The target state is `target_state = [x_d, y_d, yaw_d]`.
- Target position bearing is sampled independently from desired final robot heading.
- Curriculum changes only task/reset difficulty over training.
- Observation noise and domain-randomization magnitudes stay at their configured max values from the start.
- The post-policy action path is intentionally simple:
  raw policy output -> optional action delay -> `action_scale`/default-joint offset mapping -> joint position target
- The observation "last action" term is the latest executed delayed command, not the newest raw model output.

## Repository Layout

- [go2_pendulum_env_cfg.py](/home/john/Documents/isaaclab_projects/go2_pendulum/source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/go2_pendulum_env_cfg.py): task config, reward scales, noise, domain randomization, simulation settings
- [go2_pendulum_env.py](/home/john/Documents/isaaclab_projects/go2_pendulum/source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/go2_pendulum_env.py): DirectRLEnv implementation, observations, rewards, dones, reset logic, curriculum
- [rsl_rl_ppo_cfg.py](/home/john/Documents/isaaclab_projects/go2_pendulum/source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/agents/rsl_rl_ppo_cfg.py): PPO runner config
- [__init__.py](/home/john/Documents/isaaclab_projects/go2_pendulum/source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/__init__.py): Gym registration for `Template-Go2-Pendulum-Direct-v0`

## Environment Setup

Activate the Isaac Lab environment:

```bash
source ~/environments/isaac/bin/activate
```

Install the extension in editable mode:

```bash
python -m pip install -e source/go2_pendulum
```

## Common Commands

Verify that the task is registered:

```bash
python scripts/list_envs.py
```

Train with RSL-RL:

```bash
python scripts/rsl_rl/train.py --task=Template-Go2-Pendulum-Direct-v0 --headless
```

Play a trained checkpoint:

```bash
python scripts/rsl_rl/play.py --task=Template-Go2-Pendulum-Direct-v0 --num_envs=16
```

Run quick smoke tests:

```bash
python scripts/zero_agent.py --task=Template-Go2-Pendulum-Direct-v0
python scripts/random_agent.py --task=Template-Go2-Pendulum-Direct-v0
```

SKRL entry points are also available:

```bash
python scripts/skrl/train.py --task=Template-Go2-Pendulum-Direct-v0 --headless
python scripts/skrl/play.py --task=Template-Go2-Pendulum-Direct-v0 --num_envs=16
```

## Key Config Knobs

Useful settings live in [go2_pendulum_env_cfg.py](/home/john/Documents/isaaclab_projects/go2_pendulum/source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/go2_pendulum_env_cfg.py).

- `goal_randomization_dist_min/max`: target distance sampling
- `goal_randomization_angle_min/max`: target position bearing in the environment frame
- `goal_yaw_randomization_min/max`: desired robot heading at the target
- `pendulum_angle_min/max`: initial pendulum reset angle sampling
- `action_scale`: mapping from policy output to desired joint-position offsets
- `enable_action_delay`, `action_delay_steps_min/max`: simulated control delay
- `body_lin_vel_noise`, `body_ang_vel_noise`, `orientation_noise`, `position_noise`: actor observation noise
- `enable_domain_randomization` and related mass/friction/motor/bias fields: sim-to-real randomization

## Observation And Action Reference

This section is the same deployment-facing information that now lives at the top of [go2_pendulum_env_cfg.py](/home/john/Documents/isaaclab_projects/go2_pendulum/source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/go2_pendulum_env_cfg.py).

### Observation Space

The actor and critic both use a 56-D observation vector with the same ordering:

- `0:3`: body linear velocity in base frame `[vx, vy, vz]`
- `3:6`: body angular velocity in base frame `[wx, wy, wz]`
- `6:9`: projected gravity in base frame `[gx, gy, gz]`
- `9:12`: target-state error in base frame `[target_x_error_body, target_y_error_body, wrap_to_pi(target_yaw - base_yaw)]`
- `12:24`: leg joint position offsets from the default pose in this exact joint order:
  `FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint, FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint, FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint`
- `24:36`: leg joint velocities in the same exact order:
  `FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint, FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint, FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint`
- `36:38`: pendulum joint positions:
  `pendulum_joint1, pendulum_joint2`
- `38:40`: pendulum joint velocities:
  `pendulum_joint1, pendulum_joint2`
- `40:52`: latest executed action command in the same exact leg-joint order:
  `FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint, FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint, FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint`
- `52:56`: gait clock inputs:
  `sin(phase_FL), sin(phase_FR), sin(phase_RL), sin(phase_RR)`

Important details:

- The actor observation is noisy and can be delayed.
- The critic observation is clean ground truth.
- The "last action" term is the latest executed delayed command, not the newest raw policy output.

### Action Space

The policy outputs a 12-D floating-point action ordered as:

`FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint, FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint, FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint`

These values are interpreted as joint-position offsets relative to the default standing pose, not absolute joint targets.

The environment converts the action to desired joint positions as:

```text
q_des = q_default + action_scale * action_executed
```

with:

- `action_scale = 0.25` rad per unit action
- `action_executed =` the delayed command after the action-delay model
- no downstream filter or clamp in the environment after delay

The current default joint-position offsets are:

```text
[
   0.1, -0.1,  0.1, -0.1,   # FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint
   0.8,  0.8,  1.0,  1.0,   # FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint
  -1.5, -1.5, -1.5, -1.5,   # FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint
]
```

Example:

```text
q_des_FR_thigh = 0.8 + 0.25 * action[FR_thigh_joint]
```

## Using A Trained Policy

At deployment time, the control loop is:

1. Build the 56-D observation in the exact order above.
2. Run the actor policy to get the 12-D joint-offset action.
3. Apply the same action-delay model used in deployment.
4. Convert the delayed action to desired joint positions with `q_des = q_default + 0.25 * action_executed`.
5. Send those desired joint positions to the robot controller in the same joint order.

If you are trying to match training as closely as possible, use the measured delayed action in the observation history slot and preserve the same joint ordering exactly.
