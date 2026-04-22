# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Lab extension that trains a Unitree Go2 quadruped with an inverted pendulum to balance and optionally track target points. Built as a standalone extension outside the core Isaac Lab repo. Uses asymmetric actor-critic with fixed-max observation/domain randomization and a curriculum for task difficulty progression.

**Task ID:** `Template-Go2-Pendulum-Direct-v0`

## Isaaclab environment
The isaaclab environment can be activated via
```
source ~/environments/isaac/bin/activate
```

## Common Commands

```bash
# Install the extension (editable mode, requires Isaac Lab Python env)
python -m pip install -e source/go2_pendulum

# Train (asymmetric actor-critic with fixed max noise + difficulty curriculum)
python scripts/rsl_rl/train.py --task=Template-Go2-Pendulum-Direct-v0 --headless

# Play/evaluate a trained checkpoint
python scripts/rsl_rl/play.py --task=Template-Go2-Pendulum-Direct-v0 --num_envs=16

# Verify env registration
python scripts/list_envs.py

# Quick smoke test with zero/random actions
python scripts/zero_agent.py --task=Template-Go2-Pendulum-Direct-v0
python scripts/random_agent.py --task=Template-Go2-Pendulum-Direct-v0
```

Training logs go to `logs/rsl_rl/go2_pendulum_direct/`. Hydra outputs go to `outputs/`.

## Architecture

### Two-file environment pattern
The environment is a single `DirectRLEnv` subclass split across two files:
- **`go2_pendulum_env_cfg.py`** — `Go2PendulumEnvCfg(DirectRLEnvCfg)`: all hyperparameters, reward scales, termination thresholds, articulation/sensor configs, noise magnitudes, domain randomization fields, and curriculum parameters.
- **`go2_pendulum_env.py`** — `Go2PendulumEnv(DirectRLEnv)`: implements the DirectRL interface methods (`_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`), plus `_update_curriculum` and `_DIFFICULTY_PRESETS`.

### Asymmetric actor-critic
`_get_observations()` returns `{"policy": actor_obs, "critic": critic_obs}`. Both are 56-dim tensors with the same structure (body vel, gravity, state error, joints, pendulum, actions, clock). The actor's observations have noise/bias/delay applied; the critic's are always ground-truth. `state_space` in the config is set equal to `observation_space`.

### Curriculum system
`_update_curriculum()` is called each step and uses `common_step_counter / curriculum_total_steps` as progress (0→1). `curriculum_total_steps` is computed in `train.py` from `max_iterations * num_steps_per_env`.
- **Observation noise / DR:** sensor noise, bias, mass/COM randomization, and motor-gain randomization are applied at their configured max magnitudes from the start of training.
- **Difficulty curriculum:** 5 levels applied over training progress in 20% increments. Levels change reset/task difficulty, termination thresholds, pendulum joint limits, and external push strength. Presets are in `_DIFFICULTY_PRESETS` dict.

### Agent config
`agents/rsl_rl_ppo_cfg.py` — PPO runner config with `obs_groups = {"policy": ["policy"], "critic": ["critic"]}`, actor/critic `[256,256,64]`, ELU, 1500 iterations.

### USD model
The Go2 + pendulum USD lives at `source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/go2_model/go2_pendulum.usd`. A pendulum-free variant (`go2.usd`) is selected when `use_pendulum=False`.

## Key Design Details

- **Action pipeline:** raw policy output → per-joint bounding → delay buffer → optional LPF → joint position targets. Actions are offsets from default joint positions scaled by `action_scale`.
- **Observation space:** 48 (base state + leg joints) + 4 (clock/gait) + 2×N_pendulum_joints = 56 dims. Both actor and critic use the same structure.
- **Domain randomization:** material friction, base mass/COM, motor gains, sensor bias+drift, external wrench pushes, observation delay — all configured as flat fields on the env cfg (disabled by default, ready to enable for sim-to-real).

## Code Style

- Line length: 120 (Black)
- Import sorting: isort with Black profile
- Pre-commit hooks: Black, flake8, isort, pyupgrade (py310+), codespell
- BSD-3-Clause license header required on all `.py` files
