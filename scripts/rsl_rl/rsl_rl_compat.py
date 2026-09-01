# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Narrow compatibility handling for Isaac Lab adapter fields across RSL-RL releases."""

import inspect

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent


def _remove_known_unsupported_keys(
    section_cfg: dict,
    constructor: type,
    candidate_keys: tuple[str, ...],
) -> list[str]:
    """Remove only allowlisted keys that are not explicit constructor parameters."""
    parameters = inspect.signature(constructor.__init__).parameters
    removed = [key for key in candidate_keys if key in section_cfg and key not in parameters]
    for key in removed:
        section_cfg.pop(key)
    return removed


def sanitize_rsl_rl_config(runner_cfg: dict) -> dict:
    """Adapt known Isaac Lab-only fields without hiding unrelated configuration errors."""
    algorithm_cfg = runner_cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict) and algorithm_cfg.get("class_name") == "PPO":
        removed = _remove_known_unsupported_keys(
            algorithm_cfg,
            PPO,
            ("optimizer", "share_cnn_encoders"),
        )
        if removed:
            print(f"[INFO] Removed unsupported RSL-RL PPO adapter keys: {', '.join(removed)}.")

    policy_cfg = runner_cfg.get("policy")
    if isinstance(policy_cfg, dict):
        policy_classes = {
            "ActorCritic": ActorCritic,
            "ActorCriticRecurrent": ActorCriticRecurrent,
        }
        policy_class = policy_classes.get(policy_cfg.get("class_name"))
        if policy_class is not None:
            removed = _remove_known_unsupported_keys(
                policy_cfg,
                policy_class,
                ("state_dependent_std",),
            )
            if removed:
                print(
                    f"[INFO] Removed unsupported RSL-RL {policy_class.__name__} adapter key: "
                    "state_dependent_std."
                )

    return runner_cfg
