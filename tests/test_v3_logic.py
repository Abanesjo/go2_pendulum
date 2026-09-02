# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure contract-v3 logic tests that do not launch Isaac Sim.

The environment module imports Isaac APIs at module import time.  These tests
extract the small tensor-only functions under test from its AST, so they can run
in a normal ``unittest`` process while still exercising the checked-in source.
"""

from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / (
    "source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/go2_pendulum_env.py"
)
TRAIN_PATH = REPO_ROOT / "scripts/rsl_rl/train.py"


class _MathUtils:
    @staticmethod
    def wrap_to_pi(value: torch.Tensor) -> torch.Tensor:
        return torch.remainder(value + math.pi, 2.0 * math.pi) - math.pi


def _environment_tree() -> ast.Module:
    return ast.parse(ENV_PATH.read_text(encoding="utf-8"), filename=str(ENV_PATH))


def _environment_class(tree: ast.Module) -> ast.ClassDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Go2PendulumEnv"
    )


def _load_environment_symbols(
    *, top_level: tuple[str, ...] = (), methods: tuple[str, ...] = ()
) -> dict[str, object]:
    tree = _environment_tree()
    class_node = _environment_class(tree)
    selected: list[ast.stmt] = []
    selected.extend(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in top_level
    )
    selected.extend(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    )
    expected = len(top_level) + len(methods)
    if len(selected) != expected:
        raise AssertionError(f"Expected {expected} extracted symbols, found {len(selected)}.")
    namespace = {"math": math, "math_utils": _MathUtils, "torch": torch}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(ENV_PATH), "exec"), namespace)
    return namespace


def _class_value(name: str):
    tree = _environment_tree()
    class_node = _environment_class(tree)
    assignment = next(
        node
        for node in class_node.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    return eval(
        compile(ast.Expression(assignment.value), str(ENV_PATH), "eval"),
        {"dict": dict, "math": math},
    )


class V3FormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        symbols = _load_environment_symbols(
            top_level=("_interpolate_curriculum_value", "_required_goal_progress"),
            methods=(
                "_compute_goal_error_terms",
                "_compute_navigation_command",
                "_apply_curriculum_progress",
                "_apply_difficulty_preset",
                "_update_curriculum",
            ),
        )
        cls.interpolate = staticmethod(symbols["_interpolate_curriculum_value"])
        cls.required_progress = staticmethod(symbols["_required_goal_progress"])
        cls.compute_goal_error = staticmethod(symbols["_compute_goal_error_terms"])
        cls.compute_command = staticmethod(symbols["_compute_navigation_command"])
        cls.apply_progress = staticmethod(symbols["_apply_curriculum_progress"])
        cls.apply_preset = staticmethod(symbols["_apply_difficulty_preset"])
        cls.update_curriculum = staticmethod(symbols["_update_curriculum"])

    def test_required_progress_piecewise_formula(self) -> None:
        checkpoint = torch.tensor([0.08, 0.09, 0.10, 0.20, 0.35, 0.50])
        expected = torch.tensor([0.00, 0.01, 0.02, 0.03, 0.06, 0.08])
        torch.testing.assert_close(self.required_progress(checkpoint, 0.08), expected)

    def test_forward_command_is_not_attenuated_by_near_goal_yaw_blend(self) -> None:
        dummy = SimpleNamespace(
            num_envs=4,
            device="cpu",
            cfg=SimpleNamespace(),
            target_state=torch.tensor(
                [[0.10, 0.0, 0.0], [0.50, 0.0, 0.0], [0.0, 0.50, 0.0], [-0.50, 0.0, 0.0]]
            ),
            _stand_mode=torch.zeros(4, dtype=torch.bool),
        )
        dummy._compute_goal_error_terms = self.compute_goal_error.__get__(dummy)
        _, command, move_gate, stand_gate = self.compute_command(
            dummy, torch.zeros(4, 2), torch.zeros(4), False
        )
        torch.testing.assert_close(command[:2, 0], torch.tensor([0.15, 0.60]))
        self.assertLess(abs(float(command[2, 0])), 1.0e-6)
        self.assertEqual(float(command[3, 0]), 0.0)
        torch.testing.assert_close(move_gate[:2], torch.tensor([0.25, 1.0]))
        torch.testing.assert_close(stand_gate, 1.0 - move_gate)

    def test_stand_latch_keeps_small_pose_correction_but_closes_move_gate(self) -> None:
        dummy = SimpleNamespace(
            num_envs=1,
            device="cpu",
            cfg=SimpleNamespace(),
            target_state=torch.tensor([[0.04, 0.0, 0.0]]),
            _stand_mode=torch.zeros(1, dtype=torch.bool),
        )
        dummy._compute_goal_error_terms = self.compute_goal_error.__get__(dummy)
        _, command, move_gate, stand_gate = self.compute_command(
            dummy, torch.zeros(1, 2), torch.zeros(1), True
        )
        self.assertTrue(dummy._stand_mode.item())
        torch.testing.assert_close(command, torch.tensor([[0.04, 0.0, 0.0]]))
        self.assertEqual(float(move_gate.item()), 0.0)
        self.assertEqual(float(stand_gate.item()), 1.0)

    def test_curriculum_exact_anchors_and_midpoint(self) -> None:
        dummy = SimpleNamespace(
            cfg=SimpleNamespace(),
            _CURRICULUM_ANCHOR_PROGRESS=_class_value("_CURRICULUM_ANCHOR_PROGRESS"),
            _DIFFICULTY_PRESETS=_class_value("_DIFFICULTY_PRESETS"),
        )
        dummy._validate_range = lambda _name, _value: None
        dummy._apply_curriculum_progress = self.apply_progress.__get__(dummy)
        for level, progress in enumerate(dummy._CURRICULUM_ANCHOR_PROGRESS, start=1):
            self.apply_progress(dummy, progress)
            preset = dummy._DIFFICULTY_PRESETS[level]
            self.assertEqual(dummy.cfg.goal_distance_mixture, preset["goal_distance_mixture"])
            self.assertEqual(dummy.cfg.goal_chain_distance_mixture, preset["goal_chain_distance_mixture"])
            self.assertAlmostEqual(dummy.cfg.domain_randomization_scale, preset["domain_randomization_scale"])
            self.assertEqual(dummy._current_difficulty_level, level)

        self.apply_progress(dummy, 0.325)
        for actual, expected in zip(dummy.cfg.goal_distance_mixture, (0.30, 0.375, 0.325)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(dummy.cfg.goal_chain_distance_mixture, (0.10, 0.35, 0.55)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(dummy.cfg.domain_randomization_scale, 0.375)
        self.assertEqual(dummy.cfg.push_force_x_range, (-5.0, 5.0))

    def test_curriculum_final_action_reaches_one_and_resume_uses_full_horizon(self) -> None:
        recorded: list[float] = []
        dummy = SimpleNamespace(
            cfg=SimpleNamespace(
                enable_curriculum=True,
                curriculum_total_steps=100,
                curriculum_start_step=40,
            ),
            common_step_counter=59,
            _apply_curriculum_progress=recorded.append,
        )
        self.update_curriculum(dummy)
        self.assertEqual(recorded, [1.0])

        tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"), filename=str(TRAIN_PATH))
        horizon_assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "curriculum_total_steps"
                for target in node.targets
            )
        ]
        self.assertEqual(len(horizon_assignments), 1)
        value = horizon_assignments[0].value
        self.assertIsInstance(value, ast.BinOp)
        self.assertIsInstance(value.op, ast.Add)
        self.assertEqual({value.left.id, value.right.id}, {"curriculum_start_step", "curriculum_run_steps"})


class _WatchdogDummy:
    def _seconds_to_steps(self, seconds: float) -> int:
        return max(1, math.ceil(seconds / self.step_dt))

    def _compute_base_tilt_rad(self) -> torch.Tensor:
        return torch.zeros(self.num_envs)


class V3WatchdogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        symbols = _load_environment_symbols(
            top_level=("_required_goal_progress",), methods=("_get_dones",)
        )
        _WatchdogDummy._get_dones = symbols["_get_dones"]

    @staticmethod
    def _make_dummy(distance: float = 0.50, goal_class: int = 1) -> _WatchdogDummy:
        dummy = _WatchdogDummy()
        dummy.num_envs = 1
        dummy.device = "cpu"
        dummy.step_dt = 0.02
        dummy.max_episode_length = 10_000
        dummy.episode_length_buf = torch.zeros(1, dtype=torch.long)
        dummy._steps_since_reset = torch.zeros(1, dtype=torch.long)
        dummy.cfg = SimpleNamespace(
            base_contact_grace_s=0.5,
            termination_grace_s=0.5,
            pendulum_termination_grace_s=0.5,
            base_height_min=0.28,
            base_height_terminate_duration_s=0.25,
            base_tilt_terminate_angle_rad=math.radians(60.0),
            use_pendulum=False,
            pendulum_contact_force_threshold=1.0,
            pendulum_terminate_angle_rad=math.radians(20.0),
            pendulum_terminate_duration_s=0.25,
            absolute_position_divergence_m=2.5,
            absolute_position_divergence_duration_s=0.25,
            relative_position_divergence_margin_m=0.35,
            relative_position_divergence_duration_s=0.50,
            goal_watchdog_exempt_distance_m=0.08,
            goal_watchdog_initial_window_s=4.0,
            goal_watchdog_progress_window_s=3.0,
        )
        dummy._contact_sensor = SimpleNamespace(
            data=SimpleNamespace(net_forces_w_history=torch.zeros(1, 3, 1, 3))
        )
        dummy._pendulum_contact_sensor = None
        dummy._base_id = [0]
        dummy.robot = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[0.0, 0.0, 0.33]]),
                projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
                joint_pos=torch.zeros(1, 0),
            )
        )
        dummy._pendulum_dof_ids = torch.empty(0, dtype=torch.long)
        dummy._terrain = SimpleNamespace(env_origins=torch.zeros(1, 3), terrain_origins=None)
        dummy.scene = SimpleNamespace(env_origins=torch.zeros(1, 3))
        dummy.target_state = torch.tensor([[distance, 0.0, 0.0]])
        dummy._absolute_position_failure_steps = None
        dummy._base_height_failure_steps = None
        dummy._pendulum_angle_failure_steps = None
        dummy._current_goal_class = torch.tensor([goal_class])
        dummy._goal_initial_distance = torch.tensor([distance])
        dummy._goal_checkpoint_distance = torch.tensor([distance])
        dummy._goal_best_distance = torch.tensor([distance])
        dummy._goal_watchdog_elapsed_steps = torch.zeros(1, dtype=torch.long)
        dummy._goal_watchdog_first_window = torch.ones(1, dtype=torch.bool)
        dummy._goal_watchdog_was_exempt = torch.tensor([distance <= 0.08])
        dummy._goal_relative_divergence_steps = torch.zeros(1, dtype=torch.long)
        dummy._push_watchdog_pause_until_step = torch.zeros(1, dtype=torch.long)
        return dummy

    @staticmethod
    def _step(dummy: _WatchdogDummy) -> bool:
        return bool(dummy._get_dones()[0].item())

    def test_stationary_goal_fails_on_exact_initial_tick_200(self) -> None:
        dummy = self._make_dummy()
        self.assertFalse(any(self._step(dummy) for _ in range(199)))
        self.assertTrue(self._step(dummy))
        self.assertTrue(dummy._goal_no_progress_terminated.item())

    def test_passed_initial_window_then_fails_on_subsequent_tick_150(self) -> None:
        dummy = self._make_dummy()
        for index in range(200):
            dummy.robot.data.root_pos_w[0, 0] = 0.08 * (index + 1) / 200
            self.assertFalse(self._step(dummy))
        self.assertFalse(dummy._goal_watchdog_first_window.item())
        self.assertEqual(dummy._goal_watchdog_elapsed_steps.item(), 0)
        self.assertFalse(any(self._step(dummy) for _ in range(149)))
        self.assertTrue(self._step(dummy))

    def test_planted_goal_is_watchdog_and_relative_divergence_exempt(self) -> None:
        dummy = self._make_dummy(goal_class=0)
        self.assertFalse(any(self._step(dummy) for _ in range(250)))

    def test_exempt_region_exit_starts_fresh_150_tick_window(self) -> None:
        dummy = self._make_dummy(distance=0.07)
        self.assertFalse(self._step(dummy))
        dummy.target_state[0, 0] = 0.09
        dummy._goal_initial_distance[0] = 0.09
        self.assertFalse(self._step(dummy))
        self.assertEqual(dummy._goal_watchdog_elapsed_steps.item(), 0)
        self.assertFalse(dummy._goal_watchdog_first_window.item())
        self.assertFalse(any(self._step(dummy) for _ in range(149)))
        self.assertTrue(self._step(dummy))

    def test_relative_divergence_fails_on_exact_tick_25(self) -> None:
        dummy = self._make_dummy(distance=0.36)
        dummy._goal_initial_distance[0] = 0.0
        self.assertFalse(any(self._step(dummy) for _ in range(24)))
        self.assertTrue(self._step(dummy))
        self.assertTrue(dummy._relative_position_terminated.item())

    def test_push_pause_freezes_partial_watchdog_window(self) -> None:
        dummy = self._make_dummy()
        self.assertFalse(any(self._step(dummy) for _ in range(199)))
        dummy._push_watchdog_pause_until_step[:] = dummy._steps_since_reset + 25
        self.assertFalse(any(self._step(dummy) for _ in range(25)))
        self.assertEqual(dummy._goal_watchdog_elapsed_steps.item(), 199)
        self.assertTrue(self._step(dummy))


class V3StaticStructureTests(unittest.TestCase):
    def test_reward_accumulator_keys_exactly_match_reward_dictionary(self) -> None:
        tree = _environment_tree()
        class_node = _environment_class(tree)
        init_method = next(
            node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        reward_method = next(
            node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "_get_rewards"
        )
        sum_keys = next(
            ast.literal_eval(node.value)
            for node in ast.walk(init_method)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "episode_sum_keys" for target in node.targets)
        )
        reward_keys = next(
            [ast.literal_eval(key) for key in node.value.keys]
            for node in ast.walk(reward_method)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "rewards" for target in node.targets)
        )
        reward_keys.append("termination_penalty")
        self.assertEqual(len(sum_keys), len(set(sum_keys)))
        self.assertEqual(set(sum_keys), set(reward_keys))


if __name__ == "__main__":
    unittest.main()
