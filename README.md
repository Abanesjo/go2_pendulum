# Go2 Pendulum Project

Isaac Lab extension for training a Unitree Go2 quadruped with a 2-DoF inverted pendulum. The task combines balance, locomotion, and target-reaching with an asymmetric actor-critic observation setup.

The environment is registered with two RSL-RL policy variants:

- `Template-Go2-Pendulum-Direct-v0`: feed-forward MLP policy
- `Template-Go2-Pendulum-GRU-Direct-v0`: recurrent GRU-MLP policy

## What This Repo Contains

- A standalone Isaac Lab task under `source/go2_pendulum/.../go2_pendulum/`
- A direct RL environment split into config and implementation files
- RSL-RL and SKRL training/play entry points
- Utility scripts for env listing and zero/random-action smoke tests

## Policy Contract v3

Policy contract v3 retains the 56-D actor input and 12-D action dimensions, but changes the navigation-command semantics and training objective. It is intentionally checkpoint-incompatible with both v1 and v2. Start new MLP and GRU runs; do not resume an older checkpoint.

The main v3 changes are:

- Forward speed is no longer attenuated by the near-goal yaw blend. The blend now only transitions the yaw controller from path heading to final heading.
- Locomotion uses normalized command-error costs, signed goal progress, residual-distance and elapsed-time costs, and a one-time settled-arrival bonus. A stationary policy no longer receives free positive command reward.
- Stand-only rewards use the binary stand latch, and positive air-time reward is suppressed unless the robot actually tracks the commanded motion.
- A no-progress watchdog terminates locomotion goals that stall, while separate relative- and absolute-divergence guards catch motion away from the target.
- A smoothly interpolated mixed curriculum includes walking goals from iteration zero and increases their fraction, bearing range, yaw range, pendulum difficulty, randomization, and pushes over training.
- The established sim-to-real observation, finite-difference mocap, gait-clock, planted-stance, action-delivery, and ONNX interfaces remain in place.

The Gym suffix and policy-contract version are independent. `Template-Go2-Pendulum-Direct-v0` and `Template-Go2-Pendulum-GRU-Direct-v0` retain `-v0` because that is the stable Gym registration/API identifier. Policy v3 is set by `POLICY_CONTRACT_VERSION` in the environment config, saved in `params/env.yaml`, exported in `policy_contract.yaml`, and separated operationally by the `_v3` RSL-RL experiment directories.

> **ROS 2 scope:** this repository trains and exports the policy. No files in `go2_pendulum_ros2` are modified here. The deployment section below is an implementation contract for a separate ROS 2 change.

## Repository Layout

- [go2_pendulum_env_cfg.py](source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/go2_pendulum_env_cfg.py): task config, policy contract, reward scales, noise, domain randomization, and simulation settings
- [go2_pendulum_env.py](source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/go2_pendulum_env.py): DirectRLEnv implementation, observations, action delivery, rewards, resets, gait, and curriculum
- [rsl_rl_ppo_cfg.py](source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/agents/rsl_rl_ppo_cfg.py): MLP and GRU PPO runner configs
- [__init__.py](source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/__init__.py): Gym registration for both task variants

## Environment Setup

Activate the Isaac Lab environment:

```bash
conda activate isaac
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

Train a fresh v3 MLP policy with RSL-RL:

```bash
python scripts/rsl_rl/train.py --task=Template-Go2-Pendulum-Direct-v0 --headless
```

Train a fresh v3 GRU-MLP policy with RSL-RL:

```bash
python scripts/rsl_rl/train.py --task=Template-Go2-Pendulum-GRU-Direct-v0 --headless
```

The default output roots are:

- MLP: `logs/rsl_rl/go2_pendulum_direct_v3/`
- GRU: `logs/rsl_rl/go2_pendulum_gru_direct_v3/`

MLP and GRU checkpoints are mutually incompatible. Policy v1 and v2 checkpoints are also incompatible with v3. Resume only from a v3 checkpoint with the same architecture and observation contract.

Play and export a trained MLP checkpoint:

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Go2-Pendulum-Direct-v0 \
  --checkpoint=/absolute/path/to/model_24999.pt \
  --num_envs=1 \
  --headless
```

Play and export a GRU checkpoint:

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Go2-Pendulum-GRU-Direct-v0 \
  --checkpoint=/absolute/path/to/model_24999.pt \
  --num_envs=1 \
  --headless
```

`play.py` defaults to a canonical, deterministic `nominal` profile at curriculum anchor 5 with the anchor's mixed reset distribution. Use `--goal-profile stand`, `short`, or `walk` to isolate one goal class, `--eval-profile randomized` for the in-distribution randomization envelope, or `--eval-profile stress` for held-out 1.5x DR and 1.25x push tests. `--difficulty 1` through `5` maps to the five curriculum anchors. Evaluation disables episode mirroring and training-only goal chaining, so each environment measures full-horizon convergence and holding on one fixed goal.

Playing, resuming, or exporting also verifies `params/env.yaml` beside the checkpoint. A missing or non-v3 `policy_contract_version` is rejected even when an older network has the same 56-input/12-output shape.

`play.py` writes the deployment artifacts beside the checkpoint under `exported/`:

- `policy.onnx`
- `policy.pt`
- `policy_contract.yaml`

The YAML contract records the version, exact observation layout, joint order, navigation/gait/action constants, whether normalization is embedded, and the MLP or GRU ONNX tensor signatures. Treat it as the machine-readable source of truth shipped with the model.

The commands above use the default nominal/mixed evaluation at anchor 5. Evaluate goal classes separately before deployment, for example:

```bash
python scripts/rsl_rl/play.py \
  --task=Template-Go2-Pendulum-Direct-v0 \
  --checkpoint=/absolute/path/to/model_24999.pt \
  --num_envs=64 \
  --headless \
  --eval-profile=randomized \
  --difficulty=5 \
  --goal-profile=walk
```

`--eval-profile` accepts `nominal`, `randomized`, or `stress`; `--goal-profile` accepts `mixed`, `stand`, `short`, or `walk`; and `--difficulty` accepts 1 through 5. Use nominal for deterministic regression, randomized for in-distribution robustness, and stress for held-out robustness. Export is performed by the same `play.py` invocation.

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

The v3 ONNX and `policy_contract.yaml` deployment path documented below is the RSL-RL `play.py` export path.

## V3 Training Behavior

The two task IDs use one environment and differ only in policy architecture. Important v3 behavior lives in [go2_pendulum_env_cfg.py](source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/go2_pendulum_env_cfg.py) and [go2_pendulum_env.py](source/go2_pendulum/go2_pendulum/tasks/direct/go2_pendulum/go2_pendulum_env.py).

- The rollout curriculum duration is derived from the final `max_iterations * num_steps_per_env` after CLI overrides. Difficulty values are linearly interpolated between progress anchors at 0.00, 0.20, 0.45, 0.70, and 1.00 instead of changing abruptly.
- Goal distance ranges remain fixed: stand is 0-0.05 m, short is 0.10-0.35 m, and walk is 0.35-1.50 m. Goal bearing is sampled relative to the current base yaw, initially within +/-45 degrees and eventually over the full circle.
- Training chains goals without resetting the robot or policy state after the settled-arrival dwell and planted hold. Recurrent state, action/filter history, gait phase, physics, and domain randomization remain continuous. `play.py` disables chaining for fixed-goal evaluation.
- Every locomotion goal starts a progress watchdog. It permits 4 s for initial progress, then requires further best-distance improvement every 3 s. At each checkpoint distance `rho_checkpoint`, the required improvement is `min(0.08, max(0.025, 0.20*(rho_checkpoint - 0.05)), rho_checkpoint - 0.08)` meters. The watchdog is disabled inside 0.08 m and paused during a scripted push plus 0.5 s afterward.
- The current 2.5 m limit remains an absolute arena-divergence guard. A separate guard terminates when distance stays more than 0.35 m above the goal's initial distance for 0.5 s. No-progress, relative-divergence, and absolute-divergence causes are logged separately.
- Locomotion command tracking is a zero-at-correct normalized cost rather than a positive Gaussian. Goal progress, remaining distance, elapsed locomotion time, and settled arrival directly determine whether walking is worthwhile. Stand settling, contact, foot-lift, and symmetry rewards apply only while the binary stand latch is active.
- Each reset still selects nominal, uniformly randomized, or maximum-randomization conditions. The default fractions are 20%, 70%, and 10%; robustness remains at 0.25 scale through the 0.20 curriculum anchor before increasing.
- Gait air time is consistent with the commanded frequency and duty factor. Contact tracking uses a 5 N threshold, 0.08 m swing clearance, and 0.02 m stance height.
- Mirrored episodes swap left/right observations and actions with the appropriate sign changes, discouraging a permanently preferred side while preserving valid turning.
- PPO clips normalized actions to `[-2, 2]`, enables actor/critic observation normalization, and uses separate v3 experiment directories.

The curriculum anchors are:

| Progress | Reset stand/short/walk | Chain stand/short/walk | Relative bearing | Goal yaw | Pendulum max | DR | Push XY |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 50/40/10 | 10/55/35 | +/-45 deg | +/-10 deg | 3 deg | 0.25 | off |
| 0.20 | 35/40/25 | 10/40/50 | +/-75 deg | +/-20 deg | 3 deg | 0.25 | off |
| 0.45 | 25/35/40 | 10/30/60 | +/-120 deg | +/-45 deg | 5 deg | 0.50 | +/-10 N |
| 0.70 | 20/25/55 | 10/20/70 | +/-180 deg | +/-75 deg | 7 deg | 0.75 | +/-20 N |
| 1.00 | 20/20/60 | 10/20/70 | +/-180 deg | +/-90 deg | 10 deg | 1.00 | +/-40 N |

The randomized ranges are placeholders until they can be calibrated from real ROS bags. Keep nominal samples in every curriculum level; simply maximizing every randomization usually degrades precise standing and goal convergence.

## Exact 56-D Actor Observation

Run the policy at exactly 50 Hz, with `dt = 0.02 s`. All tensors are `float32`. The canonical leg-joint order is:

```text
FL_hip_joint, FR_hip_joint, RL_hip_joint, RR_hip_joint,
FL_thigh_joint, FR_thigh_joint, RL_thigh_joint, RR_thigh_joint,
FL_calf_joint, FR_calf_joint, RL_calf_joint, RR_calf_joint
```

The actor input is:

| Slice | Size | Exact actor value |
|---|---:|---|
| `0:3` | 3 | Base linear velocity `[vx, vy, vz]` in the current base frame, from the 50 Hz motion-capture position finite difference |
| `3:6` | 3 | IMU gyroscope `[wx, wy, wz]` in the base frame |
| `6:9` | 3 | Projected gravity `R_imu^T [0, 0, -1]` in the base frame |
| `9:12` | 3 | Actor command: locomotion uses `[forward_velocity_command, 0.0, yaw_rate_command]`; latched stand uses bounded body-frame `[x_correction, y_correction, yaw_correction]` |
| `12:24` | 12 | Leg joint positions minus the default positions, in canonical order |
| `24:36` | 12 | Leg joint velocities in canonical order |
| `36:38` | 2 | Mocap-reconstructed `[pendulum_joint1, pendulum_joint2]` angles |
| `38:40` | 2 | Wrapped 50 Hz finite differences of those pendulum angles |
| `40:52` | 12 | Normalized joint target actually delivered by the complete action pipeline, in canonical order |
| `52:56` | 4 | `[sin(2*pi*phase), cos(2*pi*phase), move_gate, stand_gate]` |

The critic is also 56-D but is privileged and is never constructed on the robot. It uses clean simulator quantities and puts `[goal_dx_body, goal_dy_body, final_yaw_error]` in `9:12` instead of the actor command.

The default joint positions in canonical order are:

```text
[
   0.1, -0.1,  0.1, -0.1,
   0.8,  0.8,  1.0,  1.0,
  -1.5, -1.5, -1.5, -1.5,
]
```

### Goal command and arrival hysteresis

The ROS observation builder must reproduce the same stateful controller as training. Given world-frame base pose `(x, y, yaw)` and target `(target_x, target_y, target_yaw)`, compute:

```text
dx = target_x - x
dy = target_y - y
rho = hypot(dx, dy)
bearing = atan2(dy, dx)
alpha = wrap_to_pi(bearing - yaw)
beta = wrap_to_pi(target_yaw - bearing)
goal_yaw_error = wrap_to_pi(target_yaw - yaw)
goal_dx_body = cos(yaw)*dx + sin(yaw)*dy
goal_dy_body = -sin(yaw)*dx + cos(yaw)*dy

k_rho = 1.5
k_alpha = 2.5
k_beta = -0.8
k_final_yaw = 2.0
max_forward_velocity = 0.6       # m/s
max_yaw_rate = 1.2              # rad/s
```

Define a smooth navigation blend that is zero at 0.05 m and one at 0.20 m. In v3 this blend applies only to the yaw transition:

```text
u = clamp((rho - 0.05) / (0.20 - 0.05), 0, 1)
blend = u*u*(3 - 2*u)

yaw_nav = clamp(k_alpha*alpha + k_beta*beta, -1.2, 1.2)
yaw_final = clamp(k_final_yaw*goal_yaw_error, -1.2, 1.2)
yaw_rate_command = blend*yaw_nav + (1 - blend)*yaw_final

forward_velocity_command =
    clamp(k_rho*rho, 0, 0.6) * max(cos(alpha), 0)

locomotion_command = [forward_velocity_command, 0, yaw_rate_command]
```

Also set forward velocity explicitly to zero when `abs(alpha) >= pi/2`. This lets the robot turn before walking when the goal lies behind it. Do not multiply forward speed by `blend`: at an aligned goal 0.10 m away, v3 commands 0.15 m/s rather than attenuating the command near the stand boundary.

Maintain one latched `stand_mode` state:

- Enter stand when `rho <= 0.05 m` and `abs(goal_yaw_error) <= 5 deg`.
- Exit stand when `rho >= 0.08 m` or `abs(goal_yaw_error) >= 8 deg`.

Outside stand, actor observation `9:12` is `locomotion_command`. While latched in stand, keep the gait planted but replace that slice with bounded signed correction cues:

```text
x_correction = clamp(1.0*goal_dx_body, -0.08, 0.08)       # m/s
y_correction = clamp(1.0*goal_dy_body, -0.08, 0.08)       # m/s
yaw_correction = clamp(1.5*goal_yaw_error, -0.15, 0.15)   # rad/s
actor_command = [x_correction, y_correction, yaw_correction]
```

These cues request small base/body corrections while all feet remain planted; they do not reopen the locomotion gate. At the exact goal they are zero, so the learned target is true stillness.

Reset this latch whenever policy state is reset. Do not implement the thresholds as a stateless dead band; the gap between enter and exit limits prevents goal-boundary chatter.

### Gait clock

Outside latched stand mode, define a continuous motion gate from both translation and turning:

```text
motion_fraction = clamp(
    max(
        forward_velocity_command / 0.6,
        abs(yaw_rate_command) / 1.2,
    ),
    0,
    1,
)
move_gate = motion_fraction
stand_gate = 1 - motion_fraction
gait_frequency = 1.5 + (2.5 - 1.5)*motion_fraction       # Hz
duty_factor = 0.62 + (0.52 - 0.62)*motion_fraction
```

For non-stand environments, update the command and gates, advance normalized phase, and then construct the current observation:

```text
phase = (phase + gait_frequency*0.02) mod 1
clock = [sin(2*pi*phase), cos(2*pi*phase), move_gate, stand_gate]
```

In latched stand mode, freeze phase and set `move_gate=0`, `stand_gate=1` while retaining the frozen sine/cosine phase:

```text
clock = [sin(2*pi*phase), cos(2*pi*phase), 0, 1]
```

Training randomizes initial phase uniformly. The gait is a diagonal trot: FL/RR share one contact phase and FR/RL are offset by half a cycle. In stand mode, all four feet are requested in contact and swing/air-time shaping is disabled.

## Motion-Capture and Finite-Difference Contract

Policy contract v3 intentionally does not use a Savitzky-Golay filter. The following paths are relative to the `src/go2_pendulum/` ROS package root. The separate `go2_pendulum_ros2` implementation must remove Savgol smoothing and differentiation from both:

- `go2_bridge/scripts/go2_bridge_node.py`
- `go2_bridge/scripts/go2_bridge_passive_node.py`

This README documents that required follow-up; those ROS files are not changed in this repository.

Synchronize raw `/pose/base_link` and `/pose/pendulum_ee` packets with timestamp skew no greater than 5 ms. At each 50 Hz policy tick, sample the latest valid synchronized pair. For base position `p_base_w`, current base quaternion `R_base`, and pendulum-end position `p_ee_w`:

```text
base_velocity_w = (p_base_w[t] - p_base_w[t-1]) / 0.02
base_velocity_b = R_base[t]^T * base_velocity_w
```

Use the current quaternion, not the previous quaternion, for the world-to-base rotation. Emit zero base velocity on the first tick after every reset/rearm.

Reconstruct pendulum angles from raw synchronized geometry:

```text
hinge_offset_b = [-0.05, 0.0, 0.06]
r_b = R_base^T * (p_ee_w - p_base_w) - hinge_offset_b

q1 = atan2(-r_b.y, r_b.z)
q2 = atan2(r_b.x, hypot(r_b.y, r_b.z))

dq1 = wrap_to_pi(q1[t] - q1[t-1]) / 0.02
dq2 = wrap_to_pi(q2[t] - q2[t-1]) / 0.02
```

Publish/use raw `q1, q2`; do not retain the Savgol position estimate. Compute `dq1, dq2` at the policy rate, and emit zero pendulum velocity on the first tick after reset/rearm. Reset both finite-difference histories on policy lifecycle changes, stale-data trips, and ROS clock rewind.

The policy-rate controller must own this sampling and differentiation: `go2_bringup/src/rl_controller_node.cpp` should consume both raw pose streams (or an exactly synchronized raw-pose pair), reconstruct `q1/q2`, and take one wrapped difference per 20 ms inference tick. Do not differentiate in a bridge callback running near the mocap publication rate and then pass that higher-rate derivative into the policy.

## Exact Action Delivery Contract

The ONNX actor output is a normalized, unitless 12-D vector in canonical joint order. Deployment must apply the following stages in exactly this order:

1. Clip each raw actor output to `[-2, 2]`.
2. Training only: apply the sampled packet hold and integer action delay. Do not deliberately inject these faults on the robot.
3. Apply one first-order low-pass filter to normalized actions. Deployment uses 4 Hz; training samples the cutoff uniformly from 3 to 5 Hz each episode.
4. Convert to a joint target with `q_candidate = q_default + 0.25*a_filtered`.
5. Clamp each joint target to its URDF position limits with a 0.1 rad margin on both sides.
6. Slew-limit the clamped target relative to the previous delivered target at 6 rad/s. At 50 Hz, the maximum per-tick change is 0.12 rad.
7. Send the resulting `q_delivered` to the joint-position controller.
8. Feed back `a_delivered = (q_delivered - q_default) / 0.25` in actor observation `40:52`.

Use the low-pass update:

```text
alpha_lpf = 1 - exp(-2*pi*cutoff_hz*0.02)
a_filtered[t] = a_filtered[t-1] + alpha_lpf*(a_clipped[t] - a_filtered[t-1])
```

On policy reset, initialize the normalized filter/history to zero and the previous delivered joint target to `q_default`. There must be only one deployment LPF; leaving an additional legacy filter in the bridge changes both latency and the action-history observation.

The observation does **not** contain the raw ONNX output, the clipped action, or merely the delayed action. It contains the normalized target that survived all clamps and was actually delivered.

## ONNX Runtime Contract

`policy_contract.yaml` must travel with `policy.onnx`. The export uses ONNX opset 18 and a fixed batch size of 1. The exported normalizer is embedded in the ONNX graph; pass the raw 56-D values described above and do not normalize them a second time in ROS.

The fixed-batch MLP signature is:

```text
input:
  obs      float32 [1, 56]
output:
  actions  float32 [1, 12]
```

The recurrent signature is:

```text
inputs:
  obs      float32 [1, 56]
  h_in     float32 [L, 1, H]
outputs:
  actions  float32 [1, 12]
  h_out    float32 [L, 1, H]
```

Read `L` and `H` from `policy_contract.yaml` rather than hard-coding them. The default GRU uses `L=1` and `H=256`. Keep one persistent hidden-state tensor per live policy instance and copy `h_out` to the next tick's `h_in`.

Zero the GRU hidden state on:

- policy start and policy stop;
- sit, stand, and emergency transitions;
- ROS clock rewind;
- any stale-data safety trip and the subsequent explicit rearm.

Reset the goal-controller latch, gait phase/clock, finite-difference history, delivered-action history, LPF, and slew state at the same lifecycle boundaries. An MLP has no hidden tensor, but all of those non-network states still require reset.

## Stale-Data Safety for `go2_pendulum_ros2`

Before every inference tick, require:

- base and pendulum-end mocap timestamps within 5 ms of one another; and
- every required observation source, including the synchronized mocap pair, joint state, and IMU, to be no more than 100 ms old.

If either check fails:

1. Stop ONNX inference immediately.
2. Latch policy inactive and reset all policy/controller histories, including GRU state.
3. Command the safe default standing targets rather than replaying the last policy target.
4. Remain latched out even if packets resume.
5. Require an explicit operator/service rearm after all data is fresh; do not auto-resume.

Apply the same reset-and-rearm behavior after ROS time moves backward. Emergency damping remains higher priority than the policy's safe-standing fallback.

## ROS 2 Integration Checklist

The following work belongs in the separate `go2_pendulum_ros2` repository and is not performed here:

1. Copy `policy.onnx` and `policy_contract.yaml` together into the selected model directory.
2. Update the ONNX wrapper in `go2_bringup/src/rl_controller_node.cpp` to validate tensor names/shapes and support both the MLP and GRU signatures.
3. Replace raw goal-error observation channels with the exact stateful goal-command controller above.
4. Replace the four legacy per-foot sine channels with sine/cosine plus move/stand gates, including phase freeze in stand.
5. Remove Savgol code and parameters from both active and passive bridges; provide raw synchronized base/pendulum-end poses. Make the policy-rate C++ controller reconstruct pendulum angles and take exactly one finite difference per 50 Hz inference tick.
6. Replace the legacy raw-action history and command filter with the exact clip/LPF/URDF-clamp/slew pipeline and delivered-action feedback.
7. Add the 5 ms pair-skew, 100 ms freshness, safe-standing latch, and explicit-rearm checks.
8. Reset all recurrent and non-network state on every lifecycle and safety boundary listed above.

Before enabling motor commands, validate the implementation with recorded ROS data:

- observation vectors are exactly 56 finite `float32` values in the documented order;
- MLP and GRU ONNX outputs are exactly 12 finite values;
- replaying one observation sequence produces the same commands as a reference Python/ONNX run;
- entering the goal tolerance does not chatter between move and stand;
- a stale or skewed mocap packet stops inference within one 20 ms control tick;
- no delivered joint target exceeds its URDF margin or changes by more than 0.12 rad per tick;
- the GRU output after every reset matches a run started from zero hidden state.
