# Portable hardware-validation workloads

These are genuine, deterministic CPU computations suitable for a first PC +
Jetson Orin scheduling/data-transfer test. They require only Python's standard
library. There are no downloaded model weights, CUDA assumptions, simulated
task delays, motor commands, or fabricated inference/energy measurements.

## Dependency topology and contract

```text
hil_sensor ──observations──> hil_mapping ──map──> hil_planning
    │                           │                     │
    └────────truth──────────────┼─────> hil_validation <──trajectory
                                └──map───────^
```

`PORT_TYPES` in `__init__.py` exports each task's input/output type mapping.
`execute(task_type, inputs, seed)` consumes JSON objects by input port and returns
JSON objects by output port. All payloads carry `schema_version: 1`, a `kind`, and
a shared scene identity. Derived artifacts include canonical JSON SHA256 hashes
of their actual upstream inputs. Acquisition uses the seed; later stages use
their inputs, not a regenerated fixture hidden behind the seed.

| Task | Actual work | Outputs |
| --- | --- | --- |
| `hil_sensor` | Generate three obstacle walls and raycast 5,120 range beams from 20 known survey poses, adding bounded 4 mm range noise | `observations`, independent analytic `truth` |
| `hil_mapping` | Integrate traversed ray cells and measured hit cells into a 96 × 64 occupancy grid | `map` |
| `hil_planning` | Inflate occupied/unknown cells for a circular robot footprint; run four-neighbor Manhattan A*; generate stop/turn/translate motion primitives | `trajectory` |
| `hil_validation` | Check every continuous motion segment against independent obstacle rectangles and world boundaries, plus map clearance, provenance, goal, speed, acceleration, heading and timing | `validation` |

The start and goal are separated by alternating walls. Changing the seed changes
wall ends/range readings and can change the resulting path. Missing observations
remain **unknown and blocked**: absence of a hit never means an entire region
is free. A no-hit beam clears only cells along its measured range.

## What is and is not real

- **Synthetic acquisition:** a static, two-dimensional survey with exact known
  sensor poses. Multiple virtual viewpoints cover the room; this is not a claim
  that a single stationary sensor sees through walls. Range noise is a bounded
  test fixture, not a calibrated physical sensor model. There is no localization
  or SLAM.
- **Real computation:** ray/rectangle intersections, ray-grid integration,
  obstacle inflation, priority-queue graph search, trajectory construction, and
  collision/kinematics checks execute on the machine running the worker. There
  is no sleep-based workload or precomputed completion result.
- **Kinematics only:** a circular planar robot stops before each turn. Linear
  segments use a triangular/trapezoidal velocity profile with bounded linear
  acceleration; turns have bounded yaw rate. Angular acceleration, dynamics,
  actuation tracking, moving obstacles, and physical safety certification are
  outside this MVP. The generated trajectory is **not sent to a robot**.
- `planned_motion_duration_s` is the time a hypothetical robot would need to
  follow the trajectory, **not task compute time**. The Agent measures actual
  process elapsed time separately. This small portable workload does not measure
  YOLO/VLA performance, GPU speed, energy savings, or production scheduling
  benefit.

`hil_validation` raises `WorkloadError` for an invalid result, producing a failed
task rather than a successful task whose result quietly says `valid: false`.
The truth hash provides accidental-corruption/provenance checking, not security
against an adversary who can replace every artifact.

## Process boundary

The Agent can execute one isolated worker per task:

```bash
python -m examples.hardware_workloads.worker
```

It reads one JSON object from standard input until EOF:

```json
{"task_type":"hil_sensor","inputs":{},"seed":19}
```

On success it writes exactly one JSON object, keyed by output port, to stdout.
Errors go to stderr with a nonzero process exit status. It does not resolve file
paths, fetch URLs, import arbitrary plugins, or run caller-supplied commands.
The Agent owns real artifact transfer and supplies decoded upstream JSON.

Each payload is limited to 2 MiB. Grid dimensions, ray counts, inflation radius,
obstacle counts, trajectory size and numeric values are bounded; the Agent must
also enforce a subprocess wall-clock timeout and cancellation. A complete
default dataset is much smaller than the cap and needs no network downloads.

## Tests

From the repository root, in the normal test environment:

```bash
python -m pytest -q tests/test_hardware_workloads.py
```

Coverage includes reproducibility, cross-seed feasible paths, input dependence,
unknown-space handling, blocked paths, corrupt trajectories, excessive speed/
acceleration/yaw rate, provenance mismatch, malformed/bounded inputs, subprocess
protocol errors, and collision *between* waypoints that an endpoint-only check
would miss. Host tests establish portable software behavior, not Jetson hardware
validation; the two-machine run remains required.
