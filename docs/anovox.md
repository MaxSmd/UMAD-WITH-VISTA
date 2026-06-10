# AnoVox benchmark — structure, technical configuration, recreation steps

A reference document for the AnoVox dataset this project consumes. Covers what
the data looks like on disk, how the AnoVox generation framework builds it, the
exact technical settings used here, and how to recreate the dataset from
scratch on this machine.

The 16-scenario evaluation set this project uses lives at
[data/anovox/Outputs/Final_Output_2026_05_26-21_18/](../data/anovox/Outputs/Final_Output_2026_05_26-21_18/).

---

## 1. What AnoVox is

AnoVox (Bogdoll et al., 2024) is a CARLA-based **anomaly detection benchmark for
autonomous driving**. It generates synthetic driving scenarios in which a rare
object — an "anomaly actor" — is placed somewhere along the ego vehicle's route.
The ego vehicle drives toward that anomaly under autopilot, while a multimodal
sensor suite records the scene. Ground truth is provided in three forms:

- **Semantic camera** — per-pixel class IDs, with anomalous objects assigned
  dedicated class IDs (29–34, plus a generic catch-all 100).
- **Instance camera** — per-pixel instance IDs encoded in RGB.
- **Voxel grid** — 3-D ground-truth voxelization (this project does not use it).

The benchmark intentionally randomizes the anomaly object, its placement, the
ego spawn point, the weather, and the NPC traffic, so each scenario differs.
Static objects (boxes, animals, mundane debris, etc.) are the main evaluation
modality used in the UMAD paper.

Codebase: [external/anovox/](../external/anovox/) — upstream clone of the
AnoVox repo, plus our additions ([resume_run.py](../external/anovox/resume_run.py)).

---

## 2. On-disk layout

```
data/anovox/Outputs/                          (= external/anovox/Data/Outputs, symlink)
└── Final_Output_<TIMESTAMP>/
    ├── Scenario_Configuration_Files/
    │   ├── scenario_config_map_Town01.json
    │   ├── scenario_config_map_Town02.json
    │   └── ...                               (one JSON per used town)
    ├── Scenario_<UUID>/                      (one dir per scenario, UUID v4)
    │   ├── ACTION/                           (200 CSVs; ego-vehicle steering/throttle/brake/speed per tick)
    │   ├── ANOMALY/                          (200 CSVs; anomaly actor's location/rotation per tick)
    │   ├── RGB-CAM(0, 0, 1.8)(0, 0, 0)_<sensor-uuid>/   (200 PNGs, front camera)
    │   ├── DEPTH_CAM(0, 0, 1.8)(0, 0, 0)_<sensor-uuid>/ (200 PNGs, depth)
    │   ├── SEMANTIC-CAM(0, 0, 1.8)(0, 0, 0)_<sensor-uuid>/ (200 PNGs, semantic-id map)
    │   ├── INSTANCE-CAM(0, 0, 1.8)(0, 0, 0)_<sensor-uuid>/ (200 PNGs, instance IDs in RGB)
    │   ├── LIDAR(0, 0, 1.8)(0, 0, 0)_<sensor-uuid>/        (200 .npy point clouds)
    │   ├── SEMANTIC-LIDAR(0, 0, 1.8)(0, 0, 0)_<sensor-uuid>/ (200 .npy semantic point clouds)
    │   ├── ROUTE_MAP/                        (200 PNGs; bird's-eye view of the planned route)
    │   └── sensor_setup.json                 (sensor positions / parameters for this scenario)
    └── Scenario_<UUID>/...
```

Sensor directory names embed the sensor's position `(x, y, z)`, rotation, and a
sensor-instance UUID — that's why they look unusual. The position
`(0, 0, 1.8)` is the on-vehicle offset in meters; rotation `(0, 0, 0)` means
forward-facing on the roof.

**Frame indexing.** Frame numbers do **not** restart at 0 per scenario — each
sensor's filename ends in `..._<global_carla_frame_id>.{png,npy,csv}`. Frame IDs
are sequential within a scenario but offset per simulation run. To recover the
in-scenario index (0–199), sort filenames by the trailing integer and take
their position in that order.

**Anomaly visibility.** The anomaly object is placed along the route, so it
only enters the camera frustum partway through the scenario. The number of
frames in which it is visible is typically much less than 200.

### 2.1 Class IDs in the semantic camera

The semantic camera is a single-channel `L` PNG containing CARLA class IDs.
The 0–28 IDs are Cityscapes-style classes (road, sidewalk, building, vegetation,
pedestrian, vehicle…). The anomaly classes are:

| ID | Name | Color (RGB) |
|----|------|-------------|
| 29 | `home`     | `(245, 29, 0)` |
| 30 | `animal`   | `(245, 30, 0)` |
| 31 | `nature`   | `(245, 31, 0)` |
| 32 | `special`  | `(245, 32, 0)` |
| 33 | `airplane` | `(245, 33, 0)` |
| 34 | `falling`  | `(245, 34, 0)` |
| 100 | `anomaly` (generic catch-all) | `(245, 0, 0)` |

The full label table is at
[external/anovox/Definitions.py:LABELS](../external/anovox/Definitions.py).
This project's binary anomaly mask is `class_id in (29..34, 100)` — implemented
in [src/vista_umad/anovox.py:ANOMALY_CLASS_IDS](../src/vista_umad/anovox.py#L35).

### 2.2 Image resolutions

- RGB / depth / semantic / instance camera: **768 × 512 px**, FOV 90°
  (FOCAL ≈ 384 px). Set in
  [external/anovox/Definitions.py](../external/anovox/Definitions.py) (`IMAGE_WIDTH`,
  `IMAGE_HEIGHT`, `CAMERA_FOV`).
- LiDAR: **64-channel ray-cast**, 100 m range, +15°/-25° FOV,
  1 000 000 points/sec, σ=0.1 noise. Typical scan is ~80 000 points
  `[N, 4]` float32 (xyz + intensity).
- Semantic LiDAR: same geometry, `[N, 4]` float64 with class IDs in the 4th
  column.

---

## 3. Per-frame data formats

| Folder | File | Format | Notes |
|--------|------|--------|-------|
| `ACTION/` | `ACTION_<frame>.csv` | `;`-separated key/value pairs | `frame_id`, `scenario_id`, `throttle`, `steer`, `brake`, `speed`, etc. |
| `ANOMALY/` | `ANOMALY_<frame>.csv` | `;`-separated key/value pairs | `frame_id`, `scenario_id`, `anomaly_id`, `location`, `rotation` |
| `RGB-CAM/` | `RGB-CAM..._<frame>.png` | RGB PNG 768×512 | front camera |
| `DEPTH_CAM/` | `DEPTH_CAM..._<frame>.png` | logarithmic-depth-encoded PNG | CARLA's standard depth encoding |
| `SEMANTIC-CAM/` | `..._<frame>.png` | single-channel `L` PNG | class IDs (0–34, 100) |
| `INSTANCE-CAM/` | `..._<frame>.png` | RGB PNG | R = class_id, G+B encode instance_id (little-endian 16-bit) |
| `LIDAR/` | `..._<frame>.npy` | `[N, 4]` float32 | xyz + intensity, sensor frame |
| `SEMANTIC-LIDAR/` | `..._<frame>.npy` | `[N, 4]` float64 | xyz + semantic class id |
| `ROUTE_MAP/` | `ROUTE_MAP_<frame>.png` | RGB PNG | bird's-eye-view of planned route |

---

## 4. Scenario configuration JSON

Each Town has one JSON config in `Scenario_Configuration_Files/`. The structure:

```json
{
  "scenario_definition": {
    "map": "Town01",
    "scenarios": [
      {
        "id": "<uuid v4>",
        "anomaly_config": {
          "anomalytype": "STATIC",
          "anomaly_bp_name": "static.prop.o_cardboardbox22_home",
          "distance_to_waypoint": 78,
          "rotation": 237.305...
        },
        "ego_spawnpoint": {"location": {...}, "rotation": {...}},
        "ego_end_spawnpoint": {"location": {...}, "rotation": {...}},
        "ego_route": {"locations": [...]},
        "weather_preset": "SOFT_RAIN_SUNSET",
        "npc_vehicle_amount": 100,
        "npc_walker_amount": 50
      },
      { ... }
    ]
  }
}
```

`anomaly_bp_name` is a CARLA blueprint string. The suffix (`_home`, `_animal`,
`_special`, `_nature`, etc.) determines which class ID the anomaly maps to in
the semantic camera. The set of available blueprints is determined by the
custom AnoVox CARLA build (not stock CARLA assets).

`distance_to_waypoint` is the number of waypoints from the ego spawn point at
which the anomaly is placed (default range `[65, 105]`, see
[Definitions.py:DISTANCE_INTERVAL](../external/anovox/Definitions.py)).

---

## 5. Technical settings

All knobs live in
[external/anovox/Definitions.py](../external/anovox/Definitions.py).
Defaults relevant to this project:

| Setting | Value | Effect |
|---------|-------|--------|
| `PORT` | `2000` | CARLA RPC port |
| `FIXED_DELTA_SECONDS` | `0.1` | Sim time step (10 Hz logical) |
| `SUBSTEPPING` | `True` | Physics substepping |
| `MAX_SUBSTEP` | `10` | CARLA 0.9.14 max substeps per tick |
| `MAX_SUBSTEP_DELTA_TIME` | `0.01` | CARLA 0.9.14 max substep delta |
| `MAX_TICKCOUNT` | `215` | **15 spawn ticks + 200 driving ticks** → 200 frames/scenario |
| `TICK_COUNT_MODULO_VALUE` | `1` | Record every tick (vs every Nth) |
| `EGO_VEHICLE` | `vehicle.lincoln.mkz_2020` | CARLA blueprint for the ego |
| `DISTANCE_INTERVAL` | `[65, 105]` | Waypoints between ego spawn and anomaly |

**Total scenario duration**: 200 ticks × 0.1 s = 20 s of simulated driving per
scenario. Real wall-clock per scenario observed here: **~2.5–3 minutes**.

### 5.1 Per-town NPC density

[Definitions.py:TOWN_CONFIGS](../external/anovox/Definitions.py) controls the
number of NPC vehicles and walkers per town. **This is the dominant cause of
scenario failure in our runs.** Each NPC vehicle is spawned at a random
waypoint and given a route; if the town's lane network can't sustain the
requested count, AnoVox raises `ValueError("No target waypoints available for
NPC vehicle")` and the scenario is marked unrecoverable.

| Town | NPC vehicles | NPC walkers | Map shipped in CARLA build | Empirical success rate (this project) |
|------|-------------:|------------:|----------------------------|--------------------------------------:|
| Town01 | 100 | 200 | yes | 3/3 |
| Town02 |  50 | 100 | yes | 2/2 |
| Town03 | 200 | 150 | yes | 3/4 |
| Town04 | **250** | 100 | yes | **1/4 — Town04 is unreliable at the default NPC density** |
| Town05 | 150 | 150 | yes | 2/2 |
| Town06 | 150 |  50 | **no** (missing) | n/a |
| Town07 |  50 | 100 | **no** (missing) | n/a |
| Town10HD | 150 | 150 | yes | 7/9 |

To make Town04 reliably succeed, lower `npc_vehicle_amount` to ~100. To re-add
Town06/Town07 you would need a different CARLA build that ships those maps.

### 5.2 Anomaly types and categories

```python
class AnomalyTypes(Enum):
    NORMALITY = "normality"                              # no anomaly (baseline data)
    STATIC = "static"                                    # this project's only mode
    SUDDEN_BREAKING_OF_VEHICLE_AHEAD = "..."
```

For `STATIC`, the categories drawn from per scenario are configured via
[Definitions.py:categories](../external/anovox/Definitions.py):

```python
categories = ["home", "animal", "special"]   # what we used
# also available, commented out: 'nature', 'airplane', 'falling'
```

Each STATIC scenario draws a random anomaly blueprint from these categories.
The category determines which anomaly class ID (29–34) the object will carry in
the semantic camera.

### 5.3 Weather presets

[Definitions.py:WeatherPresets](../external/anovox/Definitions.py) enumerates
14 CARLA weather presets (clear noon, cloudy noon, wet noon, hard rain noon,
clear sunset, ..., soft rain sunset). One is chosen at random per scenario.

### 5.4 Sensor setup (`MONO_SENSOR_SETS`)

This project uses the MONO_SENSOR_SETS profile — one of four profiles defined
in [external/anovox/EgoVehicleSetup.py](../external/anovox/EgoVehicleSetup.py).
It is a single forward camera plus a single roof-mounted LiDAR, with depth /
semantic / instance cameras sharing the RGB camera's pose. Sensor offsets are
`(x=0, y=0, z=1.8)` from the ego vehicle origin — roof height, forward-facing.

Camera and LiDAR parameters are in
[external/anovox/EgoVehiculeSensorDefaults.py](../external/anovox/EgoVehiculeSensorDefaults.py):

```python
CAMERA_ARGUMENTS = {"image_height": 512, "image_width": 768, "camera_fov": 90.0}
LIDAR_ARGUMENTS = {"range": 100.0, "upper_fov": 15.0, "lower_fov": -25.0,
                   "channels": 64, "rotation_frequency": 200.0,
                   "points_per_second": 1000000, "noise_stddev": 0.1, ...}
```

---

## 6. The generation pipeline

`main.py --run` does two phases:

1. **Config generation** (`ScenarioDefinitionGenerator.generate_all_scenario_config_files`):
   * Reads `NBR_OF_SCENARIOS` and `USED_MAPS` from Definitions.py.
   * For each map, allocates a share of the total scenario count (round-robin
     remainder to the last map). E.g. `NBR_OF_SCENARIOS=10` across 4 maps →
     `{Town01: 2, Town03: 2, Town04: 2, Town10HD: 4}`.
   * For each scenario in each map: randomly picks the ego spawn point, route
     length, an anomaly category and blueprint, an anomaly placement waypoint
     in `DISTANCE_INTERVAL`, a rotation, an NPC vehicle/walker count from
     `TOWN_CONFIGS[map]`, and a weather preset.
   * Writes one JSON per map to
     `Final_Output_<TIMESTAMP>/Scenario_Configuration_Files/scenario_config_map_<Town>.json`.

2. **Execution** (`ScenarioMain.run_all_scenarios_from_configs`):
   * For each JSON file: switches CARLA to that map (`World.change_world`),
     then iterates each scenario in the file.
   * Per scenario: sets weather → spawns ego → attaches sensors → builds route
     → spawns NPCs → spawns anomaly → runs 215 ticks (15 spawn-stabilization +
     200 recorded). Sensor outputs stream to disk per tick.
   * On a Python-level error (pedestrian crash, NPC waypoint failure, anomaly
     spawn failure, ego sudden break): logs the error, appends the UUID to
     `external/anovox/scenario_id_fails.txt`, destroys everything, sleeps 10s,
     continues with the next scenario.
   * On a C-level CARLA segfault: the whole `main.py` process dies. There is
     no built-in recovery — this is what motivated
     [external/anovox/resume_run.py](../external/anovox/resume_run.py) +
     [`/tmp/anovox_run_gen.sh`](../external/anovox/) (see §8).

---

## 7. Recreating the dataset on this machine

Prerequisites already in place:

- CARLA 0.9.14 custom AnoVox build at
  [external/carla/CarlaUE4.sh](../external/carla/CarlaUE4.sh).
- Bundled Vulkan loader at
  `external/lib/extracted/usr/lib/x86_64-linux-gnu/libvulkan.so.1` — the system
  has no libvulkan, so it must be on `LD_LIBRARY_PATH` for CARLA to launch.
- AnoVox venv at `external/anovox/.venv/` (Python 3.8.20 via `uv`; has the
  `carla` 0.9.14 wheel installed).

### 7.1 Single-batch recreation (start to finish)

```bash
cd ~/umad-with-vista

# 1. (Optional) Edit scenario count / town selection.
#    NBR_OF_SCENARIOS controls total count; USED_MAPS controls which towns.
sed -i 's/^NBR_OF_SCENARIOS = .*/NBR_OF_SCENARIOS = 16/' external/anovox/Definitions.py

# 2. Start CARLA inside a detachable tmux window. LD_LIBRARY_PATH is required.
tmux new-session -d -s anovox-gen -n carla
tmux set-window-option -t anovox-gen:carla remain-on-exit on
tmux send-keys -t anovox-gen:carla \
  'cd ~/umad-with-vista && \
   export LD_LIBRARY_PATH=$PWD/external/lib/extracted/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH && \
   external/carla/CarlaUE4.sh --carla-world-port=2000 -RenderOffScreen 2>&1 | \
   tee /tmp/carla_server.log' C-m

# 3. Wait for the RPC port (CARLA needs ~5–10 s to bind).
until ss -ltn | grep -q ':2000'; do sleep 1; done

# 4. Run main.py inside a second tmux window, tee'd to a log.
tmux new-window -t anovox-gen -n gen
tmux set-window-option -t anovox-gen:gen remain-on-exit on
tmux send-keys -t anovox-gen:gen \
  'cd ~/umad-with-vista/external/anovox && \
   .venv/bin/python main.py --run 2>&1 | tee /tmp/anovox_gen.log' C-m

# 5. Detach (laptop-safe). Reattach later with `tmux attach -t anovox-gen`.
```

Output lands in `external/anovox/Data/Outputs/Final_Output_<NOW>/` — symlinked
to `data/anovox/Outputs/Final_Output_<NOW>/`.

### 7.2 Crash-resilient recreation

`main.py` has no resume logic, and we observed one CARLA segfault in 18
scenarios (~5 % rate) plus a 30 % pedestrian/NPC failure rate. For a
hands-off run, use [external/anovox/resume_run.py](../external/anovox/resume_run.py)
inside a retry loop (the wrapper script
[`/tmp/anovox_run_gen.sh`](../external/anovox/)). The wrapper:

1. Inventories the target `Final_Output_<TIMESTAMP>/` dir: a scenario is
   *complete* if its `RGB-CAM*/` directory has ≥ 200 PNGs.
2. Loads `external/anovox/scenario_id_fails.txt` (unrecoverable failures —
   AnoVox appends here itself when raising `ValueError('pedestrian_crash')`,
   `ValueError("No target waypoints available for NPC vehicle")`, etc.).
3. Deletes any partial scenario dirs (segfault stubs) so AnoVox can restart
   them cleanly on the next pass.
4. Filters the Town JSONs in-place, removing UUIDs in `done ∪ failed`.
5. Calls `ScenarioMain.run_all_scenarios_from_configs(remaining_paths)`
   directly (bypasses `main.py`'s brittle `len(file_paths) == len(USED_MAPS)`
   sanity check, so a town can have 0 remaining scenarios).
6. Exits 0 when nothing remains; non-zero on segfault (exit 139) or other
   crash, which the outer bash loop catches and retries up to 12 times.

Run it instead of step 4 above:

```bash
tmux send-keys -t anovox-gen:gen \
  'bash /tmp/anovox_run_gen.sh' C-m
```

`/tmp/anovox_run_gen.sh` targets the `ANOVOX_OUTDIR` env var (defaults to the
21-18 directory used in this project). To resume into a different dir:

```bash
ANOVOX_OUTDIR=/path/to/Final_Output_<NEW>/ bash /tmp/anovox_run_gen.sh
```

### 7.3 Output verification

```bash
DIR=data/anovox/Outputs/Final_Output_<TIMESTAMP>

# Count viable scenarios (those with the full 200 RGB frames):
for s in $DIR/Scenario_*-*; do
  [ -d "$s" ] || continue
  n=$(ls "$s"/RGB-CAM*/*.png 2>/dev/null | wc -l)
  if [ "$n" -ge 200 ]; then echo "$(basename "$s")"; fi
done | wc -l
```

---

## 8. Failure modes observed

Five distinct unrecoverable failure modes seen across our 26 scenario attempts:

| Failure | Raised by | Trigger | Mitigation |
|---|---|---|---|
| `ValueError('pedestrian_crash')` | [ScenarioMain.py:339](../external/anovox/Scenarios/ScenarioMain.py#L339) | Ego hits a pedestrian during the 200 driving ticks | Inherent randomness; just re-roll |
| `ValueError("No target waypoints available for NPC vehicle")` | [Static.py:115](../external/anovox/Scenarios/Static.py#L115) | NPC traffic spawn finds no valid route waypoint — happens when `npc_vehicle_amount` exceeds the town's capacity | Lower `TOWN_CONFIGS[map].npc_vehicle_amount` |
| `Exception("Failed to spawn anomaly after 5 attempts")` | [AnomalyBehaviourDefinitions.py:204](../external/anovox/Scenarios/AnomalyBehaviourDefinitions.py#L204) | Anomaly blueprint cannot be placed at its randomly chosen waypoint (geometry collision, off-road) | Inherent randomness |
| `ValueError('Ego vehicle sudden breaking. Ignore scenario')` | [World.py:900](../external/anovox/Models/World/World.py#L900) | Ego autopilot deadlocks (NPCs blocking the route) | Inherent randomness |
| **CARLA segfault** | C-level inside `CarlaUE4-Linux-Shipping` | Often triggered on the next scenario after a pedestrian-crash cleanup poisoned the simulator state | Retry from outside (`resume_run.py` + bash loop) |

In all four Python-level cases AnoVox appends the UUID to
`scenario_id_fails.txt` before continuing to the next scenario. The segfault
case is the only one that kills the process; the resume wrapper exists
specifically to recover from it.

---

## 9. The 16-scenario dataset used by this project

Generated across two batches on edward (2× A40, GPU 1 for CARLA), unified by
symlinking the second batch's keepers into the first batch's dir:

- **Batch 1** ([Final_Output_2026_05_26-21_18](../data/anovox/Outputs/Final_Output_2026_05_26-21_18/)):
  `NBR_OF_SCENARIOS=16`, all 6 shipped towns enabled (Town01–Town05 + Town10HD).
  Result: 11 complete, 5 in fails.txt (incl. 2 Town04 NPC-waypoint failures).
- **Batch 2** ([Final_Output_2026_05_27-12_48](../data/anovox/Outputs/Final_Output_2026_05_27-12_48/)):
  `NBR_OF_SCENARIOS=10`, towns reduced to deficits only (Town01, Town03, Town04,
  Town10HD). Result: 7 complete, 3 in fails.txt.
- **Unified set**: the 11 batch-1 scenario dirs + 5 batch-2 symlinks (drop the
  2 batch-2 Town10HD scenarios since batch 1 already over-supplied that town).
  Final per-town count:

  ```
  Town01: 3   Town02: 2   Town03: 3   Town04: 1   Town05: 2   Town10HD: 5   = 16
  ```

  Town04 is under-represented because of the NPC density problem (1 success in
  4 attempts at the default 250-vehicle config). Lowering Town04's
  `npc_vehicle_amount` to ~100 would fix this — left as a possible future
  rebalance.

All 16 scenarios share the same generation parameters: STATIC anomaly type,
MONO_SENSOR_SETS, 200 frames at 10 Hz logical (20 s), categories
`["home", "animal", "special"]`, distance interval 65–105 waypoints.

The combined `scenario_id_fails.txt` is at
[external/anovox/scenario_id_fails.txt](../external/anovox/scenario_id_fails.txt)
(8 UUIDs as of this writing — 5 from batch 1, 3 from batch 2).
