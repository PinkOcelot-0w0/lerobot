# Kinova Gen3 quick launcher

`kinova_gen3.py` wraps the standard LeRobot commands for MuJoCo simulation, real-robot teleoperation, dataset collection, and policy rollout.

Install the optional simulation and Quest3 dependencies when needed:

```bash
pip install -e ".[quest3]"
```

Quest3 uses the XRoboToolkit Python SDK. Start XRoboToolkit and connect the headset before launching the commands below; XRoboToolkit manages its PC Service. The LeRobot launcher does not start or manage that service. SO-ARM101 control uses the serial port connected to the leader arm.

## MuJoCo teleoperation

```bash
python examples/kinova_gen3/kinova_gen3.py sim --teleop quest3
```

Use SO-ARM101 instead:

```bash
python examples/kinova_gen3/kinova_gen3.py sim --teleop so101 --teleop-port /dev/ttyUSB0
```

The MuJoCo viewer opens by default. Add `--camera wrist --display-data` to render and display the wrist camera, or `--headless` to suppress the viewer.
Quest control runs at 60 Hz by default for lower input latency; override with `--fps` if needed.
If a pure left/right controller motion causes diagonal wrist drift, first compare with `--position-only`; this separates position-axis mapping from orientation tracking in the IK target.

## Real Gen3 teleoperation

Start with the built-in low speed cap and verify motion direction before increasing it:

```bash
python examples/kinova_gen3/kinova_gen3.py real \
  --robot-ip 192.168.1.10 \
  --teleop quest3 \
  --max-joint-speed-deg-s 5
```

## Dataset collection

Records locally by default. The wrapper only uploads when `--push-to-hub` is explicitly requested with a Hub repo id.

```bash
python examples/kinova_gen3/kinova_gen3.py record \
  --target sim \
  --teleop quest3 \
  --camera wrist \
  --episodes 10 \
  --episode-seconds 30 \
  --task "Pick up the object and place it in the tray"
```

Change `--target sim` to `--target real` to record from the physical Gen3. Use `--repo-id <username>/<dataset>` together with `--push-to-hub` to upload.

## Policy execution

```bash
python examples/kinova_gen3/kinova_gen3.py rollout \
  --target sim \
  --policy /path/to/kinova_gen3_policy \
  --camera wrist \
  --task "Pick up the object" \
  --duration 30
```

Policy checkpoints need observation and action features compatible with the Kinova dataset, including matching camera keys when vision is used. Use `--target real` for hardware and retain a conservative speed limit during initial tests.
