# Kinova Gen3 quick launcher

`kinova_gen3.py` wraps the standard LeRobot commands for MuJoCo simulation, real-robot teleoperation, dataset collection, and policy rollout.

For Quest3 simulation, install the WebXR stack and MuJoCo. The physical Gen3
teleoperator still uses XRoboToolkit and its separate optional package:

```bash
pip install -e ".[vr-teleop]"
```

Quest3 simulation uses the WebXR VR teleop kit and its local relay. Physical Gen3 control still uses the XRoboToolkit Python SDK; start XRoboToolkit and connect the headset before real-robot commands so it can manage its PC Service. SO-ARM101 control uses the serial port connected to the leader arm.

## MuJoCo teleoperation

Quest3 simulation uses the WebXR VR teleop kit. Start its relay in another terminal:

```bash
vr-teleop-relay
```

For USB, forward the relay port with `adb reverse tcp:8443 tcp:8443`, then open
`http://localhost:8443/` in the Quest browser. For LAN use, start the relay with
HTTPS certificates and open its HTTPS URL in the headset. The physical Gen3
command below continues to use the existing XRoboToolkit teleoperator.

```bash
python examples/kinova_gen3/kinova_gen3.py sim --teleop quest3
```

Use SO-ARM101 instead:

```bash
python examples/kinova_gen3/kinova_gen3.py sim --teleop so101 --teleop-port /dev/ttyUSB0
```

The MuJoCo viewer opens by default. Add `--camera wrist --display-data` to render and display the wrist camera in LeRobot; enable **Camera streaming** in the VR page settings to show that simulated wrist view inside Quest. Add `--headless` to suppress the MuJoCo viewer.
The simulation relay defaults to `ws://127.0.0.1:8443/ws`; override with `--vr-url` if the relay is on another host. Quest control runs at 60 Hz by default; override with `--fps` if needed.
If a pure left/right controller motion causes diagonal wrist drift, first compare with `--position-only`; this separates position-axis mapping from orientation tracking in the IK target.

On a headless Linux host, set MuJoCo's EGL renderer when requesting a rendered wrist image:

```bash
MUJOCO_GL=egl python examples/kinova_gen3/kinova_gen3.py sim --teleop quest3 --camera wrist --headless
```

The right grip button engages the clutch, the right trigger controls the gripper, A/X enables precision scaling, and the right thumbstick click ramps the arm to its MuJoCo home pose. The VR page's wrist calibration corrects the controller pivot locally; the calibrated controller pose is what the Kinova IK receives. The page's live gain and joint-step controls, reach limits, and IK haptics are supported. Simulated grasp haptics use gripper tracking error as a force proxy, not a physical torque sensor.

## Real Gen3 teleoperation

The launcher starts at 3 deg/s. The driver rejects configured limits above 3 deg/s and independently clamps every joint-speed command to that ceiling. It also sends stop commands if measurements become invalid or stale or a command fails. Start with the default and verify motion direction in a clear workspace:

```bash
python examples/kinova_gen3/kinova_gen3.py real \
  --robot-ip 192.168.1.10 \
  --teleop quest3
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
