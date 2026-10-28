# VR Teleoperation Kit

The uploaded WebXR VR kit is currently kept under `lerobot.vr_teleop_kit`.
The examples use that package path, so they work from this LeRobot checkout
after installing the project in editable mode.

## Install

```bash
conda activate lerobot
cd ~/code/lerobot
pip install -e ".[vr-teleop]"
```

This extra installs the WebSocket relay, FastAPI/Uvicorn, WebRTC, PyAV, and
MuJoCo dependencies. Re-run the command after changing `pyproject.toml` so the
`vr-teleop-relay` entry point is installed.

## Start the relay

```bash
vr-teleop-relay
```

The default address is `http://127.0.0.1:8443/`. For a Quest connected by USB:

```bash
adb reverse tcp:8443 tcp:8443
```

Then open `http://localhost:8443/` in the Quest browser. For LAN access, use
the relay's HTTPS options and open the corresponding `https://<host-ip>:8443/`
URL in the headset.

The standalone `pure_sim.py` and `teleop_bi_dk1.py` examples still use DK1.
The Kinova launcher uses the same relay and WebXR page through a dedicated
7-DoF Gen3 IK and teleoperator adapter. Its simulation entry point is:

```bash
python examples/kinova_gen3/kinova_gen3.py sim --teleop quest3
```

It supports the clutch, precision modifier, home ramp, wrist-pivot calibration,
reach limits, live IK settings, IK haptics, simulated gripper-resistance haptics,
and an optional MuJoCo wrist camera streamed into the Quest through WebRTC. The
gripper-force calibration button calibrates the simulator's tracking-error
proxy. The `real` command continues to use the existing XRoboToolkit
teleoperator.
