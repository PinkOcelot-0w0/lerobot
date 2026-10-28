"""Damped-least-squares inverse kinematics for the 7-DoF Kinova Gen3."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .kinova_model import KINOVA_EE_SITE, KINOVA_JOINT_NAMES, build_kinova_gen3_model

# Continuous Gen3 joints are bounded to one full turn in this VR model. The
# remaining joints use the explicit limits in the packaged MJCF.
_UNBOUNDED_JOINT_LIMITS = (-2.0 * np.pi, 2.0 * np.pi)
_GEN3_JOINT_LIMITS = np.array(
    [
        _UNBOUNDED_JOINT_LIMITS,
        (-2.24, 2.24),
        _UNBOUNDED_JOINT_LIMITS,
        (-2.57, 2.57),
        _UNBOUNDED_JOINT_LIMITS,
        (-2.09, 2.09),
        _UNBOUNDED_JOINT_LIMITS,
    ],
    dtype=np.float64,
)


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=np.float64))
    return matrix.reshape(3, 3)


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(matrix, dtype=np.float64).reshape(-1))
    return quaternion


class KinovaGen3IKSolver:
    """Solve bounded Gen3 end-effector pose updates with a 7-DoF DLS step.

    ``solve`` returns seven joint angles in radians. ``max_joint_step_rad``
    bounds the total change returned by one call; when used as an interactive
    solver, callers should choose it together with their actual control rate.
    The hardware driver's independent speed limit remains authoritative.
    """

    def __init__(
        self,
        model_xml: str | Path | None = None,
        gripper_xml: str | Path | None = None,
        *,
        damping: float = 0.08,
        posture_gain: float = 0.02,
        rest_qpos: np.ndarray | list[float] | None = None,
        max_joint_step_rad: float | list[float] | np.ndarray = 0.03,
        iterations: int = 6,
    ) -> None:
        """Create a solver using the packaged Gen3 and Robotiq model."""
        if damping <= 0.0:
            raise ValueError("damping must be greater than zero")
        if posture_gain < 0.0:
            raise ValueError("posture_gain must be non-negative")
        step_limit = np.asarray(max_joint_step_rad, dtype=np.float64)
        if step_limit.ndim == 0:
            step_limit = np.full(7, float(step_limit), dtype=np.float64)
        if step_limit.shape != (7,) or not np.all(np.isfinite(step_limit)) or np.any(step_limit <= 0.0):
            raise ValueError("max_joint_step_rad must be positive finite scalar or length-7 vector")
        if iterations <= 0:
            raise ValueError("iterations must be greater than zero")

        self.model, self.data = build_kinova_gen3_model(model_xml, gripper_xml)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.joint_ids = np.array(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in KINOVA_JOINT_NAMES],
            dtype=np.int32,
        )
        if np.any(self.model.jnt_type[self.joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Kinova Gen3 IK requires seven scalar hinge joints")
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.dof_addresses = self.model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, KINOVA_EE_SITE)
        self.wrist_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "spherical_wrist_1_link")
        if self.wrist_body_id < 0:
            raise ValueError("Kinova Gen3 model is missing the spherical wrist body")
        self.joint_limits = _GEN3_JOINT_LIMITS.copy()
        self.damping = float(damping)
        self.posture_gain = float(posture_gain)
        self.max_joint_step_rad = step_limit
        self.iterations = int(iterations)

        if rest_qpos is None:
            self.rest_qpos = np.zeros(7, dtype=np.float64)
            home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if home_id >= 0:
                self.rest_qpos = self.model.key_qpos[home_id, self.qpos_addresses].copy()
        else:
            self.rest_qpos = np.asarray(rest_qpos, dtype=np.float64).reshape(7).copy()
        self.rest_qpos = np.clip(self.rest_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1])

        self.last_position_error_m = 0.0
        self.last_rotation_error_rad = 0.0
        self.last_joint_limit_pressure = 0.0
        self.last_singularity_proximity = 0.0
        self.last_wrist_singularity_proximity = 0.0

    def _forward(self, qpos: np.ndarray) -> None:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(7)
        self.data.qpos[self.qpos_addresses] = qpos
        mujoco.mj_forward(self.model, self.data)

    def fk(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(pinch_site_position, pinch_site_quaternion_wxyz)``."""
        self._forward(qpos)
        position = self.data.site_xpos[self.site_id].copy()
        rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
        return position, _matrix_to_quat_wxyz(rotation)

    def wrist_pivot_xpos(self, qpos: np.ndarray) -> np.ndarray:
        """Return the current joint-5 wrist pivot position in the robot base frame."""
        self._forward(qpos)
        return self.data.xpos[self.wrist_body_id].copy()

    def solve(
        self,
        target_pos: np.ndarray,
        target_quat_wxyz: np.ndarray,
        qpos_seed: np.ndarray,
    ) -> np.ndarray:
        """Return a bounded 7-joint update toward the requested end-effector pose."""
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_quat_wxyz = np.asarray(target_quat_wxyz, dtype=np.float64).reshape(4)
        q_seed = np.asarray(qpos_seed, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(target_pos)) or not np.all(np.isfinite(target_quat_wxyz)):
            raise ValueError("IK target pose must contain only finite values")
        if not np.all(np.isfinite(q_seed)):
            raise ValueError("IK seed must contain only finite joint angles")
        quaternion_norm = float(np.linalg.norm(target_quat_wxyz))
        if quaternion_norm < 1e-9:
            raise ValueError("IK target quaternion must have non-zero length")
        target_quat_wxyz = target_quat_wxyz / quaternion_norm
        target_rotation = _quat_wxyz_to_matrix(target_quat_wxyz)

        q = np.clip(q_seed, self.joint_limits[:, 0], self.joint_limits[:, 1])
        jac_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jac_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        task_jacobian = np.zeros((6, 7), dtype=np.float64)
        task_error = np.zeros(6, dtype=np.float64)
        limit_pressure = 0.0

        for _ in range(self.iterations):
            self._forward(q)
            current_position = self.data.site_xpos[self.site_id].copy()
            current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
            position_error = target_pos - current_position
            rotation_error_matrix = target_rotation @ current_rotation.T
            rotation_error_quat = _matrix_to_quat_wxyz(rotation_error_matrix)
            rotation_error = np.zeros(3, dtype=np.float64)
            mujoco.mju_quat2Vel(rotation_error, rotation_error_quat, 1.0)

            self.last_position_error_m = float(np.linalg.norm(position_error))
            self.last_rotation_error_rad = float(np.linalg.norm(rotation_error))
            if self.last_position_error_m < 1e-4 and self.last_rotation_error_rad < 1e-3:
                break

            mujoco.mj_jacSite(self.model, self.data, jac_position, jac_rotation, self.site_id)
            for index, dof_address in enumerate(self.dof_addresses):
                task_jacobian[:3, index] = jac_position[:, dof_address]
                task_jacobian[3:, index] = jac_rotation[:, dof_address]
            singular_values = np.linalg.svd(task_jacobian, compute_uv=False)
            self.last_singularity_proximity = float(np.clip(1.0 - singular_values[-1] / 0.08, 0.0, 1.0))
            wrist_jacobian = task_jacobian[3:, 3:]
            wrist_singular_values = np.linalg.svd(wrist_jacobian, compute_uv=False)
            self.last_wrist_singularity_proximity = float(
                np.clip(1.0 - wrist_singular_values[-1] / 0.15, 0.0, 1.0)
            )
            task_error[:3] = position_error
            task_error[3:] = rotation_error

            # DLS bounds singular configurations. A small null-space posture
            # term uses Gen3's seventh DoF without overwhelming the pose task.
            system = task_jacobian @ task_jacobian.T + self.damping**2 * np.eye(6)
            try:
                inverse_task = np.linalg.solve(system, np.eye(6))
            except np.linalg.LinAlgError:
                break
            jacobian_pseudoinverse = task_jacobian.T @ inverse_task
            delta = jacobian_pseudoinverse @ task_error
            nullspace = np.eye(7) - jacobian_pseudoinverse @ task_jacobian
            delta += nullspace @ (self.posture_gain * (self.rest_qpos - q))

            q_unclipped = q + delta
            q = np.clip(q_unclipped, self.joint_limits[:, 0], self.joint_limits[:, 1])
            limit_pressure = max(limit_pressure, float(np.linalg.norm(q_unclipped - q)))

        total_delta = np.clip(q - q_seed, -self.max_joint_step_rad, self.max_joint_step_rad)
        q_bounded = np.clip(q_seed + total_delta, self.joint_limits[:, 0], self.joint_limits[:, 1])
        self.last_joint_limit_pressure = max(
            limit_pressure,
            float(np.linalg.norm(q_seed + total_delta - q_bounded)),
        )
        return q_bounded
