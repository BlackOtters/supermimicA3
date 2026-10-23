"""Versioned lower-body Sequence-AMP contract for DEPSIC A3 V1.

This module deliberately contains no policy observations and no actor input.
It defines the training-only data exchange between a hand-curated A3 expert
cache, the mjlab environment, and the AMP discriminator.  The contract is
kept small on purpose: it covers the lower-body motion chain and support
phase, while DEPSIC's existing arm teacher remains outside AMP.

The source A3 PKLs are kinematic references, not physics rollouts.  Their
contact feature is consequently named ``kinematic_support_phase`` throughout
the cache metadata.  Policy samples use the matching binary support semantic
derived from mjlab contact forces.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Final
import xml.etree.ElementTree as et

import numpy as np

from mimic_lite.depsic_a3_contract import (
    A3_31DOF_JOINT_NAMES,
    A3_POLICY_JOINT_NAMES,
    A3_T3D0_MJCF_JOINT_NAMES,
)


# The runtime/cache retain a 30-frame FIFO for the established A3 features.
# The forward route additionally consumes four adjacent 105-D lower-body
# link-state frames with its own validity horizon.  Bump the contract whenever
# either raw feature definition changes so incompatible caches are rejected.
A3_AMP_CONTRACT_VERSION: Final[int] = 22
A3_AMP_SEQUENCE_LENGTH: Final[int] = 30
A3_AMP_FORWARD_SEQUENCE_LENGTH: Final[int] = 4
A3_AMP_BACKWARD_SEQUENCE_LENGTH: Final[int] = 20
A3_AMP_LATERAL_SEQUENCE_LENGTH: Final[int] = 20
A3_AMP_TURN_SEQUENCE_LENGTH: Final[int] = 10
A3_AMP_SQUAT_SEQUENCE_LENGTH: Final[int] = 4
A3_AMP_CONDITION_FIELDS: Final[tuple[str, ...]] = (
    "lin_vel_x",
    "lin_vel_y",
    "ang_vel_z",
    "torso_height",
)
A3_AMP_CONDITION_DIM: Final[int] = len(A3_AMP_CONDITION_FIELDS)
# Match the independently sampled A3 LBC ranges.  These are fixed semantic
# scales rather than fitted expert moments, so a narrow expert set cannot
# hide a command distribution mismatch through normalization leakage.
A3_AMP_CONDITION_MEAN: Final[tuple[float, ...]] = (0.0, 0.0, 0.0, 0.855)
A3_AMP_CONDITION_STD: Final[tuple[float, ...]] = (0.55, 0.55, 1.57, 0.225)
# The conditional AMP labels must stay inside the command space that the
# policy can actually sample.  We clip only planar velocity labels at cache
# build time; the motion timeline and all kinematic features remain untouched.
A3_AMP_PLANAR_VELOCITY_LIMIT: Final[float] = 0.55
A3_AMP_MODE_NAMES: Final[tuple[str, ...]] = (
    "locomotion_forward",
    "locomotion_backward",
    "locomotion_lateral_left",
    "locomotion_lateral_right",
    "locomotion_turn_left",
    "locomotion_turn_right",
    "squat",
    "neutral",
)
A3_AMP_MODE_COUNT: Final[int] = len(A3_AMP_MODE_NAMES)
A3_AMP_ACTIVE_MODE_COUNT: Final[int] = 7
A3_AMP_SQUAT_MODE: Final[int] = 6
A3_AMP_NEUTRAL_MODE: Final[int] = 7
# ``h`` remains a discriminator condition and selects squat versus neutral
# only inside the stationary branch. Keep this in physical command units and
# share the threshold with the runtime router.
A3_AMP_SQUAT_SPEED_THRESHOLD: Final[float] = 0.10
A3_AMP_SQUAT_HEIGHT_THRESHOLD: Final[float] = 1.0

# The order is a model/training contract. AMP observes the waist RPY followed
# by the two six-DoF legs. Including waist state lets the style prior constrain
# the torso/leg coordination that is visible in natural gait. No arm/head DoF
# enters AMP.
A3_AMP_LOWER_JOINT_NAMES: Final[tuple[str, ...]] = (
    A3_POLICY_JOINT_NAMES[:3] + A3_POLICY_JOINT_NAMES[17:]
)
A3_AMP_KEYPOINT_BODY_NAMES: Final[tuple[str, ...]] = (
    "torso_Link",
    "left_hip_pitch_Link",
    "left_knee_Link",
    "left_ankle_pitch_Link",
    "left_ankle_roll_Link",
    "right_hip_pitch_Link",
    "right_knee_Link",
    "right_ankle_pitch_Link",
    "right_ankle_roll_Link",
)
A3_AMP_SUPPORT_BODY_NAMES: Final[tuple[str, str]] = (
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
)
A3_AMP_024_SERIAL_SUPPORT_BODY_NAMES: Final[tuple[str, ...]] = (
    "left_ankle_roll_Link",
    "left_foot_forefoot_Link",
    "left_foot_toe_Link",
    "right_ankle_roll_Link",
    "right_foot_forefoot_Link",
    "right_foot_toe_Link",
)
A3_AMP_024_SERIAL_PASSIVE_JOINT_POSITIONS: Final[dict[str, float]] = {
    "left_foot_forefoot_joint": 0.0,
    "left_foot_toe_joint": 0.0,
    "right_foot_forefoot_joint": 0.0,
    "right_foot_toe_joint": 0.0,
}
A3_AMP_DEFAULT_ASSET_PROFILE: Final[str] = "a3"
A3_AMP_024_SERIAL_ASSET_PROFILE: Final[str] = "a3_024_serial"
# Ultra keeps the exact lower-body AMP feature names/layout, but its full
# MJCF/URDF has different torso/head/hand inertias.  It therefore needs a
# separate cache identity even though the 69-D and 105-D arrays have the
# same shape as ordinary A3.
A3_AMP_ULTRA_T3D0_ASSET_PROFILE: Final[str] = "a3_ultra_t3d0"
A3_AMP_ASSET_PROFILES: Final[frozenset[str]] = frozenset((
    A3_AMP_DEFAULT_ASSET_PROFILE,
    A3_AMP_024_SERIAL_ASSET_PROFILE,
    A3_AMP_ULTRA_T3D0_ASSET_PROFILE,
))
# The forward route's richer observation uses seven lower-body links: pelvis
# plus hip-roll, knee and ankle-roll on each side. Pose is expressed relative
# to the torso anchor; linear/angular velocity is expressed in each body's
# own local frame.
A3_FORWARD_AMP_ANCHOR_BODY_NAME: Final[str] = "torso_Link"
A3_FORWARD_AMP_BODY_NAMES: Final[tuple[str, ...]] = (
    "pelvis_link",
    "left_hip_roll_Link", "left_knee_Link", "left_ankle_roll_Link",
    "right_hip_roll_Link", "right_knee_Link", "right_ankle_roll_Link",
)
A3_FORWARD_AMP_FRAME_DIM: Final[int] = len(A3_FORWARD_AMP_BODY_NAMES) * 15
A3_FORWARD_AMP_TRANSITION_DIM: Final[int] = (
    A3_AMP_FORWARD_SEQUENCE_LENGTH * A3_FORWARD_AMP_FRAME_DIM
)

def build_forward_amp_frames(*, dof: np.ndarray, root_pos_w: np.ndarray,
                             root_rot_xyzw: np.ndarray, policy_dt: float,
                             kinematics: "A3LowerBodyKinematics") -> np.ndarray:
    """Build lower-body link states for the routed forward prior.

    Each link contributes torso-relative position, rotation first two columns
    (6-D orientation), and local linear/angular velocity.  Features use the
    field-major layout: all positions, all orientations, all linear
    velocities, then all angular velocities. Torso and arms are
    deliberately excluded; pelvis remains visible relative to torso.
    """
    poses = kinematics.link_poses_world(root_pos_w=root_pos_w,
                                        root_rot_xyzw=root_rot_xyzw, dof=dof)
    anchor = poses[A3_FORWARD_AMP_ANCHOR_BODY_NAME]
    anchor_r = anchor[:, :3, :3]
    positions = []
    orientations = []
    linear_velocities = []
    angular_velocities = []
    for name in A3_FORWARD_AMP_BODY_NAMES:
        pose = poses[name]
        # Express each body origin and orientation in the torso frame.
        rel = np.einsum(
            "tji,tj->ti",
            anchor_r,
            pose[:, :3, 3] - anchor[:, :3, 3],
            optimize=True,
        )
        rot = np.einsum("tij,tjk->tik", np.swapaxes(anchor_r, 1, 2), pose[:, :3, :3], optimize=True)
        rot6 = rot[:, :, :2].reshape(len(dof), 6)
        vel_w = np.gradient(pose[:, :3, 3], policy_dt, axis=0, edge_order=1)
        lin_local = np.einsum("tji,tj->ti", pose[:, :3, :3], vel_w, optimize=True)
        # Finite-difference local angular velocity from relative rotations.
        omega = np.zeros((len(dof), 3), dtype=np.float32)
        if len(dof) > 1:
            dr = np.einsum(
                "tji,tjk->tik",
                pose[:-1, :3, :3],
                pose[1:, :3, :3],
                optimize=True,
            )
            omega[1:, 0] = (dr[:, 2, 1] - dr[:, 1, 2]) / (2.0 * policy_dt)
            omega[1:, 1] = (dr[:, 0, 2] - dr[:, 2, 0]) / (2.0 * policy_dt)
            omega[1:, 2] = (dr[:, 1, 0] - dr[:, 0, 1]) / (2.0 * policy_dt)
        positions.append(rel)
        orientations.append(rot6)
        linear_velocities.append(lin_local)
        angular_velocities.append(omega)
    result = np.concatenate(
        tuple(positions + orientations + linear_velocities + angular_velocities),
        axis=-1,
    ).astype(np.float32)
    if result.shape != (len(dof), A3_FORWARD_AMP_FRAME_DIM):
        raise RuntimeError(f"forward AMP feature contract broken: got {result.shape}")
    return np.ascontiguousarray(result)

# Per-frame feature fields.  Root linear velocity is present in the root/body
# frame so the discriminator can distinguish how the lower-body motion
# realizes a commanded velocity, rather than only seeing its joint geometry.
A3_AMP_FEATURE_FIELDS: Final[tuple[str, ...]] = (
    "lower_joint_pos",
    "lower_joint_vel",
    "keypoints_relative_to_pelvis",
    "projected_gravity",
    "torso_height",
    "root_linear_velocity_body",
    "root_angular_velocity_body",
    "left_right_support",
)
A3_AMP_FEATURE_DIM: Final[int] = (
    len(A3_AMP_LOWER_JOINT_NAMES) * 2
    + len(A3_AMP_KEYPOINT_BODY_NAMES) * 3
    + 3
    + 1
    + 3
    + 3
    + 2
)
# ``build_a3_amp_frames`` concatenates 15 q, 15 dq, 27 keypoint
# coordinates, three projected-gravity values, then the scalar torso height.
# Give the height an explicit contract index so cache tooling never infers it
# from a brittle negative offset again.
A3_AMP_TORSO_HEIGHT_INDEX: Final[int] = (
    len(A3_AMP_LOWER_JOINT_NAMES) * 2
    + len(A3_AMP_KEYPOINT_BODY_NAMES) * 3
    + 3
)
A3_AMP_SEQUENCE_DIM: Final[int] = A3_AMP_SEQUENCE_LENGTH * A3_AMP_FEATURE_DIM
A3_AMP_LATERAL_SEQUENCE_DIM: Final[int] = A3_AMP_LATERAL_SEQUENCE_LENGTH * A3_AMP_FEATURE_DIM
A3_AMP_TURN_SEQUENCE_DIM: Final[int] = A3_AMP_TURN_SEQUENCE_LENGTH * A3_AMP_FEATURE_DIM
A3_AMP_SQUAT_SEQUENCE_DIM: Final[int] = A3_AMP_SQUAT_SEQUENCE_LENGTH * A3_AMP_FEATURE_DIM


@dataclass(frozen=True)
class A3AmpCache:
    """Validated, normalized expert cache loaded by the training policy."""

    frames: np.ndarray
    clip_starts: np.ndarray
    clip_lengths: np.ndarray
    clip_splits: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class A3SupportContactGeometry:
    """Foot collision spheres expressed in their ankle-roll link frames.

    A3 T3D0 represents each rigid sole by a small set of collision spheres.
    These are the only points that can physically touch MuJoCo's ground, so
    expert grounding and kinematic support labels must use them rather than
    the ankle-link origins.
    """

    body_names: tuple[str, str]
    centers_b: np.ndarray
    radii: np.ndarray


@dataclass(frozen=True)
class A3MeshSupportContactGeometry:
    """Collision-mesh vertices used by the 024 serial/passive-foot cache.

    ``vertices_b`` stores vertices in each support link's local frame after
    the MJCF geom transform has been applied.  Unlike the rigid A3 sphere
    profile, a mesh has no meaningful radius; support labels operate directly
    on its lowest surface vertices.
    """

    body_names: tuple[str, ...]
    vertices_b: tuple[np.ndarray, ...]
    margins_m: tuple[float, ...]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def amp_schema_dict(
    *,
    sequence_length: int = A3_AMP_SEQUENCE_LENGTH,
    mask_root_linear_velocity: bool = False,
    mask_ankle_joint_state: bool = False,
    asset_profile: str = A3_AMP_DEFAULT_ASSET_PROFILE,
) -> dict[str, object]:
    """Return the complete, hashable AMP feature definition.

    The two mask values are explicit even though this version fixes them to
    the requested velocity-visible / ankle-visible setting.  Caches trained
    under a different setting are rejected instead of silently reusing an
    incompatible normalization/statistics file.
    """
    if sequence_length != A3_AMP_SEQUENCE_LENGTH:
        raise ValueError(
            f"A3 AMP contract requires sequence_length={A3_AMP_SEQUENCE_LENGTH}, "
            f"got {sequence_length}"
        )
    if mask_root_linear_velocity:
        raise ValueError("A3 AMP v1 fixes mask_root_linear_velocity=false")
    if mask_ankle_joint_state:
        raise ValueError("A3 AMP v1 fixes mask_ankle_joint_state=false")
    if asset_profile not in A3_AMP_ASSET_PROFILES:
        raise ValueError(f"Unknown A3 AMP asset profile {asset_profile!r}")
    support_names = (
        A3_AMP_SUPPORT_BODY_NAMES
        if asset_profile == A3_AMP_DEFAULT_ASSET_PROFILE
        else A3_AMP_024_SERIAL_SUPPORT_BODY_NAMES
    )
    result = {
        "version": A3_AMP_CONTRACT_VERSION,
        "sequence_length": sequence_length,
        "feature_dim": A3_AMP_FEATURE_DIM,
        "feature_fields": A3_AMP_FEATURE_FIELDS,
        "forward_sequence_length": A3_AMP_FORWARD_SEQUENCE_LENGTH,
        "backward_sequence_length": A3_AMP_BACKWARD_SEQUENCE_LENGTH,
        "lateral_sequence_length": A3_AMP_LATERAL_SEQUENCE_LENGTH,
        "turn_sequence_length": A3_AMP_TURN_SEQUENCE_LENGTH,
        "squat_sequence_length": A3_AMP_SQUAT_SEQUENCE_LENGTH,
        "condition_fields": A3_AMP_CONDITION_FIELDS,
        "condition_mean": A3_AMP_CONDITION_MEAN,
        "condition_std": A3_AMP_CONDITION_STD,
        "planar_velocity_label_limit": A3_AMP_PLANAR_VELOCITY_LIMIT,
        "mode_names": A3_AMP_MODE_NAMES,
        "active_mode_count": A3_AMP_ACTIVE_MODE_COUNT,
        "mode_count": A3_AMP_MODE_COUNT,
        "lower_joint_names": A3_AMP_LOWER_JOINT_NAMES,
        "keypoint_body_names": A3_AMP_KEYPOINT_BODY_NAMES,
        "support_body_names": support_names,
        "forward_anchor_body_name": A3_FORWARD_AMP_ANCHOR_BODY_NAME,
        "forward_body_names": A3_FORWARD_AMP_BODY_NAMES,
        "forward_frame_dim": A3_FORWARD_AMP_FRAME_DIM,
        "forward_feature_layout": "field_major_pos_ori6_linvel_local_angvel_local",
        "mask_root_linear_velocity": mask_root_linear_velocity,
        "mask_ankle_joint_state": mask_ankle_joint_state,
        "sequence_order": "oldest_to_newest",
        "expert_contact_provenance": "kinematic_support_phase",
        "expert_root_alignment": "per_frame_lowest_foot_collision_surface_to_ground",
    }
    # Preserve the established default schema hash for existing A3 caches;
    # the serial profile gets an explicit discriminator so it cannot be
    # confused with a rigid-A3 cache.
    if asset_profile == A3_AMP_024_SERIAL_ASSET_PROFILE:
        result.update({
            "asset_profile": asset_profile,
            "support_body_side_indices": (0, 0, 0, 1, 1, 1),
            "support_collision_shape": "convex_mesh_vertices",
            "passive_joint_positions_rad": A3_AMP_024_SERIAL_PASSIVE_JOINT_POSITIONS,
        })
    elif asset_profile == A3_AMP_ULTRA_T3D0_ASSET_PROFILE:
        result.update({
            "asset_profile": asset_profile,
            "runtime_head_pitch_joint": "ultra_head_pitch_joint",
            "passive_hand_joints": "excluded_from_policy_amp_and_replay",
        })
    return result


def amp_schema_hash(**kwargs) -> str:
    return hashlib.sha256(_canonical_json(amp_schema_dict(**kwargs)).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def a3_asset_fingerprint(
    *, mjcf_path: str | Path, urdf_path: str | Path,
    asset_profile: str = A3_AMP_DEFAULT_ASSET_PROFILE,
) -> str:
    """Fingerprint the two A3 descriptions used by cache FK and mjlab.

    The path itself is deliberately excluded.  Expert caches are intended to
    travel with a run between machines/checkouts, where a byte-identical
    bundled asset normally lives below a different absolute project root.
    Names plus content hashes retain a useful provenance record without
    making cache validity depend on that incidental location.
    """
    paths = (Path(mjcf_path).expanduser(), Path(urdf_path).expanduser())
    if asset_profile not in A3_AMP_ASSET_PROFILES:
        raise ValueError(f"Unknown A3 AMP asset profile {asset_profile!r}")
    payload = {
        "mjcf": {"name": paths[0].name, "sha256": sha256_file(paths[0])},
        "urdf": {"name": paths[1].name, "sha256": sha256_file(paths[1])},
    }
    if asset_profile != A3_AMP_DEFAULT_ASSET_PROFILE:
        payload["asset_profile"] = asset_profile
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def canonicalize_a3_dof(payload: Mapping[str, object]) -> np.ndarray:
    """Return an A3 PKL's 31 DoFs in :data:`A3_31DOF_JOINT_NAMES` order."""
    if "dof" not in payload:
        raise ValueError("A3 AMP clip is missing dof")
    raw = np.asarray(payload["dof"], dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != len(A3_31DOF_JOINT_NAMES):
        raise ValueError(f"A3 AMP requires dof [T, 31], got {raw.shape}")
    names = tuple(str(name) for name in payload.get("joint_names", A3_T3D0_MJCF_JOINT_NAMES))
    if len(names) != len(A3_31DOF_JOINT_NAMES) or len(set(names)) != len(names):
        raise ValueError("A3 AMP clip has invalid joint_names")
    missing = [name for name in A3_31DOF_JOINT_NAMES if name not in names]
    if missing:
        raise ValueError(f"A3 AMP clip is missing joints: {missing}")
    indices = {name: index for index, name in enumerate(names)}
    return np.ascontiguousarray(raw[:, [indices[name] for name in A3_31DOF_JOINT_NAMES]])


def apply_a3_joint_offsets(
    dof: np.ndarray,
    joint_offsets_rad: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return a copy of canonical A3 DoFs with explicit constant offsets.

    A hand-reviewed expert may require a small retargeting correction.  It is
    part of the manifest provenance, never written back to the source PKL.
    Applying it before root grounding is essential: changing an ankle pitch
    changes the physical sole height used by the AMP support labels.
    """
    if dof.ndim != 2 or dof.shape[1] != len(A3_31DOF_JOINT_NAMES):
        raise ValueError(f"A3 joint offsets require canonical dof [T, 31], got {dof.shape}")
    if not joint_offsets_rad:
        return dof
    indices = {name: index for index, name in enumerate(A3_31DOF_JOINT_NAMES)}
    result = np.array(dof, dtype=np.float32, copy=True)
    for name, value in joint_offsets_rad.items():
        if name not in indices:
            raise ValueError(f"A3 AMP joint offset names an unknown joint: {name}")
        offset = float(value)
        if not math.isfinite(offset):
            raise ValueError(f"A3 AMP joint offset for {name} must be finite, got {value!r}")
        result[:, indices[name]] += offset
    return result


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    result = np.empty((*rpy.shape[:-1], 3, 3), dtype=np.float32)
    result[..., 0, 0] = cy * cp
    result[..., 0, 1] = cy * sp * sr - sy * cr
    result[..., 0, 2] = cy * sp * cr + sy * sr
    result[..., 1, 0] = sy * cp
    result[..., 1, 1] = sy * sp * sr + cy * cr
    result[..., 1, 2] = sy * sp * cr - cy * sr
    result[..., 2, 0] = -sp
    result[..., 2, 1] = cp * sr
    result[..., 2, 2] = cp * cr
    return result


def _axis_angle_matrix(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-8)
    x, y, z = axis
    c, s = np.cos(angles), np.sin(angles)
    one_c = 1.0 - c
    result = np.empty((*angles.shape, 3, 3), dtype=np.float32)
    result[..., 0, 0] = c + x * x * one_c
    result[..., 0, 1] = x * y * one_c - z * s
    result[..., 0, 2] = x * z * one_c + y * s
    result[..., 1, 0] = y * x * one_c + z * s
    result[..., 1, 1] = c + y * y * one_c
    result[..., 1, 2] = y * z * one_c - x * s
    result[..., 2, 0] = z * x * one_c - y * s
    result[..., 2, 1] = z * y * one_c + x * s
    result[..., 2, 2] = c + z * z * one_c
    return result


def _xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(~np.isfinite(quat)) or np.any(norm < 1.0e-6):
        raise ValueError("A3 AMP root_rot is non-finite or zero-norm")
    x, y, z, w = np.moveaxis(quat / norm, -1, 0)
    result = np.empty((*quat.shape[:-1], 3, 3), dtype=np.float32)
    result[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    result[..., 0, 1] = 2.0 * (x * y - z * w)
    result[..., 0, 2] = 2.0 * (x * z + y * w)
    result[..., 1, 0] = 2.0 * (x * y + z * w)
    result[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    result[..., 1, 2] = 2.0 * (y * z - x * w)
    result[..., 2, 0] = 2.0 * (x * z - y * w)
    result[..., 2, 1] = 2.0 * (y * z + x * w)
    result[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return result


def _matrix_to_xyzw(matrix: np.ndarray) -> np.ndarray:
    """Stable enough vectorized 3x3 rotation-matrix -> xyzw conversion."""
    matrix = np.asarray(matrix, dtype=np.float32)
    flat = matrix.reshape(-1, 3, 3)
    result = np.empty((len(flat), 4), dtype=np.float32)
    for index, value in enumerate(flat):
        trace = float(np.trace(value))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            result[index] = (
                (value[2, 1] - value[1, 2]) / scale,
                (value[0, 2] - value[2, 0]) / scale,
                (value[1, 0] - value[0, 1]) / scale,
                0.25 * scale,
            )
        else:
            diag = np.diag(value)
            axis = int(np.argmax(diag))
            if axis == 0:
                scale = math.sqrt(max(1.0 + value[0, 0] - value[1, 1] - value[2, 2], 1.0e-12)) * 2.0
                result[index] = (
                    0.25 * scale,
                    (value[0, 1] + value[1, 0]) / scale,
                    (value[0, 2] + value[2, 0]) / scale,
                    (value[2, 1] - value[1, 2]) / scale,
                )
            elif axis == 1:
                scale = math.sqrt(max(1.0 + value[1, 1] - value[0, 0] - value[2, 2], 1.0e-12)) * 2.0
                result[index] = (
                    (value[0, 1] + value[1, 0]) / scale,
                    0.25 * scale,
                    (value[1, 2] + value[2, 1]) / scale,
                    (value[0, 2] - value[2, 0]) / scale,
                )
            else:
                scale = math.sqrt(max(1.0 + value[2, 2] - value[0, 0] - value[1, 1], 1.0e-12)) * 2.0
                result[index] = (
                    (value[0, 2] + value[2, 0]) / scale,
                    (value[1, 2] + value[2, 1]) / scale,
                    0.25 * scale,
                    (value[1, 0] - value[0, 1]) / scale,
                )
    result /= np.linalg.norm(result, axis=-1, keepdims=True).clip(1.0e-8)
    return result.reshape(*matrix.shape[:-2], 4)


def _nlerp_xyzw(lower: np.ndarray, upper: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    upper = np.where(np.sum(lower * upper, axis=-1, keepdims=True) < 0.0, -upper, upper)
    result = lower + (upper - lower) * alpha[..., None]
    return result / np.linalg.norm(result, axis=-1, keepdims=True).clip(1.0e-8)


def resample_a3_clip(
    *,
    dof: np.ndarray,
    root_pos_w: np.ndarray,
    root_rot_xyzw: np.ndarray,
    source_fps: float,
    policy_dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample a source clip to the policy clock before making AMP frames."""
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"A3 AMP clip has invalid fps={source_fps}")
    if policy_dt <= 0.0:
        raise ValueError("policy_dt must be positive")
    if dof.ndim != 2 or len(dof) < A3_AMP_SEQUENCE_LENGTH:
        raise ValueError(f"A3 AMP clip needs at least {A3_AMP_SEQUENCE_LENGTH} source frames")
    if root_pos_w.shape != (len(dof), 3) or root_rot_xyzw.shape != (len(dof), 4):
        raise ValueError("A3 AMP root signals are not aligned to dof")
    duration = (len(dof) - 1) / source_fps
    target_times = np.arange(0.0, duration + policy_dt * 0.25, policy_dt, dtype=np.float32)
    if len(target_times) < A3_AMP_SEQUENCE_LENGTH:
        raise ValueError(
            f"A3 AMP clip is shorter than one {A3_AMP_SEQUENCE_LENGTH}-frame policy sequence"
        )
    source_times = np.arange(len(dof), dtype=np.float32) / float(source_fps)
    upper = np.searchsorted(source_times, target_times, side="right").clip(1, len(dof) - 1)
    lower = upper - 1
    alpha = (target_times - source_times[lower]) / (source_times[upper] - source_times[lower]).clip(1.0e-8)
    dof_out = dof[lower] + (dof[upper] - dof[lower]) * alpha[:, None]
    root_pos_out = root_pos_w[lower] + (root_pos_w[upper] - root_pos_w[lower]) * alpha[:, None]
    root_rot_out = _nlerp_xyzw(root_rot_xyzw[lower], root_rot_xyzw[upper], alpha)
    return (
        np.ascontiguousarray(dof_out, dtype=np.float32),
        np.ascontiguousarray(root_pos_out, dtype=np.float32),
        np.ascontiguousarray(root_rot_out, dtype=np.float32),
    )


@dataclass(frozen=True)
class _UrdfJoint:
    name: str
    parent: str
    child: str
    joint_type: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


class A3LowerBodyKinematics:
    """Minimal vectorized URDF FK for A3 AMP expert keypoints.

    It intentionally uses the same bundled A3 URDF that defines the current
    rigid A3 mjlab task.  This is not a general robot IK library; refusing a
    missing required body is preferable to silently training a cache from an
    incompatible retargeted model.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        passive_joint_positions: Mapping[str, float] | None = None,
        replay_joint_aliases: Mapping[str, str] | None = None,
    ):
        source = Path(urdf_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"A3 AMP URDF not found: {source}")
        joints: list[_UrdfJoint] = []
        for node in et.parse(source).getroot().findall("joint"):
            parent, child = node.find("parent"), node.find("child")
            if parent is None or child is None:
                continue
            origin, axis = node.find("origin"), node.find("axis")
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ", dtype=np.float32)
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ", dtype=np.float32)
            joint_axis = np.fromstring(axis.attrib.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" ", dtype=np.float32)
            if xyz.shape != (3,) or rpy.shape != (3,) or joint_axis.shape != (3,):
                raise ValueError(f"A3 AMP malformed URDF joint {node.attrib.get('name')}")
            joints.append(_UrdfJoint(
                name=node.attrib["name"], parent=parent.attrib["link"], child=child.attrib["link"],
                joint_type=node.attrib.get("type", "fixed"), xyz=xyz, rpy=rpy, axis=joint_axis,
            ))
        self._joints = tuple(joints)
        self._joint_by_name = {joint.name: joint for joint in joints}
        self._passive_joint_positions = {
            str(name): float(value)
            for name, value in (passive_joint_positions or {}).items()
        }
        self._replay_joint_aliases = {
            str(runtime_name): str(replay_name)
            for runtime_name, replay_name in (replay_joint_aliases or {}).items()
        }
        for runtime_name, replay_name in self._replay_joint_aliases.items():
            if runtime_name not in self._joint_by_name:
                raise ValueError(
                    f"A3 AMP replay joint alias has unknown runtime joint {runtime_name!r}"
                )
            if replay_name not in A3_31DOF_JOINT_NAMES:
                raise ValueError(
                    f"A3 AMP replay joint alias has unknown replay joint {replay_name!r}"
                )
        for name, value in self._passive_joint_positions.items():
            if name not in self._joint_by_name or not math.isfinite(value):
                raise ValueError(f"A3 AMP passive joint default is invalid: {name}={value!r}")
        self._required = (
            set(A3_AMP_KEYPOINT_BODY_NAMES)
            | set(A3_FORWARD_AMP_BODY_NAMES)
            | {A3_FORWARD_AMP_ANCHOR_BODY_NAME}
        )
        known = {"pelvis_link"} | {joint.child for joint in joints}
        missing = sorted(self._required.difference(known))
        if missing:
            raise ValueError(f"A3 AMP URDF is missing keypoint bodies: {missing}")

    @staticmethod
    def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        shape = rotation.shape[:-2]
        result = np.zeros((*shape, 4, 4), dtype=np.float32)
        result[..., :3, :3] = rotation
        result[..., :3, 3] = translation
        result[..., 3, 3] = 1.0
        return result

    def link_poses_world(
        self,
        *,
        root_pos_w: np.ndarray,
        root_rot_xyzw: np.ndarray,
        dof: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if dof.ndim != 2 or dof.shape[1] != len(A3_31DOF_JOINT_NAMES):
            raise ValueError(f"A3 AMP FK needs dof [T, 31], got {dof.shape}")
        frames = len(dof)
        root_rotation = _xyzw_to_matrix(root_rot_xyzw)
        poses: dict[str, np.ndarray] = {
            "pelvis_link": self._transform(root_rotation, root_pos_w),
        }
        q_by_name = {name: dof[:, index] for index, name in enumerate(A3_31DOF_JOINT_NAMES)}
        unresolved = list(self._joints)
        while unresolved:
            next_unresolved: list[_UrdfJoint] = []
            made_progress = False
            for joint in unresolved:
                parent_pose = poses.get(joint.parent)
                if parent_pose is None:
                    next_unresolved.append(joint)
                    continue
                origin = self._transform(
                    np.broadcast_to(_rpy_matrix(joint.rpy), (frames, 3, 3)),
                    np.broadcast_to(joint.xyz, (frames, 3)),
                )
                if joint.joint_type in {"revolute", "continuous"}:
                    if joint.name in q_by_name:
                        angle = q_by_name[joint.name]
                    elif joint.name in self._replay_joint_aliases:
                        angle = q_by_name[self._replay_joint_aliases[joint.name]]
                    elif joint.name in self._passive_joint_positions:
                        angle = np.full(
                            frames, self._passive_joint_positions[joint.name], dtype=np.float32
                        )
                    else:
                        raise ValueError(f"A3 AMP FK needs joint {joint.name}")
                    motion = self._transform(
                        _axis_angle_matrix(joint.axis, angle),
                        np.zeros((frames, 3), dtype=np.float32),
                    )
                    poses[joint.child] = parent_pose @ origin @ motion
                else:
                    poses[joint.child] = parent_pose @ origin
                made_progress = True
            if not made_progress:
                names = [joint.name for joint in next_unresolved]
                raise ValueError(f"A3 AMP FK could not resolve URDF tree: {names}")
            unresolved = next_unresolved
        return poses

    def keypoints_world(
        self,
        *,
        root_pos_w: np.ndarray,
        root_rot_xyzw: np.ndarray,
        dof: np.ndarray,
    ) -> np.ndarray:
        poses = self.link_poses_world(
            root_pos_w=root_pos_w, root_rot_xyzw=root_rot_xyzw, dof=dof
        )
        return np.stack([poses[name][..., :3, 3] for name in A3_AMP_KEYPOINT_BODY_NAMES], axis=1)

    def support_contact_points_world(
        self,
        *,
        root_pos_w: np.ndarray,
        root_rot_xyzw: np.ndarray,
        dof: np.ndarray,
        geometry: A3SupportContactGeometry,
    ) -> np.ndarray:
        """Return contact-sphere centres with shape ``[T, 2, P, 3]``."""
        if geometry.body_names != A3_AMP_SUPPORT_BODY_NAMES:
            raise ValueError(
                "A3 AMP support geometry body names do not match the versioned "
                f"contract: expected {A3_AMP_SUPPORT_BODY_NAMES}, got {geometry.body_names}"
            )
        if geometry.centers_b.ndim != 3 or geometry.centers_b.shape[:2] != (2, 13) or geometry.centers_b.shape[-1] != 3:
            raise ValueError(
                "A3 AMP requires exactly 13 local collision centres per foot; "
                f"got {geometry.centers_b.shape}"
            )
        if geometry.radii.shape != (2, 13):
            raise ValueError(f"A3 AMP support radii must be [2, 13], got {geometry.radii.shape}")
        poses = self.link_poses_world(
            root_pos_w=root_pos_w, root_rot_xyzw=root_rot_xyzw, dof=dof
        )
        points = []
        for index, name in enumerate(geometry.body_names):
            pose = poses.get(name)
            if pose is None:
                raise ValueError(f"A3 AMP URDF is missing support body {name}")
            points.append(
                np.einsum(
                    "tij,pj->tpi", pose[..., :3, :3], geometry.centers_b[index], optimize=True
                ) + pose[:, None, :3, 3]
            )
        return np.stack(points, axis=1).astype(np.float32)

    def mesh_support_points_world(
        self,
        *,
        root_pos_w: np.ndarray,
        root_rot_xyzw: np.ndarray,
        dof: np.ndarray,
        geometry: A3MeshSupportContactGeometry,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return packed 024 mesh support vertices and margins.

        The 024 rear/forefoot/toe meshes have different vertex counts.  The
        cache only needs their lowest surfaces, so each side is packed into a
        fixed-width array after concatenating the three link-local meshes.
        """
        if geometry.body_names != A3_AMP_024_SERIAL_SUPPORT_BODY_NAMES:
            raise ValueError(
                "A3 024 AMP mesh support body names do not match the versioned "
                f"contract: expected {A3_AMP_024_SERIAL_SUPPORT_BODY_NAMES}, "
                f"got {geometry.body_names}"
            )
        if (
            len(geometry.vertices_b) != len(geometry.body_names)
            or len(geometry.margins_m) != len(geometry.body_names)
        ):
            raise ValueError("A3 024 AMP mesh support geometry is incomplete")
        poses = self.link_poses_world(
            root_pos_w=root_pos_w, root_rot_xyzw=root_rot_xyzw, dof=dof
        )
        side_points: list[np.ndarray] = []
        side_margins: list[np.ndarray] = []
        for side in range(2):
            pieces: list[np.ndarray] = []
            margins: list[np.ndarray] = []
            for index in range(side * 3, side * 3 + 3):
                name = geometry.body_names[index]
                vertices = geometry.vertices_b[index]
                pose = poses.get(name)
                if pose is None:
                    raise ValueError(f"A3 024 AMP URDF is missing support body {name}")
                if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
                    raise ValueError(f"A3 024 AMP mesh vertices for {name} are invalid")
                pieces.append(
                    np.einsum(
                        "tij,pj->tpi", pose[..., :3, :3], vertices, optimize=True
                    ) + pose[:, None, :3, 3]
                )
                margins.append(
                    np.full(len(vertices), geometry.margins_m[index], dtype=np.float32)
                )
            side_points.append(np.concatenate(pieces, axis=1))
            side_margins.append(np.concatenate(margins))
        if side_points[0].shape[1] != side_points[1].shape[1]:
            raise ValueError("A3 024 AMP requires symmetric left/right mesh vertex counts")
        return (
            np.stack(side_points, axis=1).astype(np.float32),
            np.stack(side_margins).astype(np.float32),
        )


def load_a3_support_contact_geometry(mjcf_path: str | Path) -> A3SupportContactGeometry:
    """Load the T3D0 foot collision spheres from the versioned MJCF asset."""
    source = Path(mjcf_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"A3 AMP MJCF not found: {source}")
    root = et.parse(source).getroot()
    centers_by_foot: list[np.ndarray] = []
    radii_by_foot: list[np.ndarray] = []
    for side, body_name in zip(("left", "right"), A3_AMP_SUPPORT_BODY_NAMES, strict=True):
        body = next((node for node in root.findall(".//body") if node.attrib.get("name") == body_name), None)
        if body is None:
            raise ValueError(f"A3 AMP MJCF is missing support body {body_name}")
        spheres = []
        for geom in body.findall("geom"):
            name = geom.attrib.get("name", "")
            if not (name.startswith(f"{side}_foot") and name.endswith("_collision")):
                continue
            if geom.attrib.get("type") != "sphere":
                raise ValueError(f"A3 AMP expects {name} to be a sphere collision geom")
            position = np.fromstring(geom.attrib.get("pos", "0 0 0"), sep=" ", dtype=np.float32)
            size = np.fromstring(geom.attrib.get("size", ""), sep=" ", dtype=np.float32)
            if position.shape != (3,) or size.shape != (1,) or size[0] <= 0.0:
                raise ValueError(f"A3 AMP malformed foot collision geom {name}")
            spheres.append((name, position, float(size[0])))
        spheres.sort(key=lambda item: item[0])
        if len(spheres) != 13:
            raise ValueError(
                f"A3 AMP requires 13 {side} foot collision spheres in {source}, got {len(spheres)}"
            )
        centers_by_foot.append(np.stack([item[1] for item in spheres]))
        radii_by_foot.append(np.asarray([item[2] for item in spheres], dtype=np.float32))
    return A3SupportContactGeometry(
        body_names=A3_AMP_SUPPORT_BODY_NAMES,
        centers_b=np.stack(centers_by_foot).astype(np.float32),
        radii=np.stack(radii_by_foot).astype(np.float32),
    )


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert MuJoCo's scalar-first unit quaternion to a 3x3 matrix."""
    w, x, y, z = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm((w, x, y, z)))
    if not math.isfinite(norm) or norm < 1.0e-8:
        raise ValueError("A3 024 AMP mesh geom has an invalid quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray((
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    ), dtype=np.float32)


def load_a3_024_serial_mesh_support_geometry(
    mjcf_path: str | Path,
) -> A3MeshSupportContactGeometry:
    """Read the actual six 024 passive-foot collision meshes from MuJoCo.

    This deliberately compiles the supplied MJCF rather than parsing a mesh
    filename convention.  It therefore includes all mesh scale, rotation and
    local-geom offsets that the training model actually uses.
    """
    source = Path(mjcf_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"A3 024 AMP MJCF not found: {source}")
    try:
        import mujoco
    except ModuleNotFoundError as error:  # pragma: no cover - environment issue
        raise ModuleNotFoundError(
            "Building the A3 024 AMP cache requires the mujoco Python package"
        ) from error
    model = mujoco.MjModel.from_xml_path(str(source))
    vertices_by_body: list[np.ndarray] = []
    margins_by_body: list[float] = []
    for body_name in A3_AMP_024_SERIAL_SUPPORT_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"A3 024 AMP MJCF is missing support body {body_name}")
        pieces: list[np.ndarray] = []
        piece_margin: float | None = None
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if not int(model.geom_contype[geom_id]) or not int(model.geom_conaffinity[geom_id]):
                continue
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                raise ValueError(
                    f"A3 024 AMP support geom {geom_id} on {body_name} must be a mesh"
                )
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id < 0:
                raise ValueError(f"A3 024 AMP support geom {geom_id} has no mesh data")
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float32)
            if vertices.shape != (count, 3) or count < 3:
                raise ValueError(f"A3 024 AMP mesh support geom {geom_id} has invalid vertices")
            rotation = _quat_wxyz_to_matrix(model.geom_quat[geom_id])
            pieces.append(vertices @ rotation.T + np.asarray(model.geom_pos[geom_id], dtype=np.float32))
            piece_margin = float(model.geom_margin[geom_id])
        if not pieces:
            raise ValueError(f"A3 024 AMP support body {body_name} has no collidable mesh")
        if len(pieces) != 1:
            raise ValueError(
                f"A3 024 AMP requires one collision mesh on {body_name}, got {len(pieces)}"
            )
        # The one-mesh invariant above makes this unambiguous and avoids
        # accidentally retaining the last unrelated geom's margin if the
        # source asset is edited in the future.
        if piece_margin is None or not math.isfinite(piece_margin) or piece_margin < 0.0:
            raise ValueError(f"A3 024 AMP support mesh {body_name} has an invalid contact margin")
        vertices_by_body.append(np.concatenate(pieces, axis=0).astype(np.float32))
        margins_by_body.append(piece_margin)
    if sum(len(vertices_by_body[index]) for index in range(3)) != sum(
        len(vertices_by_body[index]) for index in range(3, 6)
    ):
        raise ValueError("A3 024 AMP requires symmetric left/right mesh support topology")
    return A3MeshSupportContactGeometry(
        body_names=A3_AMP_024_SERIAL_SUPPORT_BODY_NAMES,
        vertices_b=tuple(vertices_by_body),
        margins_m=tuple(margins_by_body),
    )


def _angular_velocity_body(root_rotation: np.ndarray, dt: float) -> np.ndarray:
    """First-order body angular velocity from consecutive world rotations."""
    if len(root_rotation) < 2:
        raise ValueError("A3 AMP needs two root rotations")
    relative = np.swapaxes(root_rotation[:-1], -1, -2) @ root_rotation[1:]
    skew = np.stack(
        (relative[:, 2, 1] - relative[:, 1, 2], relative[:, 0, 2] - relative[:, 2, 0], relative[:, 1, 0] - relative[:, 0, 1]),
        axis=-1,
    )
    cosine = ((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5).clip(-1.0, 1.0)
    angle = np.arccos(cosine)
    sine = np.sin(angle)
    scale = np.full_like(angle, 0.5 / dt)
    nonzero = np.abs(sine) > 1.0e-5
    scale[nonzero] = angle[nonzero] / (2.0 * sine[nonzero] * dt)
    omega = skew * scale[:, None]
    return np.concatenate((omega, omega[-1:]), axis=0).astype(np.float32)


def a3_amp_mode_from_condition(condition: np.ndarray) -> np.ndarray:
    """Route commands to a direction-specific AMP prior or neutral silence.

    The four command components always remain discriminator conditions.
    At low speed, only commands below 1.0 m select the squat prior;
    stationary higher-height commands select neutral, whose AMP coverage is
    exactly zero.  Otherwise the dominant magnitude among ``vx``, ``vy``
    and ``wz / 1.57 * 0.55`` selects direction and sign.
    """
    condition = np.asarray(condition, dtype=np.float32)
    if condition.shape[-1] != A3_AMP_CONDITION_DIM:
        raise ValueError(
            f"A3 AMP condition needs [..., {A3_AMP_CONDITION_DIM}], got {condition.shape}"
        )
    vx, vy, wz, height = np.moveaxis(condition, -1, 0)
    mode = np.zeros(condition.shape[:-1], dtype=np.int64)
    scaled_wz = wz / 1.57 * 0.55
    speed = np.sqrt(np.square(vx) + np.square(vy) + np.square(scaled_wz))
    dominant = np.stack((np.abs(vx), np.abs(vy), np.abs(scaled_wz)), axis=-1).argmax(axis=-1)
    mode = np.where(dominant == 0, np.where(vx >= 0.0, 0, 1), mode)
    mode = np.where(dominant == 1, np.where(vy >= 0.0, 2, 3), mode)
    mode = np.where(dominant == 2, np.where(wz >= 0.0, 4, 5), mode)
    stationary = speed < A3_AMP_SQUAT_SPEED_THRESHOLD
    mode = np.where(
        stationary,
        np.where(height < A3_AMP_SQUAT_HEIGHT_THRESHOLD, A3_AMP_SQUAT_MODE, A3_AMP_NEUTRAL_MODE),
        mode,
    )
    return mode


def build_a3_amp_conditions(
    *,
    root_pos_w: np.ndarray,
    root_rot_xyzw: np.ndarray,
    torso_height: np.ndarray,
    policy_dt: float,
) -> np.ndarray:
    """Infer expert-compatible lower-body command labels from kinematics.

    AMASS has no independently sampled LBC label.  The conditional prior
    therefore uses the part of the task command observable in a kinematic
    clip: root-local planar velocity, yaw rate and torso height.  Deliberately
    exclude roll/pitch/torso-yaw targets: those are independently sampled
    posture commands in DEPSIC and cannot be reconstructed from this small
    expert corpus without fabricating labels.
    """
    if root_pos_w.shape != (len(root_rot_xyzw), 3):
        raise ValueError("A3 AMP conditions require aligned root position/rotation")
    if torso_height.shape != (len(root_pos_w), 1):
        raise ValueError("A3 AMP conditions require torso_height [T, 1]")
    rotation = _xyzw_to_matrix(root_rot_xyzw)
    velocity_w = np.gradient(root_pos_w, policy_dt, axis=0, edge_order=1)
    yaw = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    velocity_b_x = cosine * velocity_w[:, 0] + sine * velocity_w[:, 1]
    velocity_b_y = -sine * velocity_w[:, 0] + cosine * velocity_w[:, 1]
    omega_b = _angular_velocity_body(rotation, policy_dt)
    result = np.stack((velocity_b_x, velocity_b_y, omega_b[:, 2]), axis=-1)
    result = np.concatenate((result, torso_height), axis=-1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("A3 AMP conditions contain non-finite values")
    return result


def kinematic_support_phase(
    foot_points_w: np.ndarray,
    *,
    dt: float,
    radii: np.ndarray | None = None,
    on_height: float = 0.04,
    off_height: float = 0.06,
    on_tangent_speed: float = 1.0,
    off_tangent_speed: float = 1.5,
    vertical_speed: float = 1.0,
) -> np.ndarray:
    """Infer a binary support phase from kinematic foot trajectories.

    ``foot_points_w`` may be the legacy one-point-per-foot tensor ``[T, 2, 3]``
    or physical sole collision centres ``[T, 2, P, 3]``.  With collision
    points, the lowest collision *surface* is used per foot and frame.  The
    ground is z=0 after :func:`align_a3_root_to_ground`; this is still an
    explicitly kinematic label, never a fabricated physical force.
    """
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("A3 AMP support dt must be finite and positive")
    # Differentiate material points before selecting a contact point. Taking
    # gradients after argmin switches between heel/toe spheres manufactures
    # foot velocity even for an almost stationary, flat sole.
    point_velocity = np.gradient(foot_points_w, dt, axis=0, edge_order=1)
    if foot_points_w.ndim == 3:
        if foot_points_w.shape[1:] != (2, 3):
            raise ValueError(f"A3 AMP support labels need [T, 2, 3], got {foot_points_w.shape}")
        feet_pos_w = foot_points_w
        velocity = point_velocity
    elif foot_points_w.ndim == 4:
        if foot_points_w.shape[1] != 2 or foot_points_w.shape[-1] != 3:
            raise ValueError(
                "A3 AMP support collision points need [T, 2, P, 3], "
                f"got {foot_points_w.shape}"
            )
        if radii is None:
            # Mesh support points are already surface vertices.  Select the
            # lowest vertex per foot without applying a sphere radius.
            surface_height = foot_points_w[..., 2]
            lowest = surface_height.argmin(axis=-1)
            feet_pos_w = np.take_along_axis(
                foot_points_w, lowest[..., None, None], axis=2
            ).squeeze(axis=2).copy()
        else:
            if radii.shape != foot_points_w.shape[1:3]:
                raise ValueError(
                    "A3 AMP support collision radii need [2, P] matching points, "
                    f"got {radii.shape}"
                )
            surface_height = foot_points_w[..., 2] - radii[None]
            lowest = surface_height.argmin(axis=-1)
            feet_pos_w = np.take_along_axis(
                foot_points_w, lowest[..., None, None], axis=2
            ).squeeze(axis=2).copy()
            feet_pos_w[..., 2] = np.take_along_axis(
                surface_height, lowest[..., None], axis=-1
            ).squeeze(axis=-1)
        velocity = np.take_along_axis(
            point_velocity, lowest[..., None, None], axis=2
        ).squeeze(axis=2)
    else:
        raise ValueError(
            "A3 AMP support labels need [T, 2, 3] or [T, 2, P, 3], "
            f"got {foot_points_w.shape}"
        )
    if not (0.0 <= on_height <= off_height):
        raise ValueError("A3 AMP support height thresholds must satisfy 0 <= on <= off")
    tangent_speed = np.linalg.norm(velocity[..., :2], axis=-1)
    vertical = np.abs(velocity[..., 2])
    height = feet_pos_w[..., 2]
    on = (height <= on_height) & (tangent_speed <= on_tangent_speed) & (vertical <= vertical_speed)
    hold = (height <= off_height) & (tangent_speed <= off_tangent_speed) & (vertical <= vertical_speed * 1.5)
    result = np.zeros_like(on, dtype=np.float32)
    state = on[0].copy()
    result[0] = state
    for index in range(1, len(result)):
        state = np.where(state, hold[index], on[index])
        result[index] = state
    return result


def align_a3_root_to_ground(
    *,
    dof: np.ndarray,
    root_pos_w: np.ndarray,
    root_rot_xyzw: np.ndarray,
    kinematics: A3LowerBodyKinematics,
    support_geometry: A3SupportContactGeometry | A3MeshSupportContactGeometry,
) -> tuple[np.ndarray, np.ndarray]:
    """Ground each kinematic frame at its lowest physical sole surface.

    The source clip is replay-only kinematics, not a physics rollout.  Follow
    the requested convention exactly: at every frame the lower of the two
    soles touches ``z=0``.  This intentionally removes root-z bounce from
    the AMP height feature; the discriminated gait remains in joint motion,
    pelvis-relative keypoints, gravity, angular velocity, and support phase.
    """
    if root_pos_w.shape != (len(dof), 3):
        raise ValueError(f"A3 AMP grounding needs root_pos [T, 3], got {root_pos_w.shape}")
    if isinstance(support_geometry, A3MeshSupportContactGeometry):
        contact_points, contact_margins = kinematics.mesh_support_points_world(
            root_pos_w=root_pos_w,
            root_rot_xyzw=root_rot_xyzw,
            dof=dof,
            geometry=support_geometry,
        )
        floor_offsets = (
            contact_points[..., 2] - contact_margins[None]
        ).min(axis=(1, 2))
    else:
        contact_points = kinematics.support_contact_points_world(
            root_pos_w=root_pos_w,
            root_rot_xyzw=root_rot_xyzw,
            dof=dof,
            geometry=support_geometry,
        )
        floor_offsets = (contact_points[..., 2] - support_geometry.radii[None]).min(axis=(1, 2))
    if not np.isfinite(floor_offsets).all():
        raise ValueError("A3 AMP grounding found a non-finite foot collision height")
    aligned_root = np.array(root_pos_w, dtype=np.float32, copy=True)
    aligned_root[:, 2] -= floor_offsets
    return aligned_root, floor_offsets.astype(np.float32)


def build_a3_amp_frames(
    *,
    dof: np.ndarray,
    root_pos_w: np.ndarray,
    root_rot_xyzw: np.ndarray,
    policy_dt: float,
    kinematics: A3LowerBodyKinematics,
    support_geometry: A3SupportContactGeometry | A3MeshSupportContactGeometry,
    support_kwargs: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Build raw, unnormalized AMP frames from one 50-Hz A3 expert clip."""
    if not np.isfinite(dof).all() or not np.isfinite(root_pos_w).all():
        raise ValueError("A3 AMP expert clip contains non-finite values")
    root_rotation = _xyzw_to_matrix(root_rot_xyzw)
    keypoints_w = kinematics.keypoints_world(
        root_pos_w=root_pos_w, root_rot_xyzw=root_rot_xyzw, dof=dof
    )
    keypoints_b = np.einsum(
        "tji,tkj->tki", root_rotation, keypoints_w - root_pos_w[:, None, :], optimize=True
    )
    lower_indices = [A3_31DOF_JOINT_NAMES.index(name) for name in A3_AMP_LOWER_JOINT_NAMES]
    q = dof[:, lower_indices]
    dq = np.gradient(q, policy_dt, axis=0, edge_order=1).astype(np.float32)
    gravity_world = np.array((0.0, 0.0, -1.0), dtype=np.float32)
    projected_gravity = np.einsum("tji,j->ti", root_rotation, gravity_world, optimize=True)
    torso_index = A3_AMP_KEYPOINT_BODY_NAMES.index("torso_Link")
    torso_height = keypoints_w[:, torso_index, 2:3]
    root_velocity_w = np.gradient(root_pos_w, policy_dt, axis=0, edge_order=1)
    root_velocity_b = np.einsum(
        "tji,tj->ti", root_rotation, root_velocity_w, optimize=True
    ).astype(np.float32)
    omega_b = _angular_velocity_body(root_rotation, policy_dt)
    if isinstance(support_geometry, A3MeshSupportContactGeometry):
        support_points_w, support_radii = kinematics.mesh_support_points_world(
            root_pos_w=root_pos_w,
            root_rot_xyzw=root_rot_xyzw,
            dof=dof,
            geometry=support_geometry,
        )
    else:
        support_points_w = kinematics.support_contact_points_world(
            root_pos_w=root_pos_w,
            root_rot_xyzw=root_rot_xyzw,
            dof=dof,
            geometry=support_geometry,
        )
        support_radii = support_geometry.radii
    support = kinematic_support_phase(
        support_points_w,
        radii=support_radii,
        dt=policy_dt,
        **dict(support_kwargs or {}),
    )
    result = np.concatenate(
        (
            q,
            dq,
            keypoints_b.reshape(len(dof), -1),
            projected_gravity,
            torso_height,
            root_velocity_b,
            omega_b,
            support,
        ),
        axis=-1,
    ).astype(np.float32)
    if result.shape != (len(dof), A3_AMP_FEATURE_DIM):
        raise RuntimeError(f"A3 AMP feature contract broken: got {result.shape}")
    return np.ascontiguousarray(result)


def clip_balanced_mean_std(frames_by_clip: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Feature moments where each selected expert clip has equal weight."""
    if not frames_by_clip:
        raise ValueError("A3 AMP needs at least one expert clip")
    means = np.stack([frames.mean(axis=0) for frames in frames_by_clip])
    mean = means.mean(axis=0)
    second = np.stack([(frames * frames).mean(axis=0) for frames in frames_by_clip]).mean(axis=0)
    std = np.sqrt(np.maximum(second - mean * mean, 1.0e-6)).astype(np.float32)
    return mean.astype(np.float32), std


__all__ = [
    "A3_AMP_CONTRACT_VERSION",
    "A3_AMP_CONDITION_DIM",
    "A3_AMP_CONDITION_FIELDS",
    "A3_AMP_CONDITION_MEAN",
    "A3_AMP_CONDITION_STD",
    "A3_AMP_PLANAR_VELOCITY_LIMIT",
    "A3_AMP_FEATURE_DIM",
    "A3_AMP_FEATURE_FIELDS",
    "A3_AMP_KEYPOINT_BODY_NAMES",
    "A3_AMP_LOWER_JOINT_NAMES",
    "A3_AMP_SEQUENCE_DIM",
    "A3_AMP_TORSO_HEIGHT_INDEX",
    "A3_AMP_FORWARD_SEQUENCE_LENGTH",
    "A3_FORWARD_AMP_ANCHOR_BODY_NAME",
    "A3_FORWARD_AMP_BODY_NAMES",
    "A3_FORWARD_AMP_FRAME_DIM",
    "A3_FORWARD_AMP_TRANSITION_DIM",
    "A3_AMP_BACKWARD_SEQUENCE_LENGTH",
    "A3_AMP_LATERAL_SEQUENCE_LENGTH",
    "A3_AMP_LATERAL_SEQUENCE_DIM",
    "A3_AMP_TURN_SEQUENCE_LENGTH",
    "A3_AMP_TURN_SEQUENCE_DIM",
    "A3_AMP_SQUAT_SEQUENCE_LENGTH",
    "A3_AMP_SQUAT_SEQUENCE_DIM",
    "A3_AMP_SQUAT_SPEED_THRESHOLD",
    "A3_AMP_SQUAT_HEIGHT_THRESHOLD",
    "A3_AMP_SEQUENCE_LENGTH",
    "A3_AMP_ACTIVE_MODE_COUNT",
    "A3_AMP_MODE_COUNT",
    "A3_AMP_MODE_NAMES",
    "A3_AMP_SQUAT_MODE",
    "A3_AMP_NEUTRAL_MODE",
    "A3_AMP_SUPPORT_BODY_NAMES",
    "A3_AMP_024_SERIAL_SUPPORT_BODY_NAMES",
    "A3_AMP_024_SERIAL_PASSIVE_JOINT_POSITIONS",
    "A3_AMP_DEFAULT_ASSET_PROFILE",
    "A3_AMP_024_SERIAL_ASSET_PROFILE",
    "A3_AMP_ULTRA_T3D0_ASSET_PROFILE",
    "A3_AMP_ASSET_PROFILES",
    "A3AmpCache",
    "A3LowerBodyKinematics",
    "A3MeshSupportContactGeometry",
    "a3_asset_fingerprint",
    "a3_amp_mode_from_condition",
    "apply_a3_joint_offsets",
    "amp_schema_dict",
    "amp_schema_hash",
    "build_a3_amp_frames",
    "build_a3_amp_conditions",
    "canonicalize_a3_dof",
    "clip_balanced_mean_std",
    "align_a3_root_to_ground",
    "kinematic_support_phase",
    "load_a3_support_contact_geometry",
    "load_a3_024_serial_mesh_support_geometry",
    "resample_a3_clip",
    "sha256_file",
]
