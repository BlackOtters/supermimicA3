"""Convert the named A3 corpus-v2 qpos format into a checked, compact motion library.

NumPy only: conversion requires neither a GPU nor an installed simulator. Write
to a POSIX filesystem (e.g. CPFS), then copy the completed files to object storage.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np

from a3_supermimic.joint_names import POLICY_JOINTS, HEAD_JOINTS

FORMAT = "a3_supermimic_packed_v1"
JOINTS = POLICY_JOINTS + HEAD_JOINTS
ROOT_NAMES = ("root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz")
ARRAYS = {"root_q": (7, "float32"), "root_v": (6, "float32"),
          "joint_q": (31, "float16"), "joint_v": (31, "float16")}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_names(names):
    return tuple("ultra_head_pitch_joint" if str(n) == "head_pitch_joint" else str(n) for n in names)


def quat_mul(a, b):
    w = a[..., :1]*b[..., :1] - (a[..., 1:]*b[..., 1:]).sum(-1, keepdims=True)
    v = a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate((w, v), -1)


def rotation_difference(a, b):
    """Shortest world-frame rotation taking orientation b to a (both wxyz)."""
    r = quat_mul(a, b * np.array([1., -1., -1., -1.]))
    r = np.where(r[..., :1] < 0, -r, r)
    s = np.linalg.norm(r[..., 1:], axis=-1, keepdims=True)
    scale = np.divide(2*np.arctan2(s, r[..., :1]), s, out=np.full_like(s, 2.), where=s > 1.e-10)
    return r[..., 1:] * scale


def convert_qpos(qpos, names, fps):
    """Central velocities; one-sided endpoints; no smoothing, IK or limit clipping."""
    names = canonical_names(names)
    if len(names) != 31 or set(names) != set(JOINTS):
        raise ValueError("Expected each of the 29 A3 policy joints and 2 head joints exactly once")
    if not np.isclose(float(fps), 50., rtol=0., atol=1.e-6):
        raise ValueError(f"This corpus must already be 50 Hz, got {fps}")
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 38 or len(qpos) < 2 or not np.isfinite(qpos).all():
        raise ValueError("Expected finite qpos (T>=2,38)")
    norm = np.linalg.norm(qpos[:, 3:7], axis=-1)
    norm_error = float(np.abs(norm-1).max())
    if norm_error > 1.e-3:
        raise ValueError(f"Non-unit wxyz root quaternion: max norm error {norm_error}")
    q = qpos[:, 3:7] / norm[:, None]
    joints = qpos[:, [7+names.index(n) for n in JOINTS]]
    dt = 1./float(fps)
    v = np.gradient(qpos[:, :3], dt, axis=0)
    dq = np.gradient(joints, dt, axis=0)
    w = np.empty((len(q), 3), dtype=np.float64)
    w[0] = rotation_difference(q[1], q[0])/dt
    w[-1] = rotation_difference(q[-1], q[-2])/dt
    if len(q) > 2:
        w[1:-1] = rotation_difference(q[2:], q[:-2])/(2*dt)
    # R(q)^T omega_world, matching MuJoCo free-joint qvel.
    uv = np.cross(-q[:, 1:], w)
    w_body = w + 2*(q[:, :1]*uv + np.cross(-q[:, 1:], uv))
    pos = qpos[:, :3].copy()
    origin_xy = pos[0, :2].tolist()
    pos[:, :2] -= pos[0, :2].copy()
    result = {"root_q": np.concatenate((pos, q), -1).astype(np.float32),
              "root_v": np.concatenate((v, w_body), -1).astype(np.float32),
              "joint_q": joints.astype(np.float16), "joint_v": dq.astype(np.float16)}
    if not all(np.isfinite(a).all() for a in result.values()):
        raise ValueError("Converted coordinates/velocities overflowed their storage dtype")
    stats = {"root_quat_norm_error": norm_error, "max_joint_speed": float(np.abs(dq).max()),
             "max_root_linear_speed": float(np.linalg.norm(v, axis=-1).max()),
             "max_root_angular_speed": float(np.linalg.norm(w, axis=-1).max()), "origin_xy": origin_xy}
    return result, stats


def convert_file(job):
    path, relative, names = job
    raw = path.read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        if not {"qpos", "fps", "joint_names"}.issubset(z.files):
            raise ValueError(f"{path}: missing qpos/fps/joint_names")
        if tuple(str(x) for x in z["joint_names"].tolist()) != tuple(names):
            raise ValueError(f"{path}: joint_names do not match the source manifest")
        arrays, stats = convert_qpos(z["qpos"], names, np.asarray(z["fps"]).item())
    return arrays, {"source": relative, "source_sha256": hashlib.sha256(raw).hexdigest(), **stats}


def read_packed_manifest(path, verify=False):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest["format"] != FORMAT or manifest["fps"] != 50 or tuple(manifest["joint_names"]) != JOINTS:
        raise ValueError(f"Unsupported motion-library contract: {path}")
    total = 0
    seen = set()
    for clip in manifest["clips"]:
        if clip["start"] != total or clip["frames"] < 2 or clip["name"] in seen:
            raise ValueError("Invalid clip boundaries or duplicate names")
        seen.add(clip["name"])
        total += clip["frames"]
    if not total or total != manifest["num_frames"]:
        raise ValueError("Packed frame count mismatch")
    for key, (width, dtype) in ARRAYS.items():
        entry = manifest["arrays"][key]
        file = path.parent / (key+".npy")
        a = np.load(file, mmap_mode="r", allow_pickle=False)
        if a.shape != (total, width) or str(a.dtype) != dtype:
            raise ValueError(f"Invalid {key} shape/dtype")
        if entry["shape"] != list(a.shape) or entry["dtype"] != dtype:
            raise ValueError(f"Invalid {key} metadata")
        if verify and sha256_file(file) != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {file}")
        if verify:
            for start in range(0, total, 262144):
                block = a[start:start+262144]
                if not np.isfinite(block).all():
                    raise ValueError(f"Nonfinite values in {key} at {start}")
                if key == "root_q" and not np.allclose(np.linalg.norm(block[:, 3:7], axis=-1), 1., atol=1.e-3):
                    raise ValueError("Non-unit packed root quaternion")
    return manifest


def convert_corpus(source, output, workers=16, limit_per_group=None):
    source, output = Path(source), Path(output)
    if workers < 1:
        raise ValueError("workers must be positive")
    manifests = sorted(source.glob("*/manifest.json"))
    if not manifests:
        raise ValueError("Expected corpus-v2/<dataset>/manifest.json")
    jobs, sources, expected_frames = [], [], 0
    for path in manifests:
        raw = path.read_bytes()
        m = json.loads(raw)
        if m["format_version"] != 2 or m["qpos_dim"] != 38 or not np.isclose(m["timestep"], .02):
            raise ValueError(f"Unsupported manifest: {path}")
        if tuple(m["qpos_names"][:7]) != ROOT_NAMES:
            raise ValueError(f"Root coordinate convention mismatch: {path}")
        names = tuple(m["qpos_names"][7:])
        if len(names) != 31 or set(canonical_names(names)) != set(JOINTS):
            raise ValueError(f"Invalid joint coordinates: {path}")
        files = sorted((path.parent/m["motions_subdir"]).glob("*.npz"))
        if len(files) != m["num_motions"]:
            raise ValueError(f"{path}: expected {m['num_motions']} files, found {len(files)}")
        if limit_per_group is not None:
            files = files[:limit_per_group]
        jobs.extend((f, str(f.relative_to(source)), names) for f in files)
        expected_frames += m["num_frames"]
        sources.append({"group": path.parent.name, "manifest_sha256": hashlib.sha256(raw).hexdigest(), "manifest": m})
    if limit_per_group is not None:
        expected_frames = 0
        for path, _, _ in jobs:
            with np.load(path, allow_pickle=False) as z:
                expected_frames += len(z["qpos"])
    output.mkdir(parents=True, exist_ok=False)
    arrays = {key: np.lib.format.open_memmap(output/(key+".npy"), mode="w+", dtype=dtype,
              shape=(expected_frames, width)) for key, (width, dtype) in ARRAYS.items()}
    rows, cursor, started = [], 0, time.monotonic()
    stats = {k: 0. for k in ("root_quat_norm_error", "max_joint_speed", "max_root_linear_speed", "max_root_angular_speed")}
    print(f"CONVERT_START clips={len(jobs)} frames={expected_frames} workers={workers}", flush=True)
    # Bound queued results so long clips / a slow object-store read cannot fill RAM.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = deque()
        iterator = iter(jobs)
        for _ in range(min(len(jobs), 2*workers)):
            pending.append(pool.submit(convert_file, next(iterator)))
        while pending:
            data, row = pending.popleft().result()
            n = len(data["root_q"])
            if cursor+n > expected_frames:
                raise ValueError("Source frames exceed manifest totals")
            for key, a in data.items():
                arrays[key][cursor:cursor+n] = a
            relative = Path(row["source"])
            row.update(name=relative.parts[0]+"__"+relative.stem, start=cursor, frames=n)
            rows.append(row)
            cursor += n
            for key in stats:
                stats[key] = max(stats[key], row[key])
            job = next(iterator, None)
            if job is not None:
                pending.append(pool.submit(convert_file, job))
            if len(rows) % 1000 == 0 or len(rows) == len(jobs):
                print(f"CONVERT {len(rows)}/{len(jobs)} frames={cursor} seconds={time.monotonic()-started:.1f}", flush=True)
    if cursor != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, converted {cursor}")
    entries = {}
    for key, a in arrays.items():
        a.flush()
        entries[key] = {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256_file(output/(key+".npy"))}
    manifest = {"format": FORMAT, "fps": 50, "joint_names": JOINTS, "num_frames": cursor,
                "num_clips": len(rows), "total_hours": cursor/50/3600,
                "synthetic_fixture": False, "arrays": entries, "clips": rows,
                "source_root": str(source), "source_manifests": sources, "statistics": stats,
                "conversion": {"velocity": "central difference, one-sided endpoints; SO(3) world derivative rotated to body",
                               "root_xy": "subtract each clip's initial xy; preserve z and full trajectory",
                               "resampling": False, "filtering": False, "joint_clipping": False,
                               "split": "all supplied clips; original/mirror grouping retained in source names",
                               "script_sha256": sha256_file(__file__)}}
    temporary = output/"library.pending.json"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))+"\n")
    read_packed_manifest(temporary, verify=True)
    temporary.replace(output/"library.json")
    print(f"CONVERT_AND_VERIFY_OK clips={len(rows)} frames={cursor} hours={manifest['total_hours']:.6f} output={output}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("source", type=Path)
    convert.add_argument("output", type=Path)
    convert.add_argument("--workers", type=int, default=16)
    convert.add_argument("--limit-per-group", type=int)
    verify = sub.add_parser("verify")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "convert":
        convert_corpus(args.source, args.output, args.workers, args.limit_per_group)
    else:
        m = read_packed_manifest(args.directory/"library.json", verify=True)
        print(f"VERIFIED clips={m['num_clips']} frames={m['num_frames']} hours={m['total_hours']:.6f}")


if __name__ == "__main__":
    main()
