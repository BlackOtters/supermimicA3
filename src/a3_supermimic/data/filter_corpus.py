"""Appendix E screening of the A3 packed corpus, with per-clip provenance.

Keyword list and rule structure adapted from CuboiLeo/SuperMimic-Reproduced,
scripts/data/filter_bones_seed.py at d9be385 (MIT; see LICENSE.upstream).
This performs kinematics only. It never changes reference coordinates or velocities.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import re
import time

import mujoco
import numpy as np

from a3_supermimic.data.convert_corpus import ARRAYS, read_packed_manifest, sha256_file


KEYWORDS = sorted(set([
    "bed", "bike", "chair", "climb", "com_up_50cm", "sitting", "step_on", "seat", "table", "_sit_", "sit_",
    "ladder", "crutch", "_bed_", "_ride_", "scooter", "stepdown", "box_hspu", "50cm_box_", "on_box",
    "fall_from", "on_1m", "form_box", "off_1m", "230m", "jump_over_obstacle_", "lift_crate_come_up_",
    "stair", "box_jump", "box_dips", "walking_on_edge", "push_obstacle",
    "_box_", "box_", "_stair", "jump_off_50cm", "off_50cm", "fence", "look_over",
]))
UPSTREAM_COMMIT = "d9be385b3eeb13a638fb33fb200d3bfd9be9839f"


@dataclass(frozen=True)
class FilterConfig:
    fps: int = 50
    elevated_z: float = .20
    elevated_frames: int = 30
    penetration_hard_z: float = -.12
    penetration_soft_z: float = -.08
    penetration_fraction: float = .80
    floating_z: float = .02
    floating_frames: int = 25
    upright_cos: float = .9
    standing_low: float = .8179254191745672
    standing_high: float = 1.15872767716397
    still_root_speed: float = .1
    still_joint_rms: float = .5


def semantic_name(name):
    """Remove augmentation suffixes and hash tokens before keyword matching."""
    stem = re.sub(r"__(original|mirror)$", "", Path(name).stem.lower())
    stem = re.sub(r"(?<![a-z0-9])[a-f0-9]{8,}(?![a-z0-9])", "", stem)
    return stem


def name_reasons(name):
    normalized = semantic_name(name)
    return ["name:" + k for k in KEYWORDS if k in normalized]


def longest_run(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    lengths = edges[1::2] - edges[::2]
    if not len(lengths):
        return 0, -1
    index = int(lengths.argmax())
    return int(lengths[index]), int(edges[2 * index])


def classify_geometry(zmin, foot_min, root_q, root_v, joint_rms, cfg):
    upright = 1 - 2 * np.sum(root_q[:, 4:6] ** 2, axis=1) > cfg.upright_cos
    still = (np.linalg.norm(root_v[:, :3], axis=1) < cfg.still_root_speed) & (joint_rms < cfg.still_joint_rms)
    standing = (root_q[:, 2] > cfg.standing_low) & (root_q[:, 2] < cfg.standing_high)
    gate = upright & still & standing
    elevated_n, elevated_start = longest_run(zmin > cfg.elevated_z)
    floating_n, floating_start = longest_run(gate & (foot_min > cfg.floating_z))
    original_g1_gate = upright & still & (root_q[:, 2] > .6) & (root_q[:, 2] < .85)
    g1_n, _ = longest_run(original_g1_gate & (foot_min > cfg.floating_z))
    penetration_fraction = float(np.mean(zmin < cfg.penetration_soft_z))
    reasons = []
    if elevated_n >= cfg.elevated_frames:
        reasons.append("elevated_support")
    if np.min(zmin) < cfg.penetration_hard_z or penetration_fraction > cfg.penetration_fraction:
        reasons.append("ground_penetration")
    if floating_n >= cfg.floating_frames:
        reasons.append("floating_feet")
    return reasons, {
        "lowest_body_z_m": float(np.min(zmin)), "lowest_body_frame": int(np.argmin(zmin)),
        "below_soft_penetration_fraction": penetration_fraction,
        "elevated_max_run_frames": elevated_n, "elevated_start_frame": elevated_start,
        "floating_max_run_frames": floating_n, "floating_start_frame": floating_start,
        "floating_with_unadapted_g1_standing_gate_frames": g1_n,
        "minimum_foot_distance_m": float(np.min(foot_min)),
        "maximum_both_feet_clearance_m": float(np.max(foot_min)),
        "standing_still_frames": int(gate.sum()),
        "standing_still_foot_distance_median_m": float(np.median(foot_min[gate])) if gate.any() else None,
    }


class KinematicScreen:
    def __init__(self, scene, joint_names):
        self.model = mujoco.MjModel.from_binary_path(str(scene))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qids = np.array([m.joint("robot/" + n).qposadr[0] for n in joint_names])
        self.jids = np.array([m.joint("robot/" + n).id for n in joint_names])
        self.policy_ids = np.array([i for i, n in enumerate(joint_names) if "head" not in n])
        self.body_ids = np.array([b for b in range(m.nbody)
                                  if m.body(b).name.startswith("robot/") and not m.body(b).name.endswith("_rotor")])
        self.foot_ids = np.array([g for g in range(m.ngeom)
                                  if re.fullmatch(r"robot/(left|right)_foot\d+_collision", m.geom(g).name)])
        if len(self.policy_ids) != 29 or len(self.foot_ids) != 26:
            raise ValueError("Expected A3's 29 policy joints and 26 spherical foot colliders")
        if not np.all(m.geom_type[self.foot_ids] == mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError("This A3 screening implementation requires its spherical foot colliders")
        self.radii = m.geom_size[self.foot_ids, 0].copy()
        self.plane_id = m.geom("terrain").id
        if m.geom_type[self.plane_id] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise ValueError("Expected flat terrain")

    def trajectories(self, root_q, joint_q):
        n = len(root_q)
        zmin = np.empty(n)
        foot_min = np.empty(n)
        self.data.qpos[:] = self.model.qpos0
        for i in range(n):
            self.data.qpos[:7] = root_q[i]
            self.data.qpos[self.qids] = joint_q[i]
            mujoco.mj_kinematics(self.model, self.data)
            zmin[i] = np.min(self.data.xpos[self.body_ids, 2])
            # Exact signed sphere-to-horizontal-plane distances, validated against mj_geomDistance.
            foot_min[i] = np.min(self.data.geom_xpos[self.foot_ids, 2] - self.radii)
        return zmin, foot_min


_WORKER = None


def init_worker(library, scene, joint_names, cfg):
    global _WORKER
    arrays = {k: np.load(Path(library) / (k + ".npy"), mmap_mode="r") for k in ARRAYS}
    _WORKER = (arrays, KinematicScreen(scene, joint_names), cfg)


def screen_one(job):
    index, clip = job
    arrays, screen, cfg = _WORKER
    sl = slice(clip["start"], clip["start"] + clip["frames"])
    rq = np.asarray(arrays["root_q"][sl], dtype=np.float64)
    jq = np.asarray(arrays["joint_q"][sl], dtype=np.float64)
    jv = np.asarray(arrays["joint_v"][sl], dtype=np.float64)
    rv = np.asarray(arrays["root_v"][sl], dtype=np.float64)
    if not all(np.isfinite(x).all() for x in (rq, jq, jv, rv)):
        raise ValueError(f"Nonfinite source arrays: {clip['name']}")
    zmin, foot = screen.trajectories(rq, jq)
    joint_rms = np.sqrt(np.mean(jv[:, screen.policy_ids] ** 2, axis=1))
    geometry, metrics = classify_geometry(zmin, foot, rq, rv, joint_rms, cfg)
    names = name_reasons(Path(clip["source"]).stem)
    raw_matches = ["name:" + k for k in KEYWORDS if k in clip["name"].lower()]
    limits = screen.model.jnt_range[screen.jids]
    excess = np.maximum(jq - limits[:, 1], limits[:, 0] - jq).clip(0.)
    metrics.update(max_joint_limit_excess_rad=float(excess.max()), max_joint_speed_rad_s=float(np.abs(jv).max()),
                   max_root_speed_m_s=float(np.linalg.norm(rv[:, :3], axis=1).max()))
    reasons = names + geometry
    return {"source_index": index, "name": clip["name"], "source": clip["source"],
            "source_sha256": clip["source_sha256"], "source_start": clip["start"], "frames": clip["frames"],
            "corpus": clip["source"].split("/")[0], "keep": not reasons, "reasons": reasons,
            "primary_reason": reasons[0] if reasons else "keep", "geometry_reasons": geometry,
            "semantic_name": semantic_name(Path(clip["source"]).stem),
            "name_coverage": "opaque_amass_id" if clip["source"].startswith("amass/") else "descriptive",
            "avoided_hash_matches": sorted(set(raw_matches) - set(names)), "metrics": metrics}


def summarize(rows, source, cfg, scene, elapsed):
    kept = [r for r in rows if r["keep"]]
    primary = Counter(r["primary_reason"] for r in rows)
    groups = {}
    for group in sorted(set(r["corpus"] for r in rows)):
        subset = [r for r in rows if r["corpus"] == group]
        ok = [r for r in subset if r["keep"]]
        groups[group] = {"input_clips": len(subset), "kept_clips": len(ok), "removed_clips": len(subset)-len(ok),
                         "kept_hours": sum(r["frames"] for r in ok)/50/3600,
                         "removed_hours": sum(r["frames"] for r in subset if not r["keep"])/50/3600}
    return {"input_clips": len(rows), "kept_clips": len(kept), "removed_clips": len(rows)-len(kept),
            "input_frames": source["num_frames"], "kept_frames": sum(r["frames"] for r in kept),
            "input_hours": source["total_hours"], "kept_hours": sum(r["frames"] for r in kept)/50/3600,
            "primary_reasons": dict(primary), "all_reason_counts_overlapping": dict(Counter(reason for r in rows for reason in r["reasons"])),
            "groups": groups, "avoided_hash_false_positive_clips": sum(bool(r["avoided_hash_matches"]) for r in rows),
            "a3_floating_flags_missed_by_g1_height_gate": sum("floating_feet" in r["geometry_reasons"] and r["metrics"]["floating_with_unadapted_g1_standing_gate_frames"] < cfg.floating_frames for r in rows),
            "thresholds": asdict(cfg), "upstream_commit": UPSTREAM_COMMIT, "scene_sha256": sha256_file(scene),
            "script_sha256": sha256_file(__file__), "elapsed_seconds": elapsed,
            "physics_steps": 0, "fps": 50, "coordinates_modified": False,
            "limitations": ["Kinematic feasibility only, not proof of dynamic trackability.",
                            "Opaque AMASS identifiers have no reliable action captions; geometry is checked on every frame.",
                            "Difficult floor-compatible motions are not removed merely because the current policy fails.",
                            "Originals and mirrors are screened independently; no train/test split is introduced."]}


def pack_filtered(library, output, manifest, rows, summary):
    output.mkdir(exist_ok=False)
    kept = [r for r in rows if r["keep"]]
    if not kept:
        raise ValueError("No clips survived; refusing to create an empty library")
    total = sum(r["frames"] for r in kept)
    src = {k: np.load(library/(k+".npy"), mmap_mode="r") for k in ARRAYS}
    dst = {k: np.lib.format.open_memmap(output/(k+".npy"), mode="w+", dtype=dtype, shape=(total,width))
           for k,(width,dtype) in ARRAYS.items()}
    clips, cursor = [], 0
    for row in kept:
        clip = dict(manifest["clips"][row["source_index"]])
        n = clip["frames"]
        for k in ARRAYS:
            dst[k][cursor:cursor+n] = src[k][clip["start"]:clip["start"]+n]
        clip.update(parent_start=clip["start"], parent_index=row["source_index"], start=cursor)
        clips.append(clip)
        cursor += n
    entries = {}
    for k,a in dst.items():
        a.flush()
        entries[k] = {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256_file(output/(k+".npy"))}
    result = dict(manifest)
    result.update(num_clips=len(clips), num_frames=total, total_hours=total/50/3600, clips=clips, arrays=entries,
                  statistics={k: max(c[k] for c in clips) for k in manifest.get("statistics", {})},
                  filtering={"method": "SuperMimic Appendix E with A3 geometry and scaled standing-height gate",
                             "parent_manifest_sha256": sha256_file(library/"library.json"),
                             "parent_library": str(library), "summary": summary,
                             "audit_sha256": sha256_file(output.parent/"audit.jsonl")})
    pending=output/"library.pending.json"
    pending.write_text(json.dumps(result,ensure_ascii=False,separators=(",",":"))+"\n")
    read_packed_manifest(pending,verify=True)
    # Verify every retained slice against its parent, including velocities and clip boundaries.
    for clip in clips:
        for k in ARRAYS:
            if not np.array_equal(dst[k][clip["start"]:clip["start"]+clip["frames"]],
                                  src[k][clip["parent_start"]:clip["parent_start"]+clip["frames"]]):
                raise ValueError(f"Repacking modified {clip['name']} / {k}")
    pending.replace(output/"library.json")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--scene",type=Path,required=True)
    parser.add_argument("--calibration",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=12)
    parser.add_argument("--limit",type=int)
    parser.add_argument("--audit-only",action="store_true")
    args=parser.parse_args()
    manifest=read_packed_manifest(args.library/"library.json",verify=True)
    calibration=json.loads(args.calibration.read_text())
    if sha256_file(args.scene)!=calibration["a3_model_sha256"]:
        raise ValueError("Screening model differs from calibrated model")
    cfg=FilterConfig(standing_low=calibration["standing_height_range"][0],standing_high=calibration["standing_height_range"][1])
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"calibration.json").write_text(json.dumps(calibration,indent=2)+"\n")
    clips=manifest["clips"][:args.limit]
    started=time.monotonic()
    rows=[]
    print(f"SCREEN_START clips={len(clips)} workers={args.workers}",flush=True)
    with mp.get_context("spawn").Pool(args.workers,initializer=init_worker,
                initargs=(args.library,args.scene,manifest["joint_names"],cfg)) as pool, (args.output/"audit.jsonl").open("w") as f:
        for row in pool.imap(screen_one,enumerate(clips),chunksize=32):
            rows.append(row)
            f.write(json.dumps(row,separators=(",",":"))+"\n")
            if len(rows)%1000==0 or len(rows)==len(clips):
                f.flush()
                print(f"SCREEN {len(rows)}/{len(clips)} kept={sum(r['keep'] for r in rows)} seconds={time.monotonic()-started:.1f}",flush=True)
    summary=summarize(rows,manifest,cfg,args.scene,time.monotonic()-started)
    if args.limit:
        summary["partial_audit"]=True
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    for label,keep in (("kept",True),("rejected",False)):
        (args.output/(label+".txt")).write_text("".join(r["name"]+"\n" for r in rows if r["keep"]==keep))
    if not args.audit_only:
        if args.limit:
            raise ValueError("Partial audits cannot publish a filtered corpus")
        result=pack_filtered(args.library,args.output/"library",manifest,rows,summary)
        print(f"PACK_VERIFIED clips={result['num_clips']} frames={result['num_frames']} hours={result['total_hours']:.6f}",flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
