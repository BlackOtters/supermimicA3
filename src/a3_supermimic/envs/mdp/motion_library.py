"""Validated A3 reference motions; compact generalized coordinates plus online FK."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import torch
from a3_supermimic.robot import POLICY_JOINTS, HEAD_JOINTS, ENCODERS, ANKLES
from a3_supermimic.kinematics.rotations import projected_gravity, quat_apply_inverse, heading_quat
from a3_supermimic.data.convert_corpus import read_packed_manifest, sha256_file, ARRAYS


@dataclass
class Reference:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor


@dataclass
class Bodies:
    pos_w: torch.Tensor
    quat_w: torch.Tensor
    lin_vel_w: torch.Tensor
    ang_vel_w: torch.Tensor


def discover_npz(directory, limit=None, names=None, manifest=None):
    if manifest is not None:
        raise ValueError("Use an explicit clip directory/list; G1 manifests are not A3 contracts")
    packed = Path(directory)/"library.json"
    if packed.exists():
        if limit is not None or names is not None:
            raise ValueError("Packed libraries are complete datasets; use a separate converted subset for limit/clips")
        return [packed]
    files = sorted(Path(directory).glob("*.npz"))
    if names is not None:
        wanted = set(names)
        files = [p for p in files if p.stem in wanted]
        missing = wanted - {p.stem for p in files}
        if missing:
            raise ValueError(f"Missing requested clips: {sorted(missing)}")
    return files[:limit] if limit else files


def read_motion(path, model, layout):
    with np.load(path, allow_pickle=False) as z:
        required = {"fps", "joint_names", "joint_pos", "joint_vel", "root_pos", "root_quat_wxyz",
                    "root_lin_vel_w", "root_ang_vel_b"}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}; convert using a3-sm-data first")
        fps = float(np.asarray(z["fps"]).item())
        if not np.isclose(fps, 50.):
            raise ValueError(f"{path}: expected 50 Hz, got {fps}; resample explicitly")
        names = [str(n) for n in z["joint_names"].tolist()]
        names = ["ultra_head_pitch_joint" if n == "head_pitch_joint" else n for n in names]
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: duplicate joint names")
        if set(names) != set(POLICY_JOINTS + HEAD_JOINTS):
            raise ValueError(f"{path}: expected all 29 policy + 2 head joints by name; no motor/finger coordinates")
        n = len(z["root_pos"])
        if n < 2:
            raise ValueError(f"{path}: a motion needs at least two frames")
        expected = {"root_pos": (n, 3), "root_quat_wxyz": (n, 4), "root_lin_vel_w": (n, 3),
                    "root_ang_vel_b": (n, 3), "joint_pos": (n, 31), "joint_vel": (n, 31)}
        arrays = {}
        for name, shape in expected.items():
            x = np.asarray(z[name], dtype=np.float32)
            if x.shape != shape or not np.isfinite(x).all():
                raise ValueError(f"{path}: {name} must be finite with shape {shape}, got {x.shape}")
            arrays[name] = x
        quat = arrays["root_quat_wxyz"]
        if not np.allclose(np.linalg.norm(quat, axis=-1), 1., atol=1.e-3):
            raise ValueError(f"{path}: root_quat_wxyz is not unit-normalized")
        qpos = np.broadcast_to(model.qpos0, (n, model.nq)).copy().astype(np.float32)
        qvel = np.zeros((n, model.nv), dtype=np.float32)
        qpos[:, :3], qpos[:, 3:7] = arrays["root_pos"], quat
        # Every clip has a local horizontal origin; preserve its whole trajectory.
        qpos[:, :2] -= qpos[0, :2].copy()
        qvel[:, :3], qvel[:, 3:6] = arrays["root_lin_vel_w"], arrays["root_ang_vel_b"]
        for name, qi, vi in zip(POLICY_JOINTS + HEAD_JOINTS, layout.policy_q + layout.head_q, layout.policy_v + layout.head_v):
            col = names.index(name)
            qpos[:, qi], qvel[:, vi] = arrays["joint_pos"][:, col], arrays["joint_vel"][:, col]
        # Strain-free reference rotors; simulator reset chooses elastic or rigid.
        for output, qi, vi in zip(ANKLES, layout.encoder_q, layout.encoder_v):
            col = names.index(output)
            qpos[:, qi], qvel[:, vi] = arrays["joint_pos"][:, col], arrays["joint_vel"][:, col]
        synthetic = bool(z["synthetic_fixture"]) if "synthetic_fixture" in z else False
    return qpos, qvel, synthetic


class MotionLibrary:
    def __init__(self, files, model, layout, fk, tracked_indices, default_joint_pos, *, allow_synthetic=False):
        if not files:
            raise ValueError("No A3 motions supplied")
        self.layout, self.fk, self.tracked_indices = layout, fk, tracked_indices
        self.device = fk.device
        self.default_joint_pos = default_joint_pos
        if len(files) == 1 and Path(files[0]).name == "library.json":
            metadata = read_packed_manifest(files[0], verify=True)
            if metadata.get("synthetic_fixture") and not allow_synthetic:
                raise ValueError("Synthetic library is forbidden for real training")
            self.names = [c["name"] for c in metadata["clips"]]
            lengths = [c["frames"] for c in metadata["clips"]]
            qcols = np.array(layout.policy_q + layout.head_q)
            vcols = np.array(layout.policy_v + layout.head_v)
            self.fingerprint = sha256_file(files[0])
            for key in ARRAYS:
                array = np.load(Path(files[0]).parent/(key+".npy"), mmap_mode="c", allow_pickle=False)
                setattr(self, key, torch.as_tensor(array, device=self.device))
        else:
            self.names, lengths, qpos, qvel = [], [], [], []
            digest = hashlib.sha256()
            qcols = np.array(layout.policy_q + layout.head_q)
            vcols = np.array(layout.policy_v + layout.head_v)
            for file in files:
                q, v, synthetic = read_motion(file, model, layout)
                if synthetic and not allow_synthetic:
                    raise ValueError(f"{file}: synthetic smoke fixture is forbidden for real training/evaluation")
                self.names.append(Path(file).stem)
                lengths.append(len(q))
                qpos.append((q[:, :7].copy(), q[:, qcols].astype(np.float16)))
                qvel.append((v[:, :6].copy(), v[:, vcols].astype(np.float16)))
                digest.update(Path(file).name.encode()); digest.update(Path(file).read_bytes())
            if len(set(self.names)) != len(self.names):
                raise ValueError("Motion clip names must be unique")
            self.fingerprint = digest.hexdigest()
            # Root pose/velocities retain float32; only the 31 articulated reference
            # angles/rates use fp16. Passive fingers and rotor copies are expanded
            # on demand, so dataset memory does not scale with simulator-only DOFs.
            self.root_q = torch.as_tensor(np.concatenate([q[0] for q in qpos]), device=self.device)
            self.root_v = torch.as_tensor(np.concatenate([v[0] for v in qvel]), device=self.device)
            self.joint_q = torch.as_tensor(np.concatenate([q[1] for q in qpos]).astype(np.float16), device=self.device)
            self.joint_v = torch.as_tensor(np.concatenate([v[1] for v in qvel]).astype(np.float16), device=self.device)
        self.qpos0 = torch.tensor(model.qpos0.copy(), dtype=torch.float32, device=self.device)
        self.num_frames = len(self.root_q)
        self.all_q = torch.tensor(qcols, device=self.device)
        self.all_v = torch.tensor(vcols, device=self.device)
        self.encoder_q = layout.tensor_ids("encoder_q", self.device)
        self.encoder_v = layout.tensor_ids("encoder_v", self.device)
        self.ankle_cols = torch.tensor([POLICY_JOINTS.index(n) for n in ANKLES], device=self.device)
        self.lengths = torch.tensor(lengths, device=self.device)
        self.starts = torch.cat((torch.zeros(1, dtype=torch.long, device=self.device), self.lengths.cumsum(0)[:-1]))
        self.ends = self.starts + self.lengths - 1
        self.frame_clip = torch.repeat_interleave(torch.arange(len(self.names), device=self.device), self.lengths)
        self.policy_q = layout.tensor_ids("policy_q", self.device)
        self.policy_v = layout.tensor_ids("policy_v", self.device)
        self.head_q = layout.tensor_ids("head_q", self.device)
        # One bin per second; shorter terminal bins remain valid reset candidates.
        # Build bins on CPU in bulk: avoid one GPU launch / synchronization per clip.
        cpu_lengths = np.asarray(lengths, dtype=np.int64)
        cpu_starts = np.concatenate(([0], cpu_lengths.cumsum()[:-1]))
        bin_counts = (cpu_lengths+49)//50
        bin_offsets = np.concatenate(([0], bin_counts.cumsum()[:-1]))
        bins = np.repeat(cpu_starts, bin_counts) + 50*(
            np.arange(bin_counts.sum()) - np.repeat(bin_offsets, bin_counts))
        self.bin_starts = torch.as_tensor(bins, device=self.device)
        self.bin_ends = torch.minimum(self.bin_starts + 49, self.last_frame_of(self.bin_starts))

    def generalized(self, frame):
        q = self.qpos0.expand(*frame.shape, -1).clone()
        v = torch.zeros(*frame.shape, self.layout.nv, device=self.device)
        q[..., :7], v[..., :6] = self.root_q[frame], self.root_v[frame]
        count = len(self.all_q)
        q[..., self.all_q] = self.joint_q[frame][..., :count].float()
        v[..., self.all_v] = self.joint_v[frame][..., :count].float()
        if not self.layout.compact:
            q[..., self.encoder_q] = self.joint_q[frame][..., self.ankle_cols].float()
            v[..., self.encoder_v] = self.joint_v[frame][..., self.ankle_cols].float()
        return q, v

    def last_frame_of(self, frame):
        return self.ends[self.frame_clip[frame]]

    def ref(self, frame):
        return Reference(self.root_q[frame, :3], self.root_q[frame, 3:7])

    def bodies(self, frame):
        q, v = self.generalized(frame)
        out = self.fk(q, v)
        ids = self.tracked_indices
        return Bodies(out["pos"][..., ids, :], out["quat"][..., ids, :], out["lin_vel"][..., ids, :], out["ang_vel"][..., ids, :])

    def head_targets(self, frame):
        return self.joint_q[frame][..., 29:31].float()

    def goal_window(self, frame, future_frames=10):
        f = torch.minimum(frame[..., None] + torch.arange(future_frames, device=self.device), self.last_frame_of(frame)[..., None])
        q, v = self.root_q[f], self.root_v[f]
        feature = torch.cat((self.joint_q[f][..., :29].float() - self.default_joint_pos, self.joint_v[f][..., :29].float(),
                             projected_gravity(q[..., 3:7]), v[..., 3:6],
                             quat_apply_inverse(heading_quat(q[..., 3:7]), v[..., :3]), q[..., 2:3]), -1)
        return feature.flatten(-2)
