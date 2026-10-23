"""Named A3 contracts. Never infer the robot's joint order from array width."""
from dataclasses import dataclass
import hashlib
import json
import mujoco
import numpy as np
import torch

# Importing mimic_lite registers simulator components; keep numerical tools usable
# without importing that package before active_adaptation.set_backend().
from a3_supermimic.joint_names import POLICY_JOINTS, HEAD_JOINTS
ANKLES = tuple(f"{s}_ankle_{a}_joint" for s in ("left", "right") for a in ("pitch", "roll"))
ENCODERS = tuple(n.replace("_joint", "_motor_joint") for n in ANKLES)
TRACKED_BODIES = (
    "pelvis_link", "torso_Link",
    *tuple(f"{s}_{b}_Link" for s in ("left", "right") for b in
           ("hip_roll", "knee", "ankle_roll", "shoulder_roll", "elbow", "wrist_yaw")),
)
BASE_TASK = "depsic-a3-ultra-v2-amp-s2r-four-ankle-mixed"
TRACKING_ANCHOR = "torso_Link"  # Upstream e5 reward anchor; termination still uses the pelvis root.
CONTRACT_VERSION = 9
UPSTREAM_COMMIT = "66f44a4f8f2c8dd31d00434c80e979eea2572622"


def check_contract_version(checkpoint):
    if checkpoint.get("contract", {}).get("version") != CONTRACT_VERSION:
        raise ValueError("Checkpoint does not match v9 orientation-only goals, analytic termination and WM-only warmup; use its cached source or start a new run")


@dataclass
class RobotLayout:
    nq: int
    nv: int
    joint_names: tuple[str, ...]
    policy_q: tuple[int, ...]
    policy_v: tuple[int, ...]
    head_q: tuple[int, ...]
    head_v: tuple[int, ...]
    encoder_q: tuple[int, ...]
    encoder_v: tuple[int, ...]
    fingerprint: str

    @classmethod
    def from_model(cls, model):
        names = [model.joint(j).name.split("/")[-1] for j in range(model.njnt)]
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1 or model.jnt_qposadr[free[0]] != 0 or model.jnt_dofadr[free[0]] != 0:
            raise ValueError("Expected one A3 floating base at qpos/qvel address zero")
        if any(t != mujoco.mjtJoint.mjJNT_HINGE for j, t in enumerate(model.jnt_type) if j != free[0]):
            raise ValueError("A3 world model supports the selected free-base/hinge topology only")
        def addresses(subset, field):
            return tuple(int(field[names.index(n)]) for n in subset)
        compact = set(names[1:]) == set(POLICY_JOINTS)
        # Include topology and physical constants, not merely dimensions.
        h = hashlib.sha256()
        h.update(json.dumps(names).encode())
        bodies = [i for i in range(1, model.nbody) if model.body(i).name.startswith("robot/")]
        if not bodies:
            bodies = list(range(1, model.nbody))
        parents = [(model.body(i).name.split("/")[-1], model.body(int(model.body_parentid[i])).name.split("/")[-1]) for i in bodies]
        h.update(json.dumps(parents).encode())
        for field in ("body_pos", "body_quat", "body_mass", "body_inertia"):
            h.update(np.asarray(getattr(model, field))[bodies].tobytes())
        for field in ("jnt_type", "jnt_axis", "jnt_pos", "jnt_range", "dof_armature"):
            h.update(np.asarray(getattr(model, field)).tobytes())
        return cls(model.nq, model.nv, tuple(names[j] for j in range(model.njnt) if j != free[0]),
                   addresses(POLICY_JOINTS, model.jnt_qposadr), addresses(POLICY_JOINTS, model.jnt_dofadr),
                   () if compact else addresses(HEAD_JOINTS, model.jnt_qposadr), () if compact else addresses(HEAD_JOINTS, model.jnt_dofadr),
                   () if compact else addresses(ENCODERS, model.jnt_qposadr), () if compact else addresses(ENCODERS, model.jnt_dofadr), h.hexdigest())

    @property
    def physical_dim(self):
        return self.nq + self.nv

    @property
    def state_dim(self):
        # Last issued policy actions + head targets: the delay history at a
        # 20 ms boundary is fully determined when the largest lag is <=10 ms.
        return self.physical_dim + (0 if self.compact else 32)

    @property
    def compact(self):
        return len(self.joint_names) == 29 and not self.head_q and not self.encoder_q

    @property
    def action_dim(self):
        return 29 + len(self.head_q)

    def tensor_ids(self, field, device):
        return torch.tensor(getattr(self, field), dtype=torch.long, device=device)


def setup_backend():
    import active_adaptation as aa
    try:
        backend = aa.get_backend()
    except RuntimeError:
        aa.set_backend("mjlab")
    else:
        if backend != "mjlab":
            raise ValueError("A3 SuperMimic requires the mjlab backend")
    from active_adaptation.registry import Registry
    Registry.verbose = False
    import mimic_lite  # noqa: F401


def load_robot_model(compact=False):
    setup_backend()
    if compact:
        from a3_supermimic.envs.compact_robot import compact_rigid_spec
        return compact_rigid_spec().compile()
    from mimic_lite.assets.a3_ultra_four_ankle import four_ankle_spec
    return four_ankle_spec().compile()
