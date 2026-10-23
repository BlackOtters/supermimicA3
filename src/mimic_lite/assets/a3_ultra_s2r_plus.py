"""Ultra with the released 035 two-stage foot and MuJoCo contact package.

Only the foot subtree is transplanted; Ultra's actuators, hand topology and all
policy/link frames remain unchanged. Internal hand self-contacts are filtered.
The contact settings below are the
released MuJoCo counterpart, NOT PhysX's dimensional 69050 N/m and 650 Ns/m
interpreted as solref. Cross-engine F-x equivalence still requires bench QA.
"""

from copy import deepcopy
from itertools import combinations
import re
import xml.etree.ElementTree as ET

import mujoco

from active_adaptation.registry import Registry
from mimic_lite.assets.a3_024 import A3_024_MJCF_PATH, PASSIVE_FOOT_JOINT_NAMES
from mimic_lite.assets.a3_ultra import make_a3_ultra_depsic_v1_cfg, ultra_mjcf_path


SUPPORT_BODIES = tuple(
    name for side in ("left", "right") for name in
    (f"{side}_ankle_roll_Link", f"{side}_foot_forefoot_Link", f"{side}_foot_toe_Link")
)
# Released MJCF: floor inherits main geom solref/solimp; feet override both.
GROUND_SOLREF = (0.005, 1.0)
GROUND_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)


def hand_internal_body_pairs(root):
    """Exclude each hand's internal contacts, not contacts outside that hand.

    Fingers are passive and outside the policy contract. Their cylinder/palm
    self-contacts can produce invalid Warp CCD witness points far from origin.
    Keep all geoms and masks, so hands still collide with the world and robot.
    """
    for side in ("left", "right"):
        name = f"{side}_wrist_yaw_Link"
        hand = root.find(f".//body[@name='{name}']")
        if hand is None:
            raise ValueError(f"Missing Ultra hand subtree: {name}")
        bodies = [body.attrib["name"] for body in hand.iter("body")]
        yield from combinations(bodies, 2)


def passive_foot_spec() -> mujoco.MjSpec:
    source = ultra_mjcf_path()
    root = ET.parse(source).getroot()
    for element in list(root):
        if element.tag in {"actuator", "keyframe"} or (
            element.tag == "include" and element.get("file") == "../../terrain/mujoco/flat.xml"
        ):
            root.remove(element)
    released = ET.parse(A3_024_MJCF_PATH).getroot()
    mesh_root = A3_024_MJCF_PATH.parent / released.find("compiler").get("meshdir")
    assets = root.find("asset")
    for mesh in released.findall("asset/mesh"):
        if re.fullmatch(r"(left|right)_foot_(rear|forefoot|toe)_(visual|collision_convex)", mesh.get("name", "")):
            mesh = deepcopy(mesh)
            mesh.set("file", str((mesh_root / mesh.get("file")).resolve()))
            assets.append(mesh)
    for side in ("left", "right"):
        name = f"{side}_ankle_roll_Link"
        dst = root.find(f".//body[@name='{name}']")
        src = released.find(f".//body[@name='{name}']")
        # Preserve the Ultra ankle joint and foot site. Replace its one-piece
        # inertia/visual/13 sphere colliders with the measured three-piece foot.
        for child in list(dst):
            if child.tag in {"inertial", "geom"}:
                dst.remove(child)
        for child in src:
            if child.tag in {"inertial", "geom", "body"}:
                dst.append(deepcopy(child))
        for index, geom in enumerate(dst.iter("geom")):
            geom.set("name", f"{side}_passive_foot_geom_{index}")
            if geom.get("class") == "collision":
                # Explicit: do not inherit Ultra's different friction default.
                geom.set("friction", "1 1 0.005")
                # Warp's margin-inflated convex CCD can report a 13.8 cm
                # penetration for a separated shoe/box (captured regression).
                # Use the actual shoe surface; keep the fitted spring/contact
                # compliance, but remove the released 1 mm detection skin.
                geom.set("margin", "0")
        for joint in dst.iter("joint"):
            if joint.get("name") in PASSIVE_FOOT_JOINT_NAMES:
                # Ultra's global joint defaults must not add motor friction.
                joint.set("frictionloss", "0")
        # Non-adjacent rear/toe meshes must not self-contact inside one shoe.
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        ET.SubElement(contact, "exclude", body1=name, body2=f"{side}_foot_toe_Link")
    existing = {frozenset((e.get("body1"), e.get("body2"))) for e in contact.findall("exclude")}
    for body1, body2 in hand_internal_body_pairs(root):
        if frozenset((body1, body2)) not in existing:
            ET.SubElement(contact, "exclude", body1=body1, body2=body2)
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.modelfiledir = str(source.parent)
    return spec


def make_a3_ultra_s2r_plus_cfg(backend: str):
    cfg, sensors = make_a3_ultra_depsic_v1_cfg(backend)
    spec = passive_foot_spec()
    cfg.spec_fn = lambda: spec
    # The parent adapter's catch-all CollisionCfg enables visual meshes too.
    # Respect the MJCF masks: in particular a passive shoe must collide using
    # its three convex collision meshes, not duplicate visual hulls.
    cfg.collisions = ()
    cfg.joint_names_simulation = [j.name for j in spec.joints if j.name != "free_joint"]
    cfg.body_names_simulation = [b.name for b in spec.bodies if b.name != "world"]
    return cfg, sensors


def make_035_terrain(*, seed: int = 42, num_rows: int = 4, num_cols: int = 2):
    from mjlab.terrains import BoxRandomGridTerrainCfg, HfRandomUniformTerrainCfg
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

    return TerrainGeneratorCfg(
        seed=seed, size=(8.0, 8.0), border_width=20.0,
        # mjlab's curriculum layout uses one column per TYPE and distributes
        # worlds by proportion (6:1). It does not enable command curriculum.
        # Random 4x7 patches made 6532 geoms; the Warp SAP int32 prefix scan
        # overflowed at 4096 worlds even though contact buffers were not full.
        num_rows=num_rows, num_cols=num_cols, curriculum=True, add_lights=False,
        sub_terrains={
            "boxes": BoxRandomGridTerrainCfg(
                proportion=0.3, grid_width=0.45,
                grid_height_range=(0.001, 0.005), platform_width=2.0,
            ),
            "random_rough": HfRandomUniformTerrainCfg(
                proportion=0.05, noise_range=(0.001, 0.005),
                # 035's noise_step=.02 / vertical_scale=.005 quantizes
                # outside the requested 1--5 mm range. Honor the physical
                # range instead of copying that discretization artifact.
                noise_step=0.001, vertical_scale=0.001,
                # Limit prism candidates for torso/limb contacts (Warp caps
                # each hfield pair at 50). Keep 1 mm vertical resolution.
                horizontal_scale=0.2, border_width=0.25,
                # A 4 mm height range must not imply a 4 mm-thick floor.
                base_thickness_ratio=250.0,
            ),
        },
    )


Registry.instance().register("asset", "a3-ultra-depsic-v1-s2r-plus", make_a3_ultra_s2r_plus_cfg)
