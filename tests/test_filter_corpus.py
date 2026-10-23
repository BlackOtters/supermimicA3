import numpy as np

from a3_supermimic.data.filter_corpus import FilterConfig, classify_geometry, longest_run, name_reasons


def signals(n=50):
    root = np.zeros((n, 7))
    root[:, 2] = 1.0
    root[:, 3] = 1.0
    return np.full(n, .07), np.zeros(n), root, np.zeros((n, 6)), np.zeros(n)


def test_names_preserve_acrobatics_and_ignore_hash_fragments():
    assert name_reasons("230228__jump_off_50cm_001__original")
    assert name_reasons("230228__sit_on_chair__mirror")
    assert not name_reasons("AMASS_ACCAD_0123bed456789abc_123456789abc__original")
    for name in ("cartwheel", "handstand", "safety_roll", "kozak_dance", "monkey_jump"):
        assert not name_reasons(name)


def test_duration_requires_consecutive_frames():
    mask = np.r_[np.ones(24), np.zeros(1), np.ones(24)]
    assert longest_run(mask) == (24, 0)
    assert longest_run(np.zeros(5)) == (0, -1)
    z, foot, root, vel, rms = signals(50)
    foot[:] = .03
    vel[24, 0] = .2
    vel[49, 0] = .2
    reasons, _ = classify_geometry(z, foot, root, vel, rms, FilterConfig())
    assert "floating_feet" not in reasons
    vel[24, 0] = 0
    reasons, _ = classify_geometry(z, foot, root, vel, rms, FilterConfig())
    assert "floating_feet" in reasons


def test_elevated_duration_and_penetration_boundaries():
    z, foot, root, vel, rms = signals(100)
    z[:29] = .21
    assert "elevated_support" not in classify_geometry(z, foot, root, vel, rms, FilterConfig())[0]
    z[29] = .21
    assert "elevated_support" in classify_geometry(z, foot, root, vel, rms, FilterConfig())[0]
    z[:] = .07
    z[:80] = -.09
    assert "ground_penetration" not in classify_geometry(z, foot, root, vel, rms, FilterConfig())[0]
    z[80] = -.09
    assert "ground_penetration" in classify_geometry(z, foot, root, vel, rms, FilterConfig())[0]
    z[:] = .07
    z[0] = -.121
    assert "ground_penetration" in classify_geometry(z, foot, root, vel, rms, FilterConfig())[0]


def test_floating_excludes_moving_inverted_and_one_supported_foot():
    z, foot, root, vel, rms = signals()
    foot[:] = .03
    reasons, metrics = classify_geometry(z, foot, root, vel, rms, FilterConfig())
    assert reasons == ["floating_feet"]
    assert metrics["floating_with_unadapted_g1_standing_gate_frames"] == 0
    vel[:, 2] = .2
    assert not classify_geometry(z, foot, root, vel, rms, FilterConfig())[0]
    vel[:] = 0
    rms[:] = .6
    assert not classify_geometry(z, foot, root, vel, rms, FilterConfig())[0]
    rms[:] = 0
    root[:, 3:7] = [0, 1, 0, 0]
    assert not classify_geometry(z, foot, root, vel, rms, FilterConfig())[0]
    root[:, 3:7] = [1, 0, 0, 0]
    foot[:] = -.003  # Minimum over both feet: one supported foot prevents rejection.
    assert not classify_geometry(z, foot, root, vel, rms, FilterConfig())[0]


def test_repack_preserves_surviving_frames_and_boundaries(tmp_path):
    import json
    from a3_supermimic.data.convert_corpus import ARRAYS, FORMAT, JOINTS, read_packed_manifest, sha256_file
    from a3_supermimic.data.filter_corpus import pack_filtered

    parent = tmp_path / "parent"
    parent.mkdir()
    entries = {}
    arrays = {}
    for key, (width, dtype) in ARRAYS.items():
        a = np.arange(6 * width, dtype=np.float64).reshape(6, width).astype(dtype) / 100
        if key == "root_q":
            a[:, 3:] = [1, 0, 0, 0]
        arrays[key] = a
        np.save(parent / (key + ".npy"), a)
        entries[key] = {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha256_file(parent / (key + ".npy"))}
    clips = [{"name": f"clip_{i}", "start": 2*i, "frames": 2} for i in range(3)]
    source = {"format": FORMAT, "fps": 50, "joint_names": JOINTS, "num_clips": 3,
              "num_frames": 6, "arrays": entries, "clips": clips}
    (parent / "library.json").write_text(json.dumps(source))
    output = tmp_path / "screened"
    output.mkdir()
    (output / "audit.jsonl").write_text("{}\n")
    rows = [{"source_index": i, "keep": i != 0, "frames": 2} for i in range(3)]
    pack_filtered(parent, output / "library", source, rows, {})
    result = read_packed_manifest(output / "library/library.json", verify=True)
    assert result["num_clips"] == 2 and result["num_frames"] == 4
    assert [c["start"] for c in result["clips"]] == [0, 2]
    assert [c["parent_start"] for c in result["clips"]] == [2, 4]
    for key in ARRAYS:
        np.testing.assert_array_equal(np.load(output / "library" / (key + ".npy")), arrays[key][2:])
