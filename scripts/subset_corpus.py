"""Select a corpus from an already converted/filtered packed motion library."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

import numpy as np

from a3_supermimic.data.convert_corpus import ARRAYS, read_packed_manifest, sha256_file


def select_corpus(source: Path, output: Path, corpus: str):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifest_path = source / "library.json"
    parent_hash = sha256_file(manifest_path)
    manifest = read_packed_manifest(manifest_path, verify=True)
    selected = [(i, c) for i, c in enumerate(manifest["clips"])
                if c["source"].split("/")[0] == corpus]
    if not selected:
        raise ValueError(f"No clips from {corpus!r} in {source}")
    if not all(c["name"].startswith(corpus + "__") for _, c in selected):
        raise ValueError("Corpus labels disagree between source paths and clip names")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=output.name + ".pending-", dir=output.parent))
    total = sum(c["frames"] for _, c in selected)
    arrays = {k: np.load(source / (k + ".npy"), mmap_mode="r", allow_pickle=False) for k in ARRAYS}
    copied = {k: np.lib.format.open_memmap(pending / (k + ".npy"), mode="w+", dtype=dtype,
                                         shape=(total, width)) for k, (width, dtype) in ARRAYS.items()}
    clips, cursor = [], 0
    for index, clip in selected:
        start, n = clip["start"], clip["frames"]
        for key in ARRAYS:
            copied[key][cursor:cursor+n] = arrays[key][start:start+n]
        row = dict(clip)
        row.update(start=cursor, subset_parent_index=index, subset_parent_start=start)
        clips.append(row)
        cursor += n
    entries = {}
    for key, a in copied.items():
        a.flush()
        entries[key] = {"shape": list(a.shape), "dtype": str(a.dtype),
                        "sha256": sha256_file(pending / (key + ".npy"))}
    result = dict(manifest)
    result.update(num_clips=len(clips), num_frames=total, total_hours=total / manifest["fps"] / 3600,
                  clips=clips, arrays=entries,
                  statistics={k: max(c[k] for c in clips) for k in manifest.get("statistics", {})},
                  source_manifests=[m for m in manifest.get("source_manifests", []) if m["group"] == corpus],
                  subset={"corpus": corpus, "parent_manifest_sha256": parent_hash,
                          "parent_library": str(source), "coordinates_modified": False,
                          "augmentations": dict(Counter(c["name"].rsplit("__", 1)[-1] for c in clips)),
                          "parent_filtering_preserved": "filtering" in manifest})
    manifest_out = pending / "library.json"
    manifest_out.write_text(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    read_packed_manifest(manifest_out, verify=True)
    for clip in clips:
        for key in ARRAYS:
            if not np.array_equal(copied[key][clip["start"]:clip["start"]+clip["frames"]],
                                  arrays[key][clip["subset_parent_start"]:clip["subset_parent_start"]+clip["frames"]]):
                raise ValueError(f"Repacking changed {clip['name']} / {key}")
    assert sha256_file(manifest_path) == parent_hash, "Parent manifest changed during extraction"
    pending.rename(output)
    report = {"corpus": corpus, "clips": len(clips), "frames": total, "hours": result["total_hours"],
              "augmentations": result["subset"]["augmentations"], "parent_manifest_sha256": parent_hash,
              "manifest_sha256": sha256_file(output / "library.json"),
              "all_slices_exact": True, "output": str(output)}
    (output / "subset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corpus", required=True)
    args = parser.parse_args()
    print(json.dumps(select_corpus(args.source, args.output, args.corpus), indent=2))


if __name__ == "__main__":
    main()
