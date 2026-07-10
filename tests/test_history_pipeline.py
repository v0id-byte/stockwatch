import json

from scripts.build_training_set import _training_stock_paths


def test_training_stock_paths_follow_exact_manifest_codes(tmp_path):
    stock_dir = tmp_path / "stocks"
    stock_dir.mkdir()
    for code in ("000001", "000002", "000003"):
        (stock_dir / f"{code}.parquet").touch()
    (tmp_path / "history_manifest.json").write_text(json.dumps({
        "codes": ["000001", "000003"],
        "constituent_membership_kind": "current_snapshot_not_point_in_time",
    }))

    paths, meta = _training_stock_paths(tmp_path, stock_dir)

    assert [path.stem for path in paths] == ["000001", "000003"]
    assert meta["source"] == "history_manifest.codes"
    assert meta["membership_kind"] == "current_snapshot_not_point_in_time"


def test_training_stock_paths_fall_back_for_legacy_manifest(tmp_path):
    stock_dir = tmp_path / "stocks"
    stock_dir.mkdir()
    for code in ("000001", "000002"):
        (stock_dir / f"{code}.parquet").touch()

    paths, meta = _training_stock_paths(tmp_path, stock_dir)

    assert [path.stem for path in paths] == ["000001", "000002"]
    assert meta["source"] == "stock_directory_legacy_fallback"
