import importlib
import zipfile


def test_training_retention_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAINING_RETENTION_ENABLED", "1")
    monkeypatch.setenv("TRAINING_DATA_DIR", str(tmp_path / "training-data"))
    monkeypatch.setenv("TRAINING_ADMIN_TOKEN", "secret")

    from webapp.api import training_retention

    tr = importlib.reload(training_retention)
    record_dir = tr.retain_upload(
        analysis_id="abc123",
        source_type="comtrade_pair",
        files=[
            tr.RetainedUploadFile("cfg_file", "case.cfg", "text/plain", b"cfg"),
            tr.RetainedUploadFile("dat_file", "case.dat", "application/octet-stream", b"dat"),
        ],
        metadata={"station_name": "GI TEST"},
    )

    assert record_dir is not None
    assert (record_dir / "case.cfg").read_bytes() == b"cfg"
    assert (record_dir / "case.dat").read_bytes() == b"dat"
    assert (record_dir / "metadata.json").exists()

    tr.append_feedback(
        {
            "analysis_id": "abc123",
            "relay_type": "21",
            "ai_correct": False,
            "actual_label": "POHON",
            "include_for_training": True,
        }
    )
    status = tr.get_training_status()
    assert status["raw_record_count"] == 1
    assert status["feedback_count"] == 1

    archive_path = tr.build_training_archive()
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert any(name.endswith("case.cfg") for name in names)
    assert "labels/feedback.csv" in names
    archive_path.unlink()

    result = tr.clear_training_archive()
    assert result["removed_raw_records"] == 1
    assert tr.get_training_status()["raw_record_count"] == 0
