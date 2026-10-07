import hashlib
from fractions import Fraction
from types import SimpleNamespace

import pytest

from parser_core.journal import Journal
from scripts.capture_predeclared_windows import checked_resume_frames, selected_frames


def test_predeclared_sampling_keeps_unique_ordered_windows():
    assert selected_frames([{"first_frame": 0, "last_frame": 9, "step": 3},
                            {"first_frame": 20, "last_frame": 24, "step": 2}], 25) == {
                                0, 3, 6, 9, 20, 22, 24}


@pytest.mark.parametrize("windows", [
    [{"first_frame": 0, "last_frame": 9, "step": 0}],
    [{"first_frame": 0, "last_frame": 9, "step": 1},
     {"first_frame": 9, "last_frame": 20, "step": 1}],
    [{"first_frame": -1, "last_frame": 2, "step": 1}],
    [{"first_frame": 0, "last_frame": 26, "step": 1}],
])
def test_bad_windows_fail_before_capture(windows):
    with pytest.raises(ValueError):
        selected_frames(windows, 25)


def test_resume_checks_complete_frame_and_crop_hashes(tmp_path):
    run = tmp_path / "raw"
    journal = Journal(run / "journal.sqlite3")
    journal.add_source("source1", "video.mp4", "sourcehash", 10)
    journal.add_session("slot1", "source1", "layout1", {"sampling_plan_sha256": "planhash"},
                        "UNRESOLVED")
    journal.add_frame("source1", 0, 0, 1, 30, 2560, 1440)
    for kind in ("table", "chat"):
        relative = f"crops/{kind}.png"
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(kind.encode())
        journal.add_observation(kind, "source1", 0, "slot1", kind,
                                [0, 0, 10, 10], relative,
                                hashlib.sha256(path.read_bytes()).hexdigest(), "RAW")
    journal.close()
    assert checked_resume_frames(run, "source1", {0}, {1: "slot1"}, "planhash") == {
        0: (0, 1, 30, 2560, 1440)}
    with pytest.raises(ValueError, match="sampling plan differs"):
        checked_resume_frames(run, "source1", {0}, {1: "slot1"}, "other")
    (run / "crops/chat.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Stored crop changed"):
        checked_resume_frames(run, "source1", {0}, {1: "slot1"}, "planhash")


def test_capture_roi_failure_does_not_commit_an_incomplete_frame(tmp_path, monkeypatch):
    from PIL import Image
    from scripts import capture_predeclared_windows as module

    video = tmp_path / "source.mp4"
    video.write_bytes(b"mock source")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text("frozen mock plan", encoding="utf-8")
    plan = {"video": str(video), "source_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "slots": [1], "windows": [{"first_frame": 0, "last_frame": 29, "step": 1}]}
    layout = SimpleNamespace(source_width=100, source_height=80, layout_id="mock_layout",
        slots=[SimpleNamespace(slot_id=1, table_box=(0, 0, 100, 80), chat_box=(0, 0, 40, 60))])
    monkeypatch.setattr(module, "load_fixed_layout", lambda *args: layout)

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(frames=30, time_base=Fraction(1, 30))])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def decode(self, stream):
            for index in range(30):
                yield SimpleNamespace(pts=index, time_base=Fraction(1, 30), width=100, height=80,
                    to_image=lambda: Image.new("RGB", (100, 80), "black"))

    monkeypatch.setattr(module.av, "open", lambda *args: Container())
    add_observation = Journal.add_observation
    def fail_after_table(self, *args, **kwargs):
        result = add_observation(self, *args, **kwargs)
        if args[2] == 12 and args[4] == "table":
            raise RuntimeError("ROI writer failed")
        return result
    monkeypatch.setattr(Journal, "add_observation", fail_after_table)
    run = tmp_path / "raw"
    with pytest.raises(RuntimeError, match="ROI writer failed"):
        module.capture(plan, plan_path, run)
    journal = Journal(run / "journal.sqlite3")
    try:
        assert journal.db.execute("SELECT COUNT(*),MAX(frame_id) FROM frames").fetchone() == (10, 9)
        assert journal.db.execute("SELECT COUNT(*) FROM observations").fetchone() == (20,)
        assert journal.db.execute("SELECT state FROM sources").fetchone() == ("FAILED",)
        assert not journal.db.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        journal.close()
