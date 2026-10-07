"""Transactional, idempotent journal for source evidence and later interpretations."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import nullcontext
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL, state TEXT NOT NULL CHECK(state IN ('OPEN','EOF','PARTIAL','FAILED')),
    UNIQUE(sha256, size_bytes)
);
CREATE TABLE IF NOT EXISTS table_sessions (
    table_session_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(source_id),
    layout_id TEXT NOT NULL, geometry_json TEXT NOT NULL,
    identity_status TEXT NOT NULL CHECK(identity_status IN ('CONFIGURED','CONFIRMED','UNRESOLVED'))
);
CREATE TABLE IF NOT EXISTS frames (
    source_id TEXT NOT NULL REFERENCES sources(source_id), frame_id INTEGER NOT NULL,
    source_pts INTEGER, time_base_num INTEGER, time_base_den INTEGER,
    width INTEGER NOT NULL, height INTEGER NOT NULL, state TEXT NOT NULL,
    PRIMARY KEY(source_id, frame_id),
    CHECK(source_pts IS NULL OR (time_base_num IS NOT NULL AND time_base_den IS NOT NULL
                                 AND time_base_num > 0 AND time_base_den > 0))
);
CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, frame_id INTEGER NOT NULL,
    table_session_id TEXT NOT NULL REFERENCES table_sessions(table_session_id),
    roi_type TEXT NOT NULL, bbox_json TEXT NOT NULL, transform_json TEXT NOT NULL,
    crop_path TEXT NOT NULL, crop_sha256 TEXT NOT NULL, quality_json TEXT NOT NULL,
    status TEXT NOT NULL, reason TEXT,
    FOREIGN KEY(source_id, frame_id) REFERENCES frames(source_id, frame_id),
    UNIQUE(source_id, frame_id, table_session_id, roi_type, bbox_json)
);
CREATE TABLE IF NOT EXISTS recognitions (
    recognition_id TEXT PRIMARY KEY, observation_id TEXT NOT NULL REFERENCES observations(observation_id),
    revision INTEGER NOT NULL CHECK(revision > 0), raw_text TEXT, engine TEXT NOT NULL,
    model_version TEXT, preprocessing_json TEXT NOT NULL, raw_score TEXT,
    alternatives_json TEXT NOT NULL, status TEXT NOT NULL, reason TEXT,
    UNIQUE(observation_id, engine, revision)
);
CREATE TABLE IF NOT EXISTS physical_rows (
    physical_row_id TEXT PRIMARY KEY, table_session_id TEXT NOT NULL REFERENCES table_sessions(table_session_id),
    chat_epoch TEXT NOT NULL, first_source_pts INTEGER, last_source_pts INTEGER,
    state TEXT NOT NULL, reason TEXT
);
CREATE TABLE IF NOT EXISTS row_observations (
    physical_row_id TEXT NOT NULL REFERENCES physical_rows(physical_row_id),
    observation_id TEXT NOT NULL REFERENCES observations(observation_id),
    ordinal INTEGER NOT NULL, PRIMARY KEY(physical_row_id, observation_id)
);
CREATE TABLE IF NOT EXISTS row_decisions (
    physical_row_id TEXT NOT NULL REFERENCES physical_rows(physical_row_id),
    observation_id TEXT NOT NULL REFERENCES observations(observation_id),
    algorithm_version TEXT NOT NULL, decision TEXT NOT NULL, reason TEXT,
    shift_px INTEGER, alignment_score REAL, zero_score REAL,
    PRIMARY KEY(physical_row_id, observation_id, algorithm_version)
);
CREATE TABLE IF NOT EXISTS row_suppressions (
    source_id TEXT NOT NULL, frame_id INTEGER NOT NULL,
    table_session_id TEXT NOT NULL REFERENCES table_sessions(table_session_id),
    top_local_y INTEGER NOT NULL, bottom_local_y INTEGER NOT NULL,
    algorithm_version TEXT NOT NULL, reason TEXT NOT NULL,
    PRIMARY KEY(source_id, frame_id, table_session_id,
                top_local_y, bottom_local_y, algorithm_version),
    FOREIGN KEY(source_id, frame_id) REFERENCES frames(source_id, frame_id),
    CHECK(0 <= top_local_y AND top_local_y < bottom_local_y)
);
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY, table_session_id TEXT NOT NULL REFERENCES table_sessions(table_session_id),
    completeness TEXT NOT NULL, start_status TEXT NOT NULL, end_status TEXT NOT NULL,
    reason TEXT, assembler_version TEXT NOT NULL DEFAULT 'legacy_unknown'
);
CREATE TABLE IF NOT EXISTS message_fragments (
    message_id TEXT NOT NULL REFERENCES messages(message_id), ordinal INTEGER NOT NULL,
    physical_row_id TEXT NOT NULL REFERENCES physical_rows(physical_row_id),
    recognition_id TEXT REFERENCES recognitions(recognition_id),
    PRIMARY KEY(message_id, ordinal)
);
CREATE TABLE IF NOT EXISTS event_revisions (
    event_id TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision > 0),
    table_session_id TEXT NOT NULL REFERENCES table_sessions(table_session_id),
    message_id TEXT REFERENCES messages(message_id), observation_id TEXT REFERENCES observations(observation_id),
    event_type TEXT NOT NULL, actor TEXT, seat_id TEXT, amount_decimal TEXT, amount_unit TEXT,
    amount_role TEXT, cards_json TEXT NOT NULL, order_key TEXT NOT NULL,
    observed_source_pts INTEGER, action_time_status TEXT NOT NULL,
    status TEXT NOT NULL, reason TEXT, normalizer_version TEXT NOT NULL,
    evidence_json TEXT NOT NULL, PRIMARY KEY(event_id, revision),
    CHECK(message_id IS NOT NULL OR observation_id IS NOT NULL),
    CHECK((amount_decimal IS NULL AND amount_unit IS NULL) OR
          (amount_decimal IS NOT NULL AND amount_unit IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS hand_revisions (
    hand_id TEXT NOT NULL, revision INTEGER NOT NULL CHECK(revision > 0),
    table_session_id TEXT NOT NULL REFERENCES table_sessions(table_session_id),
    status TEXT NOT NULL CHECK(status IN ('DRAFT','PARTIAL','REJECTED','VERIFIED')),
    reasons_json TEXT NOT NULL, event_versions_json TEXT NOT NULL,
    state_json TEXT NOT NULL, validation_json TEXT NOT NULL,
    PRIMARY KEY(hand_id, revision)
);
CREATE TABLE IF NOT EXISTS exports (
    export_id TEXT PRIMARY KEY, hand_id TEXT NOT NULL, hand_revision INTEGER NOT NULL,
    format TEXT NOT NULL, exporter_version TEXT NOT NULL, content_sha256 TEXT,
    path TEXT, status TEXT NOT NULL CHECK(status IN ('DONE','REFUSED','STALE')),
    reason TEXT, FOREIGN KEY(hand_id, hand_revision) REFERENCES hand_revisions(hand_id, revision),
    UNIQUE(hand_id, hand_revision, format, exporter_version)
);
CREATE TABLE IF NOT EXISTS gaps (
    gap_id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(source_id),
    table_session_id TEXT REFERENCES table_sessions(table_session_id),
    first_pts INTEGER, last_pts INTEGER, reason TEXT NOT NULL
);
"""


def stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


class Journal:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=120)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        self.db.execute("CREATE INDEX IF NOT EXISTS observations_roi_frame ON observations(roi_type,frame_id,table_session_id)")
        self.db.execute("CREATE INDEX IF NOT EXISTS row_observations_first ON row_observations(physical_row_id,ordinal)")
        message_columns = {row[1] for row in self.db.execute("PRAGMA table_info(messages)")}
        if "assembler_version" not in message_columns:
            self.db.execute("""ALTER TABLE messages ADD COLUMN assembler_version TEXT
                NOT NULL DEFAULT 'legacy_unknown'""")
        self._batched = False

    def close(self) -> None:
        if self._batched:
            self.db.rollback()
        self.db.close()

    def begin_batch(self) -> None:
        if self._batched:
            raise RuntimeError("Batch already active")
        self.db.execute("BEGIN IMMEDIATE")
        self._batched = True

    def end_batch(self, commit: bool = True) -> None:
        if not self._batched:
            raise RuntimeError("No active batch")
        try:
            self.db.commit() if commit else self.db.rollback()
        finally:
            self._batched = False

    def _write_scope(self):
        return nullcontext() if self._batched else self.db

    def add_source(self, source_id: str, path: str, sha256: str, size: int) -> None:
        with self._write_scope():
            old = self.db.execute("SELECT sha256,size_bytes FROM sources WHERE source_id=?",
                                  (source_id,)).fetchone()
            if old is not None and old != (sha256, size):
                raise ValueError(f"Source identity conflict: {source_id}")
            self.db.execute("INSERT INTO sources VALUES (?,?,?,?,?) ON CONFLICT(source_id) DO NOTHING",
                            (source_id, path, sha256, size, "OPEN"))

    def set_source_state(self, source_id: str, state: str) -> None:
        with self._write_scope():
            self.db.execute("UPDATE sources SET state=CASE WHEN state='EOF' THEN 'EOF' ELSE ? END WHERE source_id=?",
                            (state, source_id))

    def add_session(self, session_id: str, source_id: str, layout_id: str,
                    geometry: dict, identity_status: str = "CONFIGURED") -> None:
        geometry_json = json.dumps(geometry, sort_keys=True)
        with self._write_scope():
            old = self.db.execute("""SELECT source_id,layout_id,geometry_json,identity_status
                FROM table_sessions WHERE table_session_id=?""", (session_id,)).fetchone()
            expected = (source_id, layout_id, geometry_json, identity_status)
            if old is not None and old != expected:
                raise ValueError(f"Table session identity conflict: {session_id}")
            self.db.execute("INSERT INTO table_sessions VALUES (?,?,?,?,?) ON CONFLICT(table_session_id) DO NOTHING",
                            (session_id, source_id, layout_id, geometry_json, identity_status))

    def add_frame(self, source_id: str, frame_id: int, pts: int | None,
                  time_base_num: int | None, time_base_den: int | None,
                  width: int, height: int, state: str = "DECODED") -> None:
        with self._write_scope():
            old = self.db.execute("""SELECT source_pts,time_base_num,time_base_den,width,height,state
                FROM frames WHERE source_id=? AND frame_id=?""", (source_id, frame_id)).fetchone()
            expected = (pts, time_base_num, time_base_den, width, height, state)
            if old is not None and old != expected:
                raise ValueError(f"Frame identity conflict: {source_id}:{frame_id}")
            self.db.execute("INSERT INTO frames VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(source_id,frame_id) DO NOTHING",
                            (source_id, frame_id, pts, time_base_num, time_base_den,
                             width, height, state))

    def add_observation(self, observation_id: str, source_id: str, frame_id: int,
                        session_id: str, roi_type: str, bbox: list[int], crop_path: str,
                        crop_sha256: str, status: str, reason: str | None = None,
                        quality: dict | None = None) -> None:
        bbox_json = json.dumps(bbox, separators=(",", ":"))
        quality_json = json.dumps(quality or {}, sort_keys=True)
        with self._write_scope():
            old = self.db.execute("""SELECT source_id,frame_id,table_session_id,roi_type,bbox_json,crop_path,crop_sha256,quality_json,status,reason
                FROM observations WHERE observation_id=?""", (observation_id,)).fetchone()
            expected = (source_id, frame_id, session_id, roi_type, bbox_json,
                        crop_path, crop_sha256, quality_json, status, reason)
            if old is not None and old != expected:
                raise ValueError(f"Observation identity conflict: {observation_id}")
            self.db.execute("""INSERT INTO observations VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(observation_id) DO NOTHING""",
                (observation_id, source_id, frame_id, session_id, roi_type, bbox_json,
                 json.dumps({"type": "identity", "coordinate_space": "source_frame"}),
                 crop_path, crop_sha256, quality_json, status, reason))

    def add_recognition(self, observation_id: str, raw_text: str | None, engine: str,
                        model_version: str, preprocessing: dict, status: str,
                        reason: str | None = None) -> tuple[str, int]:
        preprocessing_json = json.dumps(preprocessing, sort_keys=True)
        with self._write_scope():
            latest = self.db.execute("""SELECT recognition_id,revision,raw_text,model_version,
                preprocessing_json,status,reason FROM recognitions
                WHERE observation_id=? AND engine=? ORDER BY revision DESC LIMIT 1""",
                (observation_id, engine)).fetchone()
            if latest is not None and latest[2:] == (raw_text, model_version,
                                                       preprocessing_json, status, reason):
                return latest[0], latest[1]
            revision = 1 if latest is None else latest[1] + 1
            recognition_id = stable_id("rec", observation_id, engine, revision)
            self.db.execute("""INSERT INTO recognitions VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                            (recognition_id, observation_id, revision, raw_text, engine,
                             model_version, preprocessing_json, None, "[]", status, reason))
            return recognition_id, revision

    def add_physical_row(self, row_id: str, session_id: str, chat_epoch: str,
                         source_pts: int | None, state: str, reason: str | None = None) -> None:
        with self._write_scope():
            old = self.db.execute("""SELECT table_session_id,chat_epoch,state,reason
                FROM physical_rows WHERE physical_row_id=?""", (row_id,)).fetchone()
            if old is not None and old != (session_id, chat_epoch, state, reason):
                raise ValueError(f"Physical row identity conflict: {row_id}")
            self.db.execute("""INSERT INTO physical_rows VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(physical_row_id) DO NOTHING""",
                (row_id, session_id, chat_epoch, source_pts, source_pts, state, reason))
            if source_pts is not None:
                self.db.execute("""UPDATE physical_rows SET
                    first_source_pts=MIN(COALESCE(first_source_pts,?),?),
                    last_source_pts=MAX(COALESCE(last_source_pts,?),?)
                    WHERE physical_row_id=?""",
                    (source_pts, source_pts, source_pts, source_pts, row_id))

    def link_row_observation(self, row_id: str, observation_id: str, frame_id: int) -> None:
        with self._write_scope():
            self.db.execute("""INSERT INTO row_observations VALUES (?,?,?)
                ON CONFLICT(physical_row_id,observation_id) DO NOTHING""",
                (row_id, observation_id, frame_id))

    def add_row_decision(self, row_id: str, observation_id: str, version: str,
                         decision: str, reason: str | None, shift_px: int | None,
                         score: float | None, zero_score: float | None) -> None:
        values = (row_id, observation_id, version, decision, reason,
                  shift_px, score, zero_score)
        with self._write_scope():
            old = self.db.execute("""SELECT physical_row_id,observation_id,
                algorithm_version,decision,reason,shift_px,alignment_score,zero_score
                FROM row_decisions WHERE physical_row_id=? AND observation_id=?
                AND algorithm_version=?""", (row_id, observation_id, version)).fetchone()
            if old is not None and old != values:
                raise ValueError(f"Row decision conflict: {row_id}:{observation_id}")
            self.db.execute("""INSERT INTO row_decisions VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(physical_row_id,observation_id,algorithm_version) DO NOTHING""",
                values)

    def add_row_suppression(self, source_id: str, frame_id: int,
                            session_id: str, span: tuple[int, int],
                            version: str, reason: str) -> None:
        values = (source_id, frame_id, session_id, span[0], span[1], version, reason)
        with self._write_scope():
            old = self.db.execute("""SELECT * FROM row_suppressions
                WHERE source_id=? AND frame_id=? AND table_session_id=?
                  AND top_local_y=? AND bottom_local_y=? AND algorithm_version=?""",
                values[:6]).fetchone()
            if old is not None and old != values:
                raise ValueError(f"Row suppression conflict: {source_id}:{frame_id}:{session_id}")
            self.db.execute("""INSERT INTO row_suppressions VALUES (?,?,?,?,?,?,?)
                ON CONFLICT DO NOTHING""", values)

    def add_gap(self, gap_id: str, source_id: str, session_id: str,
                first_pts: int | None, last_pts: int | None, reason: str) -> None:
        with self._write_scope():
            self.db.execute("""INSERT INTO gaps VALUES (?,?,?,?,?,?)
                ON CONFLICT(gap_id) DO NOTHING""",
                (gap_id, source_id, session_id, first_pts, last_pts, reason))

    def add_message(self, message_id: str, session_id: str, completeness: str,
                    start_status: str, end_status: str, reason: str | None,
                    assembler_version: str) -> None:
        values = (message_id, session_id, completeness, start_status,
                  end_status, reason, assembler_version)
        with self._write_scope():
            old = self.db.execute("SELECT * FROM messages WHERE message_id=?",
                                  (message_id,)).fetchone()
            if old is not None and old != values:
                raise ValueError(f"Message identity conflict: {message_id}")
            self.db.execute("""INSERT INTO messages VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(message_id) DO NOTHING""", values)

    def add_message_fragment(self, message_id: str, ordinal: int,
                             row_id: str, recognition_id: str | None) -> None:
        values = (message_id, ordinal, row_id, recognition_id)
        with self._write_scope():
            old = self.db.execute("""SELECT * FROM message_fragments
                WHERE message_id=? AND ordinal=?""", (message_id, ordinal)).fetchone()
            if old is not None and old != values:
                raise ValueError(f"Message fragment conflict: {message_id}:{ordinal}")
            self.db.execute("""INSERT INTO message_fragments VALUES (?,?,?,?)
                ON CONFLICT(message_id,ordinal) DO NOTHING""", values)

    def append_event(self, record: dict) -> tuple[str, int]:
        """Append a changed interpretation while preserving earlier revisions."""
        columns = ("event_id", "revision", "table_session_id", "message_id",
                   "observation_id", "event_type", "actor", "seat_id",
                   "amount_decimal", "amount_unit", "amount_role", "cards_json",
                   "order_key", "observed_source_pts", "action_time_status",
                   "status", "reason", "normalizer_version", "evidence_json")
        event_id = record["event_id"]
        with self._write_scope():
            latest = self.db.execute("""SELECT * FROM event_revisions
                WHERE event_id=? ORDER BY revision DESC LIMIT 1""", (event_id,)).fetchone()
            revision = 1 if latest is None else latest[1] + 1
            values = tuple(record.get(name) if name != "revision" else revision
                           for name in columns)
            if latest is not None and latest[:1] + latest[2:] == values[:1] + values[2:]:
                return event_id, latest[1]
            self.db.execute("""INSERT INTO event_revisions VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
            return event_id, revision

    def append_hand(self, record: dict) -> tuple[str, int]:
        columns = ("hand_id", "revision", "table_session_id", "status",
                   "reasons_json", "event_versions_json", "state_json", "validation_json")
        hand_id = record["hand_id"]
        with self._write_scope():
            latest = self.db.execute("""SELECT * FROM hand_revisions
                WHERE hand_id=? ORDER BY revision DESC LIMIT 1""", (hand_id,)).fetchone()
            revision = 1 if latest is None else latest[1] + 1
            values = tuple(record.get(name) if name != "revision" else revision
                           for name in columns)
            if latest is not None and latest[:1] + latest[2:] == values[:1] + values[2:]:
                return hand_id, latest[1]
            self.db.execute("INSERT INTO hand_revisions VALUES (?,?,?,?,?,?,?,?)", values)
            return hand_id, revision

    def counts(self) -> dict[str, int]:
        return {table: self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("sources", "table_sessions", "frames", "observations",
                              "recognitions", "physical_rows", "row_decisions",
                              "row_suppressions", "messages", "event_revisions",
                              "hand_revisions", "exports", "gaps")}
