"""Local, persistent PaddleOCR-VL worker bridge for isolated GPU dependencies."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from pathlib import Path


class VLChatClient:
    def __init__(self, python: Path, model_dir: Path, weights_sha256: str,
                 stderr_path: Path):
        self.python = python
        self.model_dir = model_dir
        self.weights_sha256 = weights_sha256
        self.stderr_path = stderr_path
        self.process: subprocess.Popen[str] | None = None
        self.stderr_file = None
        self.sequence = 0
        self.lines: queue.Queue[str | None] = queue.Queue()

    def _read_stdout(self) -> None:
        process = self.process
        line_queue = self.lines
        assert process is not None and process.stdout is not None
        for line in process.stdout:
            line_queue.put(line)
        line_queue.put(None)

    def _next_line(self, timeout: int) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError(f"PaddleOCR-VL worker exceeded {timeout}s") from exc
        if line is None:
            raise RuntimeError("PaddleOCR-VL worker exited before responding")
        return line

    def _start(self) -> None:
        if self.process is not None:
            return
        if not self.python.is_file() or not self.model_dir.is_dir():
            raise FileNotFoundError("PaddleOCR-VL Python or model directory is missing")
        self.lines = queue.Queue()
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_file = self.stderr_path.open("a", encoding="utf-8")
        command = [str(self.python), "-u",
                   str(Path(__file__).resolve().parents[1] / "scripts" / "vl_chat_worker.py"),
                   "--model-dir", str(self.model_dir),
                   "--weights-sha256", self.weights_sha256]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr_file, text=True, encoding="utf-8", errors="replace",
            bufsize=1, creationflags=flags,
        )
        threading.Thread(target=self._read_stdout, daemon=True).start()
        # The 1.9 GB model is on a HDD here; verifying its SHA-256 alone took
        # 250 seconds on the Smoke12 machine. This is a one-time worker startup.
        ready_line = self._next_line(600)
        ready = json.loads(ready_line)
        if ready.get("type") != "ready":
            raise RuntimeError(f"Unexpected PaddleOCR-VL worker response: {ready}")

    def read(self, path: Path, source_sha256: str, right_trim_px: int) -> dict:
        self._start()
        assert self.process is not None
        assert self.process.stdin is not None and self.process.stdout is not None
        self.sequence += 1
        request_id = str(self.sequence)
        self.process.stdin.write(json.dumps({"request_id": request_id,
            "path": str(path), "source_sha256": source_sha256,
            "right_trim_px": right_trim_px}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response_line = self._next_line(130)
        response = json.loads(response_line)
        if response.get("type") != "result" or response.get("request_id") != request_id:
            raise RuntimeError(f"Unexpected PaddleOCR-VL worker response: {response}")
        return response

    def close(self) -> None:
        if self.process is not None:
            if self.process.stdin:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
            self.process = None
        if self.stderr_file is not None:
            self.stderr_file.close()
            self.stderr_file = None
