"""Client for the isolated persistent PP-OCRv6 tiny worker."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


class TinyChatClient:
    def __init__(self, python: Path, model_dir: Path, model_name: str,
                 weights_sha256: str, device: str, stderr_path: Path):
        self.python = python
        self.model_dir = model_dir
        self.model_name = model_name
        self.weights_sha256 = weights_sha256
        self.device = device
        self.stderr_path = stderr_path
        self.process: subprocess.Popen[str] | None = None
        self.stderr_file = None
        self.sequence = 0
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.ready: dict = {}

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _next_line(self, timeout: int) -> str:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError(f"PP-OCRv6 tiny worker exceeded {timeout}s") from exc
        if line is None:
            raise RuntimeError("PP-OCRv6 tiny worker exited before responding")
        return line

    def _next_json(self, timeout: int) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"PP-OCRv6 tiny worker exceeded {timeout}s")
            line = self._next_line(max(1, int(remaining)))
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if self.stderr_file is not None:
                    self.stderr_file.write("stdout_noise: " + line)
                    self.stderr_file.flush()
                continue
            if isinstance(value, dict):
                return value

    def _start(self) -> None:
        if self.process is not None:
            return
        if not self.python.is_file() or not self.model_dir.is_dir():
            raise FileNotFoundError("PP-OCRv6 tiny Python or model directory is missing")
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_file = self.stderr_path.open("a", encoding="utf-8")
        command = [str(self.python), "-u",
                   str(Path(__file__).resolve().parents[1] / "scripts" / "tiny_chat_worker.py"),
                   "--model-dir", str(self.model_dir), "--model-name", self.model_name,
                   "--weights-sha256", self.weights_sha256, "--device", self.device]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=self.stderr_file, text=True,
            encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        self.ready = self._next_json(300)
        if self.ready.get("type") != "ready":
            raise RuntimeError(f"Unexpected tiny worker response: {self.ready}")

    def read_batch(self, items: list[dict], batch_size: int) -> dict:
        self._start()
        assert self.process is not None and self.process.stdin is not None
        self.sequence += 1
        request_id = str(self.sequence)
        self.process.stdin.write(json.dumps({"request_id": request_id,
            "batch_size": batch_size, "items": items}, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response = self._next_json(180)
        if response.get("type") != "result" or response.get("request_id") != request_id:
            raise RuntimeError(f"Unexpected tiny worker response: {response}")
        if response.get("status") == "ERROR":
            raise RuntimeError(response.get("error", "Tiny worker failed"))
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
