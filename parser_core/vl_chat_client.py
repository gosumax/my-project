"""Local, persistent PaddleOCR-VL worker bridge for isolated GPU dependencies."""

from __future__ import annotations

import json
import base64
import hashlib
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

    def read(self, path: Path, source_sha256: str, right_trim_px: int,
             prompt: str = "OCR:", board_sha: str | None = None,
             board_evidence_id: str | None = None) -> dict:
        return self._request({"path": str(path)}, source_sha256, right_trim_px,
                             prompt, board_sha, board_evidence_id)

    def read_bytes(self, source: bytes, source_sha256: str, right_trim_px: int,
                   prompt: str = "OCR:", board_sha: str | None = None,
                   board_evidence_id: str | None = None) -> dict:
        if hashlib.sha256(source).hexdigest() != source_sha256:
            raise ValueError("VL RAM crop SHA mismatch")
        return self._request({"png_base64": base64.b64encode(source).decode("ascii")},
                             source_sha256, right_trim_px, prompt, board_sha, board_evidence_id)

    def read_batch_bytes(self, items: list[dict]) -> list[dict]:
        if not 1 <= len(items) <= 8:
            raise ValueError("VL batch must contain 1..8 CHAT crops")
        requests = []
        for item in items:
            source = item["source"]
            sha = item["source_sha256"]
            if hashlib.sha256(source).hexdigest() != sha:
                raise ValueError("VL RAM crop SHA mismatch")
            self.sequence += 1
            requests.append({"request_id": str(self.sequence),
                "png_base64": base64.b64encode(source).decode("ascii"), "source_sha256": sha,
                "right_trim_px": int(item.get("right_trim_px", 12)), "prompt": "OCR:"})
        self._start()
        assert self.process is not None and self.process.stdin is not None
        batch_id = ":".join(r["request_id"] for r in requests)
        self.process.stdin.write(json.dumps({"type": "batch", "batch_id": batch_id,
                                            "requests": requests}) + "\n")
        self.process.stdin.flush()
        response = json.loads(self._next_line(130))
        if response.get("type") != "batch_result" or response.get("batch_id") != batch_id:
            raise RuntimeError("Foreign VL batch response")
        returned = response.get("results", [])
        if len(returned) != len(requests) or len({r.get("request_id") for r in returned}) != len(requests):
            raise RuntimeError("VL batch result count or identity mismatch")
        by_id = {r.get("request_id"): r for r in returned}
        ordered = []
        for request in requests:
            result = by_id.get(request["request_id"])
            if result is None or result.get("source_sha256") != request["source_sha256"]:
                raise RuntimeError("VL batch response SHA mismatch")
            ordered.append(result)
        return ordered

    def _request(self, transport: dict, source_sha256: str, right_trim_px: int,
                 prompt: str, board_sha: str | None, board_evidence_id: str | None) -> dict:
        self._start()
        assert self.process is not None
        assert self.process.stdin is not None and self.process.stdout is not None
        self.sequence += 1
        request_id = str(self.sequence)
        self.process.stdin.write(json.dumps({"request_id": request_id,
            **transport, "source_sha256": source_sha256,
            "right_trim_px": right_trim_px, "prompt": prompt,
            "board_sha": board_sha, "board_evidence_id": board_evidence_id},
            ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        response_line = self._next_line(130)
        response = json.loads(response_line)
        if response.get("type") != "result" or response.get("request_id") != request_id:
            raise RuntimeError(f"Unexpected PaddleOCR-VL worker response: {response}")
        if "png_base64" in transport and response.get("source_sha256") != source_sha256:
            raise RuntimeError("VL RAM response SHA does not match its request")
        if board_sha is not None and (response.get("board_sha") != board_sha or
                                      response.get("source_sha256") != source_sha256 or
                                      response.get("board_evidence_id") != board_evidence_id):
            raise RuntimeError("BOARD VL response SHA does not match its request")
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
