import json
import time
from pathlib import Path

class Telemetry:
    def __init__(self, session_id: str, out_dir: str = "../../data/telemetry"):
        self.session_id = session_id
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        self.file_path = out_path / f"{session_id}.jsonl"
        self.frame_idx = 0
        self._f = open(self.file_path, "a")

    def log(self, tracking_valid: bool, confidence: float = None,
            inference_ms: float = None, extra: dict = None):
        record = {
            "session_id": self.session_id,
            "frame_id": self.frame_idx,
            "ts": time.time(),
            "tracking_valid": tracking_valid,
            "confidence": confidence,
            "inference_ms": inference_ms,
        }
        if extra:
            record.update(extra)
        self._f.write(json.dumps(record) + "\n")
        self.frame_idx += 1

    def close(self):
        self._f.close()
