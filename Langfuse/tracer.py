# tracer.py
import json
import uuid
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional

TRACE_FILE = Path("traces.jsonl")
_TRACE_LOCK = threading.Lock()

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def write_trace(record: Dict[str, Any]) -> None:
    """Append a trace record as JSONL safely."""
    with _TRACE_LOCK:
        TRACE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

class LocalTracer:
    """Simple tracer to log runs to a local JSONL file."""

    def __init__(self, session_id: Optional[str] = None, metadata: Optional[Dict]=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.metadata = metadata or {}

    def start_run(self, name: str, input_data: Any) -> str:
        run_id = str(uuid.uuid4())
        rec = {
            "event": "start",
            "time": now_iso(),
            "run_id": run_id,
            "session_id": self.session_id,
            "name": name,
            "input": input_data,
            "metadata": self.metadata
        }
        write_trace(rec)
        return run_id

    def end_run(self, run_id: str, output_data: Any, success: bool = True, extra: Optional[Dict]=None):
        rec = {
            "event": "end",
            "time": now_iso(),
            "run_id": run_id,
            "session_id": self.session_id,
            "output": output_data,
            "success": success,
            "extra": extra or {}
        }
        write_trace(rec)

    # convenience context manager
    def run_context(self, name: str, input_data: Any):
        class Ctx:
            def __init__(self, tracer, name, input_data):
                self.tracer = tracer
                self.name = name
                self.input = input_data
                self.run_id = None
            def __enter__(self):
                self.run_id = self.tracer.start_run(self.name, self.input)
                return self
            def __exit__(self, exc_type, exc, tb):
                if exc:
                    self.tracer.end_run(self.run_id, {"error": str(exc)}, success=False)
                else:
                    # nothing to do here — caller should call end_run via ctx
                    pass
        return Ctx(self, name, input_data)
