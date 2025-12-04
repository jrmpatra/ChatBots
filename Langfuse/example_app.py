# example_app.py
from tracer import LocalTracer
import time

tr = LocalTracer(metadata={"user": "localdev"})

# Example: instrumenting a hypothetical inference call
run_id = tr.start_run("openai_call", {"prompt": "Say hi"})
time.sleep(0.2)  # simulate work
tr.end_run(run_id, {"response": "Hello!"})

# Or use the context manager:
with tr.run_context("composed_task", {"step": 1}) as ctx:
    # do work...
    result = {"final": "ok"}
    # when done, call end_run explicitly:
    ctx.tracer.end_run(ctx.run_id, result)
