# viewer.py
from flask import Flask, render_template_string, request
from pathlib import Path
import json

app = Flask(__name__)
TRACE_FILE = Path("traces.jsonl")

TEMPLATE_INDEX = """
<!doctype html>
<title>Local Traces</title>
<h1>Traces ({{count}})</h1>
<form method="get">
  <input name="q" placeholder="search run_id / name / metadata" value="{{q or ''}}" style="width:40rem;">
  <button>Search</button>
</form>
<ul>
{% for t in traces %}
  <li><b>{{t.get('time')}}</b> — <i>{{t.get('event')}}</i> — {{t.get('name') or t.get('run_id')}} — <a href="/trace/{{t['run_id']}}">view</a></li>
{% endfor %}
</ul>
"""

TEMPLATE_TRACE = """
<!doctype html>
<title>Trace {{run_id}}</title>
<a href="/">← back</a>
<h2>Trace {{run_id}}</h2>
<pre>{{trace_json}}</pre>
"""

def load_traces():
    if not TRACE_FILE.exists():
        return []
    out = []
    with TRACE_FILE.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    # newest first
    out.reverse()
    return out

@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    traces = load_traces()
    if q:
        qlow = q.lower()
        traces = [t for t in traces if qlow in json.dumps(t).lower()]
    return render_template_string(TEMPLATE_INDEX, traces=traces, count=len(traces), q=q)

@app.route("/trace/<run_id>")
def trace_view(run_id):
    traces = load_traces()
    # show all events for run_id
    run_events = [t for t in traces if t.get("run_id") == run_id]
    if not run_events:
        return "Not found", 404
    return render_template_string(TEMPLATE_TRACE, run_id=run_id, trace_json=json.dumps(run_events, indent=2))

if __name__ == "__main__":
    app.run(debug=True, port=8080)
