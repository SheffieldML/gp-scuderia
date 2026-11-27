"""Run a command and record its outcome as JSON for the report.

Usage: python scripts/record_check.py <out.json> <name> <description> -- <command...>
"""
import json, pathlib, subprocess, sys, time
out, name, desc = sys.argv[1:4]
cmd = sys.argv[sys.argv.index("--") + 1:]
t0 = time.time()
p = subprocess.run(cmd, capture_output=True, text=True)
text = (p.stdout + p.stderr)
sys.stdout.write(text)
lines = [l for l in text.splitlines() if l.strip()]
pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(out).write_text(json.dumps({
    "check": name, "description": desc, "command": " ".join(cmd), "ok": p.returncode == 0,
    "returncode": p.returncode, "seconds": round(time.time() - t0, 1),
    "ok_lines": sum(l.startswith(("ok", "PASS")) for l in lines),
    "fail_lines": sum(l.startswith(("FAIL", "FAILED")) for l in lines),
    "tail": lines[-12:]}, indent=2) + "\n")
sys.exit(p.returncode)
