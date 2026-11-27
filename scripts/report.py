"""Build report.md and report.html from compare.py / record_check.py JSON files.

Usage: python scripts/report.py <results-dir> [--out dir]
Result files are <results-dir>/**/*.json. Comparison files have a 'framework' key and check files
a 'check' key.
"""
import argparse, html, json, pathlib

ap = argparse.ArgumentParser()
ap.add_argument("results")
ap.add_argument("--out", default=".")
a = ap.parse_args()
docs = [json.loads(p.read_text()) for p in sorted(pathlib.Path(a.results).rglob("*.json"))]
comps = [d for d in docs if "framework" in d]
checks = [d for d in docs if "check" in d]
cols = sorted({(d["framework"], d["mode"]) for d in comps})
goldens = sorted({r["golden"] for d in comps for r in d["results"]})
by = {(d["framework"], d["mode"], r["golden"]): r for d in comps for r in d["results"]}


def frac(r):
    dev = r.get("deviation")
    return max(dev[k] / r["tolerance"][k] for k in dev) if dev else None


def cell(r):
    """(symbol text, css class)"""
    if r is None:
        return "-", "na"
    if not r["ok"]:
        return "UNEXPECTED " + r["status"], "bad"
    if r["status"] == "refused":
        return "refused", "ref"
    f = frac(r)
    return "match %.0f%%" % (100 * f), "good"


unexpected = [(d["framework"], d["mode"], r) for d in comps for r in d["results"] if not r["ok"]]
bad_checks = [c for c in checks if not c["ok"]]
verdict = "PASS" if not unexpected and not bad_checks else "FAIL"

md = ["# gp-scuderia conformance report", "",
      "**Verdict: %s** — %d unexpected result(s), %d failing check(s)." % (verdict, len(unexpected), len(bad_checks)), "",
      "Cells show the largest deviation as a percentage of the golden's tolerance (100% is the limit). "
      "`refused` means the adapter raised `UnsupportedByFramework` as the registry expects.", "",
      "| golden | " + " | ".join("%s (%s)" % c for c in cols) + " |",
      "|---|" + "---|" * len(cols)]
for g in goldens:
    md.append("| %s | %s |" % (g, " | ".join(cell(by.get((f, m, g)))[0] for f, m in cols)))
md += ["", "## Versions", "", "| framework | mode | python | packages |", "|---|---|---|---|"]
for d in comps:
    md.append("| %s | %s | %s | %s |" % (d["framework"], d["mode"], d["python"],
              ", ".join("%s %s" % (k, v) for k, v in d["versions"].items() if v)))
if unexpected:
    md += ["", "## Unexpected results", ""]
    for f, m, r in unexpected:
        md.append("- **%s (%s) / %s**: expected %s, got %s. %s" % (f, m, r["golden"], r["expected"], r["status"], r.get("reason", "")))
if checks:
    md += ["", "## Other checks", "", "| check | result | seconds | description |", "|---|---|---|---|"]
    for c in checks:
        md.append("| %s | %s | %s | %s |" % (c["check"], "ok" if c["ok"] else "FAIL", c["seconds"], c["description"]))
refs = sorted({(f, m, r["golden"], r["reason"]) for f, m, r in
               [(d["framework"], d["mode"], r) for d in comps for r in d["results"]] if r["status"] == "refused"})
if refs:
    md += ["", "## Refusals and why", ""]
    seen = set()
    for f, m, g, why in refs:
        if (f, g) not in seen:
            seen.add((f, g)); md.append("- %s / %s: %s" % (f, g, why))
text = "\n".join(md) + "\n"
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "report.md").write_text(text)

css = ("body{font-family:system-ui,sans-serif;margin:2em;max-width:1100px}table{border-collapse:collapse}"
       "td,th{border:1px solid #ccc;padding:4px 8px;font-size:14px}.good{background:#d8f2d8}.ref{background:#ececec}"
       ".bad{background:#f6c6c6;font-weight:bold}.na{color:#999}.PASS{color:#157a15}.FAIL{color:#b00}")
h = ["<html><head><meta charset=utf-8><title>gp-scuderia conformance</title><style>%s</style></head><body>" % css,
     "<h1>gp-scuderia conformance report</h1><h2 class=%s>%s</h2>" % (verdict, verdict),
     "<p>Cell: largest deviation as a share of the golden's tolerance. Refused = expected "
     "<code>UnsupportedByFramework</code>.</p><table><tr><th>golden</th>%s</tr>" % "".join("<th>%s<br>%s</th>" % c for c in cols)]
for g in goldens:
    row = "".join('<td class="%s" title="%s">%s</td>' % (lambda t, c, r: (c, html.escape((r or {}).get("reason", "")), t))(*cell(by.get((f, m, g))), by.get((f, m, g))) for f, m in cols)
    h.append("<tr><td>%s</td>%s</tr>" % (g, row))
h.append("</table><h2>Versions</h2><ul>")
for d in comps:
    h.append("<li><b>%s</b> (%s, python %s): %s</li>" % (d["framework"], d["mode"], d["python"],
             html.escape(", ".join("%s %s" % (k, v) for k, v in d["versions"].items() if v))))
h.append("</ul>")
if checks:
    h.append("<h2>Other checks</h2><table><tr><th>check</th><th>result</th><th>s</th><th>description</th></tr>")
    for c in checks:
        h.append('<tr><td>%s</td><td class="%s">%s</td><td>%s</td><td>%s</td></tr>' % (
            c["check"], "good" if c["ok"] else "bad", "ok" if c["ok"] else "FAIL", c["seconds"], html.escape(c["description"])))
    h.append("</table>")
h.append("</body></html>")
(out / "report.html").write_text("\n".join(h))
print(text)
raise SystemExit(0 if verdict == "PASS" else 1)
