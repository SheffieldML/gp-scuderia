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
benches = [d for d in docs if "benchmark_run" in d]
timings = [d for d in docs if "timing_run" in d]
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
bench_bad = [(d["benchmark_framework"], d["mode"], r) for d in benches for r in d["results"] if not r["ok"]]
verdict = "PASS" if not unexpected and not bad_checks and not bench_bad else "FAIL"

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
if benches:
    bcols = sorted({(d["benchmark_framework"], d["mode"]) for d in benches})
    bnames = sorted({r["name"] for d in benches for r in d["results"]})
    bby = {(d["benchmark_framework"], d["mode"], r["name"]): r for d in benches for r in d["results"]}

    def bcell(r):
        if r is None:
            return "-"
        if not r["ok"]:
            return "UNEXPECTED " + r["status"]
        if r["status"] in ("skipped", "known"):
            return r["status"]
        f = frac(r)
        return "match %.0f%%" % (100 * f) if f is not None else r["status"]

    md += ["", "## Accuracy benchmarks", "",
           "Seeded generated models, reference from the GPy oracle. `skipped` means the framework lacks a "
           "required capability; `known` is a recorded, explained mismatch.", "",
           "| case | " + " | ".join("%s (%s)" % c for c in bcols) + " |", "|---|" + "---|" * len(bcols)]
    for n in bnames:
        md.append("| %s | %s |" % (n, " | ".join(bcell(bby.get((f, m, n))) for f, m in bcols)))
    kn = sorted({(f, r["name"], r.get("reason", "")) for f, m, r in
                 [(d["benchmark_framework"], d["mode"], r) for d in benches for r in d["results"]] if r["status"] == "known"})
    if kn:
        md += ["", "Known mismatches:", ""] + ["- %s / %s: %s" % k for k in kn]
    for f, m, r in bench_bad:
        md.append("- **UNEXPECTED %s (%s) / %s**: %s" % (f, m, r["name"], r.get("reason", "")))
def sparse_tables():
    """Per framework: log likelihood of DTCVar and FITC against the exact value, by inducing set size."""
    out = []
    for d in sorted(benches, key=lambda d: d["benchmark_framework"]):
        for name, t in (d.get("sparse") or {}).items():
            if t["exact"] is None:
                continue
            fmt = lambda v: "-" if v is None else "%.4f" % v
            out.append((d["benchmark_framework"], name, t["exact"],
                        [(r["M"], fmt(r["dtcvar"] is not None and r["dtcvar"] - t["exact"] or None),
                          fmt(r["fitc_gap"])) for r in t["rows"]]))
    return out


if sparse_tables():
    md += ["", "## Sparse approximation: gap to the exact log likelihood", "",
           "Reported, not asserted: log likelihood of each approximation minus the exact value, by number of "
           "inducing points M (nested sets). DTCVar is a lower bound, so its gap is at most zero.", ""]
    for f, name, ex, rows in sparse_tables():
        md += ["**%s** (%s, exact = %.4f)" % (f, name, ex), "", "| M | DTCVar - exact | FITC - exact |", "|---|---|---|"]
        md += ["| %d | %s | %s |" % r for r in rows] + [""]
def timing_tables():
    """benchmark -> (columns, rows). A column is (framework, threads); a row is a case, in size order.
    A cell is (median seconds, ratio to the fastest framework in that row at the same thread setting)."""
    out = {}
    for b in sorted({r["benchmark"] for d in timings for r in d["results"]}):
        cols = sorted({(d["benchmark_framework"], d["threads"]) for d in timings
                       if any(r["benchmark"] == b and r["status"] == "timed" for r in d["results"])},
                      key=lambda c: (c[1] != "1", c[0]))
        cells = {}
        for d in timings:
            for r in d["results"]:
                if r["benchmark"] == b and r["status"] == "timed":
                    cells[(r["case"], d["benchmark_framework"], d["threads"])] = (r["seconds_median"], r["seconds_iqr"], r["sane"], r["n"], r["input_dim"])
        cases = sorted({c for c, _, _ in cells}, key=lambda c: next(v[3:] for k, v in cells.items() if k[0] == c))
        rows = []
        for c in cases:
            best = {}
            for (cc, f, th), v in cells.items():
                if cc == c:
                    best[th] = min(best.get(th, v[0]), v[0])
            rows.append((c, [(cells.get((c, f, th)), best.get(th)) for f, th in cols]))
        out[b] = (cols, rows)
    return out


def tcell(v):
    if v[0] is None:
        return "-"
    (med, iqr, sane, *_), best = v
    return "%.3gs (x%.1f)%s" % (med, med / best, "" if sane else " !")


if timings:
    hw = timings[0]["hardware"]
    md += ["", "## Timing (informational; never gates the build)", "",
           "Median wall-clock seconds to build the model from the wire document, evaluate the log likelihood and predict, "
           "with the ratio to the fastest framework in the row at the same thread setting. `1` is capped to one thread "
           "(best effort through environment variables), `all` is uncapped. `!` marks a log likelihood far from the "
           "reference, so the speed is not comparing like with like. Tier: %s. Hardware: %s, %s logical CPUs, numpy BLAS %s, %s." % (
               ", ".join(sorted({d["tier"] for d in timings})), hw["processor"] or hw["machine"], hw["logical_cpus"],
               hw["numpy_blas"], hw["dtype"]), ""]
    for b, (cols, rows) in timing_tables().items():
        md += ["**%s**" % b, "", "| case | %s |" % " | ".join("%s (%s thr)" % c for c in cols), "|---|%s|" % "|".join("---" for _ in cols)]
        md += ["| %s | %s |" % (c, " | ".join(tcell(v) for v in vs)) for c, vs in rows] + [""]
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
if benches:
    h.append("<h2>Accuracy benchmarks</h2><table><tr><th>case</th>%s</tr>" % "".join("<th>%s<br>%s</th>" % c for c in bcols))
    for n in bnames:
        row = ""
        for f, m in bcols:
            r = bby.get((f, m, n))
            cls = "na" if r is None else "bad" if not r["ok"] else "ref" if r["status"] in ("skipped", "known") else "good"
            row += '<td class="%s" title="%s">%s</td>' % (cls, html.escape((r or {}).get("reason", "")), bcell(r))
        h.append("<tr><td>%s</td>%s</tr>" % (n, row))
    h.append("</table>")
for f, name, ex, rows in sparse_tables():
    h.append("<h2>Sparse gap: %s (%s)</h2><table><tr><th>M</th><th>DTCVar - exact</th><th>FITC - exact</th></tr>%s</table>" % (
        f, name, "".join("<tr><td>%d</td><td>%s</td><td>%s</td></tr>" % r for r in rows)))
if timings:
    h.append("<h2>Timing</h2><p>Informational; never gates the build. Median seconds (ratio to the fastest framework at the same thread setting).</p>")
    for b, (cols, rows) in timing_tables().items():
        h.append("<h3>%s</h3><table><tr><th>case</th>%s</tr>" % (b, "".join("<th>%s<br>%s thr</th>" % c for c in cols)))
        for c, vs in rows:
            h.append("<tr><td>%s</td>%s</tr>" % (c, "".join("<td>%s</td>" % tcell(v) for v in vs)))
        h.append("</table>")
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
