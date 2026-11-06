"""Command-line entry points.

pwcet-analyse  : CSV of cycle counts -> gates -> fits -> held-out validation -> JSON report.
pwcet-synth    : write a synthetic trace in the firmware serial format (no hardware needed).

Exit codes of pwcet-analyse: 0 analysis ran (whatever the verdict: a refusal is a result),
2 bad input, 3 hardware session rejected as a measurement failure (not analysed).
"""
import argparse
import json
import sys
import time
import numpy as np
from . import analysis as an
from . import synthetic as syn


def _fmt(v):
    return f"{v:,.1f}" if isinstance(v, float) else str(v)


def summary(r: dict) -> str:
    lines = [f"origin: {r['origin']} ({r['note']})",
             f"fit set n={r['fit_set']['n']:,}  holdout n={r['holdout_set']['n']:,}  "
             f"fit max={_fmt(r['fit_set']['max'])}  distinct={r['fit_set']['distinct']:,}",
             f"gates: {r['gates']['verdict']}"
             + (f" (failed: {', '.join(r['gates']['failed'])})" if r['gates']['failed'] else "")]
    for t in r["gates"]["tests"]:
        lines.append(f"  {t['name']:<10} p={t['pvalue']:.4g}{'  REJECT' if t['reject'] else ''}")
    b = r["baseline_max_plus_margin"]
    lines.append(f"max observed + {b['margin']:.0%}: {_fmt(b['bound'])} "
                 f"(no probability attached; holdout exceedances {b['holdout_exceed']})")
    lines.append(f"verdict: {r['verdict']}")
    for name, est in r["estimates"].items():
        if "pwcet" not in est:
            why = est.get("reason") or est.get("error") or "no threshold passed goodness of fit"
            lines.append(f"{name}: no bound ({why})")
            continue
        extra = (f" q={est['chosen_quantile']} shape={est['shape']:.3f} gof p={est['gof_pvalue']:.3f}"
                 if name == "gpd" else f" shape={est['shape']:.3f} gof p={est['gof_pvalue']:.3f}"
                 if name == "gev" else f" cv={est['cv']:.3f} k={est['n_exceed']}")
        lines.append(f"{name}:{extra}")
        for c in r["validation"].get(name, []):
            lines.append(f"  p={c['p']:.0e} bound={_fmt(c['bound'])} holdout {c['exceed']}/{c['n']} "
                         f"(expected {c['expected']:.1f}) -> {c['verdict']}")
    lines.append(f"primary outcome: {r['primary_outcome']}")
    refuted = {m: ps for m, ps in r.get("extrapolation_refuted", {}).items() if ps}
    if refuted:
        lines.append("held-out data refute extrapolated bounds: " + "; ".join(
            f"{m} at p<={max(ps):.0e}" for m, ps in refuted.items()))
    return "\n".join(lines)


def main_analyse(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pwcet-analyse", description=__doc__.split("\n")[2])
    ap.add_argument("trace", help="fit-set trace (serial format or CSV)")
    ap.add_argument("--holdout", nargs="+", help="held-out session files (default: split the trace)")
    ap.add_argument("--origin", required=True, choices=an.ORIGINS)
    ap.add_argument("--out", help="JSON report path (default: <trace>.report.json)")
    ap.add_argument("--fit-frac", type=float, default=an.Config.fit_frac,
                    help="fit fraction when no --holdout is given (protocol: 1/3)")
    ap.add_argument("--gof-reps", type=int, default=an.Config.gof_reps)
    ap.add_argument("--gev-gof-reps", type=int, default=an.Config.gev_gof_reps)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    try:
        fit_t = an.load_trace(a.trace)
        hold_ts = [an.load_trace(h) for h in a.holdout] if a.holdout else []
    except (OSError, ValueError, IndexError) as e:
        print(f"error reading input: {e}", file=sys.stderr)
        return 2
    out = a.out or a.trace + ".report.json"
    meta = {"trace": a.trace, "holdout": a.holdout or [], "header": fit_t.header,
            "rows": fit_t.n_rows, "wrap_excluded": fit_t.n_wrap_excluded,
            "seq_missing": fit_t.n_seq_missing, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if a.origin == "hardware":
        bad = {a.trace: an.measurement_failures(fit_t)}
        bad.update({h: an.measurement_failures(t) for h, t in zip(a.holdout or [], hold_ts)})
        bad = {k: v for k, v in bad.items() if v}
        if bad:
            rep = {"origin": a.origin, "status": "measurement-failure", "reasons": bad, "meta": meta}
            with open(out, "w", encoding="utf-8") as f:
                json.dump(rep, f, indent=2)
            print(f"measurement failure, not analysed (PROTOCOL.md section 11): {bad}")
            return 3

    if hold_ts:
        fit_x = fit_t.cycles
        hold_x = np.concatenate([t.cycles for t in hold_ts])
    else:
        from .validate import split_trace
        fit_x, hold_x = split_trace(fit_t.cycles, a.fit_frac)
    cfg = an.Config(fit_frac=a.fit_frac, gof_reps=a.gof_reps,
                    gev_gof_reps=a.gev_gof_reps, seed=a.seed)
    rep = an.analyse(fit_x, hold_x, cfg, a.origin, meta)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)
    if not a.quiet:
        print(summary(rep))
        print(f"report: {out}")
    return 0


GENERATORS = ("interference", "iid", "mode-switching", "regime-shift")


def main_synth(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pwcet-synth",
                                 description="Write a synthetic trace in the serial format.")
    ap.add_argument("kind", choices=GENERATORS)
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed-gap", action="store_true",
                    help="interference: constant gap, so ISR phase drifts (dependent)")
    ap.add_argument("--miss-stickiness", type=float, default=0.0)
    ap.add_argument("--rtos-task", action="store_true",
                    help="interference: add a 100 Hz co-running task of 20k cycles")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.kind == "interference":
        x = syn.interference(a.n, gap_random=not a.fixed_gap, miss_stickiness=a.miss_stickiness,
                             task_period=1_250_000 if a.rtos_task else None,
                             task_cost=20_000.0, seed=a.seed)
    elif a.kind == "iid":
        x = np.round(syn.iid_gpd(a.n, loc=200_000, scale=500, shape=0.0, seed=a.seed))
    elif a.kind == "mode-switching":
        x = np.round(syn.mode_switching(a.n, base=200_000, delta=2_000, sigma=200, seed=a.seed))
    else:
        x = np.round(syn.regime_shift(a.n, base=200_000, delta=1_000, sigma=200, seed=a.seed))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# synthetic {a.kind} n={a.n} seed={a.seed} (pwcet-synth; not a measurement)\n")
        for i, c in enumerate(x.astype(np.int64)):
            f.write(f"T,{i},{c},{i % 8},0\n")
    print(f"wrote {a.n:,} synthetic runs to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main_analyse())
