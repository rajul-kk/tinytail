"""End-to-end analysis of one trace, following protocol/PROTOCOL.md:
load -> gates -> (only if the gates pass) GPD / GEV / MBPTA-CV / inequality (MEMIK, ATAN)
bounds -> held-out validation.

The result is a plain dict ready for JSON. Defaults equal the frozen protocol values.
"""
from dataclasses import dataclass, asdict, field
import csv
import io
import numpy as np
from . import __version__
from .evt import fit_pot, pwcet
from .gates import run_gates, GATE_ALPHA, VALID, INVALID_POOR_FIT
from . import validate as v
from . import inequality as iq
from . import tracelog

TARGET_PS = (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)
TESTABLE_PS = (1e-3, 1e-4)
ORIGINS = ("hardware", "renode", "synthetic")

# measurement-failure limits (PROTOCOL.md section 4); enforced for hardware traces only
MAX_WRAP_FRAC = 1e-3
MAX_MISSING_SEQ_FRAC = 1e-3
MIN_SESSION_ROWS = 100_000


@dataclass
class Config:
    fit_frac: float = 1 / 3
    ps: tuple = TARGET_PS
    quantiles: tuple = (0.90, 0.95, 0.975, 0.99)
    gof_reps: int = 1000
    gof_alpha: float = 0.05
    gev_blocks: int = v.GEV_N_BLOCKS
    gev_gof_reps: int = 500
    gate_alpha: float = GATE_ALPHA
    margin: float = v.MARGIN
    tolerance: float = v.TOLERANCE
    seed: int = 0


@dataclass
class Trace:
    cycles: np.ndarray
    n_rows: int = 0
    n_wrap_excluded: int = 0
    n_seq_missing: int = 0
    header: list = field(default_factory=list)
    overhead: np.ndarray = field(default_factory=lambda: np.empty(0))  # empty-region cycles
    bad_frames: int = 0  # binary format: corrupt or truncated frames skipped


def _pkey(p) -> str:
    return f"{p:.0e}"


# ------------------------------------------------------------------ loading

def parse_trace(text: str) -> Trace:
    """Accepts the firmware serial format `T,<seq>,<cycles>,<input_id>,<flags>` (flags in hex,
    bit 0 = timer wrap: such rows are excluded and counted; input_id 255 = counter-read
    overhead rows, kept apart), or a CSV with a `cycles` column, or a single column of
    numbers. Lines starting with '#' are header lines. Binary firmware logs go through
    `parse_bytes`."""
    header, serial, plain = [], [], []
    for line in io.StringIO(text):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            header.append(s)
        elif s.startswith("T,"):
            serial.append(s.split(","))
        else:
            plain.append(s)
    if serial:
        seq = [int(r[1]) for r in serial]
        cyc = [float(r[2]) for r in serial]
        ids = [int(r[3]) if len(r) > 3 else 0 for r in serial]
        flags = [int(r[4], 16) if len(r) > 4 else 0 for r in serial]
        return _from_rows(tracelog.split_rows(seq, cyc, ids, flags), header)
    rows = list(csv.reader(plain))
    if not rows:
        raise ValueError("no data rows")
    col = 0
    try:
        float(rows[0][0])
    except ValueError:  # header row
        names = [c.strip().lower() for c in rows[0]]
        col = names.index("cycles") if "cycles" in names else 0
        rows = rows[1:]
    cyc = np.array([float(r[col]) for r in rows])
    return Trace(cyc, len(rows), 0, 0, header)


def _from_rows(r: dict, header, bad_frames=0) -> Trace:
    return Trace(r["cycles"], r["n_rows"], r["n_wrap_excluded"], r["n_seq_missing"], header,
                 r["overhead"], bad_frames)


def parse_bytes(data: bytes) -> Trace:
    """Binary firmware log (`#BIN1` marker) or any text format accepted by parse_trace."""
    if tracelog.is_binary(data):
        header, rows, bad = tracelog.decode_binary(data)
        if rows.shape[0] == 0:
            raise ValueError("binary trace without valid frames")
        return _from_rows(tracelog.split_rows(*rows.T), header, bad)
    return parse_trace(data.decode("utf-8"))


def load_trace(path) -> Trace:
    with open(path, "rb") as f:
        return parse_bytes(f.read())


def measurement_failures(t: Trace) -> list:
    """Reasons a hardware session must not be analysed (PROTOCOL.md sections 4 and 11)."""
    out = []
    if t.n_rows and t.n_wrap_excluded / t.n_rows > MAX_WRAP_FRAC:
        out.append(f"timer-wrap rows {t.n_wrap_excluded} > {MAX_WRAP_FRAC:.1%}")
    if t.n_rows and t.n_seq_missing / (t.n_rows + t.n_seq_missing) > MAX_MISSING_SEQ_FRAC:
        out.append(f"missing sequence numbers {t.n_seq_missing} > {MAX_MISSING_SEQ_FRAC:.1%}")
    if t.cycles.size < MIN_SESSION_ROWS:
        out.append(f"only {t.cycles.size} valid rows (< {MIN_SESSION_ROWS})")
    return out


# ------------------------------------------------------------------ analysis

def describe(x) -> dict:
    x = np.asarray(x, float)
    return {"n": int(x.size), "min": float(x.min()), "median": float(np.median(x)),
            "mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "q0.999": float(np.quantile(x, 0.999)), "q0.9999": float(np.quantile(x, 0.9999)),
            "max": float(x.max()), "distinct": int(np.unique(x).size)}


def _validation(bounds: dict, hold, cfg: Config) -> list:
    return [c.as_dict() for c in v.validate_bounds(bounds, hold, tolerance=cfg.tolerance)]


def analyse(fit_x, hold_x, cfg: Config | None = None, origin="synthetic", meta=None) -> dict:
    cfg = cfg or Config()
    if origin not in ORIGINS:
        raise ValueError(f"origin must be one of {ORIGINS}")
    fit_x = np.asarray(fit_x, float)
    hold_x = np.asarray(hold_x, float)
    gates = run_gates(fit_x, alpha=cfg.gate_alpha)
    hwm = v.max_plus_margin(fit_x, cfg.margin)
    report = {
        "pwcet_version": __version__,
        "origin": origin,
        "note": ("synthetic trace: not a measurement" if origin == "synthetic" else
                 "Renode is not cycle-accurate: not timing evidence" if origin == "renode" else
                 "hardware measurement"),
        "meta": meta or {},
        "config": asdict(cfg),
        "fit_set": describe(fit_x),
        "holdout_set": describe(hold_x),
        "gates": gates.as_dict(),
        "baseline_max_plus_margin": {
            "bound": hwm, "margin": cfg.margin, "probabilistic": False,
            "holdout_exceed": int(np.count_nonzero(hold_x > hwm))},
        "estimates": {},
        "validation": {},
    }
    if not gates.passed:
        report["verdict"] = gates.verdict
        report["primary_outcome"] = "not-applicable (gate failed: no EVT or MBPTA bound computed)"
        return report

    ps = list(cfg.ps)
    hold_max = float(hold_x.max())

    # GPD, all candidate thresholds (sensitivity), lowest passing one is primary
    cands, chosen = [], None
    for q in cfg.quantiles:
        try:
            f = fit_pot(fit_x, q, cfg.gof_reps, cfg.seed)
        except ValueError as e:  # e.g. quantised timings with the top of the sample tied
            cands.append({"quantile": q, "error": str(e)})
            continue
        row = {"quantile": q, "threshold": f.threshold, "shape": f.shape, "scale": f.scale,
               "n_exceed": f.n_exceed, "gof_pvalue": f.gof_pvalue,
               "pwcet": {_pkey(p): float(pwcet(f, p)) for p in ps}}
        cands.append(row)
        if chosen is None and f.gof_pvalue >= cfg.gof_alpha:
            chosen = (f, row)
    gpd = {"candidates": cands, "chosen_quantile": None}
    if chosen is not None:
        f, row = chosen
        bounds = {p: float(pwcet(f, p)) for p in ps}
        gpd.update(chosen_quantile=row["quantile"], threshold=f.threshold, shape=f.shape,
                   scale=f.scale, n_exceed=f.n_exceed, gof_pvalue=f.gof_pvalue,
                   pwcet={_pkey(p): b for p, b in bounds.items()})
        gpd["pessimism_" + _pkey(min(ps))] = bounds[min(ps)] / hold_max
        report["validation"]["gpd"] = _validation(bounds, hold_x, cfg)
        report["verdict"] = VALID
    else:
        report["verdict"] = INVALID_POOR_FIT
    report["estimates"]["gpd"] = gpd

    # GEV block maxima (secondary)
    try:
        g = v.fit_gev(fit_x, cfg.gev_blocks, cfg.gev_gof_reps, cfg.seed)
        gb = {p: float(v.gev_pwcet(g, p)) for p in ps}
        gev = {"shape": g.shape, "loc": g.loc, "scale": g.scale, "block_size": g.block_size,
               "n_blocks": g.n_blocks, "gof_pvalue": g.gof_pvalue,
               "gof_pass": bool(g.gof_pvalue >= cfg.gof_alpha),
               "pwcet": {_pkey(p): b for p, b in gb.items()},
               ("pessimism_" + _pkey(min(ps))): gb[min(ps)] / hold_max}
        if chosen is not None:
            gev["ratio_to_gpd"] = {_pkey(p): gb[p] / bounds[p] for p in ps}
        report["validation"]["gev"] = _validation(gb, hold_x, cfg)
    except ValueError as e:
        gev = {"error": str(e)}
    report["estimates"]["gev"] = gev

    # MBPTA-CV baseline (PROTOCOL.md section 10 and deviation D1)
    c = v.mbpta_cv(fit_x)
    cv = {"applicable": c.applicable, "cv": c.cv, "threshold": c.threshold,
          "n_exceed": c.n_exceed, "mean_excess": c.mean_excess, "reason": c.reason}
    if c.applicable:
        cb = {p: float(v.cv_pwcet(c, p)) for p in ps}
        cv["pwcet"] = {_pkey(p): b for p, b in cb.items()}
        cv["pessimism_" + _pkey(min(ps))] = cb[min(ps)] / hold_max
        report["validation"]["mbpta_cv"] = _validation(cb, hold_x, cfg)
    report["estimates"]["mbpta_cv"] = cv

    # Inequality baselines (PROTOCOL.md deviation D2): MEMIK and ATAN, k capped by RESTK
    for m in iq.METHODS:
        f = iq.fit_inequality(fit_x, m, seed=cfg.seed)
        est = {"applicable": f.applicable, "restk_ps": list(f.restk_ps),
               "restk_maxk": list(f.restk_maxk), "restk_corr": f.restk_corr,
               "reason": f.reason}
        if f.applicable:
            ib = {p: float(iq.inequality_pwcet(f, p)) for p in ps}
            est["maxk"] = {_pkey(p): f.maxk(p) for p in ps}
            est["pwcet"] = {_pkey(p): b for p, b in ib.items()}
            est["pessimism_" + _pkey(min(ps))] = ib[min(ps)] / hold_max
            report["validation"][m] = _validation(ib, hold_x, cfg)
        report["estimates"][m] = est

    report["primary_outcome"] = primary_outcome(report)
    report["extrapolation_refuted"] = {
        m: [c["p"] for c in checks if c["verdict"] == v.FAIL and c["p"] not in TESTABLE_PS]
        for m, checks in report["validation"].items()}
    return report


def primary_outcome(report: dict) -> str:
    """PROTOCOL.md section 9: the GPD bound is pass or pass-pessimistic at both 1e-3 and 1e-4.
    A fail at a smaller p (possible even where a pass is untestable) does not change this
    outcome; it is listed separately under `extrapolation_refuted`."""
    checks = {c["p"]: c["verdict"] for c in report["validation"].get("gpd", [])}
    if not checks:
        return "no GPD bound (poor fit)"
    tested = [checks.get(p) for p in TESTABLE_PS]
    if any(t == v.FAIL for t in tested):
        return "failed held-out validation at a testable level"
    if all(t in (v.PASS, v.PASS_PESSIMISTIC) for t in tested):
        return "validated at 1e-3 and 1e-4 (smaller p are extrapolations)"
    return "inconclusive (testable levels untestable with this holdout size)"
