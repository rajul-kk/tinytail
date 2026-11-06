"""Synthetic timing traces with known properties (units arbitrary, think cycles).

i.i.d. generators expose the exact tail so tests can compare estimates with the truth.
Non-i.i.d. generators are deliberately invalid inputs that the test battery must flag.
All generators take an explicit seed.
"""
import numpy as np
from scipy.stats import genpareto


def iid_gpd(n, loc=1000.0, scale=10.0, shape=0.0, seed=0):
    """i.i.d. GPD(loc, scale, shape). shape=0 is an exponential tail. Peaks over any
    threshold are again GPD with the same shape, so the right answer is known."""
    return genpareto.rvs(shape, loc=loc, scale=scale, size=n,
                         random_state=np.random.default_rng(seed))


def true_quantile(p, loc=1000.0, scale=10.0, shape=0.0):
    """Exact value exceeded with probability p by iid_gpd samples."""
    return float(genpareto.isf(p, shape, loc=loc, scale=scale))


def ar1(n, phi=0.8, base=1000.0, sigma=5.0, seed=0):
    """Autocorrelated Gaussian trace (violates independence; stationary)."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return base + x


def regime_shift(n, at=0.5, delta=20.0, base=1000.0, sigma=5.0, seed=0):
    """Independent noise whose mean jumps by delta at fraction `at` (non-stationary)."""
    x = base + np.random.default_rng(seed).normal(0.0, sigma, n)
    x[int(n * at):] += delta
    return x


def periodic_spikes(n, period=50, height=200.0, base=1000.0, sigma=5.0, seed=0):
    """Noise plus a spike every `period` runs (periodic interference; serially dependent)."""
    x = base + np.random.default_rng(seed).normal(0.0, sigma, n)
    x[::period] += height
    return x


def mode_switching(n, p_switch=0.01, delta=40.0, base=1000.0, sigma=5.0, seed=0):
    """Two latency modes (e.g. cache-warm / cache-cold phases) that persist for about
    1/p_switch runs before flipping: a sticky Markov chain. Marginally bimodal and
    stationary in the long run, but strongly serially dependent, so EVT must be refused."""
    rng = np.random.default_rng(seed)
    flips = rng.random(n) < p_switch
    mode = np.cumsum(flips) % 2
    return base + delta * mode + rng.normal(0.0, sigma, n)


def interference(n, *, compute=200_000.0, jitter=40.0,
                 miss_prob=0.1, miss_penalty=6_000.0, miss_jitter=400.0, miss_stickiness=0.0,
                 isr_period=125_000.0, isr_cost=250.0,
                 task_period=None, task_cost=0.0,
                 gap=125_000.0, gap_random=True, seed=0):
    """Cycle-count trace of back-to-back inferences on a simulated timeline.

    One inference takes `compute` cycles plus Gaussian bus jitter. With probability
    `miss_prob` a run is cache-cold and adds `miss_penalty` (+- `miss_jitter`, Gaussian),
    giving a bimodal hit/miss distribution (RP2040 XIP cache or STM32 ART). With
    `miss_stickiness` > 0 the cold state repeats with that extra probability (dependence).
    A periodic ISR fires every `isr_period` cycles and steals `isr_cost` cycles each time it
    lands inside a run; an optional co-running RTOS task preempts every `task_period` cycles
    for `task_cost` cycles. Preemption extends the run, which can admit further ticks, so the
    tick count is iterated to a fixed point.

    Between runs the core idles for `gap` cycles: if `gap_random` the gap is uniform on
    [0, 2*gap] (phase randomisation: when gap equals the ISR period the
    ISR phase at each start is exactly uniform and independent of the previous run); otherwise it is constant, so the
    ISR phase drifts deterministically from run to run (serial dependence by construction).
    Defaults are loosely RP2040-like (125 MHz, 1 kHz tick, a 1.6 ms inference); they are
    made-up illustration values, not measurements. Returns integer-valued cycles.
    """
    rng = np.random.default_rng(seed)
    sources = [(float(isr_period), float(isr_cost))]
    if task_period:
        sources.append((float(task_period), float(task_cost)))
    offsets = [rng.uniform(0, per) for per, _ in sources]
    out = np.empty(n)
    t = 0.0
    cold_prev = False
    for i in range(n):
        p_cold = min(1.0, miss_prob + (miss_stickiness if cold_prev else 0.0))
        cold = rng.random() < p_cold
        cold_prev = cold
        d0 = compute + rng.normal(0.0, jitter)
        if cold:
            d0 += miss_penalty + rng.normal(0.0, miss_jitter)
        d = d0
        for _ in range(8):  # fixed point of duration vs number of preemptions
            extra = 0.0
            for (per, cost), off in zip(sources, offsets):
                k = np.ceil((t + d - off) / per) - np.ceil((t - off) / per)
                extra += k * cost
            if d0 + extra == d:
                break
            d = d0 + extra
        out[i] = d
        t += d + (rng.uniform(0.0, 2.0 * gap) if gap_random else gap)
    return np.round(out)
