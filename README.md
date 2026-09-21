# Order-flow long memory and price impact in Bitcoin

Trade signs are strongly predictable. Prices are close to a martingale. This
project measures both on 85.6M Binance BTCUSDT trades (March-May 2026, 92 days),
recovers the impact kernel that is supposed to reconcile them, and tests whether
the standard linear model of that kernel holds.

**Headline results (real data, per-day estimation over 92 days)**

- 85.6M aggTrades records are really **38.8M market orders**. Treating sweep
  fragments as separate trades more than doubles apparent next-order sign
  predictability (out-of-sample R^2 0.28 -> 0.65) and shifts long-memory
  estimates by -0.25 to +0.15 depending on the estimator.
- Long memory in trade signs: **gamma ~ 0.57** (GPH 0.566, ACF 0.590,
  low-bandwidth local Whittle 0.567), stable by month for the most robust
  estimators (0.54-0.61).
- During a ~7% rally in early April, short-horizon sign predictability roughly
  doubled while the long-lag exponent stayed in its normal range.
- The linear transient-impact (propagator) model is **rejected at the order
  level**: the deconvolved kernel fails to decay on 92/92 days, with bid-ask
  bounce ruled out.

Every estimator was first validated against simulated order flow with known
parameters (see "What the validation established").

---

## The one derivation that matters

Transient-impact model (Bouchaud, Gefen, Potters, Wyart 2004):

```
p_t = sum_{t' < t} G(t - t') eps_{t'} + news
```

with `C(n) = E[eps_t eps_{t+n}]`, `C(0) = 1`, and the response function
`R(l) = E[(p_{t+l} - p_t) eps_t]`. Substituting and taking expectations:

```
R(l) = sum_{n=0}^{l-1} G(l-n) C(n)  +  sum_{n>=1} [G(l+n) - G(n)] C(n)
```

Collecting terms in `G(k)`: the first sum contributes `C(l-k)` for `k <= l`,
the second contributes `C(k-l)` for `k > l` and `-C(k)` everywhere. Both of the
first two are `C(|l-k|)`, so the coefficient collapses to

```
M[l, k] = C(|l - k|) - C(k)
```

Recovering the causal kernel from the two measurable objects is then a single
linear solve, `M G = R`.

**R is not G.** The response function is contaminated by the fact that the
trade at `t` predicts later trades, which carry their own impact. Reading
impact off `R` overstates it.

---

## Install and run

```bash
pip install numpy scipy pandas pyarrow pytest
python -m pytest tests -q                 # 14 tests, ~2s
python scripts/run_validation.py          # ~2 min, writes results/validation.json
```

Get data (needs network access to data.binance.vision):

```bash
python scripts/download_binance.py --symbol BTCUSDT --start 2026-03 --end 2026-05
python scripts/download_binance.py --symbol DOGEUSDT --start 2026-03 --end 2026-05
python scripts/download_binance.py --symbol BTCUSDT --market um --start 2026-03 --end 2026-05
# one day of raw trades, for the aggregation-bias check
python scripts/download_binance.py --symbol BTCUSDT --start 2026-03-15 --end 2026-03-15 \
    --data-type trades --interval daily
```

Analyse:

```bash
python scripts/run_analysis.py --parquet "data/BTCUSDT_2026-0*.parquet" \
    --label BTCUSDT --solve-lag 4000
python scripts/run_analysis.py --zips "data/spot/DOGEUSDT/*.zip" --label DOGEUSDT
```

**Run the validation before the analysis.** The analysis reads
`results/validation.json` to calibrate the gamma bias (see finding 5).

---

## What the validation established

Every number below comes from simulated data where the true parameters were
chosen in advance. Nothing here is trustworthy on real data until it passes.

**1. Local Whittle is the estimator to use for gamma.** Against true gamma of
0.20 / 0.35 / 0.50 / 0.65 / 0.80 it returned 0.249 / 0.382 / 0.533 / 0.689 /
0.827 with SD 0.012-0.018. The naive ACF log-log fit was both biased and four
to eight times noisier -- at gamma = 0.80 it gave 0.722 +/- 0.094. Report the
naive fit as a comparison, never as the estimate.

**2. The sign ACF cannot distinguish order splitting from herding.** Two
generators built on different mechanisms -- sign of a long-memory Gaussian
process, and Lillo-Mike-Farmer metaorder splitting -- produce statistically
indistinguishable exponents. So the sign ACF alone cannot separate order
splitting from herding; that would need broker or account identifiers.

**3. The deconvolution has a boundary artifact with a measured reach.** `G` is
recovered to better than 1% at short lags and drifts upward towards the edge of
the solve grid (ratio 4.4 at the boundary). Fitting into that region flattens
beta and can drive it negative. The fit window must sit at least 20x inside the
solve grid; this is enforced in code as `EDGE_SAFETY_RATIO`.

**4. Beta is recoverable to ~0.01 and exogenous news does not bias it.** Across
true beta of 0.15 / 0.25 / 0.35 and three news levels, bias stayed within 0.02
while variance grew. News is uncorrelated with order flow, so it drops out of
`R(l)` -- it costs precision, not accuracy.

**5. The gamma bias matters out of proportion to its size.** Local Whittle's
+0.03 bias at gamma = 0.5 propagates into the efficiency prediction `(1-gamma)/2`
at half size, which is the same order as beta's standard error. On synthetic
data with true gamma = 0.50 and beta = 0.25, the uncorrected test *falsely
rejected* the efficiency relation at z = 3.02. Calibrating gamma against the
validation curve (0.536 -> 0.503) gave z = 0.32 and correct acceptance. This is
why `run_analysis.py` reads `validation.json`.

**6. `beta = (1-gamma)/2` is an approximation, not an identity.** A kernel built
at exactly that value still leaves VR(128) = 1.29, and with a correctly
specified robust standard error the martingale null is rejected decisively
(z = 23). What decay does deliver is the removal of about 97% of the excess
variance ratio a permanent-impact world would produce (VR 7.48 -> 1.17). Quote
the 97%, not "prices are a martingale".

---

## Two bugs found during validation

**A spurious factor of `n` in the Lo-MacKinlay robust variance.** `delta_j` is
already `O(1/n)` -- numerator `O(n)`, denominator `O(n^2)` -- so multiplying by
`n` inflated the standard error by `sqrt(n)` and destroyed all power. Before the
fix, VR = 1.29 reported z = 0.02. After, z = 23. The test is now calibrated
against both nulls: a pure random walk gives `|z| < 1.5`, a mean-reverting
series gives z ~ 1758.

The lesson worth stating: a test that never rejects looks like a clean result.
Always calibrate a test statistic against a case where you know it *must*
reject.

**A Davies-Harte normalisation off by sqrt(2).** Caught by checking sample
variance against the theoretical value of 1 before using the generator for
anything. Cheap check, and it would have silently rescaled every downstream
covariance.

---

## Layout

```
orderflow_impact/
  fgn.py          exact fractional Gaussian noise (Davies-Harte); reused in Project 2
  simulate.py     two order-flow generators, propagator kernel, exact response (FFT)
  longmemory.py   sign ACF, log-log fit, GPH, local Whittle
  impact.py       response function, deconvolution, exponent fitting
  efficiency.py   Lo-MacKinlay variance ratio, counterfactuals, predictability
  data.py         Binance loaders (zip + parquet), sanity checks, per-day iteration
scripts/
  download_binance.py   checksum-verified downloader, resumable
  run_validation.py     5 experiments against known ground truth
  run_analysis.py       per-day estimation and cross-day aggregation
tests/                  14 tests
```

**Why per-day estimation.** A three-month pooled series is not stationary --
activity, relative tick size and participant mix all drift. Per-day fits keep
each estimate inside a roughly stationary window, and the cross-day dispersion
gives an honest standard error that a pooled fit cannot.

---

## Data gotchas already handled

- **Sign convention.** `is_buyer_maker = True` means the buyer was the *maker*,
  so the aggressor was a seller and the sign is `-1`. Getting this backwards
  silently inverts the response function. Asserted in tests.
- **Timestamp units.** Binance spot switched from milliseconds to microseconds
  on 2025-01-01. The loader infers the unit from magnitude rather than trusting
  the date.
- **Header sniffing.** Some archive CSVs carry a header row and some do not.
- **aggTrades aggregation bias.** `aggTrades` merges consecutive fills of one
  aggressive order at one price into a single record, removing exactly the
  short-lag positive autocorrelation this project measures. Re-run on raw
  `trades` for a subsample (`--raw-trades`) to quantify the gap. This check has
  not yet been run on the BTCUSDT sample.

---

## Real-data results: BTCUSDT, March-May 2026

Input: Binance spot aggTrades for BTCUSDT, 2026-03-01 to 2026-05-31, estimated
per UTC day (92 days). Command:

```bash
python scripts/run_analysis.py --parquet "data/BTCUSDT_2026-0*.parquet" \
    --label BTCUSDT --solve-lag 4000
```

Outputs are in `results/BTCUSDT_daily.csv` (one row per day) and
`results/BTCUSDT_summary.json`; the unmerged comparison is `BTCUSDT_nomerge_*`.

### 1. Sweep fragments must be merged into market orders

Binance aggregates fills of one taker order *at one price*. An order that sweeps
several price levels still produces several consecutive records with an
identical timestamp and sign. 85,578,057 records collapse to **38,806,932
orders** (2.21 records per order; 17.9% of orders multi-fill; the largest sweep
has 2,903 fills).

| 92-day mean | merged | unmerged |
|---|---|---|
| out-of-sample next-order sign R^2 | **0.285** | 0.651 |
| hit rate | 0.731 | 0.877 |
| VR(128), event time | 3.34 | 6.81 |
| gamma, GPH | 0.566 | 0.712 |
| gamma, local Whittle m = n^0.5 | 0.567 | 0.715 |
| gamma, ACF log-log | 0.590 | 0.669 |
| gamma, local Whittle default m = n^0.65 | 0.488 | 0.234 |

One cleaning choice pushes different estimators in **opposite directions**:
long-lag estimators up by ~0.15, the default local Whittle down by 0.25. The
merge was validated on simulated data fragmented the same way: it reconstructs
the true order count exactly and recovers the true gamma.

### 2. Long memory: gamma ~ 0.57, with short-lag structure on top

The long-lag estimators agree within 0.03. Local Whittle drifts with bandwidth
(m = n^0.5: 0.567, n^0.6: 0.560, n^0.7: 0.375, n^0.8: 0.198), far more than on
clean simulated data (0.47-0.57), so the sign process has short-horizon
structure that is not a single power law. The long-lag figure is the reliable one.

By month (mean of daily estimates):

| month | GPH | LW m = n^0.5 | ACF | LW default |
|---|---|---|---|---|
| March | 0.606 | 0.598 | 0.654 | 0.587 |
| April | 0.536 | 0.547 | 0.566 | 0.488 |
| May | 0.554 | 0.555 | 0.548 | 0.388 |

The steady decline in the default estimator reflects the changing short-lag
structure, not long memory.

### 3. The April episode

From 2 to 8 April the mean price rose from about $66.8k to $71.6k (~7%). On 4-6
April out-of-sample sign R^2 was 0.58-0.68, against roughly 0.3 on surrounding
days, while GPH gamma stayed at 0.43-0.62, within its normal daily range. Prices
are continuous and trade counts normal, so this is a market episode rather than
a data fault: order flow became about twice as predictable at short horizons
without its long-memory structure changing.

### 4. The linear propagator model does not fit

The deconvolved kernel fails to decay monotonically at short lags on **92/92
days**, and the fitted exponent hits its upper bound on 57/92.

Bid-ask bounce was the leading suspect and is ruled out. The spread is one tick
($0.01): 91% of buy -> sell transitions move the trade price by exactly one tick.
The mid-price proxy `price - (spread/2) * sign` recovers the true kernel in
simulation (beta 0.244 vs true 0.25), yet on real data it leaves the failure
unchanged.

The efficiency-relation z-statistics in `BTCUSDT_summary.json` are not
meaningful: beta is not identified, so there is nothing valid to compare. Nor
are the event-time variance ratios a trading signal; trends across ~100 orders
may unfold within milliseconds.

### Caveats

- Cross-day standard errors assume independent days. Daily estimates are
  persistent, so treat them as lower bounds.
- Only the April low-gamma days were examined individually; the late-May ones
  (23, 25, 27, 30, 31 May) were not.
- Prices are trade prices with a bounce correction, not quotes.
- The analysed parquet files were converted from the monthly aggTrades archives
  (columns `time, price, qty, buyer_is_maker, first_id, last_id`);
  `scripts/download_binance.py` plus `--zips` runs on the raw archives instead.

### Next steps

Why does the propagator fail? Live hypotheses: impact depends on order size (one
order had 2,903 fills); microsecond bursts of orders behave as a single event;
and event time is the wrong clock. Each is testable: size-weighted signs, merging
orders within a short time window, and a calendar-time propagator.
