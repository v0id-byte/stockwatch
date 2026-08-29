# Structured Signed Earnings Drift / PEAD Spec

This is the implementation spec for the last clean event-alpha test in StockWatch:
structured, signed earnings-announcement drift. It is a research gate only. It
must not trigger trading, alerts, or production rankings unless it passes the
validation gates below.

## Goal

Test whether A-share earnings preannouncements / express reports contain a
long-only, post-disclosure edge after PIT alignment, A-share frictions, and
benchmarks.

Primary question:

> After a positive signed earnings-growth event is public and tradable, can an
> equal-weight long-only basket beat both the trade-date universe and CSI300
> after costs?

Terminology note: without analyst consensus or a true expectation model, this is
not clean SUE / earnings surprise. It is a structured signed earnings-growth /
earnings-drift test. The disclosure event may still be surprising, but raw YoY
growth is not the same thing as expectation surprise.

Current title-only PEAD parsing is not enough: CNINFO titles expose direction but
almost never expose magnitude. This spec uses structured AKShare fields instead.

## Hard Constraints

- Account mode: long-only cash account.
- Benchmark: CSI300 return first; CSI300 total return if a reliable local source
  is added. Report which benchmark flavor was used.
- Main horizons: 5, 20, 60 trading days, because `training_set.parquet` already
  has `forward_5d_return`, `forward_20d_return`, and `forward_60d_return`.
- Costs: configurable, default `20 bps` round-trip per completed long trade in
  research reports.
- Entry: conservative next-trading-day close entry after the disclosure calendar
  date. Do not use same-day close returns in the primary report.
- Tradability: v1 must at least report entry-day limit-up and suspension rates
  for positive and strong-positive events. A deployment-grade pass requires
  exclusion/handling; if data is missing, the report must say
  `limit_filter_applied=false`.
- Scope: standalone research scripts and artifacts only. Do not change
  `models/`, production ranking, `core/runner.py`, or dashboard behavior.

## Data Sources

Use AKShare through the existing project environment:

- `ak.stock_yjyg_em(date=<YYYYMMDD>)` for earnings preannouncements.
- `ak.stock_yjkb_em(date=<YYYYMMDD>)` for earnings express reports.
- Existing `~/.stockwatch/history/training_set.parquet` for universe and forward
  returns.
- Existing local market/kline history only if limit-up filters or extra horizons
  are implemented.

Critical caveat:

- `stock_yjyg_em` and `stock_yjkb_em` are current AKShare/Eastmoney snapshots by
  report period, not guaranteed historical vintages. They may expose revised or
  final values while retaining an earlier `公告日期`. The first implementation
  must run a small vintage audit before trusting any result. If true vintage
  cannot be verified, every report must mark the data as
  `best_effort_reconstructed_pit=true` and `true_vintage=false`.

Observed `stock_yjyg_em` columns:

- `股票代码`
- `股票简称`
- `预测指标`
- `业绩变动`
- `预测数值`
- `业绩变动幅度`
- `业绩变动原因`
- `预告类型`
- `上年同期值`
- `公告日期`

Observed `stock_yjkb_em` columns:

- `股票代码`
- `股票简称`
- `每股收益`
- `营业收入-同比增长`
- `净利润-净利润`
- `净利润-去年同期`
- `净利润-同比增长`
- `公告日期`

## New Artifacts

Suggested scripts:

- `scripts/build_pead_events.py`
  - Fetch and cache structured earnings events.
  - Output: `~/.stockwatch/history/pead_events_structured.parquet`.
  - Output report: `~/.stockwatch/history/pead_events_structured_report.json`.

- `scripts/evaluate_pead_factor.py`
  - Build event cohorts, signed scores, calendar-time long-only portfolios,
    horizon reports, and auxiliary sparse IC.
  - Output: `~/.stockwatch/history/pead_factor_report.json`.

Optional later:

- `scripts/build_benchmark_returns.py`
  - Cache CSI300 benchmark returns if no reliable local benchmark file exists.

Do not refetch AKShare inside evaluation. Fetch/cache first, evaluate second.

## Fetch / Cache Spec

`build_pead_events.py` arguments:

- `--start-year`, default `2021`
- `--end-year`, default current year
- `--output`
- `--report`
- `--sleep`, default polite delay
- `--force`

Periods:

- Generate quarter ends: `YYYY0331`, `YYYY0630`, `YYYY0930`, `YYYY1231`.
- For each period, call both `stock_yjyg_em(date=period)` and
  `stock_yjkb_em(date=period)`.
- Persist after every successful period to avoid losing progress.

Required output columns:

- `source`: `yjyg` or `yjkb`
- `code`
- `name`
- `report_period`
- `available_at`
- `event_type`
- `metric`
- `signed_score`
- `sign`
- `magnitude_pct`
- `net_profit_value`
- `last_year_value`
- `raw_type`
- `raw_change_text`
- `raw_reason`
- `snapshot_fetched_at`
- `true_vintage`: boolean, default `false` unless proven otherwise
- `vintage_note`

PIT rules:

- `available_at` must come from `公告日期`, not report-period end.
- Date-only `available_at` is not assumed tradable intraday.
- Primary evaluation enters at the close of the first trading day strictly after
  the disclosure calendar date. A Saturday/Sunday announcement enters Monday if
  Monday is the next trading day; do not map weekend to Monday and then delay
  again to Tuesday.
- Keep duplicate raw rows, then dedupe at evaluation time by
  `(code, report_period, source)` using latest `available_at`, with express
  report preferred over preannouncement when both are known on the same entry
  date.
- Preannouncement revisions are a separate PIT problem. If the snapshot only
  exposes revised values, use the revision announcement's own date if available;
  otherwise mark the event as non-vintage and include a sensitivity that keeps
  only the earliest available preannouncement per `(code, report_period)`.

## Vintage Audit

Before running the full evaluation, implement and report a quick audit:

- Pick several known `业绩预告修正` / `业绩预告更正` cases from CNINFO titles.
- Compare the structured `stock_yjyg_em` value and `公告日期` against the raw CNINFO
  revision dates.
- Report whether AKShare appears to return first disclosure, latest revision, or
  an ambiguous current snapshot.
- If ambiguous, continue only as a best-effort reconstructed-PIT study and do not
  call the result production-grade.

## Event Parsing

Filter to net profit attributable to listed-company shareholders:

- keep rows where `预测指标` contains `归属于上市公司股东的净利润`
- exclude rows where `预测指标` contains `扣除非经常性损益`, unless a separate
  `deducted_profit` variant is explicitly added

Preannouncement sign mapping:

- Strong positive: `预增`, `扭亏`, `扭亏为盈`
- Weak positive: `略增`, `减亏`
- Strong negative: `预减`, `首亏`, `续亏`, `增亏`, `由盈转亏`, `转亏`
- Weak negative: `略减`
- Neutral / skip: `不确定`, `预平`, missing or ambiguous type

Express-report sign mapping:

- Use `净利润-同比增长`.
- Positive if `净利润-同比增长 > 0`.
- Negative if `净利润-同比增长 < 0`.
- Keep exact magnitude.

Magnitude:

- Primary magnitude is absolute `业绩变动幅度` for preannouncements and absolute
  `净利润-同比增长` for express reports.
- If the source exposes a range, parse lower bound, upper bound, and midpoint.
  The primary score uses the lower bound for positive events and the worse bound
  for negative events. Midpoint is a sensitivity report only.
- If missing, try to compute from `预测数值` and `上年同期值`. If those are ranges,
  apply the same conservative-bound rule.
- Clip raw percent magnitude to `[0, 500]` before scoring.
- Turnaround / sign-flip rows (`扭亏`, `减亏`, `首亏`, `续亏`, `增亏`, `由盈转亏`)
  have pathological percentage math near zero or negative bases. Do not let
  huge computed YoY values dominate `signed_score`; either keep them as a
  direction-only cohort or assign a fixed capped magnitude and report them
  separately.

Scores:

- `sign`: `+1` or `-1`
- `magnitude_pct`: clipped absolute percentage
- `signed_score = sign * log1p(magnitude_pct / 100)`
- `strong_positive = sign > 0 and magnitude_pct >= 50`
- `strong_negative = sign < 0 and magnitude_pct >= 50`
- `own_history_adjusted_score`: optional signed YoY minus the stock's recent
  comparable YoY baseline, computed only from earlier PIT events
- `industry_adjusted_score`: optional signed YoY minus the same-period industry
  median, using the available sector map and clearly reporting whether sector
  membership is historical or current-static

Do not fill missing magnitude with zero for the primary PEAD factor. Rows without
magnitude may be kept for a direction-only sensitivity report, but they are not
the primary signal.

## Evaluation Spec

`evaluate_pead_factor.py` arguments:

- `--events-path`
- `--target-horizons`, default `5,20,60`
- `--cost-bps`, default `20`
- `--top-n`, default `20,50,100`
- `--benchmark`, default `csi300`
- `--output`

Alignment:

1. Convert `available_at` to a disclosure calendar date.
2. Set `entry_date` to the first trading day strictly after that calendar date.
3. Assume the primary strategy enters at `entry_date` close.
4. Merge `forward_Nd_return` on `entry_date`, never on the disclosure date.
5. Assert or report the return convention: `forward_Nd_return` must be
   close-to-close from the row's `trade_date`. If that cannot be verified,
   recompute returns from local kline data before using the report.

Metrics:

- Parse coverage:
  - rows fetched by source and period
  - rows retained after metric filtering
  - rows with valid sign
  - rows with valid magnitude
  - code-date event count
- Vintage/tradability diagnostics:
  - `true_vintage`
  - `best_effort_reconstructed_pit`
  - entry-date suspension count
  - entry-date open/locked limit-up count
  - same diagnostics specifically for positive and strong-positive events
- Calendar-time portfolio (primary):
  - each trading day holds all still-active positive-event positions from the
    previous `horizon` trading days
  - equal-weight active holdings each day
  - daily portfolio return, daily CSI300 return, and daily equal-weight universe
    return
  - daily excess returns vs both benchmarks
  - Newey-West adjusted t-stat on daily excess returns, using lag
    `horizon - 1` because positions overlap
  - average active names, active days, turnover proxy, max drawdown
- Sparse IC (auxiliary only):
  - daily Spearman IC of `signed_score` vs each forward horizon
  - require at least 20 event rows per date for a daily IC point
  - Newey-West adjusted t-stat with lag `horizon - 1`
- Event study:
  - all positive
  - strong positive
  - top score quantile among event rows
  - all negative
  - strong negative
  - yjyg only
  - yjkb only
- Entry-date event basket (secondary):
  - for each `entry_date`, rank positive events by `signed_score`
  - test top N = 20, 50, 100
  - equal-weight
  - deduct costs from event returns
  - compare to same-date equal-weight training universe
  - compare to CSI300 for the same holding horizon
  - report turnover proxy and average active names
- Missing-return / universe diagnostics:
  - entry events missing from training universe
  - events with missing forward return
  - delisted or long-suspended names if detectable
  - whether missing returns were dropped, imputed, or counted as failures
- Stability:
  - by year
  - by report period quarter
  - walk-forward folds:
    - train selection period: 2022-2024
    - OOS check: 2025-2026, or rolling annual folds if data supports it

## Limit-Up And Suspension Handling

First version must at least report entry-day tradability diagnostics; it cannot
pass the final gate without actual exclusion/handling.

Implementation target:

- Build entry-day market status from local kline data.
- Exclude events whose entry day cannot be bought because the stock is suspended
  or opens/locks at limit-up.
- Report:
  - `limit_filter_applied`
  - excluded event count
  - excluded positive event count
  - positive-event limit-up / suspension rate before exclusion
  - performance before and after exclusion

## Pass / Fail Gate

The primary candidate is:

`strong_positive`, structured `yjyg + yjkb`, conservative `signed_score`,
next-trading-day close entry, 20 bps round-trip cost, horizon chosen from training
folds only.

It passes research gate only if all are true:

- OOS calendar-time portfolio net excess vs CSI300 is positive.
- OOS calendar-time portfolio net excess vs equal-weight universe is positive.
- OOS calendar-time portfolio IR is positive after Newey-West adjustment.
- At least 3 calendar years have positive net excess, or the report explicitly
  explains regime concentration.
- Limit-up / suspension filter is applied, or the result is marked
  `UNTRADEABLE_RESEARCH_ONLY`.
- Top-N basket drawdown is not obviously worse than CSI300 without compensation
  in excess return.

It fails if:

- the signal only works on the short side,
- positive events are flat/negative after costs,
- the edge disappears after next-day entry,
- it depends on missing magnitude rows,
- it depends on current-snapshot fields that fail the vintage audit,
- it depends on turnaround rows with pathological percent magnitudes,
- it only wins because of size/illiquidity exposure and fails after a size/style
  neutralized diagnostic.

## Tests

Add focused tests:

- sign mapping:
  - `预增`, `扭亏` => positive
  - `预减`, `首亏`, `续亏`, `增亏`, `由盈转亏` => negative
  - `不确定`, `预平` => skipped
- magnitude:
  - uses `业绩变动幅度`
  - parses ranges and uses the conservative bound
  - clips at 500
  - treats turnaround / sign-flip rows as a separate cohort
  - does not treat missing magnitude as primary signal
- PIT:
  - report-period end is never used as `available_at`
  - date-only announcement enters the next trading day after the disclosure
    calendar date
  - after-close announcement enters the next trading day after the disclosure
    calendar date
  - weekend announcement enters the next trading day after the disclosure
    calendar date, not one extra day later
- evaluation:
  - cost is subtracted once per long event return
  - equal-weight universe benchmark is same entry date
  - same-day entry is not allowed
  - calendar-time portfolio daily holdings overlap correctly
  - sparse IC is not used as the primary pass/fail object
- regression:
  - title-only PEAD remains marked as insufficient if magnitude coverage is zero
  - reports say `true_vintage=false` unless the vintage audit proves otherwise

## Commands

Expected workflow:

```bash
.venv/bin/python scripts/build_pead_events.py --start-year 2021
.venv/bin/python scripts/evaluate_pead_factor.py \
  --events-path ~/.stockwatch/history/pead_events_structured.parquet \
  --target-horizons 5,20,60 \
  --cost-bps 20 \
  --top-n 20,50,100
.venv/bin/python -m pytest tests/test_pead_events.py tests/test_pead_factor.py -q
```

For full validation before any integration discussion:

```bash
.venv/bin/python -m pytest tests -q
```

## Implementation Order

1. Run the EM snapshot vintage audit and decide whether the study is true-vintage
   or best-effort reconstructed PIT.
2. Build structured event cache from `stock_yjyg_em` only.
3. Add entry-day limit-up / suspension diagnostics before interpreting any
   positive-event number.
4. Evaluate `yjyg` strong-positive drift with calendar-time portfolios.
5. Add `stock_yjkb_em` as a separate source and compare `yjyg` vs `yjkb`.
6. Add CSI300 benchmark, aligned by entry date and holding horizon.
7. Add actual limit-up / suspension exclusion.
8. Add size/style neutralized diagnostic only if the raw long-only result is
   positive after costs.
9. Only then consider feeding the PEAD score into broader factor research.

## Interpretation Rules

- A positive all-event IC is not enough. The calendar-time long-only
  positive-event portfolio must win after costs.
- A good in-sample result is not enough. OOS folds must hold.
- A win over CSI300 is not enough if it is just small-cap exposure. Compare to
  equal-weight universe and run size/style diagnostics.
- A win over CSI300 price index is easier than a win over CSI300 total return.
  Report benchmark flavor and treat price-index alpha as an upper-bound claim.
- Horizon choice is pre-registered: choose using training folds only, then freeze
  it for OOS. Do not pick the best OOS horizon after the fact.
- A negative or flat result is a valid endpoint. Do not keep tuning thresholds
  until one variant looks lucky.
