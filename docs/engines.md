# Two inference engines and computational cost

Supplement to the [README](../README.md) of wind-bayes-twin.

## Two engines, one model

The whole validation was run with both inference engines on the same model, noise model and injected faults:

| | Run-length + Kalman (primary) | Joint particle filter, guided | Joint particle filter, unguided |
|---|---|---|---|
| detected within 90 days | 88.8 % | 90.2 % | 90.4 % |
| median days to detection | 24 | 24 | 25 |
| first alarm names the right cause | 95 % | 93 % | 92 % |
| false alarms per turbine-year | 0.50 | 0.42 | 0.25 |
| efficiency -2 % / derating 5 % detected | 29 % / 88 % | 38 % / 92 % | 40 % / 92 % |
| 90 % interval coverage: yaw / gearbox / generator | 88-90 % / 88-90 % / 73-75 % | 85-90 % / 88 % / 71-73 % | 85-92 % / 85-88 % / 67-69 % |
| seconds per 90-day injection run (median) | 1.5 | 2.6 | 2.6 |

The two engines agree to within a few points on every scenario, which checks the factorisation into blocks and the
plug-in coupling of yaw and power: an approximation that mattered would show up as a gap. The particle filter is
slightly more sensitive to power-curve faults - it can hold an efficiency loss and a derating at the same time and
needs no plug-in - and slightly worse on the generator bearing, where Monte Carlo error adds to a noisy residual.

Without the change-point guidance (bootstrap proposals only) the particle filter targets the same posterior and
detects as many faults, but later after a step and with less precise sizes right after it: a 4° yaw offset is flagged
after 14 days instead of 8 (an 8° one after 6 instead of 4), a derating is named correctly in 66 % instead of 77 % of
the cases, and the intervals of a 3 % anemometer step cover the truth in 60 % instead of 75 % - few particles sit near
the new value when the jump happens. It raises fewer false alarms (0.25 per turbine-year), because the guidance also
follows changes that turn out to be wander. The detector buys Monte Carlo efficiency where a bootstrap filter is
weakest, just after a jump, without changing what the filter estimates.

## Computational cost

Laptop: Intel i7-10850H (6 cores), 64 GB RAM, Windows, Python 3.13; the campaigns shared the CPU with another job.

| Step | Time |
|---|---|
| read five yearly archives (1.6 million 10-minute turbine records) into Parquet | 83 s |
| `prepare`: wake tables (PyWake), reference fit and prediction, RC fits, daily statistics | 3 min |
| `calibrate`: NUTS for 7 statistics x 2 chains x 1200 iterations (Kalman likelihood in `lax.scan`) | 11-19 min |
| run-length engine: one turbine-day, five blocks, ~1,000-2,000 hypotheses | ~20 ms |
| ... with daily credible intervals and Shapley loss attribution | 40-80 ms |
| joint particle filter, 2000 particles: one turbine-day | ~40 ms |
| `validate`: 6 clean runs 2019-2021 + 624 injection runs, 6 processes | 9-12 min per engine |
| `cases` (logs, quarterly decisions) and `report` | 2 min |

Memory is about 0.5 GB per process (the prepared 10-minute data of five years). Everything after `prepare` reads only
the daily statistics and the records behind the power statistics, so the online part would run on a farm's SCADA
server in real time with room to spare.
