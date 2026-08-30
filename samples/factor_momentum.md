# Cross-Sectional Momentum: Construction and Robustness

## 1 Introduction

Cross-sectional momentum remains one of the most persistently documented
anomalies in the equity factor literature. This note reconstructs the standard
12-1 momentum factor on a liquid A-share universe, examines its decay profile,
and reports robustness under three neutralisation schemes.

![](figures/fig01_cumulative_returns.png)

*Figure 1. Cumulative long-short returns of the 12-1 momentum factor, 2015-2024.*

## 2 Data and Universe

The universe is the union of the CSI 300 and CSI 500 constituents, rebalanced
monthly. Suspended names and names with fewer than 120 trading days of history
are excluded at formation.

| Field | Source | Frequency | Coverage | Missing rate |
| --- | --- | --- | --- | --- |
| Adjusted close | Exchange feed | Daily | 2010-01 to 2024-12 | 0.02% |
| Free-float market cap | Vendor A | Daily | 2010-01 to 2024-12 | 0.11% |
| Turnover | Exchange feed | Daily | 2010-01 to 2024-12 | 0.02% |
| Book value | Filings | Quarterly | 2010-Q1 to 2024-Q4 | 1.83% |
| Analyst coverage | Vendor B | Monthly | 2012-01 to 2024-12 | 6.40% |
| Industry code | Vendor A | Static | full | 0.00% |
| Suspension flag | Exchange feed | Daily | 2010-01 to 2024-12 | 0.00% |

## 3 Factor Construction

### 3.1 Raw signal

The raw signal skips the most recent month to avoid short-horizon reversal:

```
r_{i,t}^{12-1} = log(P_{i,t-21} / P_{i,t-252})
```

### 3.2 Neutralisation

Three variants are evaluated. Each is winsorised at the 1st and 99th percentile
and standardised cross-sectionally before neutralisation.

| Variant | Size neutral | Industry neutral | Beta neutral | Turnover penalty |
| --- | --- | --- | --- | --- |
| M0 (raw) | no | no | no | 0.00 |
| M1 | yes | no | no | 0.00 |
| M2 | yes | yes | no | 0.00 |
| M3 | yes | yes | yes | 0.00 |
| M4 | yes | yes | yes | 0.05 |
| M5 | yes | yes | yes | 0.10 |

## 4 Results

### 4.1 Decile portfolios

Equal-weighted decile portfolios, monthly rebalance, 2015-01 through 2024-12.
Returns are annualised and gross of transaction costs.

| Decile | Ann. return | Ann. vol | Sharpe | Max drawdown | Turnover | Hit rate | Avg. cap (bn) | Beta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1 (losers) | -4.12% | 28.9% | -0.14 | -61.3% | 182% | 46.1% | 18.4 | 1.19 |
| D2 | -1.05% | 26.4% | -0.04 | -55.8% | 171% | 47.9% | 21.7 | 1.14 |
| D3 | 1.88% | 25.1% | 0.07 | -51.2% | 166% | 49.2% | 24.9 | 1.11 |
| D4 | 3.44% | 24.3% | 0.14 | -48.7% | 163% | 49.8% | 27.2 | 1.08 |
| D5 | 4.91% | 23.8% | 0.21 | -46.9% | 161% | 50.4% | 29.8 | 1.06 |
| D6 | 6.02% | 23.5% | 0.26 | -45.1% | 160% | 51.0% | 31.5 | 1.05 |
| D7 | 7.55% | 23.7% | 0.32 | -44.6% | 162% | 51.7% | 32.9 | 1.05 |
| D8 | 9.13% | 24.2% | 0.38 | -45.9% | 166% | 52.4% | 33.4 | 1.06 |
| D9 | 11.02% | 25.6% | 0.43 | -48.3% | 173% | 53.1% | 32.1 | 1.09 |
| D10 (winners) | 14.37% | 28.1% | 0.51 | -52.7% | 189% | 54.0% | 29.6 | 1.15 |
| D10-D1 | 18.49% | 17.3% | 1.07 | -24.1% | 371% | 58.6% | -- | -0.04 |

### 4.2 Information coefficients

| Variant | Mean IC | IC std | ICIR | t-stat | IC > 0 rate | Half-life (days) |
| --- | --- | --- | --- | --- | --- | --- |
| M0 (raw) | 0.031 | 0.118 | 0.263 | 2.88 | 59.2% | 41 |
| M1 | 0.038 | 0.109 | 0.349 | 3.82 | 61.7% | 44 |
| M2 | 0.044 | 0.101 | 0.436 | 4.77 | 63.3% | 47 |
| M3 | 0.045 | 0.099 | 0.455 | 4.98 | 63.8% | 48 |
| M4 | 0.043 | 0.097 | 0.443 | 4.85 | 63.5% | 52 |
| M5 | 0.040 | 0.096 | 0.417 | 4.56 | 62.9% | 58 |

![](figures/fig02_ic_decay.png)

*Figure 2. IC decay by forward horizon for variants M0 through M5.*

### 4.3 Sub-period stability

| Period | Ann. L/S return | Sharpe | Max DD | Regime |
| --- | --- | --- | --- | --- |
| 2015-01 to 2016-12 | 31.4% | 1.42 | -19.8% | high dispersion |
| 2017-01 to 2018-12 | 9.7% | 0.61 | -22.4% | large-cap led |
| 2019-01 to 2020-12 | 26.8% | 1.31 | -17.2% | growth rally |
| 2021-01 to 2022-12 | 4.1% | 0.24 | -24.1% | style rotation |
| 2023-01 to 2024-12 | 12.9% | 0.83 | -15.6% | range-bound |

## 5 Transaction Costs

Net-of-cost results assume a linear model of 8 bps one-way plus a square-root
impact term calibrated on participation rate.

| Cost assumption | Gross Sharpe | Net Sharpe | Break-even cost (bps) |
| --- | --- | --- | --- |
| None | 1.07 | 1.07 | -- |
| 5 bps one-way | 1.07 | 0.89 | 29.1 |
| 8 bps one-way | 1.07 | 0.78 | 29.1 |
| 15 bps one-way | 1.07 | 0.54 | 29.1 |
| 25 bps one-way | 1.07 | 0.21 | 29.1 |

## 6 Conclusion

Momentum survives size and industry neutralisation with an improved ICIR, and
degrades gracefully under a turnover penalty. The break-even cost of roughly
29 bps one-way leaves adequate headroom at institutional execution levels.
