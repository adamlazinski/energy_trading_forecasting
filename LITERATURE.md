# Literature survey: what data and code do electricity-trading papers actually use?

*Compiled 2026-07-08. Purpose: (a) know what's replicable for free vs what needs paid
data, (b) position this project against the literature, (c) interview ammunition.
Links marked ✓ were verified this session; unmarked ones are from search-result
metadata / prior knowledge — check before citing.*

## The one-sentence summary

The literature splits cleanly in two: **day-ahead forecasting runs on free data**
(and since Lago–Weron 2021 has an open benchmark everyone uses), while **continuous
intraday and order-book papers run on paid EPEX Spot transaction/M7 data** obtained
through university or employer licences — which is why that half of the literature
has almost no reproducible code+data. Imbalance/balancing papers are back on free
TSO data. Our project (free PSE/TGE/ENTSO-E, auction-level intraday IDA1–3,
imbalance CEN) sits entirely on the free side — the same data class as the
imbalance school, with the Polish 15-min single-price reform as untouched territory.

## 1. Day-ahead school (Weron / Wrocław) — free data, open code

| Paper | Market & data | Code |
|---|---|---|
| ✓ Lago, Marcjasz, De Schutter, Weron 2021, *Applied Energy* — "Forecasting day-ahead electricity prices: review + open-access benchmark" | 5 markets (EPEX-DE, EPEX-BE, EPEX-FR, Nord Pool, PJM), 6 years each, hourly DA price + 2 exogenous series (load & RES forecasts) — all bundled in the toolbox | ✓ [epftoolbox](https://github.com/jeslago/epftoolbox) — datasets, LEAR + DNN reference models, precomputed forecasts, GW-test evaluation. **The** benchmark. |
| Marcjasz, Narajewski, Weron, Ziel 2023, *Energy Economics* — "Distributional neural networks for EPF" ([arXiv 2207.02832](https://arxiv.org/abs/2207.02832)) | German EPEX DA 2015–2021, epftoolbox-style inputs + gas/coal/CO2 | Codes released with paper (GitHub, Marcjasz) — repo link in the paper PDF |
| Uniejewski, Marcjasz, Weron 2019 — "Understanding intraday electricity markets: LASSO" | German ID3 index (quarter-hourly), EPEX | Weron's group traditionally shares codes via [Weron's publications page](https://p.wz.pwr.edu.pl/~weron.rafal/Publ) (per-paper zips; page had a TLS error this session) |
| Serafin, Uniejewski, Weron 2019 — calibration-window averaging (Ave-*) | GEFCom2014 + EPEX DA | ditto |
| Maciejowska, Uniejewski, Serafin — PCA forecast averaging (Energies) | DA + intraday German prices | ditto |
| Serafin & Weron 2025 — custom loss functions for BESS trading; and [arXiv 2604.19580](https://arxiv.org/abs/2604.19580) probabilistic DA forecasts + battery strategies | German DA; battery DP on top of quantile forecasts | check paper for repo |

**Takeaway:** everything in this school is replicable for free; the trading layer is
usually a stylized battery or DA–ID switch, not an executable microstructure test.

## 2. Continuous intraday school (Ziel / Duisburg-Essen, + Statkraft) — paid EPEX data

| Paper | Market & data | Code/data status |
|---|---|---|
| Narajewski & Ziel 2019/20 — econometric ID3 forecasting ([arXiv 1812.09081](https://arxiv.org/abs/1812.09081)); transaction-arrival modelling ([arXiv 1901.09729](https://arxiv.org/abs/1901.09729)) | **EPEX transaction-level data** (every trade, German CID, hourly + quarter-hourly), ~2015–2019; target = ID3 (VWAP of trades 180–30 min pre-delivery) | Data licensed, not shared. Methods described fully; no turnkey repo |
| Hirsch & Ziel 2024 — simulation-based intraday forecasting (*Energy Journal*), cross-product effects ([arXiv 2306.13419](https://arxiv.org/abs/2306.13419)) | EPEX CID transactions via Statkraft/university licence + ENTSO-E fundamentals | Data closed. Hirsch's open-source side: online distributional regression ([arXiv 2407.08750](https://arxiv.org/abs/2407.08750)) ships as an open Python package — the method is reusable on our data |
| Kath & Ziel — "The value of forecasts" (quantile regression → DA vs ID3 switching trade) | German quarter-hourly DA auction vs ID3 | closest published analogue to our F24/F25 spread tests; EPEX data licensed |
| ✓ Serafin, Marcjasz, Weron 2022, *Energy Economics* — "Trading on short-term path forecasts of intraday electricity prices" | German CID; prediction bands from path forecasts, trade when price exits the band | paywalled; [WORMS working-paper version](https://ideas.repec.org/a/eee/eneeco/v112y2022ics014098832200281x.html) exists |
| OrderFusion ([arXiv 2502.06830](https://arxiv.org/abs/2502.06830)) — order-book encoding → probabilistic CID price prediction | CID price indices, high- and low-liquidity European markets | "methodology available" page; data almost certainly licensed EPEX order book |
| LOB directional forecasting ([arXiv 2509.04452](https://arxiv.org/abs/2509.04452)); order-book feature learning ([arXiv 2510.12685](https://arxiv.org/abs/2510.12685)); generative probabilistic CID ([arXiv 2506.00044](https://arxiv.org/abs/2506.00044)); scenario paths for optimal CID trading ([arXiv 2605.13446](https://arxiv.org/abs/2605.13446)) | all German EPEX CID, transactions or full M7 LOB | data licensed in every case |

**Takeaway:** the entire intraday-microstructure literature stands on one closed
dataset: EPEX Spot transactions / M7 order book (universities buy it; Statkraft
authors use the desk's). Nobody can publish the data. **Poland's IDA1–3 are
auctions, so TGE publishes clearing prices free — our executable-spread tests
(F25/F26) are on the *free* side of a line the German literature can't cross
without a licence.** That's a differentiator worth saying in interviews.

## 3. RL trading papers — licensed order books again

| Paper | Data |
|---|---|
| Boukas et al. 2021, *Machine Learning* — [deep RL for continuous intraday bidding](https://link.springer.com/article/10.1007/s10994-021-06020-8) | Centralized LOB, European CIM (storage-operator perspective); EPEX order-book replay |
| Wind-park operator RL on German CIM ([arXiv 2111.13609](https://arxiv.org/abs/2111.13609)) | EPEX CID transactions 2018+ |
| Feature-driven RL for PV in CID ([arXiv 2510.16021](https://arxiv.org/abs/2510.16021)) | German CID |

RL papers replay licensed order books; results are un-reproducible without the
data, and none report post-cost live-like execution with our F-list's rigor
(by-quarter breakdowns are rare in this literature).

## 4. Imbalance / balancing school — free data (our neighborhood)

| Paper | Market & data | Notes |
|---|---|---|
| Narajewski 2022 — [probabilistic German imbalance-price forecasting](https://arxiv.org/abs/2205.11439) | German reBAP, free (regelleistung.net / netztransparenz.de / ENTSO-E) | nearest methodological neighbor to our CEN forecaster |
| ✓ 2026 review — [imbalance price forecasting algorithms in Europe](https://arxiv.org/abs/2605.17054) | survey: algorithms, metrics, way forward; notes EU-wide shift to single-price 15-min settlement (exactly the Polish 2024-06-14 reform) | read against our F-list; likely confirms the field is young |
| Conformal prediction, DA + real-time balancing ([arXiv 2502.04935](https://arxiv.org/abs/2502.04935)) | balancing-market prices | conformal = cheap upgrade path for our quantiles |
| Risk-constrained trading for stochastic generation under single-price balancing ([arXiv 1708.02625](https://arxiv.org/abs/1708.02625)) | theoretical + probabilistic system-length forecasts | the "trade the imbalance exposure" framing our F21/F23 tested honestly |

**Takeaway:** this school uses exactly our data class (free TSO/ENTSO-E), and
single-price 15-min settlement is where all of Europe is heading — Poland
adopted it in 2024, so our post-reform CEN dataset is one of the earliest.

## 5. Where to get the data (summary table)

| Dataset | Access | Used by |
|---|---|---|
| epftoolbox 5-market DA bundle | **free** (GitHub) | all DA benchmark papers |
| ENTSO-E Transparency (prices, load, RES, imbalance, X-border) | **free** (API token) | imbalance school, fundamentals everywhere |
| SMARD (DE), PSE API (PL), Nord Pool (partial), OMIE (ES, fully open) | **free** | fundamentals, DA prices |
| TGE RDN/RDB + IDA1–3 auction results | **free** (scrape) | this project only, effectively |
| EPEX Spot transactions + M7 order book | **paid licence** | the whole German intraday/LOB/RL literature |
| GEFCom2014 | free (competition archive) | probabilistic-forecasting classics |

## 6. What this means for the project

1. **Our free-data, executability-guarded, by-quarter-verified F-list is
   methodologically stricter than most of what's published** — the trading
   claims in the CID literature rarely survive the "two real prints + costs on
   both legs + quarterly stability" filter we impose.
2. **Gap we occupy:** post-2024 single-price CEN in Poland — no published
   forecasting/trading paper on it yet (the 2026 review is the closest).
3. **Cheap upgrades from the literature:** Hirsch's online distributional
   regression (open package) on our CEN target; conformal prediction on our
   quantiles; Kath–Ziel's DA-vs-later-market value-of-forecast framing is
   precisely our F24/F25 — cite it when writing up why the spread class is dead.
4. **If we ever want German CID data:** it's a purchase, not a scrape. The
   Polish IDA auction structure is the free workaround — say this in interviews.
