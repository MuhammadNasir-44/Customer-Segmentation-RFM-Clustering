# Customer Segmentation — Business Analysis Report

**Prepared by:** Muhammad Nasiruddin
**Dataset:** UCI *Online Retail* — a UK-based online gift-ware retailer, Dec 2010 – Dec 2011
**Scope:** 397,884 clean transactions · 4,338 customers · £8.9M revenue

---

## Executive summary

The customer base is **not uniform** — it splits cleanly into four behavioural
segments with very different value. The single most important finding:

> **A "Champions" segment of just 716 customers (16.5% of the base) generates
> 64.9% of all revenue — roughly £5.8M.** Meanwhile the largest segment by
> headcount, "At-risk / Lapsing" (37.2% of customers), contributes only 6.2%.

This is a textbook **80/20 (Pareto)** distribution, and it has a direct
strategic consequence: **retention spend should be weighted toward protecting a
small, high-value core**, not spread evenly across the whole list. The analysis
also gives marketing four clearly-defined groups, each with an obvious and
different next action.

Two independent methods — rule-based **RFM scoring** and **K-Means clustering** —
agree on the structure, which gives confidence the segments are real and not an
artefact of one technique.

---

## The four segments

| Segment | Customers | % of base | Median recency | Median orders | Median spend | Avg RFM score (of 15) |
|---------|:---------:|:---------:|:--------------:|:-------------:|:------------:|:---------------------:|
| 🏆 **Champions** | 716 | 16.5% | 8 days | 10 | £3,734 | **14.3** |
| 🔁 **Loyal regulars** | 1,173 | 27.0% | 56 days | 4 | £1,346 | 10.6 |
| 🌱 **New / Promising** | 837 | 19.3% | 17 days | 2 | £472 | 9.3 |
| ⚠️ **At-risk / Lapsing** | 1,612 | 37.2% | 177 days | 1 | £298 | 5.3 |

*Recency = days since last purchase (lower is better). RFM score combines
Recency, Frequency and Monetary, each scored 1–5, so 15 is the maximum.*

The average RFM score column is a useful **cross-check**: the clusters line up
almost perfectly with the independent rule-based scores (Champions ≈ 14.3, At-risk
≈ 5.3). The two methods were built differently but tell the same story.

---

## Where the revenue actually comes from

| Segment | Share of customers | Share of revenue | Approx. revenue |
|---------|:------------------:|:----------------:|:---------------:|
| 🏆 Champions | 16.5% | **64.9%** | ~£5.8M |
| 🔁 Loyal regulars | 27.0% | 23.7% | ~£2.1M |
| ⚠️ At-risk / Lapsing | 37.2% | 6.2% | ~£0.55M |
| 🌱 New / Promising | 19.3% | 5.2% | ~£0.46M |

**Customer count and revenue are almost inverted.** The two smallest-value
segments by revenue (At-risk and New) make up **56% of the customer list** but
barely **11% of the money**. Champions and Loyal regulars together — 43.5% of
customers — drive **~89% of revenue**.

---

## Segment-by-segment analysis & recommendations

### 🏆 Champions — *protect at all costs*
Recent (last order ~8 days ago), frequent (~10 orders), high spend (£3,734), and
the top RFM score. This is where the money is.
- **Risk:** over-discounting them wastes margin — they buy anyway. Losing even a
  few is disproportionately expensive.
- **Actions:** loyalty/VIP programme, early access to new stock, personal service,
  and a *quiet* early-warning system if a Champion's recency starts slipping.

### 🔁 Loyal regulars — *grow toward Champions*
Solid, steady buyers (4 orders, £1,346) who haven't quite reached Champion
frequency.
- **Actions:** targeted upsell/cross-sell and replenishment reminders. Moving even
  a fraction of this 1,173-strong group up a tier has a large revenue upside,
  because the value gap to Champions is wide.

### 🌱 New / Promising — *convert the second purchase*
Bought very recently (17 days) but only once or twice — the base is young here.
- **Actions:** onboarding sequence and a second-purchase incentive. This is the
  highest-*potential* segment: today's newcomer is tomorrow's Champion, and the
  cost of a nudge is low.

### ⚠️ At-risk / Lapsing — *win back selectively, cap the spend*
The largest group by count (1,612) but lowest value: one order, ~6 months ago,
£298, lowest RFM score.
- **Actions:** a single, time-limited win-back offer — but **cap the budget**.
  Many are one-time buyers who will not return, so blanket spend here is poor ROI.
  Treat any responders as "New / Promising" and re-onboard them.

---

## Strategic priorities (in order)

1. **Defend Champions** — the highest-leverage action, since ~65% of revenue sits
   in one small segment.
2. **Promote Loyal regulars upward** — the biggest realistic growth lever.
3. **Convert New / Promising** to a second purchase — cheapest long-term investment.
4. **Win back At-risk selectively** — lowest priority; spend cautiously.

---

## Method note

- **Cleaning:** removed cancelled orders, rows without a customer ID, and
  non-sales rows (zero/negative price or quantity).
- **Features:** built an RFM table (Recency, Frequency, Monetary) per customer.
- **Two approaches:** rule-based RFM 1–5 quintile scoring, and K-Means clustering
  on log-transformed, standardised RFM values.
- **Choice of 4 segments:** supported by the elbow method and silhouette score,
  and cross-checked against hierarchical clustering (K-Means scored higher:
  silhouette 0.34 vs 0.24).

Full code, charts, and the per-customer output are in this repository; the
underlying tables are also provided as an Excel workbook
([`segment_summary.xlsx`](segment_summary.xlsx)) for non-technical stakeholders.
