# Capstone Report — Content Refresh Prioritization

- **Author:** Krish
- **Lane:** Content Refresh Prioritization (Content Opportunity Scoring)
- **Repo:** https://github.com/02017711723iot-dotcom/Flyrank_Internship_1
- **Date:** 2026-09-15

## 0. Abstract

This project asks whether observable content and search-performance signals can rank pages for human review when prioritizing content-refresh work. Using an anonymized slice of FlyRank content-performance data — 30,000 content items across 32 clients — a shallow decision tree was trained on six features (content age, days since last update, 90-day impressions, average search position, click-through rate, and word count) to score each page's likelihood of observed decline, then evaluated with a client-level holdout split that shares no client between training and test data. On that holdout, the model reached a Precision@50 of 0.56, while a simple stale-plus-visible rule — flagging pages untouched for 180 or more days with meaningful visibility — reached 0.60, a four-point gap the model did not close. The honest reading is that the transparent rule held the edge on this evaluation, and the contribution here is a leakage-aware benchmark and a decision-support ranking tool rather than a demonstrated improvement over a simpler method.

## 1. Problem framing

This supports the decision of which content page a human should review first out of a large set. The unit of analysis is a content page; the output is a ranked review score and priority order, not an automatic refresh decision. A reviewer inspects high-ranked pages and decides whether to refresh, monitor, investigate further, or take no action. The cost of a wrong call is editorial time: a false positive wastes review effort on a page that didn't need it, and a false negative delays review of a page that is actually declining. Data/ML helps here because the ranking problem spans thousands of pages and several correlated signals (age, visibility, position, CTR) that a hand-written rule can only combine crudely.

## 2. Data safety

The analysis uses `data/raw/content_refresh_anonymized.csv`: 30,000 rows, 44 columns, 32 pseudonymous clients. Only hashed `content_id` / `client_id` values are used, and only for grouping/joining — never as model features. `trend_direction` and `trend_pct` are label-derived fields and are excluded from the feature set (they are only used to construct `is_declining_label`). No client names, domains, URLs, titles, or raw queries appear anywhere in `work/`.

## 3. Baseline

The baseline is the stale-plus-visible rule carried over from the Week 4 baseline work: a page qualifies if `days_since_last_update >= 180` **and** `impressions_90d >= 500`; qualifying pages are scored by `impressions_90d`, everything else scores 0. It's a fair comparison because it's exactly the kind of manual triage rule a content team would already use, built only from observable signals. On the full 30,000-row dataset only 174 pages meet the staleness threshold and only 17 meet both conditions — the rule is narrow by design, and that scarcity shows up directly in the evaluation (Section 5).

## 4. Model / analysis

`DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)`, trained on `content_age_days`, `days_since_last_update`, `impressions_90d`, `avg_position`, `ctr`, `word_count`. Deliberately excluded: `trend_direction`, `trend_pct`, `is_declining_label` (all label-derived). Target in one sentence: `is_declining_label = 1` where the dataset's own `trend_direction` reads "down", else 0.

## 5. Evaluation

Split: `GroupShuffleSplit(test_size=0.20, random_state=42)` grouped by `client_id` — client-holdout, not random-by-row, so no client appears in both sets. Population: 22,301 rows after dropping missing feature/label/group values → 17,223 train rows (25 clients) / 5,078 test rows (7 clients), 0 client overlap.

| Approach | Precision@50 |
|---|---|
| Decision Tree ML | 0.56 |
| Stale + Visible Baseline | 0.60 |

Difference: **-0.04** — the model did not outperform the baseline on this split. Dataset-wide base rate for the declining label is 54.2%, so 0.56 sits close to that rate and should not be read as strong top-of-list discrimination on its own.

Error look: the baseline assigned a non-zero score to **zero** pages in this test split (0 of 5,078) — none of the 7 held-out test clients contained a page meeting both staleness conditions, consistent with only 17 qualifying pages existing in the entire 30,000-row dataset. Its "top 50" is therefore an ordering among tied zero-scores, not a differentiated ranking; the 0.60 figure should be read as a property of this particular split, not a settled property of the rule. This replaced an earlier row-level/stratified split (used in Weeks 5–6) that mixed a client's rows across train and test and produced a visibly inflated Precision@50 in the high 0.60s — the drop when moving to a true client holdout is itself evidence of how much of that earlier number was memorized client identity rather than transferable signal.

## 6. Interpretation

Feature importance concentrated almost entirely in two signals: `impressions_90d` (0.549) and `content_age_days` (0.255), with `avg_position` (0.114) and `ctr` (0.082) contributing less, and `days_since_last_update` and `word_count` contributing nothing to the fitted tree's splits — notably, the tree ignored the very field (`days_since_last_update`) that anchors the baseline rule it was compared against. These are feature-importance values describing how the fitted tree used its inputs, not causal weights; they don't show that impressions cause decline. The negative result — a simple rule holding its ground against a trained model — is itself a valid, reportable finding here, not a failure to hide.

## 7. Recommendation

Ranked actions, in priority order: (1) Refresh candidate — detailed human review, (2) Aging content — check freshness and intent alignment, (3) Low CTR — investigate against impressions/position, (4) Low position — check for confounds, (5) Low visibility — investigate before any content decision, (6) Review only — keep monitoring. A FlyRank editor would work the queue top-down, but treat every rank as a prompt to investigate, not an instruction — checking business importance, search intent, content quality, technical issues, and cannibalization before deciding whether to refresh, monitor, or take no action. Confidence is limited: given the -0.04 gap versus baseline and the fragility of the baseline's own number on this split (Section 5), neither ranking should be trusted as a settled winner without a larger, more balanced holdout.

## 8. Reproducibility

Fresh clone: `pip install -r requirements.txt`, then run `work/notebooks/capstone.ipynb` top to bottom. Data path: `data/raw/content_refresh_anonymized.csv`. Model seed: `random_state=42` (both the `GroupShuffleSplit` and the `DecisionTreeClassifier`). Split and metric code live in the notebook's Sections 3–4; the printed cell outputs (train/test row and client counts, Precision@50 for both approaches) are the receipts these numbers trace back to. Gap to close on the next run: the notebook does not currently print the declining-label rate within the train/test splits themselves, only the dataset-wide rate — worth adding to Section 3 for a tighter reproducibility trail.

## 9. Acknowledgments & data credit

[Built on the FlyRank ML Internship dataset](https://flyrank.ai)

---

**Claims checklist:** observed / measured / directional / decision-support language used throughout · base rate (54.2%) reported alongside Precision@50 · no causal claims · no "predicted Google's algorithm" · no client-identifying details anywhere in this file · numbers match the committed notebook's executed cell outputs (Sections 3–4 of `work/notebooks/capstone.ipynb`).
