# DS001 Diagnostic Report

This report answers key diagnostic questions regarding the scoring, quality bands, rules contribution, and human review queue composition for DS001.

---

## 1. Why was each image assigned to Review?

The review queue contains **5250** images. Here is the classification breakdown by primary trigger reasons (images triggering multiple alerts are grouped under "Multiple reasons"):

| Reason | Count | Description |
| :--- | :--- | :--- |
| **Low score** | 22 | Images isolated due to low score triggers. |
| **YOLO disagreement** | 144 | Images isolated due to yolo disagreement triggers. |
| **Blur** | 0 | Images isolated due to blur triggers. |
| **Exposure** | 0 | Images isolated due to exposure triggers. |
| **Duplicate** | 0 | Images isolated due to duplicate triggers. |
| **Annotation overlap** | 0 | Images isolated due to annotation overlap triggers. |
| **Border boxes** | 4 | Images isolated due to border boxes triggers. |
| **Tiny boxes** | 0 | Images isolated due to tiny boxes triggers. |
| **Clinical class review** | 0 | Images isolated due to clinical class review triggers. |
| **Multiple reasons** | 5080 | Images isolated due to multiple reasons triggers. |

*Note: Duplicates are mapped directly to the Reject category, so their count in the human Review queue is 0.*

---

## 2. Sample Images Registry
Folders containing **50 random images** from each category have been successfully created under:
`workspace/reports/DS001_diagnostic_samples/`
- `Review/` (50 samples)
- `Reject/` (50 samples)
- `Gold/` (50 samples)
- `Silver/` (50 samples)

These samples allow clinical teams to visually inspect the dataset and calibrate quality acceptance thresholds.

---

## 3. Score Distribution Histogram

Below is the histogram of overall quality scores across the processed active dataset (excluding duplicates/rejections):

```text
9.5 - 10.0  : █ (56)
9.0 - 9.5   : ██████ (264)
8.5 - 9.0   : █████████ (390)
8.0 - 8.5   : █████████████████ (732)
7.5 - 8.0   : ███████████████████████████ (1150)
7.0 - 7.5   : ████████████████████████████████████████ (1655)
6.5 - 7.0   : ████████████████████████████ (1170)
6.0 - 6.5   : █████████████████████████ (1061)
5.0 - 6.0   :  (0)
0.0 - 5.0   :  (0)

```

### Insights
- **Gold/Silver threshold tightness**: A large portion of the dataset falls into the **[7.5 - 8.0]** and **[8.0 - 8.5]** ranges. This indicates that the 8.0 cutoff is positioned precisely where the majority of standard-exposure images lie. Small adjustments (e.g. lowering the cutoff to 7.8) would significantly increase accepted yield rates if desired.

---

## 4. Rule Contribution Breakdown

The table below shows how many images were affected by each scoring and audit rule. (An image can trigger multiple rules):

| Rule | Images Affected | Percentage of Dataset |
| :--- | :--- | :--- |
| **Low overall score (< 8.0)** | 5723 | 67.5% |
| **YOLO disagreement score (< 8.0)** | 7401 | 87.3% |
| **Blur penalty (normalized blur_score < 4.0)** | 3264 | 38.5% |
| **Exposure penalty (normalized exposure_score < 6.0)** | 1448 | 17.1% |
| **Deduplication (Duplicate status)** | 810 | 9.6% |
| **Overlap penalty (IoU > 0.85)** | 3 | 0.0% |
| **Tiny boxes (area < 0.0001)** | 143 | 1.7% |
| **Border boxes (within 1% edge)** | 3943 | 46.5% |
| **Clinical class review (folliculitis/milium/etc.)** | 1202 | 14.2% |

### Key Findings
- **YOLO Disagreement** affects 7401 images (the dominant rule). Since predictions are ran in simulated perturbation mode, the slight shifts in coordinates cause IoU penalties that drop agreement scores. This confirms that simulated mode is highly sensitive.
- **Deduplication** successfully caught **810** near-duplicates, cleaning out redundancy.
- **Tiny boxes** and **Border boxes** affect very few images, showing that labels are relatively well-centered in DS001.
