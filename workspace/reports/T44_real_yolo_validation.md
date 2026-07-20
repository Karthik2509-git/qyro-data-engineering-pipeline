# Phase T4.4: Real YOLO Agreement Validation & Calibration Report

This report presents the validation results of replacing simulated YOLO agreement with inferences from the frozen production detector `qyro_acne_v1_best.pt` (Conf: 0.25, IoU: 0.60).

---

## 📊 Old vs New Quality Band Comparison

| Quality Band | Simulated Count | Real Count | Difference |
| :--- | :--- | :--- | :--- |
| **Gold** | 0 | 0 | +0 |
| **Silver** | 16 | 0 | -16 |
| **Review** | 845 | 337 | -508 |
| **Reject** | 7620 | 8144 | +524 |

*Observed yields: Real weights evaluated 0 images as Gold and 0 images as Silver, highlighting the calibration of the production quality engine.*

---

## 🎯 Agreement Statistics

- **Average IoU (Matched boxes)**: `0.5357`
- **Average Precision**: `0.6642`
- **Average Recall**: `0.7140`
- **Average F1 Score**: `0.6482`
- **Average Prediction Difference**: `8.46` lesions/image

---

## 🧹 Rule Contribution Breakdown
An image can trigger multiple rules. This table shows the percentage contribution of every scoring rule:

| Rule | Images Affected | Percentage of Dataset |
| :--- | :--- | :--- |
| **Low overall score (< 8.0)** | 395 | 4.66% |
| **YOLO disagreement score (< 8.0)** | 8354 | 98.50% |
| **Blur penalty (normalized blur_score < 4.0)** | 8481 | 100.00% |
| **Exposure penalty (normalized exposure_score < 6.0)** | 8481 | 100.00% |
| **Deduplication (Duplicate status)** | 58 | 0.68% |
| **Overlap penalty (IoU > 0.85)** | 3 | 0.04% |
| **Tiny boxes (area < 0.0001)** | 143 | 1.69% |
| **Border boxes (within 1% edge)** | 3943 | 46.49% |
| **Clinical class review (folliculitis/milium/etc.)** | 1202 | 14.17% |

---

## 🔍 Review Queue Decomposition
Review queue breakdown by root-cause trigger combinations:

| Trigger Combination | Image Count | Percentage of Review Queue |
| :--- | :--- | :--- |
| **Image_Quality** | 124 | 36.80% |
| **Missing_Annotations + Image_Quality** | 60 | 17.80% |
| **High_Model_Disagreement + Image_Quality** | 46 | 13.65% |
| **Border_Issues + Image_Quality** | 30 | 8.90% |
| **Low_Model_Confidence + Image_Quality** | 24 | 7.12% |
| **Missing_Annotations + Low_Model_Confidence + Image_Quality** | 19 | 5.64% |
| **Missing_Annotations + High_Model_Disagreement + Image_Quality** | 6 | 1.78% |
| **Missing_Annotations + Border_Issues + Image_Quality** | 6 | 1.78% |
| **Low_Model_Confidence + Border_Issues + Image_Quality** | 5 | 1.48% |
| **Clinical_Review + Border_Issues + Image_Quality** | 4 | 1.19% |
| **Clinical_Review + Image_Quality** | 3 | 0.89% |
| **Clinical_Review + Missing_Annotations + Border_Issues + Image_Quality** | 2 | 0.59% |
| **Clinical_Review + Missing_Annotations + Low_Model_Confidence + Border_Issues + Image_Quality** | 2 | 0.59% |
| **Missing_Annotations + High_Model_Disagreement + Low_Model_Confidence + Border_Issues + Image_Quality** | 1 | 0.30% |
| **Clinical_Review + Missing_Annotations + Image_Quality** | 1 | 0.30% |
| **Missing_Annotations + Low_Model_Confidence + Border_Issues + Image_Quality** | 1 | 0.30% |
| **Clinical_Review + Missing_Annotations + High_Model_Disagreement + Low_Model_Confidence + Border_Issues + Image_Quality** | 1 | 0.30% |
| **Clinical_Review + Low_Model_Confidence + Image_Quality** | 1 | 0.30% |
| **Clinical_Review + Missing_Annotations + Low_Model_Confidence + Image_Quality** | 1 | 0.30% |

---

## 📈 Score Distribution Histogram

```text
9.5 - 10.0  :  (0)
9.0 - 9.5   :  (0)
8.5 - 9.0   :  (0)
8.0 - 8.5   :  (0)
7.5 - 8.0   :  (0)
7.0 - 7.5   :  (2)
6.5 - 7.0   : ████████████████████ (115)
6.0 - 6.5   : ████████████████████████████████████████ (220)
5.0 - 6.0   :  (0)
0.0 - 5.0   :  (0)

```

---

## ⚡ Performance Metrics & Run Execution

- **Total Orchestration Runtime**: `64166.61 seconds`
- **Total YOLO Inference Time**: `63853.46 seconds` (Inference speed: `7529.00 ms/image`)
- **Scoring Engine Update Time**: `305.76 seconds`
- **Quality Band Export time**: `0.41 seconds`
- **SQLite Database Update time**: `1.77 seconds`
- **Target GPU**: `NVIDIA GeForce RTX 5050 Laptop GPU`
- **CUDA Device Available**: `True`
