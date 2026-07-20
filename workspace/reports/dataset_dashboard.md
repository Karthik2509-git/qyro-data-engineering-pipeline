# Dataset Health Dashboard & Report Card

*Generated on: 2026-07-07 12:36:30*

## Executive Summary
Unified status analysis of the ingested acne datasets.

## Source Ingestion Metrics
| Dataset ID | Name | Ingested Images | Accepted for Curated Pool | Yield Rate |
| :--- | :--- | :--- | :--- | :--- |
| `DS001` | Roboflow Acne Detection Dataset v1 | 8481 | 274 | 3.2% |
| `DS002` | Roboflow Pimples Detection Dataset v13 | 4409 | 149 | 3.4% |
| `DS003` | Roboflow Acne Detection E97Ja Dataset v5 | 1130 | 1 | 0.1% |
| `DS004` | Roboflow Acne Detection CHP6J Dataset v1 | 1389 | 226 | 16.3% |
| `DS005` | Roboflow Acne Vulgaris Dataset v1 | 1244 | 185 | 14.9% |

## Dashboard Metrics
## 📊 General Metrics Summary
- **Total Ingested Images**: 16653
  - **Accepted (Curated Pool)**: 835
  - **Rejected (Filter/Audit Rejections)**: 256
  - **Duplicates Resolved**: 1577
  - **Awaiting Human Triage (Review Queue)**: 13985
- **Yield Rate (Accepted/Total)**: 5.0%
- **Curated Bounding Boxes**: 5561
- **Average Lesions per Image**: 6.7
- **Average Bounding Box Area**: 0.23% of image frame
- **Average Quality Score**: 6.97/10
- **Average YOLO prediction agreement**: 2.09/10


## Diversity Indicators
## 🧬 Dataset Diversity Index (Curated Images Only)

### Acne Severity Distribution
- **Mild (< 5 lesions)**: 693 images (83.0%)
- **Moderate (5 to 20 lesions)**: 128 images (15.3%)
- **Severe (> 20 lesions)**: 14 images (1.7%)

### Lighting Conditions Category
- **Optimal Exposure**: 833 images
- **Low-Light / Dark**: 0 images
- **High-Exposure / Flash**: 0 images

### Skin Tone Lightness proxy (Fitzpatrick-like proxy)
- **L1 (Fair / Light Skin tone proxy)**: 0 images
- **L2 (Medium / Tan Skin tone proxy)**: 169 images
- **L3 (Dark / Deep Skin tone proxy)**: 664 images


## Distribution Charts
## 📈 Metric Distribution Histograms (Curated Images Only)

### Overall Curated Score Distribution (Composite Quality)
```text
[8.0 - 8.4]     | ██████████████████████████████ (256)
[8.4 - 8.8]     | ████████████████████████████████████████ (336)
[8.8 - 9.2]     | █████████████████ (144)
[9.2 - 9.6]     | ███████ (66)
[9.6 - 10.0]    | ███ (33)
```

### Resolution Dimensions (Original Width Distribution)
```text
[416.0 - 460.8] | ███████████ (185)
[460.8 - 505.6] |  (0)
[505.6 - 550.4] |  (0)
[550.4 - 595.2] |  (0)
[595.2 - 640.0] | ████████████████████████████████████████ (650)
```
