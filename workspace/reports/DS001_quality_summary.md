# DS001 Quality Ranking & Candidate Curation Report

This report presents the final quality categorization and candidate export metrics for DS001.

---

## 📊 Ingestion & Quality Breakdown

| Quality Band | Image Count | Description |
| :--- | :--- | :--- |
| **Gold** | 0 | Overall score $\ge$ 9.0 with zero warning flags. |
| **Silver** | 0 | Overall score [8.0, 9.0) with zero warning flags. |
| **Bronze** | 0 | Overall score [7.0, 8.0) with zero warning flags. |
| **Review** | 337 | Score [5.0, 7.0) or flagged by coordinates/model discrepancy/clinical warnings. |
| **Reject** | 8144 | Score < 5.0, low resolution, corruption, or identified as duplicate. |

---

## 🧹 Bounding Box Optimization Stats
- **Original Annotations Scanned**: 118693
- **Bounding Boxes Retained (Curated)**: 0
- **Bounding Boxes Removed (Ignored/Rejected/Duplicate)**: 118098
- **Bounding Boxes Pending Review**: 595
- **Percentage of Original Annotations Retained**: 0.00%
- **Average Lesions/Image in Curated Pool**: 0.00

---

## 📦 Curated Candidate Profile
- **Candidate Export Path**: `workspace/datasets/curated/DS001_candidate/`
- **Curated Dataset Size**: 0.00 MB
- **Total Exported Images**: 0 (Gold + Silver)
  - Train Split: 0
  - Valid Split: 0
  - Test Split: 0
