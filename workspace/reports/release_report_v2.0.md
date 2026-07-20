# QYRO Dataset v2.0 Release Report

This report summarizes the final release specifications, curation statistics, and split partitioning of the **QYRO Dataset v2.0**.

---

## 📈 Release Overview

- **Dataset Version**: `2.0` (Frozen Release)
- **Release Date**: 2026-07-07
- **Dataset Fingerprint SHA256**: `184dde9ec69c7e16ca5217a442fdc2026d2715dc0f015673e4ba3bcde8a3186b`
- **Total Release Images**: `835`
- **Total Release Annotations (Acne Lesions)**: `5,561`
- **Average Lesions per Image**: `6.66`
- **Clean Package Location**: `workspace/datasets/curated/dataset_v2_export/`
- **Release Archive File**: `workspace/datasets/curated/qyro_dataset_v2.0.zip`
- **Release Archive Size**: `101.07 MB`

---

## 🌐 Dataset Contribution Summary

The final curated pool consists of accepted high-quality candidates from five source datasets (DS001–DS005). The detailed yields and audit rejections are:

| Dataset ID | Source Name | Ingested | Accepted | Review Queue | Hard Rejected | Duplicates | Total BBoxes | Yield % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DS001** | Roboflow Acne Detection Dataset v1 | 8,481 | **274** | 7,235 | 29 | 943 | 1,824 | 3.23% |
| **DS002** | Roboflow Pimples Detection Dataset v13 | 4,409 | **149** | 3,995 | 4 | 261 | 867 | 3.38% |
| **DS003** | Roboflow Acne Detection E97Ja Dataset v5 | 1,130 | **1** | 853 | 223 | 53 | 1 | 0.09% |
| **DS004** | Roboflow Acne Detection CHP6J Dataset v1 | 1,389 | **226** | 1,084 | 0 | 79 | 1,741 | 16.27% |
| **DS005** | Roboflow Acne Vulgaris Dataset v1 | 1,244 | **185** | 818 | 0 | 241 | 1,128 | 14.87% |
| **Total** | | **16,653** | **835** | **13,985** | **256** | **1,577** | **5,561** | **5.01%** |

---

## 📦 Dataset Partition splits

The dataset has been shuffled and split into standard partitions using random seed `42` with strict verification checks to guarantee zero data leakage or partition contamination.

- **Train Split (70%)**: `584` images | `3,862` annotations
- **Val Split (15%)**: `125` images | `839` annotations
- **Test Split (15%)**: `126` images | `860` annotations
- **Image Resolution**: `640 x 640` (Letterbox resized with gray padding color `(114, 114, 114)`)

---

## 🧹 Duplicate & Leakage Auditing Notes

A final cross-dataset audit identified three near-duplicate image pairs in the accepted pool that breached partition boundaries. The following conflicts were resolved by deprecating the lower-quality candidates to duplicate status and rebuilding the merged pool:

1. **Conflict 1**: `DS004_00274` resolved in favor of `DS001_01808` (overall quality score `8.91` vs `8.90`).
2. **Conflict 2**: `DS001_06526` resolved in favor of `DS004_00344` (overall quality score `8.51` vs `8.26`).
3. **Conflict 3**: `DS001_08121` resolved in favor of `DS004_00920` (overall quality score `8.54` vs `8.19`).

This resolution successfully eliminated split contamination, achieving **100% split leakage verification compliance** for release.
