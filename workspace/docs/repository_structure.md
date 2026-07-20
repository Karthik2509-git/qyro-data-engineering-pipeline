# QYRO Dataset Engineering Workspace Layout

This document describes the structure of the `TEMP-QYRO` temporary workspace, outlining the file hierarchy, purpose of individual subfolders, and SQLite database schema mapping.

---

## 1. Directory Tree

```text
workspace/
├── configs/
│   └── default_dataset_policy.yaml  # Configures quality thresholds & folder paths
│
├── database/
│   └── dataset_index.sqlite         # SQLite Index Database
│
├── datasets/
│   ├── external/                    # Immutable download cache (DS001, DS002, etc.)
│   ├── raw/                         # Copied raw images and annotations
│   ├── standardized/                # Converted annotations (homogenized classes)
│   ├── audited/                     # Audited images and annotations (passed)
│   ├── curated/                     # Final selected set of image-annotation files
│   └── rejected/                    # Rejected files (kept for history/debugging)
│
├── docs/
│   ├── dataset_acceptance_policy.md # The Dataset v2 Quality Constitution
│   ├── repository_structure.md      # This file
│   └── workflow.md                  # Explanation of the 12 pipeline stages
│
├── reports/
│   └── initialization_report.md     # Phase T1 completion verification report
│
└── scripts/
    ├── import/
    │   └── import_dataset.py        # Copies data & maps IDs (e.g. DS001)
    ├── conversion/
    │   └── convert_formats.py       # Converts COCO/VOC/YOLO to schema
    ├── audit/
    │   └── audit_annotations.py     # Audits bounding box coordinates
    ├── scoring/
    │   └── scoring_engine.py        # Scores image and annotation quality
    ├── dedup/
    │   └── deduplicate.py           # Runs pHash/dHash check and links duplicates
    ├── review/
    │   └── review_queue.py          # Interfaces with Human Review Queue
    ├── export/
    │   └── export_dataset.py        # Letterboxes and outputs Dataset v2
    └── utils/
        ├── common.py                # Logging, YAML, hashing utility helpers
        └── db_manager.py            # SQLite database schema operations
```

---

## 2. Database Schema Details

The SQL schema is managed dynamically by `scripts/utils/db_manager.py`. It is structured to handle metadata indexing, scoring, logical status tracking, and future-proof segmentation masks.

### A. Table: `datasets`
Tracks ingestion data and source credentials.

| Column | Type | Description |
| :--- | :--- | :--- |
| `dataset_id` | TEXT (PK) | Unique dataset code (e.g., `DS001`, `DS002`). |
| `name` | TEXT | Display name of the dataset. |
| `source_url` | TEXT | Canonical download URL. |
| `license_type` | TEXT | License category (e.g., MIT, CC-BY-4.0). |
| `citation` | TEXT | BibTeX or paper citation string. |
| `imported_at` | TEXT | ISO8601 ingestion timestamp. |

### B. Table: `images`
Stores file attributes, evaluation metrics, and pipeline statuses.

| Column | Type | Description |
| :--- | :--- | :--- |
| `image_id` | TEXT (PK) | Generated ID matching dataset (e.g., `DS001_000001`). |
| `dataset_id` | TEXT | Reference to `datasets(dataset_id)`. |
| `original_filename` | TEXT | Original filename prior to standard naming. |
| `file_path` | TEXT | Relative path to image file in workspace. |
| `file_hash` | TEXT | MD5 checksum of the image file. |
| `perceptual_hash` | TEXT | Difference Hash (dHash) hex representation. |
| `width` | INTEGER | Original image width. |
| `height` | INTEGER | Original image height. |
| `blur_score` | REAL | Image sharpness score. |
| `exposure_score` | REAL | Histogram lighting assessment score. |
| `duplicate_risk` | REAL | Near-duplicate correlation metric. |
| `annotation_quality_score` | REAL | Bounding box layout quality score. |
| `lesion_visibility_score` | REAL | Contrast and lesion visibility score. |
| `cluster_quality_score` | REAL | Bounding box overlap and clustering audit score. |
| `overall_score` | REAL | Weighted aggregate quality score. |
| `status` | TEXT | Image status: `accepted`, `rejected`, `ignored`, `review`, `duplicate`. |
| `rejection_reason` | TEXT | Text describing the cause of reject or review routing. |
| `updated_at` | TEXT | ISO8601 update timestamp. |

### C. Table: `annotations`
Tracks bounding boxes and coordinates. Designed to support segmentation masks using polygon datasets.

| Column | Type | Description |
| :--- | :--- | :--- |
| `annotation_id` | TEXT (PK) | Unique annotation identifier. |
| `image_id` | TEXT | Reference to `images(image_id)`. |
| `class_label` | TEXT | Normalized label (mapped to `acne`). |
| `annotation_type` | TEXT | Annotation structure: `bbox` or `segmentation`. |
| `data` | TEXT | JSON string representing bounding box `[x, y, w, h]` or list of polygons `[[x1,y1,x2,y2...]]`. |
| `is_original` | INTEGER | Flag: `1` for raw file values, `0` for homogenized version. |
| `is_valid` | INTEGER | Flag: `1` for valid coordinates, `0` for failed audit. |
| `audit_reason` | TEXT | Text describing coordinate anomalies. |
| `updated_at` | TEXT | ISO8601 update timestamp. |
