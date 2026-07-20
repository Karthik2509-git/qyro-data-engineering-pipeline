# QYRO Dataset v2 Acceptance Policy (Constitution)

This document establishes the official quality and engineering standards for the ingestion and curation of images and annotations into the QYRO Dataset v2. Every dataset candidate must be audited against these rules before inclusion.

---

## 1. Core Objectives
- Ensure high-fidelity annotations for acne lesions to support precise bounding box detection.
- Establish strict data lineage, legal safety (licensing), and traceability.
- Eliminate visual anomalies (watermarks, excessive blur, extreme corruption) that degrade model training.

---

## 2. Image Quality Standards

| Parameter | Minimum Requirement | Detection Method | Action on Failure |
| :--- | :--- | :--- | :--- |
| **Resolution** | Minimum dimensions of 640x640 pixels. | Image properties check | Flag/Reject (if < 640px on either axis) |
| **Blur Threshold** | Laplacian variance score $\ge 100.0$ (standardized scale). | Variance of Laplacian ($s_{blur} < \text{threshold}$) | Route to review/reject queue |
| **Exposure / Quality** | Exposure index between 15% and 85% (excluding absolute dark/blown out). | Histogram distribution | Route to review queue |
| **Watermarks & Text** | No watermarks, copyright text, overlays, or diagnostic overlays obscuring lesions. | Visual audit (during review queue stage) | Route to reject status |
| **File Integrity** | Image must be decodable (JPEG/PNG) without truncation or corrupt headers. | `cv2.imread()` or `PIL.Image.open()` test | Hard reject |

---

## 3. Annotation & Labeling Rules

### A. Class Harmonization
- **Single Target Class**: The target schema is strictly `acne`.
- **Multi-Class Conversion**: Multi-class datasets (e.g., distinguishing *papules*, *pustules*, *nodules*, *comedones*) must be harmonized and mapped to a single class: `acne`.
- **Non-Acne Classes**: Any annotations for unrelated conditions (e.g., *eczema*, *rosacea*, *melasma*, *scars*) must be stripped or marked for ignore, unless specified as a negative class.

### B. Bounding Box Rules
1. **Tightness**: Bounding boxes must tightly fit the outer margins of the lesion. Do not leave excessive background padding ($>10\%$).
2. **No Multi-Lesion Clustering**: 
   - Each individual acne lesion must have its own separate bounding box.
   - Bounding boxes covering multiple disjoint lesions ("cluster boxes") are strictly prohibited.
   - If a raw dataset contains cluster boxes, they must be flagged for manual split in the **Human Review Queue** or rejected.
3. **Overlaps**: Overlapping boxes are permitted only when multiple distinct lesions physically overlap.
4. **Out of Bounds**: Bounding box coordinates must lie entirely within the normalized range $[0.0, 1.0]$. Negative coordinates or values $>1.0$ will fail audit.

---

## 4. Licensing and Source Traceability

Every dataset imported into the pipeline must include three explicit documents in its raw root folder:
- **`license.txt`**: Complete terms of the licensing agreement.
- **`source_url.txt`**: Canonical URL to the download page or repository.
- **`citation.txt`**: Standard BibTeX or text academic citation (if applicable).

### Allowed License Types
- **Permissive Open Source**: MIT, Apache 2.0, BSD.
- **Creative Commons**: CC0, CC-BY, CC-BY-SA, CC-BY-NC (if internal research-only).
- **Prohibited**: Proprietary or unknown terms, datasets with explicit clauses prohibiting AI training/derivative works.

---

## 5. Duplicate and Lineage Constraints

- **Perceptual Duplicate Detection**: No exact duplicate or near-duplicate images are allowed in the curated dataset.
- **Cross-Dataset Deduplication**: Images imported from Kaggle, Roboflow, or GitHub must be cross-referenced via md5 hash and perceptual hashing (pHash/dHash). If a match is found across datasets, the higher resolution/better annotated image is accepted, and the other is marked as `duplicate` referencing the master image ID.
- **Split Pollution Avoidance**: Duplicate checking is done globally to ensure that the same subject image does not end up in both the training split and validation/test splits.

---

## 6. Pipeline Ingestion Protocol

```text
       [Raw Dataset Ingested]
                 │
                 ▼
      [Automated Schema Check]  ──(Fail)──► [Rejected] (Status: rejected)
                 │
                 ▼
      [Quality & Duplication]   ──(Fail)──► [Rejected/Duplicate] (Status: duplicate/rejected)
                 │
                 ▼
      [Annotation Sanity Check] ──(Fail)──► [Human Review Queue] (Status: review)
                 │
                 ▼
       [Curation & Filtering]   ──(Pass)──► [Curated Pool] (Status: accepted)
                 │
                 ▼
      [Export & Standardization] ──► [QYRO Dataset v2 Final]
```
