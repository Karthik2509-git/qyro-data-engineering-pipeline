# Dataset Factory Version 1.0 — Future Backlog & Technical Debt

This document serves as the register of planned future enhancements and visual calibration ideas. These are deferred improvements to be addressed in subsequent versions.

---

## 🛠️ Feature Backlog

1. **Perceptual Hash Enhancement (dHash)**
   - Integrate 16x16 or 32x32 dHash sizes for increased structural detail.
   - Run duplicate searches on multiple workers using multiprocessing.

2. **Active Learning Curation Loops**
   - Automatically identify and query images where the YOLO model confidence lies between `[0.30, 0.50]` to prioritize human review.
   - Integrate labeling software webhooks to pull edits directly.

3. **Polygonal Segmentation Support**
   - Extend conversion pipeline to handle YOLOv8-seg masks and validate polygon vertex counts.

4. **Multi-Class Severity Classification**
   - Classify subtypes of acne (papules, pustules, comedones) using a dedicated ResNet classifier.

5. **Semi-Supervised Quality Scoring**
   - Train a lightweight quality classifier using the Gold and Reject annotations as positive/negative anchors.

6. **Fitzpatrick Skin Tone Auditing**
   - Replace proxy lightness categorization with a trained skin classifier.
