import os
import sys
import csv
import json
import shutil
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.utils.db_manager import DatabaseManager
from scripts.utils.common import setup_logger, create_markdown_report

def main():
    logger = setup_logger("export_quality_bands")
    logger.info("Starting Gold/Silver/Bronze ranking & candidate curation...")

    db_path = "workspace/database/dataset_index.sqlite"
    std_dir = "workspace/datasets/standardized/DS001"
    curated_dir = "workspace/datasets/curated/DS001_candidate"
    
    db = DatabaseManager(db_path)
    conn = db.conn
    cursor = conn.cursor()

    # 1. Query images with overall scores and audit status
    cursor.execute("""
        SELECT image_id, original_filename, file_path, status, overall_score, 
               blur_score, exposure_score, annotation_quality_score, yolo_agreement_score
        FROM images
        WHERE dataset_id = 'DS001';
    """)
    images = cursor.fetchall()
    logger.info(f"Loaded {len(images)} images from DB.")

    gold_list = []
    silver_list = []
    bronze_list = []
    review_list = []
    reject_list = []

    ranking_rows = []

    # Count statistics
    imported_count = len(images)
    accepted_count = 0
    review_count = 0
    rejected_count = 0

    for img in images:
        image_id = img['image_id']
        filename = img['original_filename']
        status = img['status']
        score = img['overall_score'] or 0.0
        
        # Categorize into quality bands
        # Rejected or duplicates are automatically categorized as Reject
        if status == 'rejected':
            band = "Reject"
            reject_list.append(img)
            rejected_count += 1
        elif status == 'duplicate':
            band = "Reject"
            reject_list.append(img)
            rejected_count += 1
        elif score >= 9.0:
            # Gold must also not be in active warning status
            if status == 'review':
                band = "Review"
                review_list.append(img)
                review_count += 1
            else:
                band = "Gold"
                gold_list.append(img)
                accepted_count += 1
        elif score >= 8.0:
            if status == 'review':
                band = "Review"
                review_list.append(img)
                review_count += 1
            else:
                band = "Silver"
                silver_list.append(img)
                accepted_count += 1
        elif score >= 7.0:
            if status == 'review':
                band = "Review"
                review_list.append(img)
                review_count += 1
            else:
                band = "Bronze"
                bronze_list.append(img)
                accepted_count += 1
        elif score >= 5.0 or status == 'review':
            band = "Review"
            review_list.append(img)
            review_count += 1
        else:
            band = "Reject"
            reject_list.append(img)
            rejected_count += 1

        ranking_rows.append([
            image_id,
            filename,
            status,
            score,
            img['blur_score'],
            img['exposure_score'],
            img['annotation_quality_score'],
            img['yolo_agreement_score'],
            band
        ])

    # 2. Write CSV Ranking Report
    csv_path = "workspace/reports/DS001_quality_ranking.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image ID", "Original Filename", "DB Status", "Overall Score", "Blur Score", "Exposure Score", "Annotation Quality Score", "YOLO Agreement Score", "Quality Band"])
        writer.writerows(ranking_rows)
    logger.info(f"Quality ranking CSV written to: {csv_path}")

    # 3. Curate Candidate Dataset (Copy Gold + Silver)
    splits = ["train", "valid", "test"]
    for split in splits:
        os.makedirs(os.path.join(curated_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(curated_dir, split, "labels"), exist_ok=True)

    candidates_to_copy = gold_list + silver_list
    logger.info(f"Copying {len(candidates_to_copy)} Gold & Silver candidates to {curated_dir}...")
    
    total_candidate_file_size = 0

    for img in candidates_to_copy:
        image_id = img['image_id']
        filename = img['original_filename']
        
        # Determine split
        split_name = "train"
        for s in splits:
            if f"/{s}/" in img['file_path'].replace("\\", "/"):
                split_name = s
                break

        # Standardized paths
        std_img_path = os.path.join(std_dir, split_name, "images", filename)
        std_lbl_path = os.path.join(std_dir, split_name, "labels", f"{os.path.splitext(filename)[0]}.txt")

        # Curated candidate paths
        cur_img_path = os.path.join(curated_dir, split_name, "images", filename)
        cur_lbl_path = os.path.join(curated_dir, split_name, "labels", f"{os.path.splitext(filename)[0]}.txt")

        # Copy image and label
        if os.path.exists(std_img_path):
            shutil.copy2(std_img_path, cur_img_path)
            total_candidate_file_size += os.path.getsize(cur_img_path)
        if os.path.exists(std_lbl_path):
            shutil.copy2(std_lbl_path, cur_lbl_path)

    # Write curated data.yaml
    curated_yaml = {
        "path": os.path.abspath(curated_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["acne"]
    }
    with open(os.path.join(curated_dir, "data.yaml"), "w", encoding="utf-8") as yf:
        for key, val in curated_yaml.items():
            if isinstance(val, list):
                yf.write(f"{key}: {val}\n")
            else:
                yf.write(f"{key}: {val}\n")

    # 4. Calculate bounding box statistics for the report card
    # Annotation statistics on database
    # Original bboxes (is_original = 1)
    cursor.execute("SELECT COUNT(*) FROM annotations WHERE is_original = 1 AND image_id IN (SELECT image_id FROM images WHERE dataset_id = 'DS001');")
    total_original_boxes = cursor.fetchone()[0] or 1

    # Standardized accepted bboxes (is_original = 0, is_valid = 1, status = accepted)
    cursor.execute("""
        SELECT COUNT(*) FROM annotations a
        JOIN images i ON a.image_id = i.image_id
        WHERE i.dataset_id = 'DS001' AND a.is_original = 0 AND a.is_valid = 1 AND (i.status = 'accepted');
    """)
    total_accepted_boxes = cursor.fetchone()[0] or 0

    # Standardized review bboxes (is_original = 0, status = review)
    cursor.execute("""
        SELECT COUNT(*) FROM annotations a
        JOIN images i ON a.image_id = i.image_id
        WHERE i.dataset_id = 'DS001' AND a.is_original = 0 AND (i.status = 'review');
    """)
    total_review_boxes = cursor.fetchone()[0] or 0

    total_removed_boxes = total_original_boxes - (total_accepted_boxes + total_review_boxes)
    retained_pct = (total_accepted_boxes / total_original_boxes) * 100.0
    
    # Average lesions per image in accepted set
    avg_lesions = (total_accepted_boxes / accepted_count) if accepted_count > 0 else 0.0

    # 5. Write DS001_quality_summary.md
    summary_path = "workspace/reports/DS001_quality_summary.md"
    
    summary_md = f"""# DS001 Quality Ranking & Candidate Curation Report

This report presents the final quality categorization and candidate export metrics for DS001.

---

## 📊 Ingestion & Quality Breakdown

| Quality Band | Image Count | Description |
| :--- | :--- | :--- |
| **Gold** | {len(gold_list)} | Overall score $\\ge$ 9.0 with zero warning flags. |
| **Silver** | {len(silver_list)} | Overall score [8.0, 9.0) with zero warning flags. |
| **Bronze** | {len(bronze_list)} | Overall score [7.0, 8.0) with zero warning flags. |
| **Review** | {len(review_list)} | Score [5.0, 7.0) or flagged by coordinates/model discrepancy/clinical warnings. |
| **Reject** | {len(reject_list)} | Score < 5.0, low resolution, corruption, or identified as duplicate. |

---

## 🧹 Bounding Box Optimization Stats
- **Original Annotations Scanned**: {total_original_boxes}
- **Bounding Boxes Retained (Curated)**: {total_accepted_boxes}
- **Bounding Boxes Removed (Ignored/Rejected/Duplicate)**: {total_removed_boxes}
- **Bounding Boxes Pending Review**: {total_review_boxes}
- **Percentage of Original Annotations Retained**: {retained_pct:.2f}%
- **Average Lesions/Image in Curated Pool**: {avg_lesions:.2f}

---

## 📦 Curated Candidate Profile
- **Candidate Export Path**: `workspace/datasets/curated/DS001_candidate/`
- **Curated Dataset Size**: {total_candidate_file_size / (1024*1024):.2f} MB
- **Total Exported Images**: {len(candidates_to_copy)} (Gold + Silver)
  - Train Split: {sum(1 for img in candidates_to_copy if "/train/" in img['file_path'].replace("\\", "/"))}
  - Valid Split: {sum(1 for img in candidates_to_copy if "/valid/" in img['file_path'].replace("\\", "/"))}
  - Test Split: {sum(1 for img in candidates_to_copy if "/test/" in img['file_path'].replace("\\", "/"))}
"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    logger.info(f"Quality summary written to: {summary_path}")
    
    db.close()
    print("=== DS001 QUALITY EXPORT GENERATED ===")

if __name__ == "__main__":
    main()
