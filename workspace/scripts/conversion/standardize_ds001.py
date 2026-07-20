import os
import sys
import json
import csv
import shutil
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.utils.db_manager import DatabaseManager
from scripts.utils.common import setup_logger, create_markdown_report

CLASS_NAMES = [
    'Acne', 'Blackhead', 'Conglobata', 'Crystanlline', 'Cystic', 
    'Flat_wart', 'Folliculitis', 'Keloid', 'Milium', 'Papular', 
    'Purulent', 'Scars', 'Sebo-crystan-conglo', 'Syringoma', 'Whitehead'
]

# Action definitions
KEEP_AS_ACNE_IDX = [0, 1, 2, 4, 9, 10, 14]       # Acne, Blackhead, Conglobata, Cystic, Papular, Purulent, Whitehead
REVIEW_IDX = [3, 6, 8, 12]                       # Crystanlline, Folliculitis, Milium, Sebo-crystan-conglo
IGNORE_IDX = [7, 11]                             # Keloid, Scars
REJECT_IDX = [5, 13]                             # Flat_wart, Syringoma

def main():
    logger = setup_logger("standardize_ds001")
    logger.info("Starting DS001 Smart Conversion & Mapping...")

    db_path = "workspace/database/dataset_index.sqlite"
    base_dir = "workspace/datasets/raw/DS001/Acne-newdataset-roboflow"
    std_dir = "workspace/datasets/standardized/DS001"
    
    db = DatabaseManager(db_path)
    conn = db.conn
    cursor = conn.cursor()

    # 1. Fetch images from DB
    cursor.execute("SELECT image_id, original_filename, file_path, status FROM images WHERE dataset_id = 'DS001';")
    images = cursor.fetchall()

    logger.info(f"Loaded {len(images)} images to process.")

    retained_boxes_count = 0
    removed_boxes_count = 0
    review_boxes_count = 0

    provenance_rows = []

    splits = ["train", "valid", "test"]
    for split in splits:
        os.makedirs(os.path.join(std_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(std_dir, split, "labels"), exist_ok=True)

    # Begin transaction explicitly for SQLite speed optimization
    cursor.execute("BEGIN TRANSACTION;")
    now_str = datetime.now().isoformat()

    try:
        for img in images:
            image_id = img['image_id']
            orig_filename = img['original_filename']
            src_img_path = img['file_path']
            current_status = img['status']

            # Determine split from file_path
            split_name = "train"
            for s in splits:
                if f"/{s}/" in src_img_path.replace("\\", "/"):
                    split_name = s
                    break

            # Destination paths
            dest_img_path = os.path.join(std_dir, split_name, "images", orig_filename)
            dest_lbl_path = os.path.join(std_dir, split_name, "labels", f"{os.path.splitext(orig_filename)[0]}.txt")

            # Copy image file to standardized folder
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dest_img_path)

            # Resolve raw label file path from image path
            src_lbl_path = src_img_path.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
            src_lbl_path = os.path.splitext(src_lbl_path)[0] + ".txt"

            orig_labels = []
            mapped_labels = []
            removed_classes = []
            review_classes = []
            
            yolo_lines_to_write = []
            
            has_review_class = False
            has_reject_class = False

            if os.path.exists(src_lbl_path):
                with open(src_lbl_path, "r", encoding="utf-8") as lf:
                    lines = lf.readlines()
                    
                for idx, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_idx = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                    except ValueError:
                        continue
                        
                    raw_label = CLASS_NAMES[cls_idx] if 0 <= cls_idx < len(CLASS_NAMES) else f"unknown_{cls_idx}"
                    orig_labels.append(raw_label)
                    
                    # Insert original annotation directly without commit
                    cursor.execute("""
                        INSERT OR REPLACE INTO annotations (
                            annotation_id, image_id, class_label, annotation_type, data, 
                            is_original, is_valid, audit_reason, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """, (f"ANN_{image_id}_{idx:03d}_orig", image_id, raw_label, "bbox", json.dumps({"bbox": coords}), 1, 1, None, now_str))
                    
                    # Apply mapping policy
                    if cls_idx in KEEP_AS_ACNE_IDX:
                        mapped_labels.append("acne")
                        retained_boxes_count += 1
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO annotations (
                                annotation_id, image_id, class_label, annotation_type, data, 
                                is_original, is_valid, audit_reason, updated_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (f"ANN_{image_id}_{idx:03d}_std", image_id, "acne", "bbox", json.dumps({"bbox": coords}), 0, 1, None, now_str))
                        yolo_lines_to_write.append(f"0 {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                        
                    elif cls_idx in REVIEW_IDX:
                        review_classes.append(raw_label)
                        review_boxes_count += 1
                        has_review_class = True
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO annotations (
                                annotation_id, image_id, class_label, annotation_type, data, 
                                is_original, is_valid, audit_reason, updated_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """, (f"ANN_{image_id}_{idx:03d}_std", image_id, raw_label, "bbox", json.dumps({"bbox": coords}), 0, 1, f"Class {raw_label} placed in review queue", now_str))
                        
                    elif cls_idx in IGNORE_IDX:
                        removed_classes.append(raw_label)
                        removed_boxes_count += 1
                        
                    elif cls_idx in REJECT_IDX:
                        removed_classes.append(raw_label)
                        removed_boxes_count += 1
                        has_reject_class = True

            # Write standardized annotation to file
            with open(dest_lbl_path, "w", encoding="utf-8") as lf:
                lf.writelines(yolo_lines_to_write)

            # Update image status directly without commit
            new_status = current_status
            rejection_reason = None
            
            if has_reject_class:
                new_status = "rejected"
                rejection_reason = f"Rejected due to containing non-acne clinical class: {', '.join([c for c in orig_labels if c in ['Flat_wart', 'Syringoma']])}"
            elif has_review_class:
                new_status = "review"
                rejection_reason = f"Contains class requiring review: {', '.join(review_classes)}"
                
            if new_status != current_status:
                cursor.execute("""
                    UPDATE images
                    SET status = ?, rejection_reason = ?, updated_at = ?
                    WHERE image_id = ?;
                """, (new_status, rejection_reason, now_str, image_id))

            # Record provenance for report
            provenance_rows.append([
                image_id,
                "DS001",
                ",".join(set(orig_labels)) if orig_labels else "None",
                ",".join(set(mapped_labels)) if mapped_labels else "None",
                ",".join(set(removed_classes)) if removed_classes else "None",
                new_status
            ])
            
        # Commit the single batch transaction
        conn.commit()
        logger.info("Finished writing standardized files and database inserts (Committed in one batch).")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error during batch database operations: {e}")
        db.close()
        sys.exit(1)

    # 3. Create standardized data.yaml
    std_yaml = {
        "path": os.path.abspath(std_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["acne"]
    }
    with open(os.path.join(std_dir, "data.yaml"), "w", encoding="utf-8") as yf:
        for key, val in std_yaml.items():
            if isinstance(val, list):
                yf.write(f"{key}: {val}\n")
            else:
                yf.write(f"{key}: {val}\n")

    # 4. Generate provenance CSV
    csv_path = "workspace/reports/DS001_mapping_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image ID", "Original Dataset ID", "Original Classes", "Final Mapped Classes", "Classes Removed", "Review Status"])
        writer.writerows(provenance_rows)
    logger.info(f"Mapping CSV manifest written to: {csv_path}")

    # 5. Generate Markdown summary
    summary_path = "workspace/reports/DS001_mapping_summary.md"
    total_total = retained_boxes_count + removed_boxes_count + review_boxes_count
    retained_pct = (retained_boxes_count / total_total * 100.0) if total_total > 0 else 0.0
    
    summary_md = f"""# DS001 Semantic Mapping Summary Report

This report summarizes the results of applying the approved medical class mapping policy to DS001.

---

## Mapping Metrics
- **Retained Acne Bounding Boxes**: {retained_boxes_count} ({retained_pct:.1f}% of total)
- **Excluded Bounding Boxes (Ignored/Rejected)**: {removed_boxes_count}
- **Pending Review Bounding Boxes**: {review_boxes_count}

---

## Action Mappings Profile
- **KEEP_AS_ACNE**: Mapped `Acne`, `Blackhead`, `Whitehead`, `Papular`, `Purulent`, `Cystic`, and `Conglobata` to standard `acne`.
- **REVIEW**: Flagged and isolated `Milium`, `Crystanlline`, `Sebo-crystan-conglo`, and `Folliculitis` into review queue.
- **IGNORE**: Removed `Scars` and `Keloid` from yolo annotations but tracked in original metadata.
- **REJECT**: Excluded `Flat_wart` and `Syringoma` entirely.
"""
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
    logger.info(f"Mapping summary written to: {summary_path}")

    db.close()
    print("=== DS001 STANDARDIZATION & PROVENANCE GENERATED ===")

if __name__ == "__main__":
    main()
