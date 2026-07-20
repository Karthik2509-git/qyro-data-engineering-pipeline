import os
import sys
import time
import json
import csv
import sqlite3
import random
import shutil
import subprocess
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.utils.db_manager import DatabaseManager
from scripts.utils.common import setup_logger, load_config

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torch
    from ultralytics import YOLO
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def calculate_box_iou(box1, box2):
    """Calculates Intersection-over-Union (IoU) of two YOLO boxes [xc, yc, w, h]."""
    xc1, yc1, w1, h1 = box1
    xc2, yc2, w2, h2 = box2
    
    x1, y1 = xc1 - w1 / 2, yc1 - h1 / 2
    x2, y2 = xc1 + w1 / 2, yc1 + h1 / 2
    
    x3, y3 = xc2 - w2 / 2, yc2 - h2 / 2
    x4, y4 = xc2 + w2 / 2, yc2 + h2 / 2
    
    xi_min = max(x1, x3)
    yi_min = max(y1, y3)
    xi_max = min(x2, x4)
    yi_max = min(y2, y4)
    
    inter_w = max(0.0, xi_max - xi_min)
    inter_h = max(0.0, yi_max - yi_min)
    inter_area = inter_w * inter_h
    
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def migrate_database(db_path, logger):
    """Ensure database has precision, recall, f1, missing_ratio, extra_ratio, confidence stats and primary reasoning columns."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    columns_to_add = {
        "precision": "REAL",
        "recall": "REAL",
        "f1": "REAL",
        "missing_ratio": "REAL",
        "extra_ratio": "REAL",
        "mean_confidence": "REAL",
        "median_confidence": "REAL",
        "min_confidence": "REAL",
        "max_confidence": "REAL",
        "primary_review_reason": "TEXT",
        "secondary_review_reasons": "TEXT"
    }
    for col, col_type in columns_to_add.items():
        try:
            cursor.execute(f"ALTER TABLE images ADD COLUMN {col} {col_type};")
            logger.info(f"Added column {col} to database.")
        except sqlite3.OperationalError:
            # Column already exists
            pass
    conn.commit()
    conn.close()

def main():
    logger = setup_logger("run_yolo_validation")
    logger.info("=== STARTING PHASE T4.4 VALIDATION ENGINE ===")
    
    start_time = time.time()
    random.seed(42)  # Secure stable random seed

    config_path = "workspace/configs/default_dataset_policy.yaml"
    db_path = "workspace/database/dataset_index.sqlite"
    model_path = "C:/Users/KARTHIK V/OneDrive/Desktop/QYRO-Medical-AI/models/production/qyro_acne_v1_best.pt"
    reports_dir = "workspace/reports"
    
    # 1. Database schema migration
    logger.info("Step 1: Migrating Database Schema...")
    db_migrate_start = time.time()
    migrate_database(db_path, logger)
    db_migrate_time = time.time() - db_migrate_start

    # Record OLD quality bands from database before run
    db = DatabaseManager(db_path)
    conn = db.conn
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT image_id, status, overall_score 
        FROM images 
        WHERE dataset_id = 'DS001';
    """)
    old_records = cursor.fetchall()
    
    old_bands = {"Gold": 0, "Silver": 0, "Review": 0, "Reject": 0}
    for rec in old_records:
        status = rec['status']
        score = rec['overall_score'] or 0.0
        if status == 'rejected': old_bands["Reject"] += 1
        elif status == 'duplicate': old_bands["Reject"] += 1
        elif status == 'review': old_bands["Review"] += 1
        elif score >= 9.0: old_bands["Gold"] += 1
        elif score >= 8.0: old_bands["Silver"] += 1
        elif score >= 5.0: old_bands["Review"] += 1
        else: old_bands["Reject"] += 1
        
    # Reset status of non-duplicate and non-audit-failed images back to 'accepted' for full re-evaluation
    logger.info("Resetting active image statuses back to 'accepted' for clean calibration...")
    cursor.execute("""
        UPDATE images
        SET status = 'accepted',
            rejection_reason = NULL
        WHERE dataset_id = 'DS001'
          AND status != 'duplicate'
          AND (rejection_reason IS NULL OR (rejection_reason NOT LIKE 'Annotation audit%' AND rejection_reason NOT LIKE '%corruption%'));
    """)
    conn.commit()
    
    db.close()

    # 2. Run real YOLO agreement
    logger.info("Step 2: Executing Real YOLO Agreement Audit...")
    yolo_start = time.time()
    yolo_cmd = [
        sys.executable,
        "workspace/scripts/audit/yolo_agreement.py",
        "--dataset_id", "DS001",
        "--model_path", model_path,
        "--config", config_path
    ]
    subprocess.run(yolo_cmd, check=True)
    yolo_time = time.time() - yolo_start

    # 3. Run multi-metric scoring engine
    logger.info("Step 3: Recalculating Quality Scores...")
    scoring_start = time.time()
    scoring_cmd = [
        sys.executable,
        "workspace/scripts/scoring/scoring_engine.py",
        "--dataset_id", "DS001",
        "--config", config_path
    ]
    subprocess.run(scoring_cmd, check=True)
    scoring_time = time.time() - scoring_start

    # 4. Run quality band exporter
    logger.info("Step 4: Regenerating Quality Bands and Splits...")
    export_start = time.time()
    export_cmd = [
        sys.executable,
        "workspace/scripts/export/export_quality_bands.py"
    ]
    subprocess.run(export_cmd, check=True)
    export_time = time.time() - export_start

    # 5. Evaluate and update primary / secondary reasons in database
    logger.info("Step 5: Decomposing Review Queue Triggers...")
    db_update_start = time.time()
    db = DatabaseManager(db_path)
    conn = db.conn
    cursor = conn.cursor()

    cursor.execute("""
        SELECT image_id, status, overall_score, file_path, rejection_reason,
               blur_score, exposure_score, yolo_agreement_score, mean_confidence, f1, yolo_box_count,
               mean_iou, precision, recall
        FROM images
        WHERE dataset_id = 'DS001';
    """)
    images = cursor.fetchall()

    cursor.execute("""
        SELECT image_id, class_label, data, is_original
        FROM annotations
        WHERE image_id IN (SELECT image_id FROM images WHERE dataset_id = 'DS001');
    """)
    annotations = cursor.fetchall()

    # Index annotations by image_id
    anns_by_img = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)

    # Begin batch transaction
    cursor.execute("BEGIN TRANSACTION;")
    
    # Review categories counts
    review_combinations = {}
    rule_contributions = {
        "Low overall score (< 8.0)": 0,
        "YOLO disagreement score (< 8.0)": 0,
        "Blur penalty (normalized blur_score < 4.0)": 0,
        "Exposure penalty (normalized exposure_score < 6.0)": 0,
        "Deduplication (Duplicate status)": 0,
        "Overlap penalty (IoU > 0.85)": 0,
        "Tiny boxes (area < 0.0001)": 0,
        "Border boxes (within 1% edge)": 0,
        "Clinical class review (folliculitis/milium/etc.)": 0
    }

    # Bins for Score Distribution
    score_bins = {
        "9.5 - 10.0": 0,
        "9.0 - 9.5": 0,
        "8.5 - 9.0": 0,
        "8.0 - 8.5": 0,
        "7.5 - 8.0": 0,
        "7.0 - 7.5": 0,
        "6.5 - 7.0": 0,
        "6.0 - 6.5": 0,
        "5.0 - 6.0": 0,
        "0.0 - 5.0": 0
    }

    # Tracking lists for samples
    review_pool = []
    reject_pool = []
    gold_pool = []
    silver_pool = []
    
    # Validation stats
    sum_iou = 0.0
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    sum_pred_diff = 0
    valid_stats_count = 0

    review_queue_images = []

    for img in images:
        image_id = img['image_id']
        status = img['status']
        score = img['overall_score'] or 0.0
        file_path = img['file_path']
        
        blur = img['blur_score'] or 100.0
        exposure = img['exposure_score'] or 50.0
        yolo_agreement = img['yolo_agreement_score'] or 10.0
        mean_confidence = img['mean_confidence'] or 0.0
        f1_score = img['f1'] or 0.0
        pred_box_count = img['yolo_box_count'] or 0
        rejection_reason = img['rejection_reason'] or ""

        # Aggregate stats
        if status not in ('rejected', 'duplicate'):
            sum_iou += img['mean_iou'] or 0.0
            sum_precision += img['precision'] or 0.0
            sum_recall += img['recall'] or 0.0
            sum_f1 += f1_score
            valid_stats_count += 1

        # Classify for sampling pool
        if status == 'rejected' or status == 'duplicate':
            reject_pool.append((image_id, file_path))
        elif status == 'review':
            review_pool.append((image_id, file_path))
        elif score >= 9.0:
            gold_pool.append((image_id, file_path))
        elif score >= 8.0:
            silver_pool.append((image_id, file_path))
        else:
            reject_pool.append((image_id, file_path))

        # Score distribution bins
        if status not in ('rejected', 'duplicate'):
            if score >= 9.5: score_bins["9.5 - 10.0"] += 1
            elif score >= 9.0: score_bins["9.0 - 9.5"] += 1
            elif score >= 8.5: score_bins["8.5 - 9.0"] += 1
            elif score >= 8.0: score_bins["8.0 - 8.5"] += 1
            elif score >= 7.5: score_bins["7.5 - 8.0"] += 1
            elif score >= 7.0: score_bins["7.0 - 7.5"] += 1
            elif score >= 6.5: score_bins["6.5 - 7.0"] += 1
            elif score >= 6.0: score_bins["6.0 - 6.5"] += 1
            elif score >= 5.0: score_bins["5.0 - 6.0"] += 1
            else: score_bins["0.0 - 5.0"] += 1

        # Determine Review Reason flags
        img_anns = anns_by_img.get(image_id, [])
        std_anns = [a for a in img_anns if a['is_original'] == 0]
        orig_anns = [a for a in img_anns if a['is_original'] == 1]
        
        sum_pred_diff += abs(len(std_anns) - pred_box_count)

        is_low_score = score < 8.0 and status != 'rejected'
        is_yolo_disagree = yolo_agreement < 8.0 or abs(len(std_anns) - pred_box_count) > 5
        is_blur = blur < 4.0
        is_exposure = exposure < 6.0
        is_duplicate = status == 'duplicate'

        has_overlap = False
        has_border = False
        has_tiny = False
        has_clinical = False

        boxes = []
        for ann in std_anns:
            try:
                ann_data = json.loads(ann['data'])
                if "bbox" in ann_data:
                    boxes.append(ann_data['bbox'])
            except Exception:
                pass
                
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if calculate_box_iou(boxes[i], boxes[j]) > 0.85:
                    has_overlap = True
                    break
            if has_overlap:
                break

        for ann in std_anns:
            try:
                ann_data = json.loads(ann['data'])
                if "bbox" in ann_data:
                    xc, yc, w, h = ann_data['bbox']
                    if (xc - w/2 < 0.01) or (xc + w/2 > 0.99) or (yc - h/2 < 0.01) or (yc + h/2 > 0.99):
                        has_border = True
                    if w * h < 0.0001:
                        has_tiny = True
            except Exception:
                pass

        for ann in orig_anns:
            if ann['class_label'] in ['Milium', 'Crystanlline', 'Sebo-crystan-conglo', 'Folliculitis']:
                has_clinical = True
                break

        # Accumulate rule contributions
        if is_low_score: rule_contributions["Low overall score (< 8.0)"] += 1
        if is_yolo_disagree: rule_contributions["YOLO disagreement score (< 8.0)"] += 1
        if is_blur: rule_contributions["Blur penalty (normalized blur_score < 4.0)"] += 1
        if is_exposure: rule_contributions["Exposure penalty (normalized exposure_score < 6.0)"] += 1
        if is_duplicate: rule_contributions["Deduplication (Duplicate status)"] += 1
        if has_overlap: rule_contributions["Overlap penalty (IoU > 0.85)"] += 1
        if has_tiny: rule_contributions["Tiny boxes (area < 0.0001)"] += 1
        if has_border: rule_contributions["Border boxes (within 1% edge)"] += 1
        if has_clinical: rule_contributions["Clinical class review (folliculitis/milium/etc.)"] += 1

        # Process specifically for Review status
        if status == 'review':
            # Check review categories
            active_reasons = []
            
            # Clinical mapping
            if has_clinical:
                active_reasons.append("Clinical_Review")
            # Missing annotations: missing_ratio >= 0.40
            cursor.execute("SELECT missing_ratio FROM images WHERE image_id = ?;", (image_id,))
            m_ratio = cursor.fetchone()['missing_ratio'] or 0.0
            if m_ratio >= 0.40:
                active_reasons.append("Missing_Annotations")
            # High model disagreement: f1 < 0.50 or count difference > 5
            if f1_score < 0.50 or abs(len(std_anns) - pred_box_count) > 5:
                active_reasons.append("High_Model_Disagreement")
            # Low model confidence: mean_confidence < 0.40
            if mean_confidence > 0.0 and mean_confidence < 0.40:
                active_reasons.append("Low_Model_Confidence")
            # Border issues
            if has_border:
                active_reasons.append("Border_Issues")
            # Image quality: blur or exposure
            if is_blur or is_exposure:
                active_reasons.append("Image_Quality")
                
            # Assign primary vs secondary based on priority order:
            priority_order = [
                "Clinical_Review",
                "Missing_Annotations",
                "High_Model_Disagreement",
                "Low_Model_Confidence",
                "Border_Issues",
                "Image_Quality"
            ]
            
            primary = "Mixed"
            secondary_list = []
            
            sorted_reasons = [r for r in priority_order if r in active_reasons]
            
            if len(sorted_reasons) > 1:
                primary = sorted_reasons[0]
                secondary_list = sorted_reasons[1:]
            elif len(sorted_reasons) == 1:
                primary = sorted_reasons[0]
            else:
                # Fallback to Low Score
                primary = "Image_Quality"
                
            secondary = ",".join(secondary_list) if secondary_list else "None"
            
            # Record combination for analysis
            combination_key = " + ".join(sorted_reasons) if sorted_reasons else "Unknown"
            review_combinations[combination_key] = review_combinations.get(combination_key, 0) + 1
            
            cursor.execute("""
                UPDATE images
                SET primary_review_reason = ?,
                    secondary_review_reasons = ?
                WHERE image_id = ?;
            """, (primary, secondary, image_id))

            review_queue_images.append({
                "image_id": image_id,
                "file_path": file_path,
                "primary": primary,
                "secondary": secondary,
                "score": score
            })

    conn.commit()
    db_update_time = time.time() - db_update_start
    db.close()

    # 6. Generate Review Category CSVs
    logger.info("Step 6: Exporting Review Category manifests...")
    review_manifests_dir = os.path.join(reports_dir, "review_categories")
    os.makedirs(review_manifests_dir, exist_ok=True)
    
    categories_to_write = {
        "Missing_Annotations": [],
        "Low_Model_Confidence": [],
        "High_Model_Disagreement": [],
        "Border_Issues": [],
        "Clinical_Review": [],
        "Image_Quality": [],
        "Mixed": []
    }
    
    for item in review_queue_images:
        # Check primary reason
        p = item["primary"]
        if p in categories_to_write:
            categories_to_write[p].append(item)
        # Check secondary reasons
        s_list = item["secondary"].split(",")
        for s in s_list:
            if s in categories_to_write:
                categories_to_write[s].append(item)

    for cat_name, items in categories_to_write.items():
        csv_filepath = os.path.join(review_manifests_dir, f"{cat_name}.csv")
        with open(csv_filepath, "w", newline="", encoding="utf-8") as cf:
            writer = csv.writer(cf)
            writer.writerow(["Image ID", "File Path", "Primary Reason", "Secondary Reasons", "Overall Score"])
            for row in items:
                writer.writerow([row["image_id"], row["file_path"], row["primary"], row["secondary"], row["score"]])
        logger.info(f"Wrote manifest: {csv_filepath} ({len(items)} images)")

    # 7. Create random samples folder (50 images)
    logger.info("Step 7: Creating quality validation samples...")
    validation_samples_dir = os.path.join(reports_dir, "T44_validation_samples")
    
    categories_pools = {
        "Review": review_pool,
        "Reject": reject_pool,
        "Gold": gold_pool,
        "Silver": silver_pool
    }
    
    for cat, pool in categories_pools.items():
        cat_dir = os.path.join(validation_samples_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        # Clear existing samples if any
        for f in os.listdir(cat_dir):
            os.remove(os.path.join(cat_dir, f))
            
        sample_size = min(50, len(pool))
        sampled = random.sample(pool, sample_size) if pool else []
        for index, (img_id, filepath) in enumerate(sampled):
            if os.path.exists(filepath):
                ext = os.path.splitext(filepath)[1]
                dest_name = f"{img_id}_sample{ext}"
                dest_path = os.path.join(cat_dir, dest_name)
                # Attempt hardlink first, fallback to copy
                try:
                    os.link(filepath, dest_path)
                except Exception:
                    shutil.copy2(filepath, dest_path)

    # 8. Generate Visual overlays for 50 images
    logger.info("Step 8: Generating 50 visual prediction overlays...")
    overlay_dir = os.path.join(reports_dir, "agreement_overlay")
    os.makedirs(overlay_dir, exist_ok=True)
    # Clear existing overlays
    for f in os.listdir(overlay_dir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(overlay_dir, f))

    # Sample 50 active images with annotations
    db = DatabaseManager(db_path)
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT image_id, file_path, status, overall_score, yolo_agreement_score,
               precision, recall, f1, yolo_box_count
        FROM images
        WHERE dataset_id = 'DS001' AND status != 'rejected' AND status != 'duplicate'
        ORDER BY RANDOM() LIMIT 50;
    """)
    overlay_samples = cursor.fetchall()
    
    if PIL_AVAILABLE and TORCH_AVAILABLE:
        model = YOLO(model_path)
        for index, row in enumerate(overlay_samples):
            image_id = row['image_id']
            file_path = row['file_path']
            score = row['overall_score'] or 0.0
            agreement = row['yolo_agreement_score'] or 0.0
            prec = row['precision'] or 0.0
            rec = row['recall'] or 0.0
            f1 = row['f1'] or 0.0
            
            # Run prediction inline to get coordinates
            results = model(file_path, conf=0.25, iou=0.60, verbose=False)
            pred_boxes = []
            if len(results) > 0:
                boxes = results[0].boxes
                xywhn = boxes.xywhn.cpu().numpy()
                for box in xywhn:
                    pred_boxes.append(box.tolist())
                    
            # Get GT boxes
            cursor.execute("SELECT data FROM annotations WHERE image_id = ? AND is_original = 0 AND is_valid = 1;", (image_id,))
            gt_rows = cursor.fetchall()
            gt_boxes = []
            for gt_row in gt_rows:
                ann_data = json.loads(gt_row['data'])
                if "bbox" in ann_data:
                    gt_boxes.append(ann_data['bbox'])
            
            # Load PIL Image
            with Image.open(file_path) as img:
                img_width, img_height = img.size
                draw = ImageDraw.Draw(img)
                
                # Draw Ground Truth in Green
                for box in gt_boxes:
                    xc, yc, w, h = box
                    x1 = int((xc - w/2) * img_width)
                    y1 = int((yc - h/2) * img_height)
                    x2 = int((xc + w/2) * img_width)
                    y2 = int((yc + h/2) * img_height)
                    draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                    
                # Draw Predictions in Red
                for box in pred_boxes:
                    xc, yc, w, h = box
                    x1 = int((xc - w/2) * img_width)
                    y1 = int((yc - h/2) * img_height)
                    x2 = int((xc + w/2) * img_width)
                    y2 = int((yc + h/2) * img_height)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                
                # Draw panel background (semi-transparent black rectangle)
                # Drawing in top-left
                draw.rectangle([10, 10, 310, 200], fill=(0, 0, 0, 160))
                
                # Write text
                text_lines = [
                    f"Image: {image_id}",
                    f"GT Count: {len(gt_boxes)} (Green)",
                    f"Pred Count: {len(pred_boxes)} (Red)",
                    f"Precision: {prec:.2f}",
                    f"Recall: {rec:.2f}",
                    f"F1: {f1:.2f}",
                    f"Agreement Score: {agreement:.2f}/10",
                    f"Overall Quality: {score:.2f}/10"
                ]
                
                y_offset = 15
                for line in text_lines:
                    draw.text((15, y_offset), line, fill="white")
                    y_offset += 22
                    
                # Save Overlay image
                img.save(os.path.join(overlay_dir, f"overlay_{index+1:03d}.jpg"), quality=90)
        logger.info(f"Generated 50 validation agreement overlays in: {overlay_dir}")
    else:
        logger.warning("PIL or PyTorch missing. Skipping overlays drawing.")
    
    db.close()

    # 9. Query final quality band counts from database
    db = DatabaseManager(db_path)
    cursor = db.conn.cursor()
    cursor.execute("""
        SELECT status, overall_score 
        FROM images 
        WHERE dataset_id = 'DS001';
    """)
    new_records = cursor.fetchall()
    
    new_bands = {"Gold": 0, "Silver": 0, "Review": 0, "Reject": 0}
    for rec in new_records:
        status = rec['status']
        score = rec['overall_score'] or 0.0
        if status == 'rejected': new_bands["Reject"] += 1
        elif status == 'duplicate': new_bands["Reject"] += 1
        elif status == 'review': new_bands["Review"] += 1
        elif score >= 9.0: new_bands["Gold"] += 1
        elif score >= 8.0: new_bands["Silver"] += 1
        elif score >= 5.0: new_bands["Review"] += 1
        else: new_bands["Reject"] += 1

    # 10. Write required deliverable CSVs
    logger.info("Step 10: Writing deliverables CSVs...")
    
    # A. T44_rule_breakdown.csv
    rule_csv_path = os.path.join(reports_dir, "T44_rule_breakdown.csv")
    with open(rule_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Rule", "Images Affected", "Percentage"])
        for rule, count in rule_contributions.items():
            pct = (count / len(images)) * 100.0 if len(images) > 0 else 0.0
            writer.writerow([rule, count, f"{pct:.2f}%"])
            
    # B. T44_agreement_statistics.csv
    stats_csv_path = os.path.join(reports_dir, "T44_agreement_statistics.csv")
    avg_iou = sum_iou / valid_stats_count if valid_stats_count > 0 else 0.0
    avg_prec = sum_precision / valid_stats_count if valid_stats_count > 0 else 0.0
    avg_rec = sum_recall / valid_stats_count if valid_stats_count > 0 else 0.0
    avg_f1_val = sum_f1 / valid_stats_count if valid_stats_count > 0 else 0.0
    avg_diff = sum_pred_diff / len(images) if len(images) > 0 else 0.0
    
    with open(stats_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Average IoU", f"{avg_iou:.4f}"])
        writer.writerow(["Average Precision", f"{avg_prec:.4f}"])
        writer.writerow(["Average Recall", f"{avg_rec:.4f}"])
        writer.writerow(["Average F1 Score", f"{avg_f1_val:.4f}"])
        writer.writerow(["Average Prediction Count Difference", f"{avg_diff:.2f}"])
        
    # C. T44_quality_band_comparison.csv
    band_csv_path = os.path.join(reports_dir, "T44_quality_band_comparison.csv")
    with open(band_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Quality Band", "Simulated Count", "Real Count", "Difference"])
        for band in ["Gold", "Silver", "Review", "Reject"]:
            diff = new_bands[band] - old_bands[band]
            writer.writerow([band, old_bands[band], new_bands[band], f"{diff:+}"])
            
    # D. T44_review_combinations.csv
    comb_csv_path = os.path.join(reports_dir, "T44_review_combinations.csv")
    with open(comb_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Trigger Combination", "Image Count", "Percentage of Review"])
        total_review = sum(review_combinations.values()) or 1
        for comb, count in sorted(review_combinations.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_review) * 100.0
            writer.writerow([comb, count, f"{pct:.2f}%"])

    # E. T44_validation_config.json
    config_json_path = os.path.join(reports_dir, "T44_validation_config.json")
    validation_config = {
        "model": "qyro_acne_v1_best.pt",
        "conf": 0.25,
        "iou": 0.60,
        "dataset": "DS001",
        "dataset_fingerprint": "e2f4fc3a461623930e6435d11a0394dcd97e50a9376f36c9ed4560dce3324816",
        "pipeline_version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "ultralytics_version": "8.4.33",
        "cuda": str(torch.cuda.is_available()) if TORCH_AVAILABLE else "False",
        "gpu": torch.cuda.get_device_name(0) if (TORCH_AVAILABLE and torch.cuda.is_available()) else "None"
    }
    with open(config_json_path, "w", encoding="utf-8") as jf:
        json.dump(validation_config, jf, indent=2)

    # 11. Generate Markdown Report T44_real_yolo_validation.md
    logger.info("Step 11: Generating Markdown Report...")
    total_runtime = time.time() - start_time
    
    # Render score histogram
    histogram_text = ""
    max_bin_count = max(score_bins.values()) if score_bins.values() else 1
    for key, val in score_bins.items():
        bar = "█" * int((val / max_bin_count) * 40)
        histogram_text += f"{key:<11} : {bar} ({val})\n"

    # Rule breakdown table markdown
    rule_table_lines = ["| Rule | Images Affected | Percentage of Dataset |", "| :--- | :--- | :--- |"]
    for rule, count in rule_contributions.items():
        pct = (count / len(images)) * 100.0
        rule_table_lines.append(f"| **{rule}** | {count} | {pct:.2f}% |")

    # Review combination table markdown
    comb_table_lines = ["| Trigger Combination | Image Count | Percentage of Review Queue |", "| :--- | :--- | :--- |"]
    for comb, count in sorted(review_combinations.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_review) * 100.0
        comb_table_lines.append(f"| **{comb}** | {count} | {pct:.2f}% |")

    report_path = os.path.join(reports_dir, "T44_real_yolo_validation.md")
    report_md = f"""# Phase T4.4: Real YOLO Agreement Validation & Calibration Report

This report presents the validation results of replacing simulated YOLO agreement with inferences from the frozen production detector `qyro_acne_v1_best.pt` (Conf: 0.25, IoU: 0.60).

---

## 📊 Old vs New Quality Band Comparison

| Quality Band | Simulated Count | Real Count | Difference |
| :--- | :--- | :--- | :--- |
| **Gold** | {old_bands['Gold']} | {new_bands['Gold']} | {new_bands['Gold'] - old_bands['Gold']:+d} |
| **Silver** | {old_bands['Silver']} | {new_bands['Silver']} | {new_bands['Silver'] - old_bands['Silver']:+d} |
| **Review** | {old_bands['Review']} | {new_bands['Review']} | {new_bands['Review'] - old_bands['Review']:+d} |
| **Reject** | {old_bands['Reject']} | {new_bands['Reject']} | {new_bands['Reject'] - old_bands['Reject']:+d} |

*Observed yields: Real weights evaluated {new_bands['Gold']} images as Gold and {new_bands['Silver']} images as Silver, highlighting the calibration of the production quality engine.*

---

## 🎯 Agreement Statistics

- **Average IoU (Matched boxes)**: `{avg_iou:.4f}`
- **Average Precision**: `{avg_prec:.4f}`
- **Average Recall**: `{avg_rec:.4f}`
- **Average F1 Score**: `{avg_f1_val:.4f}`
- **Average Prediction Difference**: `{avg_diff:.2f}` lesions/image

---

## 🧹 Rule Contribution Breakdown
An image can trigger multiple rules. This table shows the percentage contribution of every scoring rule:

{"\n".join(rule_table_lines)}

---

## 🔍 Review Queue Decomposition
Review queue breakdown by root-cause trigger combinations:

{"\n".join(comb_table_lines)}

---

## 📈 Score Distribution Histogram

```text
{histogram_text}
```

---

## ⚡ Performance Metrics & Run Execution

- **Total Orchestration Runtime**: `{total_runtime:.2f} seconds`
- **Total YOLO Inference Time**: `{yolo_time:.2f} seconds` (Inference speed: `{yolo_time/len(images)*1000:.2f} ms/image`)
- **Scoring Engine Update Time**: `{scoring_time:.2f} seconds`
- **Quality Band Export time**: `{export_time:.2f} seconds`
- **SQLite Database Update time**: `{db_update_time:.2f} seconds`
- **Target GPU**: `{validation_config['gpu']}`
- **CUDA Device Available**: `{validation_config['cuda']}`
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info(f"Markdown validation report written to: {report_path}")
    print("=== PHASE T4.4 COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
