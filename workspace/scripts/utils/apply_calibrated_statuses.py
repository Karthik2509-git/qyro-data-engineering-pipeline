import os
import sys
import json
import sqlite3
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.utils.db_manager import DatabaseManager
from scripts.utils.common import setup_logger, load_config

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

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

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Acne Dataset Platform - Calibration Database Status Updater")
    parser.add_argument("--dataset_id", type=str, default="DS001", help="Dataset ID to update")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("apply_calibrated_statuses")
    logger.info(f"=== STARTING SQLITE DATABASE CALIBRATION UPDATE FOR {args.dataset_id} ===")
    
    db_path = "workspace/database/dataset_index.sqlite"
    config_path = "workspace/configs/default_dataset_policy.yaml"
    
    db = DatabaseManager(db_path)
    conn = db.conn
    cursor = conn.cursor()
    
    # 1. Fetch images of dataset
    cursor.execute("""
        SELECT image_id, file_path, status, width, height, blur_score, exposure_score,
               yolo_agreement_score, f1, yolo_box_count, rejection_reason
        FROM images
        WHERE dataset_id = ?;
    """, (args.dataset_id,))
    images = cursor.fetchall()
    
    # 2. Fetch annotations
    cursor.execute("""
        SELECT image_id, class_label, data, is_original, is_valid
        FROM annotations
        WHERE image_id IN (SELECT image_id FROM images WHERE dataset_id = ?);
    """, (args.dataset_id,))
    annotations = cursor.fetchall()
    
    # Index annotations by image_id
    anns_by_img = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in anns_by_img:
            anns_by_img[img_id] = []
        anns_by_img[img_id].append(ann)
        
    logger.info(f"Loaded {len(images)} images and {len(annotations)} annotations.")
    
    # Pre-scan sharp/exposure if they are currently overwritten with norm values
    logger.info("Pre-calculating raw blur and exposure scores from image files...")
    blur_records = {}
    exposure_records = {}
    
    for idx, img in enumerate(images):
        image_id = img['image_id']
        file_path = img['file_path']
        
        if CV2_AVAILABLE and os.path.exists(file_path):
            try:
                cv_img = cv2.imread(file_path)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                raw_blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                raw_exposure = float(np.mean(gray))
            except Exception:
                raw_blur = 50.0
                raw_exposure = 127.0
        else:
            raw_blur = 50.0
            raw_exposure = 127.0
            
        blur_records[image_id] = raw_blur
        exposure_records[image_id] = raw_exposure
        
        if (idx + 1) % 2000 == 0:
            logger.info(f"Scanned {idx + 1} / {len(images)} images.")

    # 3. Update database status and scores
    cursor.execute("BEGIN TRANSACTION;")
    now_str = datetime.now().isoformat()
    
    processed_count = 0
    stats = {"Gold": 0, "Silver": 0, "Review": 0, "Reject": 0}
    
    for img in images:
        image_id = img['image_id']
        file_path = img['file_path']
        status = img['status']
        width = img['width'] or 640
        height = img['height'] or 640
        
        # Keep actual duplicates and corruption rejects as Reject
        if status == 'duplicate' or (status == 'rejected' and img['rejection_reason'] and "corruption" in img['rejection_reason'].lower()):
            stats["Reject"] += 1
            processed_count += 1
            continue
            
        blur = blur_records[image_id]
        exposure = exposure_records[image_id]
        f1_score = img['f1'] or 0.0
        yolo_agreement = img['yolo_agreement_score'] or 10.0
        
        # Calculate calibrated normalized scores
        sim_norm_blur = min(10.0, max(1.0, blur / 10.0))
        percentage_exposure = exposure / 2.55
        sim_norm_exposure = max(1.0, 10.0 - abs(percentage_exposure - 50.0) / 5.0)
        sim_image_quality = round((sim_norm_blur * 0.4) + (sim_norm_exposure * 0.4) + (10.0 * 0.2), 2)
        
        # Retrieve annotations
        img_anns = anns_by_img.get(image_id, [])
        std_anns = [a for a in img_anns if a['is_original'] == 0]
        orig_anns = [a for a in img_anns if a['is_original'] == 1]
        
        has_problematic_border = False
        has_tiny = False
        has_overlap = False
        has_clinical = False
        
        warning_reasons = []
        
        # Overlap check
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
                    warning_reasons.append("Overlapping boxes detected")
                    break
            if has_overlap:
                break
                
        # Border and tiny check
        for ann in std_anns:
            try:
                ann_data = json.loads(ann['data'])
                if "bbox" in ann_data:
                    xc, yc, w, h = ann_data['bbox']
                    # Border: touches within 0.005 and is larger than 0.04
                    left_touch = (xc - w/2 <= 0.005)
                    right_touch = (xc + w/2 >= 0.995)
                    top_touch = (yc - h/2 <= 0.005)
                    bottom_touch = (yc + h/2 >= 0.995)
                    if (left_touch or right_touch or top_touch or bottom_touch) and (w > 0.04 or h > 0.04):
                        has_problematic_border = True
                    if w * h < 0.0001:
                        has_tiny = True
            except Exception:
                pass
                
        if has_problematic_border:
            warning_reasons.append("Border box truncation detected")
        if has_tiny:
            warning_reasons.append("Tiny box area detected")
            
        # Clinical check
        for ann in orig_anns:
            if ann['class_label'] in ['Milium', 'Crystanlline', 'Sebo-crystan-conglo', 'Folliculitis']:
                has_clinical = True
                warning_reasons.append(f"Clinical Review class detected: {ann['class_label']}")
                break

        sim_ann_quality = 10.0
        if has_overlap: sim_ann_quality -= 2.0
        if has_tiny: sim_ann_quality -= 1.0
        if has_problematic_border: sim_ann_quality -= 2.0
        
        sim_yolo_agreement = f1_score * 10.0
        if sim_yolo_agreement < 5.0:
            warning_reasons.append(f"YOLO model agreement discrepancy (F1 < 0.50): {f1_score:.2f}")
            
        if sim_norm_blur < 4.0:
            warning_reasons.append(f"Blur penalty detected (raw Laplacian: {blur:.1f})")
        if sim_norm_exposure < 6.0:
            warning_reasons.append(f"Exposure penalty detected (raw brightness: {exposure:.1f})")
            
        # Calculate overall score
        overall = (sim_image_quality * 0.40) + \
                  (sim_ann_quality * 0.35) + \
                  (sim_yolo_agreement * 0.25)
        overall_score = round(max(1.0, min(10.0, overall)), 2)
        
        # Determine status
        if overall_score < 5.0:
            new_status = "rejected"
            rejection_reason = f"Poor overall composite score: {overall_score}/10"
            stats["Reject"] += 1
        elif warning_reasons:
            new_status = "review"
            rejection_reason = "; ".join(warning_reasons)
            stats["Review"] += 1
        elif overall_score >= 8.0:
            new_status = "accepted"
            rejection_reason = None
            if overall_score >= 9.0:
                stats["Gold"] += 1
            else:
                stats["Silver"] += 1
        else:
            new_status = "review"
            rejection_reason = f"Overall score ({overall_score}) below auto-accept threshold (8.0)"
            stats["Review"] += 1
            
        # Update SQLite record
        cursor.execute("""
            UPDATE images
            SET blur_score = ?,
                exposure_score = ?,
                annotation_quality_score = ?,
                yolo_agreement_score = ?,
                overall_score = ?,
                status = ?,
                rejection_reason = ?,
                updated_at = ?
            WHERE image_id = ?;
        """, (sim_norm_blur, sim_norm_exposure, sim_ann_quality, sim_yolo_agreement, overall_score,
              new_status, rejection_reason, now_str, image_id))
        processed_count += 1

    conn.commit()
    logger.info("=== CALIBRATION UPDATE COMMITTED ===")
    logger.info(f"Processed: {processed_count}")
    logger.info(f"Results: {stats}")
    db.close()

if __name__ == "__main__":
    main()
