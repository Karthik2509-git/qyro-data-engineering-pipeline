import os
import sys
import argparse
import json
import sqlite3
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.utils.common import setup_logger, load_config, create_markdown_report
from scripts.utils.db_manager import DatabaseManager

# Try importing ultralytics for real YOLOv8 inference
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Acne Dataset Pipeline - Stage 5: YOLO Agreement Audit")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID to check (e.g. DS001)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained YOLOv8 model weights (.pt)")
    parser.add_argument("--config", type=str, default="workspace/configs/default_dataset_policy.yaml", help="Path to policy config file")
    return parser.parse_args()

def calculate_box_iou(box1: list, box2: list) -> float:
    """Calculates Intersection-over-Union (IoU) of two YOLO boxes [xc, yc, w, h]."""
    xc1, yc1, w1, h1 = box1
    xc2, yc2, w2, h2 = box2
    
    # Calculate corners
    x1, y1 = xc1 - w1 / 2, yc1 - h1 / 2
    x2, y2 = xc1 + w1 / 2, yc1 + h1 / 2
    
    x3, y3 = xc2 - w2 / 2, yc2 - h2 / 2
    x4, y4 = xc2 + w2 / 2, yc2 + h2 / 2
    
    # Intersection rectangle coordinates
    xi_min = max(x1, x3)
    yi_min = max(y1, y3)
    xi_max = min(x2, x4)
    yi_max = min(y2, y4)
    
    inter_w = max(0.0, xi_max - xi_min)
    inter_h = max(0.0, yi_max - yi_min)
    inter_area = inter_w * inter_h
    
    # Union area
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0.0:
        return 0.0
        
    return inter_area / union_area

def run_prediction(image_path: str, model_instance, logger) -> list:
    """Runs YOLO prediction on image path. Returns list of {"bbox": [xc, yc, w, h], "conf": confidence}."""
    if ULTRALYTICS_AVAILABLE and model_instance is not None:
        try:
            # Set parameters exactly as required: conf=0.25, iou=0.60
            results = model_instance(image_path, conf=0.25, iou=0.60, verbose=False)
            predictions = []
            if len(results) > 0:
                boxes = results[0].boxes
                xywhn = boxes.xywhn.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for box, conf in zip(xywhn, confs):
                    predictions.append({
                        "bbox": box.tolist(),
                        "conf": float(conf)
                    })
            return predictions
        except Exception as e:
            logger.warning(f"Failed running real YOLO prediction on {image_path}: {e}.")
    return []

def migrate_database(db_path, logger):
    """Ensure database has precision, recall, f1, missing_ratio, extra_ratio, and confidence stats columns."""
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
    args = parse_args()
    logger = setup_logger("yolo_agreement")
    logger.info(f"Starting YOLO Agreement Audit for dataset {args.dataset_id}")

    if not ULTRALYTICS_AVAILABLE:
        logger.error("ultralytics package is missing. Cannot perform real YOLOv8 validation.")
        sys.exit(1)

    try:
        config = load_config(args.config)
        migrate_database(config['paths']['database_path'], logger)
        db = DatabaseManager(config['paths']['database_path'])
    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        sys.exit(1)

    # 1. Load YOLO model
    if not os.path.exists(args.model_path):
        logger.error(f"Production weights not found at absolute path: {args.model_path}")
        sys.exit(1)
        
    try:
        model = YOLO(args.model_path)
        logger.info(f"Successfully loaded YOLOv8 model weights from: {args.model_path}")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {e}")
        sys.exit(1)

    conn = db.conn
    cursor = conn.cursor()
    
    # Fetch active images
    cursor.execute("""
        SELECT image_id, file_path, status 
        FROM images 
        WHERE dataset_id = ? AND status != 'rejected';
    """, (args.dataset_id,))
    images = cursor.fetchall()

    logger.info(f"Comparing annotations vs predictions for {len(images)} images.")

    # Explicitly start transaction
    cursor.execute("BEGIN TRANSACTION;")
    now_str = datetime.now().isoformat()
    discrepancy_count = 0
    passed_count = 0
    failure_details = []

    for img in images:
        image_id = img['image_id']
        file_path = img['file_path']
        current_status = img['status']
        
        # 2. Fetch ground truth boxes from annotations table (is_original = 0, is_valid = 1)
        cursor.execute("""
            SELECT data FROM annotations 
            WHERE image_id = ? AND is_original = 0 AND is_valid = 1;
        """, (image_id,))
        ann_rows = cursor.fetchall()
        
        gt_boxes = []
        for row in ann_rows:
            ann_data = json.loads(row['data'])
            if "bbox" in ann_data:
                gt_boxes.append(ann_data['bbox'])
                
        gt_count = len(gt_boxes)
        
        # Check if already computed in previous run to save runtime
        cursor.execute("SELECT mean_confidence, yolo_agreement_score, yolo_box_count FROM images WHERE image_id = ?;", (image_id,))
        existing = cursor.fetchone()
        if existing and existing['mean_confidence'] is not None:
            pred_count = existing['yolo_box_count'] or 0
            agreement_score = existing['yolo_agreement_score'] or 10.0
            count_diff = abs(gt_count - pred_count)
            if count_diff > 5 or agreement_score < 5.0:
                discrepancy_count += 1
                failure_details.append(f"- Image `{image_id}` [DISAGREEMENT]: labels {gt_count} vs prediction {pred_count}. Score: {agreement_score}")
            else:
                passed_count += 1
            continue
            
        # 3. Get real predictions
        pred_boxes = run_prediction(file_path, model, logger)
        pred_count = len(pred_boxes)
        gt_count = len(gt_boxes)
        
        pred_confs = [p["conf"] for p in pred_boxes]
        
        # 4. Compare Predictions vs Ground Truth
        matched_ious = []
        matched_indices_pred = set()
        
        for gt_box in gt_boxes:
            best_iou = 0.0
            best_idx = -1
            
            for idx, pred_box in enumerate(pred_boxes):
                if idx in matched_indices_pred:
                    continue
                iou = calculate_box_iou(gt_box, pred_box["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
                    
            if best_iou >= 0.40: # Match threshold
                matched_ious.append(best_iou)
                matched_indices_pred.add(best_idx)
                
        matched_count = len(matched_indices_pred)
        unmatched_pred_count = pred_count - matched_count
        unmatched_annot_count = gt_count - matched_count
        
        # Calculate Precision, Recall, F1, Missing/Extra Ratios
        precision = matched_count / pred_count if pred_count > 0 else 0.0
        recall = matched_count / gt_count if gt_count > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
        
        missing_ratio = unmatched_pred_count / pred_count if pred_count > 0 else 0.0
        extra_ratio = unmatched_annot_count / gt_count if gt_count > 0 else 0.0
        
        mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        
        # Confidence statistics
        if pred_confs:
            mean_conf = sum(pred_confs) / len(pred_confs)
            sorted_confs = sorted(pred_confs)
            median_conf = sorted_confs[len(sorted_confs) // 2]
            min_conf = min(pred_confs)
            max_conf = max(pred_confs)
        else:
            mean_conf = 0.0
            median_conf = 0.0
            min_conf = 0.0
            max_conf = 0.0
        
        # 5. Compute YOLO Agreement Score (0 to 10 scale)
        agreement_score = 10.0
        if gt_count > 0 or pred_count > 0:
            # Deductions
            # Deduct for unmatched boxes
            box_penalty = (unmatched_annot_count + unmatched_pred_count) * 1.2
            # Deduct for low mean IoU
            iou_penalty = (1.0 - mean_iou) * 4.0 if matched_ious else 4.0
            
            agreement_score = max(1.0, min(10.0, 10.0 - box_penalty - iou_penalty))
            
        agreement_score = round(agreement_score, 2)
        
        # Update database agreement attributes directly
        cursor.execute("""
            UPDATE images
            SET yolo_box_count = ?,
                mean_iou = ?,
                missing_lesions = ?,
                extra_annotations = ?,
                yolo_agreement_score = ?,
                precision = ?,
                recall = ?,
                f1 = ?,
                missing_ratio = ?,
                extra_ratio = ?,
                mean_confidence = ?,
                median_confidence = ?,
                min_confidence = ?,
                max_confidence = ?,
                updated_at = ?
            WHERE image_id = ?;
        """, (pred_count, round(mean_iou, 2), unmatched_annot_count, unmatched_pred_count, agreement_score,
              round(precision, 2), round(recall, 2), round(f1, 2), round(missing_ratio, 2), round(extra_ratio, 2),
              round(mean_conf, 2), round(median_conf, 2), round(min_conf, 2), round(max_conf, 2),
              now_str, image_id))
        
        # 6. Flag for Review if severe disagreement (based on existing thresholds)
        count_diff = abs(gt_count - pred_count)
        if count_diff > 5 or agreement_score < 5.0:
            discrepancy_count += 1
            # Route to review status
            cursor.execute("""
                UPDATE images
                SET status = 'review',
                    rejection_reason = ?,
                    updated_at = ?
                WHERE image_id = ?;
            """, (f"YOLO agreement discrepancy - labels: {gt_count}, predicted: {pred_count}, score: {agreement_score}/10", now_str, image_id))
            failure_details.append(f"- Image `{image_id}` [DISAGREEMENT]: labels {gt_count} vs prediction {pred_count}. IoU: {mean_iou:.2f}. Score: {agreement_score}")
        else:
            passed_count += 1

    conn.commit()
    logger.info(f"YOLO agreement audit complete. Passed: {passed_count}, Flagged discrepancies: {discrepancy_count}.")

    # Generate Report
    report_file = os.path.join(config['paths']['reports_dir'], f"yolo_agreement_{args.dataset_id}_report.md")
    sections = {
        "Audit Agreement Statistics": (
            f"- **Images Analyzed**: {len(images)}\n"
            f"- **In Agreement (Pass)**: {passed_count}\n"
            f"- **In Disagreement (Flagged to Review Queue)**: {discrepancy_count}\n"
            f"- **Model Inference Mode**: Real weights (.pt)\n"
        ),
        "Discrepancies Audit Log": "\n".join(failure_details) if failure_details else "No major annotations vs model disagreements."
    }
    create_markdown_report(report_file, f"YOLO Agreement Audit: {args.dataset_id}", "Comparison of annotations vs model predictions.", sections)

    db.close()

if __name__ == "__main__":
    main()
