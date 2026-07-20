import os
import sys
import argparse
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.utils.common import setup_logger, load_config, create_markdown_report
from scripts.utils.db_manager import DatabaseManager

def parse_args():
    parser = argparse.ArgumentParser(description="Acne Dataset Pipeline - Stage 6: Multi-Metric Quality Scoring")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID to score (e.g. DS001)")
    parser.add_argument("--config", type=str, default="workspace/configs/default_dataset_policy.yaml", help="Path to policy config file")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("scoring_engine")
    logger.info(f"Starting Multi-Metric Scoring Engine for dataset {args.dataset_id}")

    try:
        config = load_config(args.config)
        db = DatabaseManager(config['paths']['database_path'])
    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        sys.exit(1)

    weights = config['quality_metrics']['scoring_weights']
    min_accept_score = config['quality_metrics']['acceptance_overall_score']

    conn = db.conn
    cursor = conn.cursor()
    
    # 1. Fetch images that have passed audit (not rejected)
    cursor.execute("""
        SELECT image_id, file_path, status, width, height, blur_score, exposure_score, yolo_agreement_score,
               skin_tone_category, profile_view, lighting_condition
        FROM images 
        WHERE dataset_id = ? AND status != 'rejected';
    """, (args.dataset_id,))
    images = cursor.fetchall()

    logger.info(f"Scoring {len(images)} images in the pipeline.")

    processed_count = 0
    accepted_count = 0
    review_count = 0
    rejected_count = 0

    # Explicitly start transaction
    cursor.execute("BEGIN TRANSACTION;")
    now_str = datetime.now().isoformat()

    for img in images:
        image_id = img['image_id']
        current_status = img['status']
        width = img['width'] or 640
        height = img['height'] or 640
        raw_blur = img['blur_score'] or 100.0
        raw_exposure = img['exposure_score'] or 50.0
        yolo_agreement = img['yolo_agreement_score']
        
        # Default yolo_agreement if not run yet
        if yolo_agreement is None:
            yolo_agreement = 10.0
            
        # A. Calculate Image Quality Score (Scale 0 to 10)
        # Normalize Blur (Laplacian var of 100+ is excellent)
        norm_blur = min(10.0, max(1.0, raw_blur / 20.0))
        # Normalize Exposure (Brightness close to 50% is optimal)
        percentage_exposure = raw_exposure / 2.55 if raw_exposure > 10.0 else raw_exposure
        norm_exposure = max(1.0, 10.0 - abs(percentage_exposure - 50.0) / 5.0)
        # Resolution Factor
        norm_res = 10.0 if (width >= 640 and height >= 640) else 5.0
        
        image_quality_score = round((norm_blur * 0.4) + (norm_exposure * 0.4) + (norm_res * 0.2), 2)

        # B. Calculate Annotation Quality Score (Scale 0 to 10)
        cursor.execute("SELECT is_valid, data FROM annotations WHERE image_id = ? AND is_original = 0;", (image_id,))
        anns = cursor.fetchall()
        
        ann_quality = 10.0
        invalid_count = 0
        total_boxes = len(anns)
        
        # Deductions
        for ann in anns:
            if ann['is_valid'] == 0:
                invalid_count += 1
                
        # Deduct for invalid boxes
        if total_boxes > 0:
            ann_quality -= (invalid_count / total_boxes) * 5.0
        else:
            ann_quality -= 3.0 # Empty box penalty
            
        # Density penalty
        if total_boxes > 40:
            ann_quality -= 2.0
            
        # Check rejection_reason from previous audit to apply penalties for overlaps/clusters
        cursor.execute("SELECT rejection_reason FROM images WHERE image_id = ?;", (image_id,))
        reason_row = cursor.fetchone()
        reasons = reason_row[0] if reason_row and reason_row[0] else ""
        
        if "overlapping" in reasons:
            ann_quality -= 2.0
        if "cluster" in reasons:
            ann_quality -= 1.5
            
        annotation_quality_score = round(max(1.0, min(10.0, ann_quality)), 2)

        # C. Calculate Overall Score
        overall = (image_quality_score * weights['image_quality']) + \
                  (annotation_quality_score * weights['annotation_quality']) + \
                  (yolo_agreement * weights['yolo_agreement'])
                  
        overall_score = round(max(1.0, min(10.0, overall)), 2)

        # D. Map Severity category
        if total_boxes < 5:
            severity = "mild"
        elif total_boxes <= 20:
            severity = "moderate"
        else:
            severity = "severe"
            
        # Update image diversity metadata directly
        cursor.execute("""
            UPDATE images
            SET severity_category = ?, updated_at = ?
            WHERE image_id = ?;
        """, (severity, now_str, image_id))

        # E. Save Scores in SQLite DB directly
        cursor.execute("""
            UPDATE images
            SET blur_score = ?,
                exposure_score = ?,
                duplicate_risk = 10.0,
                annotation_quality_score = ?,
                lesion_visibility_score = ?,
                cluster_quality_score = 10.0,
                yolo_agreement_score = ?,
                overall_score = ?,
                updated_at = ?
            WHERE image_id = ?;
        """, (norm_blur, norm_exposure, annotation_quality_score, image_quality_score, yolo_agreement, overall_score, now_str, image_id))

        # F. Update Status
        new_status = current_status
        rejection_reason = None
        
        if overall_score < 6.0:
            new_status = "rejected"
            rejection_reason = f"Poor overall composite score: {overall_score}/10"
            rejected_count += 1
        elif current_status == "rejected" or current_status == "duplicate":
            # Keep rejected/duplicate status intact
            pass
        else:
            # Query the warning reason from the DB
            cursor.execute("SELECT status, rejection_reason FROM images WHERE image_id = ?;", (image_id,))
            current_row = cursor.fetchone()
            db_reason = current_row['rejection_reason'] if current_row and current_row['rejection_reason'] else ""
            
            # Check for active warning flags
            has_warning_flag = any(word in db_reason.lower() for word in ["flag", "discrepancy", "blurry", "failed", "leakage"])
            
            if has_warning_flag:
                new_status = "review"
                rejection_reason = db_reason  # Retain original warning trigger reason
                review_count += 1
            elif overall_score >= min_accept_score:
                new_status = "accepted"
                accepted_count += 1
            else:
                new_status = "review"
                rejection_reason = f"Overall score ({overall_score}) below auto-accept threshold ({min_accept_score})"
                review_count += 1
            
        cursor.execute("""
            UPDATE images
            SET status = ?, rejection_reason = ?, updated_at = ?
            WHERE image_id = ?;
        """, (new_status, rejection_reason, now_str, image_id))
        processed_count += 1

    conn.commit()
    logger.info(f"Scoring finalized. Total: {processed_count}, Accepted: {accepted_count}, Review Queue: {review_count}, Rejected: {rejected_count}")

    # Generate Scoring Report
    report_file = os.path.join(config['paths']['reports_dir'], f"scoring_{args.dataset_id}_report.md")
    sections = {
        "Scoring Engine Outputs": (
            f"- **Images Scored**: {processed_count}\n"
            f"- **Auto-Accepted (Score $\\\\ge {min_accept_score}$)**: {accepted_count}\n"
            f"- **Routed to Review Queue**: {review_count}\n"
            f"- **Low Quality Rejections**: {rejected_count}\n"
        ),
        "Weights Configuration": (
            f"- **Visual Image Quality (Weight)**: {weights['image_quality']}\n"
            f"- **Annotation Quality (Weight)**: {weights['annotation_quality']}\n"
            f"- **YOLO Agreement Consensus (Weight)**: {weights['yolo_agreement']}\n"
        )
    }
    create_markdown_report(report_file, f"Multi-Metric Scoring: {args.dataset_id}", "Aggregated assessment metrics calculation run.", sections)

    db.close()

if __name__ == "__main__":
    main()
