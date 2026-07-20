import os
import sys
import argparse
import json
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.utils.common import setup_logger, load_config, create_markdown_report
from scripts.utils.db_manager import DatabaseManager

def parse_args():
    parser = argparse.ArgumentParser(description="Acne Dataset Pipeline - Stage 3: Audit Coordinates")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID to process (e.g. DS001)")
    parser.add_argument("--config", type=str, default="workspace/configs/default_dataset_policy.yaml", help="Path to policy config file")
    return parser.parse_args()

def calculate_iou(box1: list, box2: list) -> float:
    """Calculates Intersection-over-Union (IoU) of two YOLO boxes [x_center, y_center, w, h]."""
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

def main():
    args = parse_args()
    logger = setup_logger("audit_annotations")
    logger.info(f"Starting bounding box and overlap audit for dataset {args.dataset_id}")

    try:
        config = load_config(args.config)
        db = DatabaseManager(config['paths']['database_path'])
    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        sys.exit(1)

    min_box_area = config['quality_metrics']['annotations']['min_box_area_ratio']
    max_box_area = config['quality_metrics']['annotations']['max_box_area_ratio']
    overlap_threshold = config['quality_metrics']['annotations']['overlap_iou_threshold']

    conn = db.conn
    cursor = conn.cursor()
    
    # 1. Fetch images
    cursor.execute("SELECT image_id, file_path, status FROM images WHERE dataset_id = ? AND status != 'rejected';", (args.dataset_id,))
    images = cursor.fetchall()
    
    total_images_checked = len(images)
    total_passed = 0
    total_rejected = 0
    total_review = 0
    failure_details = []

    for img in images:
        image_id = img['image_id']
        current_status = img['status']
        
        # 2. Fetch standardized annotations for this image
        cursor.execute("""
            SELECT annotation_id, data, class_label 
            FROM annotations 
            WHERE image_id = ? AND is_original = 0;
        """, (image_id,))
        anns = cursor.fetchall()
        
        image_passed_audit = True
        image_flagged_for_review = False
        reasons = []
        
        # Box lists for pairwise IoU check
        boxes_list = []
        ann_ids = []
        
        # A. Basic sanity checks on individual boxes
        for ann in anns:
            ann_id = ann['annotation_id']
            ann_ids.append(ann_id)
            
            try:
                data = json.loads(ann['data'])
            except Exception:
                # Corrupt json is a hard reject
                image_passed_audit = False
                reasons.append(f"Corrupted json annotation data in {ann_id}.")
                continue
                
            if "bbox" not in data:
                # Empty annotation label check
                image_passed_audit = False
                reasons.append(f"Missing bounding box data in annotation {ann_id}.")
                continue
                
            bbox = data['bbox']
            if len(bbox) != 4:
                image_passed_audit = False
                reasons.append(f"Invalid bounding box coordinate length in {ann_id}.")
                continue
                
            xc, yc, w, h = bbox
            boxes_list.append(bbox)
            
            # Check coordinate limits
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0):
                image_passed_audit = False
                reasons.append(f"Box center ({xc:.2f}, {yc:.2f}) out of bounds in {ann_id}.")
            if w <= 0.0 or h <= 0.0:
                image_passed_audit = False
                reasons.append(f"Box dimensions ({w:.2f}, {h:.2f}) must be positive in {ann_id}.")
            if xc - w/2 < 0.0 or xc + w/2 > 1.0 or yc - h/2 < 0.0 or yc + h/2 > 1.0:
                image_passed_audit = False
                reasons.append(f"Box boundaries exceed frame edge in {ann_id}.")
                
            # Aspect ratio check (aspect ratio > 5.0)
            if w > 0 and h > 0:
                aspect1 = w / h
                aspect2 = h / w
                if aspect1 > 5.0 or aspect2 > 5.0:
                    image_flagged_for_review = True
                    reasons.append(f"Elongated box detected in {ann_id} (aspect: {max(aspect1, aspect2):.2f}).")
                    
            # Area checks
            box_area = w * h
            if box_area > max_box_area:
                # Potential giant cluster box
                image_flagged_for_review = True
                reasons.append(f"Giant box detected in {ann_id} (covers {box_area*100:.1f}% of image) - potential cluster box.")
            elif box_area < min_box_area:
                image_passed_audit = False
                reasons.append(f"Box area ({box_area:.6f}) below threshold ({min_box_area:.6f}) in {ann_id}.")

        # B. Pairwise IoU checks for overlap detection
        n_boxes = len(boxes_list)
        for i in range(n_boxes):
            for j in range(i + 1, n_boxes):
                iou = calculate_iou(boxes_list[i], boxes_list[j])
                if iou > overlap_threshold:
                    image_flagged_for_review = True
                    reasons.append(f"Heavily overlapping boxes between `{ann_ids[i]}` and `{ann_ids[j]}` (IoU: {iou:.2f}).")

        # C. Empty annotations warning
        if n_boxes == 0:
            image_flagged_for_review = True
            reasons.append("Image contains zero annotation labels.")

        # D. Update database status based on audits
        reason_str = "; ".join(reasons)
        if not image_passed_audit:
            # Hard rejection
            total_rejected += 1
            db.update_image_status(image_id, "rejected", f"Failed annotation audit: {reason_str}")
            failure_details.append(f"- Image `{image_id}` [REJECTED]: {reason_str}")
        elif image_flagged_for_review:
            # Flag for review
            total_review += 1
            db.update_image_status(image_id, "review", f"Annotation audit flag: {reason_str}")
            failure_details.append(f"- Image `{image_id}` [REVIEW QUEUE]: {reason_str}")
        else:
            # Audit passes
            total_passed += 1
            db.update_image_status(image_id, "review", "Passed annotation audit - Awaiting image metrics") # Keeps in review queue for next checks

    conn.commit()
    logger.info(f"Audit completed. Passed: {total_passed}, Rejected: {total_rejected}, Flagged for Review: {total_review}.")

    # Generate Audit Report
    report_file = os.path.join(config['paths']['reports_dir'], f"audit_{args.dataset_id}_report.md")
    sections = {
        "Audit Verification Breakdown": (
            f"- **Total Images Checked**: {total_images_checked}\n"
            f"- **Clean (Passed Annotations)**: {total_passed}\n"
            f"- **Logical Rejections**: {total_rejected}\n"
            f"- **Flagged for Review (Overlaps/Elongated/Giant)**: {total_review}\n"
        ),
        "Failed/Flagged Annotations Audit Log": "\n".join(failure_details) if failure_details else "No annotation anomalies identified."
    }
    create_markdown_report(report_file, f"Annotation Sanity Audit: {args.dataset_id}", "Sanity checks on box dimensions and overlaps.", sections)

    db.close()

if __name__ == "__main__":
    main()
