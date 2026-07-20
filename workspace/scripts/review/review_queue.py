import os
import sys
import argparse

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.utils.common import setup_logger, load_config, create_markdown_report
from scripts.utils.db_manager import DatabaseManager

def parse_args():
    parser = argparse.ArgumentParser(description="Acne Dataset Pipeline - Stage 9: Human Review Queue Manager")
    parser.add_argument("--list", action="store_true", help="List all images in the review queue")
    parser.add_argument("--update_id", type=str, default="", help="Image ID to resolve status for")
    parser.add_argument("--set_status", type=str, choices=["accepted", "rejected", "ignored"], help="Status to assign to target image")
    parser.add_argument("--reason", type=str, default="", help="Reason for status update decision")
    parser.add_argument("--config", type=str, default="workspace/configs/default_dataset_policy.yaml", help="Path to policy config file")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("review_queue")
    logger.info("Initializing Human Review Queue Manager")

    try:
        config = load_config(args.config)
        db = DatabaseManager(config['paths']['database_path'])
    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        sys.exit(1)

    conn = db.conn
    cursor = conn.cursor()

    # If updating an image ID status
    if args.update_id:
        if not args.set_status:
            logger.error("--set_status (accepted/rejected/ignored) is required when updating an image.")
            sys.exit(1)
            
        logger.info(f"Updating image {args.update_id} status to '{args.set_status}'. Reason: '{args.reason}'")
        success = db.update_image_status(args.update_id, args.set_status, args.reason or "Resolved by human review")
        
        if success:
            logger.info(f"Successfully updated image {args.update_id} status.")
        else:
            logger.error(f"Failed to update status for image {args.update_id}.")
            sys.exit(1)
            
    # List all pending reviews
    cursor.execute("""
        SELECT image_id, dataset_id, file_path, overall_score, blur_score, yolo_agreement_score, rejection_reason 
        FROM images 
        WHERE status = 'review';
    """)
    review_images = cursor.fetchall()

    logger.info(f"Current Human Review Queue: {len(review_images)} images pending review.")

    if args.list:
        print("\n=== HUMAN REVIEW QUEUE ===")
        print(f"{'Image ID':<15} | {'Dataset':<10} | {'Score':<6} | {'Blur':<6} | {'YOLO Aggr':<10} | {'Trigger Reason'}")
        print("-" * 95)
        for img in review_images:
            yolo_aggr = img['yolo_agreement_score'] if img['yolo_agreement_score'] is not None else 10.0
            print(f"{img['image_id']:<15} | {img['dataset_id']:<10} | {img['overall_score']:<6.2f} | {img['blur_score']:<6.2f} | {yolo_aggr:<10.2f} | {img['rejection_reason']}")
        print("==========================\n")

    # Generate Review Queue Report
    report_file = os.path.join(config['paths']['reports_dir'], "review_queue_report.md")
    report_title = "Human Review Queue State"
    summary = f"Currently, there are **{len(review_images)}** images requiring manual validation."
    
    table_lines = [
        "| Image ID | Dataset | Overall Score | Sharpness Score | Flagged Trigger Reason |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    for img in review_images:
        table_lines.append(f"| `{img['image_id']}` | {img['dataset_id']} | {img['overall_score']:.2f} | {img['blur_score']:.2f} | {img['rejection_reason']} |")
        
    sections = {
        "Pending Images Table": "\n".join(table_lines) if review_images else "Review Queue is currently empty."
    }
    
    create_markdown_report(report_file, report_title, summary, sections)
    logger.info(f"Review queue status report written to {report_file}")

    db.close()

if __name__ == "__main__":
    main()
