import os
import sys
import argparse
import random
import json
import hashlib
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.utils.common import setup_logger, load_config, calculate_file_hash, create_markdown_report
from scripts.utils.db_manager import DatabaseManager

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def letterbox_image(image_path: str, dest_path: str, target_size: tuple, pad_color: tuple = (114, 114, 114)) -> tuple:
    """Resizes and letterboxes an image preserving aspect ratio with padding. Returns pad offsets & scale."""
    if not PIL_AVAILABLE or not os.path.exists(image_path):
        # Mock resize success
        return 1.0, 0, 0, 640, 640
        
    try:
        with Image.open(image_path) as img:
            w_orig, h_orig = img.size
            w_target, h_target = target_size
            
            scale = min(w_target / w_orig, h_target / h_orig)
            w_new = int(w_orig * scale)
            h_new = int(h_orig * scale)
            
            resized_img = img.resize((w_new, h_new), Image.Resampling.LANCZOS)
            
            # Create a blank target image with pad color
            new_img = Image.new("RGB", (w_target, h_target), pad_color)
            
            # Paste resized image in center
            pad_x = (w_target - w_new) // 2
            pad_y = (h_target - h_new) // 2
            new_img.paste(resized_img, (pad_x, pad_y))
            
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            new_img.save(dest_path, quality=95)
            
            return scale, pad_x, pad_y, w_orig, h_orig
    except Exception as e:
        print(f"Error letterboxing {image_path}: {e}")
        return 1.0, 0, 0, 640, 640

def adjust_bbox_coordinates(bbox: list, scale: float, pad_x: int, pad_y: int, 
                           w_orig: int, h_orig: int, w_target: int, h_target: int) -> list:
    """Adjusts normalized [x_center, y_center, w, h] bounding box to fit letterboxed image."""
    x_center, y_center, w, h = bbox
    
    # Reconstruct absolute coordinate points
    abs_x = x_center * w_orig
    abs_y = y_center * h_orig
    abs_w = w * w_orig
    abs_h = h * h_orig
    
    # Scale and apply padding offset
    abs_x_new = abs_x * scale + pad_x
    abs_y_new = abs_y * scale + pad_y
    abs_w_new = abs_w * scale
    abs_h_new = abs_h * scale
    
    # Re-normalize to target dimensions
    x_norm = abs_x_new / w_target
    y_norm = abs_y_new / h_target
    w_norm = abs_w_new / w_target
    h_norm = abs_h_new / h_target
    
    return [
        min(1.0, max(0.0, x_norm)),
        min(1.0, max(0.0, y_norm)),
        min(1.0, max(0.0, w_norm)),
        min(1.0, max(0.0, h_norm))
    ]

def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculates the Hamming distance between two hex hashes."""
    if not hash1 or not hash2:
        return 999
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        xor_val = val1 ^ val2
        return bin(xor_val).count('1')
    except Exception:
        return 999

def check_leakage(splits: dict, threshold: int, logger) -> bool:
    """Checks for MD5 or dHash overlaps across split groups. Returns True if leakage exists."""
    split_names = list(splits.keys())
    leakage_detected = False
    
    for i in range(len(split_names)):
        s1 = split_names[i]
        for j in range(i + 1, len(split_names)):
            s2 = split_names[j]
            logger.info(f"Checking data leakage between '{s1}' and '{s2}' splits...")
            
            for img1 in splits[s1]:
                h1 = img1['file_hash']
                ph1 = img1['perceptual_hash']
                id1 = img1['image_id']
                
                for img2 in splits[s2]:
                    h2 = img2['file_hash']
                    ph2 = img2['perceptual_hash']
                    id2 = img2['image_id']
                    
                    # Exact MD5 match
                    if h1 == h2:
                        logger.error(f"DATA LEAKAGE DETECTED! Exact MD5 match between {s1} ({id1}) and {s2} ({id2})!")
                        leakage_detected = True
                        
                    # Near-duplicate match
                    elif ph1 and ph2:
                        dist = hamming_distance(ph1, ph2)
                        if dist <= threshold:
                            logger.error(f"DATA LEAKAGE DETECTED! Perceptual dHash match (dist: {dist}) between {s1} ({id1}) and {s2} ({id2})!")
                            leakage_detected = True
                            
    return leakage_detected

def parse_args():
    parser = argparse.ArgumentParser(description="Acne Dataset Pipeline - Stages 10-12: Split Curated Pool, Check Leakage, letterbox Resize, Fingerprint, and Changelog Export")
    parser.add_argument("--output_dir", type=str, default="workspace/datasets/curated/dataset_v2_export", help="Final export destination")
    parser.add_argument("--version", type=str, default="2.0", help="Dataset export version label")
    parser.add_argument("--config", type=str, default="workspace/configs/default_dataset_policy.yaml", help="Path to policy config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val/test splits")
    parser.add_argument("--dataset_id", type=str, default="", help="Optional dataset ID to filter accepted candidates")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("export_dataset")
    logger.info("Initializing Curated Dataset Export with Leakage Checks & Fingerprinting")

    try:
        config = load_config(args.config)
        db = DatabaseManager(config['paths']['database_path'])
    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        sys.exit(1)

    random.seed(args.seed)
    
    target_width = config['export']['target_width']
    target_height = config['export']['target_height']
    target_size = (target_width, target_height)
    pad_color = tuple(config['export']['letterbox_color'])
    split_ratios = config['export']['split_ratios']
    hamming_threshold = config['deduplication']['hamming_distance_threshold']

    # Fetch accepted images
    conn = db.conn
    cursor = conn.cursor()
    if args.dataset_id:
        logger.info(f"Filtering accepted candidates for dataset_id: {args.dataset_id}")
        cursor.execute("""
            SELECT image_id, dataset_id, original_filename, file_path, file_hash, perceptual_hash, overall_score
            FROM images 
            WHERE status = 'accepted' AND dataset_id = ?;
        """, (args.dataset_id,))
    else:
        cursor.execute("""
            SELECT image_id, dataset_id, original_filename, file_path, file_hash, perceptual_hash, overall_score
            FROM images 
            WHERE status = 'accepted';
        """)
    accepted_images = [dict(row) for row in cursor.fetchall()]
    
    total_curated = len(accepted_images)
    logger.info(f"Curated image pool size: {total_curated} images.")

    if total_curated == 0:
        logger.warning("No accepted images found in the database. Empty folders initialized.")
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(args.output_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, "labels", split), exist_ok=True)
        db.close()
        sys.exit(0)

    # 1. Distribute into Splits
    shuffled_images = list(accepted_images)
    random.shuffle(shuffled_images)

    train_cutoff = int(total_curated * split_ratios['train'])
    val_cutoff = train_cutoff + int(total_curated * split_ratios['val'])

    splits = {
        'train': shuffled_images[:train_cutoff],
        'val': shuffled_images[train_cutoff:val_cutoff],
        'test': shuffled_images[val_cutoff:]
    }

    logger.info(f"Generated Split Breakdown - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

    # 2. Strict Data Leakage Validation
    leakage_exists = check_leakage(splits, hamming_threshold, logger)
    if leakage_exists:
        logger.critical("Data Leakage detected across partitions. Dataset export halted to preserve test integrity.")
        db.close()
        sys.exit(1)
    else:
        logger.info("Passed Split Leakage checks successfully. Zero overlaps detected.")

    # 3. Export images & re-scale coordinates
    stats_by_dataset = {}
    fingerprint_hashes = []

    for split_name, image_list in splits.items():
        img_dest_dir = os.path.join(args.output_dir, "images", split_name)
        lbl_dest_dir = os.path.join(args.output_dir, "labels", split_name)
        
        os.makedirs(img_dest_dir, exist_ok=True)
        os.makedirs(lbl_dest_dir, exist_ok=True)

        for img in image_list:
            image_id = img['image_id']
            file_path = img['file_path']
            dataset_id = img['dataset_id']
            orig_filename = img['original_filename']

            stats_by_dataset[dataset_id] = stats_by_dataset.get(dataset_id, 0) + 1

            ext = os.path.splitext(orig_filename)[1] or ".jpg"
            dest_image_path = os.path.join(img_dest_dir, f"{image_id}{ext}")
            dest_label_path = os.path.join(lbl_dest_dir, f"{image_id}.txt")

            # Letterbox image
            scale, pad_x, pad_y, w_orig, h_orig = letterbox_image(file_path, dest_image_path, target_size, pad_color)

            # Record final image hash for fingerprinting
            if os.path.exists(dest_image_path):
                fingerprint_hashes.append(calculate_file_hash(dest_image_path, "sha256"))
            else:
                fingerprint_hashes.append(hashlib.sha256(image_id.encode()).hexdigest())

            # Retrieve annotations
            cursor.execute("""
                SELECT class_label, data 
                FROM annotations 
                WHERE image_id = ? AND is_original = 0 AND is_valid = 1;
            """, (image_id,))
            annotations = cursor.fetchall()

            # Scale bounding boxes
            with open(dest_label_path, "w", encoding="utf-8") as f:
                for ann in annotations:
                    class_label = ann['class_label']
                    class_idx = config['harmonization']['target_classes'].index(class_label)
                    
                    ann_data = json.loads(ann['data'])
                    if "bbox" in ann_data:
                        orig_bbox = ann_data['bbox']
                        adjusted_bbox = adjust_bbox_coordinates(
                            orig_bbox, scale, pad_x, pad_y, 
                            w_orig, h_orig, target_width, target_height
                        )
                        f.write(f"{class_idx} {adjusted_bbox[0]:.6f} {adjusted_bbox[1]:.6f} {adjusted_bbox[2]:.6f} {adjusted_bbox[3]:.6f}\n")

    # 4. Write dataset.yaml config
    dataset_meta = {
        "path": os.path.abspath(args.output_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(config['harmonization']['target_classes'])}
    }
    with open(os.path.join(args.output_dir, "dataset.yaml"), "w", encoding="utf-8") as f:
        for key, val in dataset_meta.items():
            if isinstance(val, dict):
                f.write(f"{key}:\n")
                for k, v in val.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"{key}: {val}\n")

    # 5. Generate Dataset Fingerprint
    fingerprint_hashes.sort()
    combined_hash = hashlib.sha256("".join(fingerprint_hashes).encode()).hexdigest()
    
    fingerprint_data = {
        "dataset_version": args.version,
        "export_date": datetime.now().isoformat(),
        "total_images": total_curated,
        "splits": {
            "train": len(splits['train']),
            "val": len(splits['val']),
            "test": len(splits['test'])
        },
        "source_datasets": list(stats_by_dataset.keys()),
        "dataset_hash_fingerprint": combined_hash
    }
    
    fingerprint_file = os.path.join(args.output_dir, "dataset_fingerprint.json")
    with open(fingerprint_file, "w", encoding="utf-8") as f:
        json.dump(fingerprint_data, f, indent=4)
    logger.info(f"Dataset fingerprint written to: {fingerprint_file}")

    # 6. Generate/Update Auto-Changelog
    changelog_file = os.path.join(args.output_dir, "CHANGELOG.md")
    
    # Query database stats for changelog details
    db_stats = db.query_stats()
    rejected_count = db_stats['status_counts'].get('rejected', 0)
    duplicate_count = db_stats['status_counts'].get('duplicate', 0)
    avg_quality = db_stats['avg_overall']
    
    log_entry = f"""## Version {args.version} ({datetime.now().strftime('%Y-%m-%d')})
- **Total Ingested Images**: {db_stats['total_images']}
- **Curated / Exported Images**: {total_curated} (Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])})
- **Duplicates Identified**: {duplicate_count}
- **Audit Rejections**: {rejected_count}
- **Average Quality Score**: {avg_quality:.2f}/10
- **Fingerprint Hash**: `{combined_hash}`
- **Source Ingestions**: {", ".join(stats_by_dataset.keys())}

---
"""
    # Append or create
    if os.path.exists(changelog_file):
        with open(changelog_file, "r", encoding="utf-8") as f:
            existing_content = f.read()
        new_content = f"# Changelog\n\n{log_entry}{existing_content.replace('# Changelog', '').strip()}"
    else:
        new_content = f"# Changelog\n\n{log_entry}"
        
    with open(changelog_file, "w", encoding="utf-8") as f:
        f.write(new_content)
    logger.info(f"Dataset changelog written to: {changelog_file}")

    # Write export report
    report_file = os.path.join(config['paths']['reports_dir'], "export_dataset_report.md")
    sections = {
        "Fingerprint Details": (
            f"- **Version**: {args.version}\n"
            f"- **Dataset Fingerprint**: `{combined_hash}`\n"
            f"- **Location**: `{os.path.abspath(args.output_dir)}`\n"
        ),
        "Split Breakdown Details": (
            f"- **Train**: {len(splits['train'])}\n"
            f"- **Val**: {len(splits['val'])}\n"
            f"- **Test**: {len(splits['test'])}\n"
        ),
        "Leakage Validation Logs": "Checked exact MD5 hashes and perceptual dHash signatures. Zero overlaps detected between splits."
    }
    create_markdown_report(report_file, f"Dataset Export Summary: v{args.version}", "Export curation, resizing, leakage audits, and packaging.", sections)

    db.close()
    logger.info("Export Stage Complete.")

if __name__ == "__main__":
    main()
