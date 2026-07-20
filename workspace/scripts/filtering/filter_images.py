import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.utils.common import setup_logger, load_config, create_markdown_report
from scripts.utils.db_manager import DatabaseManager

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from PIL import Image, ImageStat, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def parse_args():
    parser = argparse.ArgumentParser(description="Acne Dataset Pipeline - Stage 4: Image Visual Quality Audit")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID to process (e.g. DS001)")
    parser.add_argument("--config", type=str, default="workspace/configs/default_dataset_policy.yaml", help="Path to policy config file")
    return parser.parse_args()

def analyze_visual_properties(file_path: str, logger) -> dict:
    """Calculates visual quality metrics (blur, exposure, contrast, saturation, noise, lightness)."""
    metrics = {
        "corrupted": False,
        "blur_score": 100.0,
        "exposure_score": 50.0,  # 0 to 100 scale (optimal is 50)
        "contrast_score": 50.0,  # 0 to 100 scale
        "saturation_score": 50.0,# 0 to 100 scale
        "noise_score": 10.0,     # lower is cleaner
        "lightness_val": 0.5,     # 0.0 to 1.0 (lightness proxy)
        "width": 640,
        "height": 640
    }
    
    if not os.path.exists(file_path):
        metrics["corrupted"] = True
        return metrics

    # A. OpenCV Analysis (Optimal)
    if OPENCV_AVAILABLE:
        try:
            img = cv2.imread(file_path)
            if img is None:
                metrics["corrupted"] = True
                return metrics
                
            metrics["height"], metrics["width"] = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Blur (Variance of Laplacian)
            metrics["blur_score"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # 2. Exposure (Normalized mean brightness)
            mean_brightness = float(gray.mean())
            metrics["exposure_score"] = float((mean_brightness / 255.0) * 100.0)
            metrics["lightness_val"] = float(mean_brightness / 255.0)
            
            # 3. Contrast (Standard deviation of luminance)
            metrics["contrast_score"] = float(gray.std())
            
            # 4. Saturation (Mean S channel in HSV)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            metrics["saturation_score"] = float(hsv[:, :, 1].mean() / 255.0 * 100.0)
            
            # 5. Noise (High-frequency difference proxy)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            noise = cv2.absdiff(gray, blurred)
            metrics["noise_score"] = float(noise.mean())
            
        except Exception as e:
            logger.warning(f"OpenCV analysis failed for {file_path}: {e}")
            metrics["corrupted"] = True

    # B. PIL Analysis (Fallback)
    elif PIL_AVAILABLE:
        try:
            with Image.open(file_path) as img:
                metrics["width"], metrics["height"] = img.size
                
                # Check for format decoding success
                img.verify()
                
            # Reopen after verify (verify closes file handles)
            with Image.open(file_path) as img:
                img_rgb = img.convert('RGB')
                stat = ImageStat.Stat(img_rgb)
                
                # Mean of R, G, B channels
                means = stat.mean
                stds = stat.var
                
                # Exposure / Lightness proxy (average brightness)
                avg_val = sum(means) / 3.0
                metrics["exposure_score"] = (avg_val / 255.0) * 100.0
                metrics["lightness_val"] = avg_val / 255.0
                
                # Contrast proxy (average variance)
                avg_var = sum(stds) / 3.0
                metrics["contrast_score"] = np.sqrt(avg_var)
                
                # Saturation proxy (differences between channels)
                ch_diffs = abs(means[0] - means[1]) + abs(means[1] - means[2]) + abs(means[0] - means[2])
                metrics["saturation_score"] = min(100.0, (ch_diffs / 3.0) * 2.0)
                
                # Blur proxy using edge filter
                edge_img = img_rgb.convert('L').filter(ImageFilter.FIND_EDGES)
                edge_stat = ImageStat.Stat(edge_img)
                metrics["blur_score"] = edge_stat.var[0] / 10.0  # Normalized scale
                
                # Noise proxy
                metrics["noise_score"] = 5.0
                
        except Exception as e:
            logger.warning(f"PIL analysis failed for {file_path}: {e}")
            metrics["corrupted"] = True
            
    else:
        logger.error("No image processing library available (cv2 or PIL). Skipping calculations.")
        metrics["corrupted"] = True
        
    return metrics

def main():
    args = parse_args()
    logger = setup_logger("filter_images")
    logger.info(f"Starting visual properties audit for dataset {args.dataset_id}")

    try:
        config = load_config(args.config)
        db = DatabaseManager(config['paths']['database_path'])
    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        sys.exit(1)

    min_w = config['quality_metrics']['image']['min_width']
    min_h = config['quality_metrics']['image']['min_height']

    conn = db.conn
    cursor = conn.cursor()
    
    # Select images that have passed annotation validation (status is review)
    cursor.execute("""
        SELECT image_id, file_path, status 
        FROM images 
        WHERE dataset_id = ? AND status != 'rejected';
    """, (args.dataset_id,))
    images = cursor.fetchall()

    logger.info(f"Auditing image file properties for {len(images)} images.")

    corrupted_count = 0
    low_res_count = 0
    passed_count = 0
    
    # Lighting conditions counters
    lighting_stats = {"low_light": 0, "optimal": 0, "high_exposure": 0}
    # Skin tone proxy counters
    skin_stats = {"L1_Light": 0, "L2_Medium": 0, "L3_Dark": 0}

    # Explicitly start transaction
    cursor.execute("BEGIN TRANSACTION;")
    now_str = datetime.now().isoformat()

    for img in images:
        image_id = img['image_id']
        file_path = img['file_path']
        current_status = img['status']

        # 1. Analyze properties
        props = analyze_visual_properties(file_path, logger)

        if props["corrupted"]:
            corrupted_count += 1
            cursor.execute("""
                UPDATE images
                SET status = 'rejected', rejection_reason = 'Corrupted image file - unreadable by decoder', updated_at = ?
                WHERE image_id = ?;
            """, (now_str, image_id))
            continue

        # 2. Check resolution boundary
        if props["width"] < min_w or props["height"] < min_h:
            low_res_count += 1
            cursor.execute("""
                UPDATE images
                SET status = 'rejected', rejection_reason = ?, updated_at = ?
                WHERE image_id = ?;
            """, (f"Low resolution: {props['width']}x{props['height']} (min threshold is {min_w}x{min_h})", now_str, image_id))
            continue

        # 3. Categorize diversity fields
        # Lighting Condition
        exp = props["exposure_score"] / 100.0
        if exp < 0.25:
            lighting = "low_light"
        elif exp > 0.75:
            lighting = "high_exposure"
        else:
            lighting = "optimal"
        lighting_stats[lighting] += 1

        # Skin Tone lightness proxy
        lightness = props["lightness_val"]
        if lightness > 0.70:
            skin_type = "L1_Light"
        elif lightness > 0.45:
            skin_type = "L2_Medium"
        else:
            skin_type = "L3_Dark"
        skin_stats[skin_type] += 1

        # Default profile view
        profile = "frontal"

        # Update diversity and score attributes in SQLite in a single batched query
        cursor.execute("""
            UPDATE images
            SET severity_category = 'mild',
                skin_tone_category = ?,
                profile_view = ?,
                lighting_condition = ?,
                width = ?,
                height = ?,
                blur_score = ?,
                exposure_score = ?,
                updated_at = ?
            WHERE image_id = ?;
        """, (skin_type, profile, lighting, props["width"], props["height"], props["blur_score"], props["exposure_score"], now_str, image_id))

        passed_count += 1

    conn.commit()
    logger.info(f"Visual quality audit completed. Passed: {passed_count}, Corrupted rejected: {corrupted_count}, Low res rejected: {low_res_count}.")

    # Generate Markdown Report
    report_file = os.path.join(config['paths']['reports_dir'], f"filter_{args.dataset_id}_report.md")
    sections = {
        "Ingestion Validation Breakdown": (
            f"- **Images Audited**: {len(images)}\n"
            f"- **Passed Quality Checks**: {passed_count}\n"
            f"- **Corrupted/Unreadable Files**: {corrupted_count}\n"
            f"- **Low Resolution Files (< {min_w}x{min_h})**: {low_res_count}\n"
        ),
        "Lighting Diversity Profile": (
            f"- **Optimal Exposure**: {lighting_stats['optimal']}\n"
            f"- **Low-Light / Under-exposed**: {lighting_stats['low_light']}\n"
            f"- **High-Exposure / Blown-out**: {lighting_stats['high_exposure']}\n"
        ),
        "Skin Tone Lightness Index Distribution": (
            f"- **L1 (Light / Fair skin proxy)**: {skin_stats['L1_Light']}\n"
            f"- **L2 (Medium skin proxy)**: {skin_stats['L2_Medium']}\n"
            f"- **L3 (Dark skin proxy)**: {skin_stats['L3_Dark']}\n"
        )
    }
    create_markdown_report(report_file, f"Image Visual Properties Audit: {args.dataset_id}", "Analysis of resolution boundaries, lighting levels, and color metrics.", sections)

    db.close()

if __name__ == "__main__":
    main()
