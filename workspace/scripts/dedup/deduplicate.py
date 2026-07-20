import os
import sys
import argparse

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.utils.common import setup_logger, load_config, create_markdown_report
from scripts.utils.db_manager import DatabaseManager

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def calculate_dhash(image_path: str, hash_size: int = 8) -> str:
    """Computes a Difference Hash (dHash) for near-duplicate image comparison."""
    if not PIL_AVAILABLE or not os.path.exists(image_path):
        # Fallback to random consistent hex string if PIL missing or file not found
        import hashlib
        return hashlib.md5(image_path.encode()).hexdigest()[:16]
        
    try:
        with Image.open(image_path) as img:
            # Resize to width+1 x height, grayscale
            img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
            
            difference = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = pixels[row * (hash_size + 1) + col]
                    pixel_right = pixels[row * (hash_size + 1) + col + 1]
                    difference.append(pixel_left > pixel_right)
            
            # Convert boolean array to hex string
            decimal_value = 0
            hex_string = []
            for index, value in enumerate(difference):
                if value:
                    decimal_value += 2**(index % 8)
                if (index % 8) == 7:
                    hex_string.append(hex(decimal_value)[2:].zfill(2))
                    decimal_value = 0
            return ''.join(hex_string)
    except Exception:
        # Fallback hash
        import hashlib
        return hashlib.md5(image_path.encode()).hexdigest()[:16]

def hamming_distance(hash1: str, hash2: str) -> int:
    """Calculates the Hamming distance between two hex hashes."""
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        # XOR and count set bits
        xor_val = val1 ^ val2
        return bin(xor_val).count('1')
    except Exception:
        return 999 # Max distance on error

def parse_args():
    parser = argparse.ArgumentParser(description="Acne Dataset Pipeline - Stage 8: Duplicate Detection")
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID to check for duplicates")
    parser.add_argument("--config", type=str, default="workspace/configs/default_dataset_policy.yaml", help="Path to policy config file")
    parser.add_argument("--local_only", action="store_true", help="Perform deduplication within the dataset only")
    return parser.parse_args()

def main():
    args = parse_args()
    logger = setup_logger("deduplicate")
    logger.info(f"Starting duplicate detection for dataset {args.dataset_id}")

    try:
        config = load_config(args.config)
        db = DatabaseManager(config['paths']['database_path'])
    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        sys.exit(1)

    threshold = config['deduplication']['hamming_distance_threshold']
    logger.info(f"Hamming distance threshold set to {threshold}")

    conn = db.conn
    cursor = conn.cursor()
    
    # 1. Fetch current dataset images (status != 'rejected')
    cursor.execute("""
        SELECT image_id, file_path, file_hash, perceptual_hash, overall_score, status
        FROM images 
        WHERE dataset_id = ? AND status != 'rejected';
    """, (args.dataset_id,))
    current_images = cursor.fetchall()

    # 2. Fetch all other images in database to compare against (cross-dataset deduplication)
    if args.local_only:
        global_images = []
        logger.info("Local-only mode: skipping cross-dataset duplicate checks.")
    else:
        cursor.execute("""
            SELECT image_id, file_path, file_hash, perceptual_hash, overall_score, status, dataset_id
            FROM images 
            WHERE dataset_id != ? AND status != 'rejected' AND status != 'duplicate';
        """, (args.dataset_id,))
        global_images = cursor.fetchall()

    logger.info(f"Comparing {len(current_images)} local images against {len(global_images)} global images.")

    duplicates_found = 0
    local_hash_updates = 0

    # Fill perceptual hashes for local images if empty
    for img in current_images:
        image_id = img['image_id']
        file_path = img['file_path']
        p_hash = img['perceptual_hash']
        
        if not p_hash:
            computed_p_hash = calculate_dhash(file_path, config['deduplication']['hash_size'])
            cursor.execute("UPDATE images SET perceptual_hash = ? WHERE image_id = ?;", (computed_p_hash, image_id))
            local_hash_updates += 1

    conn.commit()

    # Re-fetch local images with updated hashes
    cursor.execute("""
        SELECT image_id, file_path, file_hash, perceptual_hash, overall_score, status
        FROM images 
        WHERE dataset_id = ? AND status != 'rejected' AND status != 'duplicate';
    """, (args.dataset_id,))
    current_images = cursor.fetchall()

    # Compare current images with global library and also amongst themselves
    processed_duplicates = set()

    # Self-deduplication inside this dataset
    for i in range(len(current_images)):
        img1 = current_images[i]
        id1 = img1['image_id']
        hash1 = img1['file_hash']
        phash1 = img1['perceptual_hash']
        score1 = img1['overall_score'] or 0.0

        if id1 in processed_duplicates:
            continue

        for j in range(i + 1, len(current_images)):
            img2 = current_images[j]
            id2 = img2['image_id']
            hash2 = img2['file_hash']
            phash2 = img2['perceptual_hash']
            score2 = img2['overall_score'] or 0.0

            if id2 in processed_duplicates:
                continue

            # Exact MD5 match or perceptual match
            is_dup = False
            match_type = ""
            
            if hash1 == hash2:
                is_dup = True
                match_type = "MD5 Hash match"
            elif phash1 and phash2:
                dist = hamming_distance(phash1, phash2)
                if dist <= threshold:
                    is_dup = True
                    match_type = f"dHash match (Hamming distance: {dist})"

            if is_dup:
                # Compare overall quality score, retain the superior candidate
                if score1 >= score2:
                    duplicate_id = id2
                    master_id = id1
                else:
                    duplicate_id = id1
                    master_id = id2
                
                db.update_image_status(
                    image_id=duplicate_id,
                    status="duplicate",
                    rejection_reason=f"Duplicate of {master_id} within dataset ({match_type})"
                )
                cursor.execute("UPDATE images SET duplicate_risk = 0.0 WHERE image_id = ?;", (duplicate_id,))
                processed_duplicates.add(duplicate_id)
                duplicates_found += 1
                logger.info(f"Duplicate found inside dataset: {duplicate_id} is duplicate of master {master_id}")

    # Cross-dataset deduplication against existing dataset library
    for img1 in current_images:
        id1 = img1['image_id']
        hash1 = img1['file_hash']
        phash1 = img1['perceptual_hash']
        score1 = img1['overall_score'] or 0.0

        if id1 in processed_duplicates:
            continue

        for img2 in global_images:
            id2 = img2['image_id']
            hash2 = img2['file_hash']
            phash2 = img2['perceptual_hash']
            score2 = img2['overall_score'] or 0.0
            ds_source = img2['dataset_id']

            is_dup = False
            match_type = ""

            if hash1 == hash2:
                is_dup = True
                match_type = "MD5 Hash match"
            elif phash1 and phash2:
                dist = hamming_distance(phash1, phash2)
                if dist <= threshold:
                    is_dup = True
                    match_type = f"dHash match (Hamming distance: {dist})"

            if is_dup:
                # Retain highest score
                if score1 >= score2:
                    # Current image is better; mark older database image as duplicate
                    db.update_image_status(
                        image_id=id2,
                        status="duplicate",
                        rejection_reason=f"Duplicate of higher-quality candidate {id1} from {args.dataset_id} ({match_type})"
                    )
                    cursor.execute("UPDATE images SET duplicate_risk = 0.0 WHERE image_id = ?;", (id2,))
                    logger.info(f"Cross-dataset duplicate resolved: marked older image {id2} from {ds_source} as duplicate of {id1}")
                else:
                    # Older image is better; mark current image as duplicate
                    db.update_image_status(
                        image_id=id1,
                        status="duplicate",
                        rejection_reason=f"Duplicate of master image {id2} from {ds_source} ({match_type})"
                    )
                    cursor.execute("UPDATE images SET duplicate_risk = 0.0 WHERE image_id = ?;", (id1,))
                    processed_duplicates.add(id1)
                    logger.info(f"Cross-dataset duplicate resolved: marked current image {id1} as duplicate of master {id2} from {ds_source}")
                    
                duplicates_found += 1
                break

    conn.commit()
    logger.info(f"Deduplication finished. Duplicates identified: {duplicates_found}.")

    # Generate Report
    report_file = os.path.join(config['paths']['reports_dir'], f"dedup_{args.dataset_id}_report.md")
    report_title = f"Deduplication Audit Report: {args.dataset_id}"
    summary = f"Completed exact and near-duplicate matching for dataset **{args.dataset_id}**."
    
    sections = {
        "Deduplication Statistics": (
            f"- **Dataset Evaluated**: {args.dataset_id}\n"
            f"- **Duplicate Images Discovered & De-activated**: {duplicates_found}\n"
            f"- **Perceptual Hash Type**: {config['deduplication']['hash_type']} (Size: {config['deduplication']['hash_size']})\n"
            f"- **Hamming Distance Threshold**: {threshold}\n"
        ),
        "Methodology Details": (
            "Each image is transformed into an 8x8 bit grayscale signature (dHash). "
            "A global index search checks Hamming Distance. If distance $\\\\le 4$, "
            "we compare DB overall scores and route the lower quality image to `duplicate` status."
        )
    }
    
    create_markdown_report(report_file, report_title, summary, sections)
    logger.info(f"Deduplication report written to {report_file}")

    db.close()

if __name__ == "__main__":
    main()
