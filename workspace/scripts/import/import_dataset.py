import os
import sys
import argparse
import shutil
import json
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.utils.common import setup_logger, load_config, calculate_file_hash, create_markdown_report
from scripts.utils.db_manager import DatabaseManager

def parse_args():
    parser = argparse.ArgumentParser(description="Acne Dataset Pipeline - Stage 1: Import & License Ingestion")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to original downloaded dataset directory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Full display name of the dataset")
    parser.add_argument("--dataset_id", type=str, required=True, help="Target Dataset ID (e.g. DS001)")
    parser.add_argument("--license_type", type=str, required=True, help="License type (e.g. MIT, CC-BY-4.0, cc-by-nc-4.0)")
    parser.add_argument("--source_url", type=str, default="", help="Canonical URL source link")
    parser.add_argument("--citation", type=str, default="", help="Academic BibTeX/citation text")
    parser.add_argument("--config", type=str, default="workspace/configs/default_dataset_policy.yaml", help="Path to policy config file")
    return parser.parse_args()

def validate_license(license_str: str, config: dict) -> tuple:
    """Checks license string against policy rules. Returns (attribution_req, commercial_allowed, status)."""
    normalized_lic = license_str.lower().strip()
    license_rules = config['licensing']['rules']
    
    # Try direct mapping
    if normalized_lic in license_rules:
        attr, comm, status = license_rules[normalized_lic]
        return attr, comm, status
        
    # Pattern matching fallbacks
    for key, value in license_rules.items():
        if key in normalized_lic:
            return value[0], value[1], value[2]
            
    # Default fallback for unknown license
    return 1, 0, "review"

def main():
    args = parse_args()
    logger = setup_logger("import_dataset")
    logger.info(f"Starting Ingestion Phase for {args.dataset_id} ({args.dataset_name})")
    
    try:
        config = load_config(args.config)
        db = DatabaseManager(config['paths']['database_path'])
    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        sys.exit(1)

    if not os.path.exists(args.dataset_path):
        logger.error(f"Source directory does not exist: {args.dataset_path}")
        sys.exit(1)

    external_dest = os.path.join(config['paths']['external_dir'], args.dataset_id)
    raw_dest = os.path.join(config['paths']['raw_dir'], args.dataset_id)

    os.makedirs(external_dest, exist_ok=True)
    os.makedirs(raw_dest, exist_ok=True)

    # 1. License Validation
    attr_required, comm_allowed, validation_status = validate_license(args.license_type, config)
    logger.info(f"License Ingestion Validation - Attribution Required: {attr_required}, Commercial Allowed: {comm_allowed}, Status: {validation_status}")

    # 2. Archive Licensing Files
    with open(os.path.join(external_dest, "license.txt"), "w", encoding="utf-8") as f:
        f.write(f"License: {args.license_type}\nDate Ingested: {datetime.now().isoformat()}\nValidation: {validation_status}\n")
    with open(os.path.join(external_dest, "source_url.txt"), "w", encoding="utf-8") as f:
        f.write(f"Source URL: {args.source_url}\n")
    if args.citation:
        with open(os.path.join(external_dest, "citation.txt"), "w", encoding="utf-8") as f:
            f.write(f"Citation:\n{args.citation}\n")

    # Register dataset in SQLite DB
    db.insert_dataset(
        dataset_id=args.dataset_id,
        name=args.dataset_name,
        source_url=args.source_url,
        license_type=args.license_type,
        citation=args.citation,
        attribution_required=attr_required,
        commercial_use_allowed=comm_allowed,
        license_validation_status=validation_status
    )

    # 3. Copy files & Generate Image Index records
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    scanned_images = []
    image_counter = 0

    # Locate and copy images
    for root, _, files in os.walk(args.dataset_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_counter += 1
                image_id = f"{args.dataset_id}_{image_counter:05d}"
                src_img_path = os.path.join(root, file)
                
                # Setup destination path
                dest_filename = f"{image_id}{os.path.splitext(file)[1]}"
                dest_img_path = os.path.join(raw_dest, dest_filename)
                
                # Handle copy and calculate hash
                try:
                    shutil.copy2(src_img_path, dest_img_path)
                    file_hash = calculate_file_hash(dest_img_path)
                except Exception as e:
                    logger.warning(f"Error copying {src_img_path}: {e}")
                    dest_img_path = src_img_path  # fallback
                    file_hash = "mock_hash_value"
                
                # Check for corresponding label file in same dir or 'labels' parent
                # We also copy label files if they exist to raw folder (YOLO/COCO format metadata)
                # (Actual format conversion happens in the convert stage)
                base_name = os.path.splitext(file)[0]
                for label_ext in ('.txt', '.xml', '.json'):
                    src_lbl_path = os.path.join(root, base_name + label_ext)
                    # Also look in a parallel 'labels' directory if YOLO structure
                    if not os.path.exists(src_lbl_path):
                        parent_dir = os.path.dirname(root)
                        src_lbl_path = os.path.join(parent_dir, 'labels', base_name + label_ext)
                        
                    if os.path.exists(src_lbl_path):
                        dest_lbl_path = os.path.join(raw_dest, f"{image_id}{label_ext}")
                        try:
                            shutil.copy2(src_lbl_path, dest_lbl_path)
                        except Exception:
                            pass
                
                # Insert metadata record
                db.insert_image(
                    image_id=image_id,
                    dataset_id=args.dataset_id,
                    original_filename=file,
                    file_path=dest_img_path,
                    file_hash=file_hash,
                    status="review" if validation_status == "review" else "review", # Awaiting full pipeline processing
                    rejection_reason="Ingested - Awaiting pipeline audit" if validation_status != "review" else "Ingested - Flagged due to license validation queue"
                )
                scanned_images.append(image_id)

    # 4. Generate Dataset Manifest Documents
    manifest_data = {
        "dataset_id": args.dataset_id,
        "name": args.dataset_name,
        "source_url": args.source_url,
        "license": args.license_type,
        "commercial_use_allowed": bool(comm_allowed),
        "attribution_required": bool(attr_required),
        "license_validation_status": validation_status,
        "total_images": len(scanned_images),
        "version": "1.0",
        "imported_at": datetime.now().isoformat()
    }
    
    # Save JSON manifest
    with open(os.path.join(raw_dest, "dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)
    with open(os.path.join(external_dest, "dataset_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)

    # Save Markdown manifest
    manifest_md = f"""# Dataset Ingestion Manifest: {args.dataset_id}

- **Name**: {args.dataset_name}
- **Source**: {args.source_url}
- **Imported**: {manifest_data['imported_at']}
- **Status**: {validation_status.upper()}
- **License**: {args.license_type}
- **Commercial Use Allowed**: {comm_allowed == 1}
- **Attribution Required**: {attr_required == 1}
- **Total Images Ingested**: {len(scanned_images)}
- **Version**: 1.0
"""
    with open(os.path.join(raw_dest, "manifest.md"), "w", encoding="utf-8") as f:
        f.write(manifest_md)
    with open(os.path.join(external_dest, "manifest.md"), "w", encoding="utf-8") as f:
        f.write(manifest_md)

    logger.info(f"Ingested {len(scanned_images)} images. Manifest files generated.")

    # 5. Write Ingestion Run Report
    report_file = os.path.join(config['paths']['reports_dir'], f"import_{args.dataset_id}_report.md")
    sections = {
        "Dataset Specifications": manifest_md,
        "Licensing Audits": (
            f"Validation status resolved to `{validation_status}`. "
            f"Attribution requirements check: `{'Required' if attr_required else 'None'}`. "
            f"Commercial reuse authorization: `{'Authorized' if comm_allowed else 'Prohibited'}`."
        )
    }
    create_markdown_report(report_file, f"Dataset Import Summary: {args.dataset_id}", f"Successfully imported dataset {args.dataset_id}.", sections)
    
    db.close()
    logger.info("Ingestion Step Complete.")

if __name__ == "__main__":
    main()
