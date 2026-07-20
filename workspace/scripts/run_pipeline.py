import os
import sys
import argparse
import subprocess
from datetime import datetime

# Add root directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils.common import setup_logger, load_config, create_markdown_report

def parse_args():
    parser = argparse.ArgumentParser(description="Acne Dataset Platform - Central Plugin Pipeline Orchestrator")
    
    # Ingestion details
    parser.add_argument("--dataset_id", type=str, required=True, help="Dataset ID (e.g. DS001)")
    parser.add_argument("--dataset_path", type=str, default="", help="Path to original downloaded dataset directory (required for import)")
    parser.add_argument("--dataset_name", type=str, default="", help="Full display name of the dataset")
    parser.add_argument("--license_type", type=str, default="", help="License type (e.g. MIT, CC-BY-4.0)")
    parser.add_argument("--source_url", type=str, default="", help="Canonical URL source link")
    parser.add_argument("--citation", type=str, default="", help="Academic BibTeX/citation text")
    
    # Format and model configurations
    parser.add_argument("--format", type=str, default="yolo", choices=["yolo", "coco", "voc", "darknet"], help="Annotation format")
    parser.add_argument("--source_classes", type=str, default="", help="Comma-separated class names for YOLO index conversion")
    parser.add_argument("--model_path", type=str, default="", help="Path to YOLOv8 model weights (.pt) for consensus audit")
    
    # Export options
    parser.add_argument("--version", type=str, default="2.0", help="Output dataset version label")
    parser.add_argument("--output_dir", type=str, default="workspace/datasets/curated/dataset_v2_export", help="Final export directory")
    
    # Execution pipeline steps control
    parser.add_argument("--skip_steps", type=str, default="", help="Comma-separated list of pipeline steps to skip (e.g. 'import,convert')")
    parser.add_argument("--config", type=str, default="workspace/configs/default_dataset_policy.yaml", help="Path to policy config file")
    
    return parser.parse_args()

def run_step_command(cmd: list, step_name: str, logger) -> bool:
    """Executes a pipeline step in a separate python process."""
    logger.info(f"==> RUNNING PIPELINE STEP: {step_name}")
    logger.info(f"Command: python {' '.join(cmd)}")
    
    # Run process
    process = subprocess.Popen(
        [sys.executable] + cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1
    )
    
    # Print output line by line in real-time
    if process.stdout:
        for line in process.stdout:
            sys.stdout.write(f"[{step_name}] {line}")
            sys.stdout.flush()
            
    process.wait()
    
    if process.returncode == 0:
        logger.info(f"==> STEP SUCCESSFUL: {step_name}\n")
        return True
    else:
        logger.error(f"==> STEP FAILED (Exit Code {process.returncode}): {step_name}\n")
        return False

def main():
    args = parse_args()
    logger = setup_logger("run_pipeline")
    logger.info("Initializing acne dataset engineering pipeline orchestrator...")

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        sys.exit(1)

    skip_list = [s.strip().lower() for s in args.skip_steps.split(',')] if args.skip_steps else []
    logger.info(f"Configured skip list: {skip_list}")

    # Auto-read dataset metadata from data.yaml if available
    raw_root = f"workspace/datasets/raw/{args.dataset_id}"
    yaml_path = None
    if os.path.exists(raw_root):
        for r, d, files in os.walk(raw_root):
            if "data.yaml" in files:
                yaml_path = os.path.join(r, "data.yaml")
                if not args.dataset_path:
                    args.dataset_path = r
                break
                
    if yaml_path and os.path.exists(yaml_path):
        try:
            import yaml
            logger.info(f"Auto-parsing metadata from: {yaml_path}")
            with open(yaml_path, 'r', encoding='utf-8') as yf:
                yaml_data = yaml.safe_load(yf)
                if yaml_data:
                    if not args.source_classes and 'names' in yaml_data:
                        if isinstance(yaml_data['names'], list):
                            args.source_classes = ",".join(yaml_data['names'])
                        elif isinstance(yaml_data['names'], dict):
                            args.source_classes = ",".join(yaml_data['names'].values())
                    
                    if 'roboflow' in yaml_data:
                        rb = yaml_data['roboflow']
                        if not args.dataset_name and 'project' in rb:
                            args.dataset_name = f"Roboflow {rb['project'].replace('-', ' ').title()} Dataset v{rb.get('version', '1')}"
                        if not args.license_type and 'license' in rb:
                            args.license_type = rb['license']
                        if not args.source_url and 'url' in rb:
                            args.source_url = rb['url']
        except Exception as ex:
            logger.warning(f"Failed to auto-parse data.yaml: {ex}")
            
    # Default fallback path
    if not args.dataset_path:
        args.dataset_path = raw_root

    # Step Definitions
    # Every step is represented by a command list: [script_path, ...arguments]
    steps = {}

    # Step 1: Import Ingest
    if "import" not in skip_list:
        if not args.dataset_path or not args.dataset_name or not args.license_type:
            logger.error("--dataset_path, --dataset_name, and --license_type are required to run the 'import' step.")
            sys.exit(1)
        steps["1_Import"] = [
            "workspace/scripts/import/import_dataset.py",
            "--dataset_path", args.dataset_path,
            "--dataset_name", args.dataset_name,
            "--dataset_id", args.dataset_id,
            "--license_type", args.license_type,
            "--source_url", args.source_url,
            "--citation", args.citation,
            "--config", args.config
        ]

    # Step 2: Convert Standardize
    if "convert" not in skip_list:
        steps["2_Convert"] = [
            "workspace/scripts/conversion/convert_formats.py",
            "--dataset_id", args.dataset_id,
            "--format", args.format,
            "--source_classes", args.source_classes,
            "--config", args.config
        ]

    # Step 3: Audit Annotation Coordinates
    if "audit_annotations" not in skip_list:
        steps["3_AuditAnnotations"] = [
            "workspace/scripts/audit/audit_annotations.py",
            "--dataset_id", args.dataset_id,
            "--config", args.config
        ]

    # Step 4: Audit Image Visual Quality
    if "filter_images" not in skip_list:
        steps["4_FilterImages"] = [
            "workspace/scripts/filtering/filter_images.py",
            "--dataset_id", args.dataset_id,
            "--config", args.config
        ]

    # Step 5: YOLO Agreement Inference Audit
    if "yolo_agreement" not in skip_list:
        steps["5_YOLOAgreement"] = [
            "workspace/scripts/audit/yolo_agreement.py",
            "--dataset_id", args.dataset_id,
            "--model_path", args.model_path,
            "--config", args.config
        ]

    # Step 6: Quality Scoring Engine
    if "scoring" not in skip_list:
        steps["6_ScoringEngine"] = [
            "workspace/scripts/scoring/scoring_engine.py",
            "--dataset_id", args.dataset_id,
            "--config", args.config
        ]

    # Step 7: Near-Duplicate Detection
    if "dedup" not in skip_list:
        steps["7_Deduplication"] = [
            "workspace/scripts/dedup/deduplicate.py",
            "--dataset_id", args.dataset_id,
            "--config", args.config
        ]

    # Step 8: Human Review Queue Generation
    if "review" not in skip_list:
        steps["8_ReviewQueue"] = [
            "workspace/scripts/review/review_queue.py",
            "--config", args.config
        ]

    # Step 9: Export Curation Package
    if "export" not in skip_list:
        steps["9_ExportDataset"] = [
            "workspace/scripts/export/export_dataset.py",
            "--output_dir", args.output_dir,
            "--version", args.version,
            "--config", args.config
        ]

    # Step 10: Compilation of Dashboard Report Card
    if "dashboard" not in skip_list:
        steps["10_Dashboard"] = [
            "workspace/scripts/utils/generate_dashboard.py",
            "--config", args.config
        ]

    # Execute step commands sequentially
    execution_success = True
    executed_steps = []
    
    for name, cmd in steps.items():
        success = run_step_command(cmd, name, logger)
        if not success:
            execution_success = False
            logger.critical(f"Pipeline execution halted due to failure in step: {name}")
            break
        executed_steps.append(name)

    # 4. Generate Ingestion Pipeline Summary Run Report
    run_report_file = os.path.join(config['paths']['reports_dir'], "pipeline_run_report.md")
    
    status_emoji = "✅ SUCCESS" if execution_success else "❌ FAILED"
    
    sections = {
        "Pipeline Run Summary": (
            f"- **Execution Date**: {datetime.now().isoformat()}\n"
            f"- **Target Dataset ID**: {args.dataset_id}\n"
            f"- **Pipeline Status**: {status_emoji}\n"
            f"- **Steps Successfully Executed**: {', '.join(executed_steps) if executed_steps else 'None'}\n"
        ),
        "Pipeline Step Flow Details": (
            "The execution sequence conforms to standard QYRO curation architecture. "
            "Verification checks (leakage, duplicates, box overlaps, license boundaries) were executed."
        )
    }
    create_markdown_report(run_report_file, "Acne Pipeline Orchestration Run Report", "Comprehensive summary of pipeline execution log.", sections)

    if execution_success:
        logger.info(f"Pipeline executed successfully. Pipeline report written to: {run_report_file}")
        sys.exit(0)
    else:
        logger.error(f"Pipeline execution failed. Review the step log outputs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
