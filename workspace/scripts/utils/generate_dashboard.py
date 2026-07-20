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
    parser = argparse.ArgumentParser(description="Acne Dataset Pipeline - Generate Dataset Health Dashboard")
    parser.add_argument("--config", type=str, default="workspace/configs/default_dataset_policy.yaml", help="Path to policy config file")
    return parser.parse_args()

def generate_ascii_histogram(data: list, bins: int = 10, title: str = "Distribution") -> str:
    """Generates a text-based ASCII histogram to show score spreads in markdown."""
    if not data:
        return "No data available."
        
    min_val, max_val = min(data), max(data)
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
        
    bin_width = (max_val - min_val) / bins
    counts = [0] * bins
    
    for val in data:
        bin_idx = int((val - min_val) / bin_width)
        if bin_idx >= bins:
            bin_idx = bins - 1
        counts[bin_idx] += 1
        
    max_count = max(counts) if counts else 0
    max_label_width = 15
    chart_lines = []
    
    for i in range(bins):
        b_start = min_val + i * bin_width
        b_end = b_start + bin_width
        bin_label = f"[{b_start:.1f} - {b_end:.1f}]"
        
        # Scale bars to fit standard page (max width 40 characters)
        bar_len = int((counts[i] / max_count) * 40) if max_count > 0 else 0
        bar = "█" * bar_len + f" ({counts[i]})"
        chart_lines.append(f"{bin_label:<{max_label_width}} | {bar}")
        
    return "\n".join(chart_lines)

def main():
    args = parse_args()
    logger = setup_logger("generate_dashboard")
    logger.info("Aggregating statistics and building Dataset Health Report Card Dashboard")

    try:
        config = load_config(args.config)
        db = DatabaseManager(config['paths']['database_path'])
    except Exception as e:
        logger.error(f"Initialization failure: {e}")
        sys.exit(1)

    conn = db.conn
    cursor = conn.cursor()
    
    # 1. Base counts queries
    db_stats = db.query_stats()
    
    # Ingestion counts per source dataset ID
    cursor.execute("""
        SELECT d.dataset_id, d.name, COUNT(i.image_id) 
        FROM datasets d
        LEFT JOIN images i ON d.dataset_id = i.dataset_id
        GROUP BY d.dataset_id;
    """)
    sources_stats = cursor.fetchall()
    
    # Active accepted counts per source dataset ID
    cursor.execute("""
        SELECT dataset_id, COUNT(image_id) 
        FROM images 
        WHERE status = 'accepted'
        GROUP BY dataset_id;
    """)
    accepted_by_source = {row[0]: row[1] for row in cursor.fetchall()}

    # 2. Diversity statistics queries
    cursor.execute("SELECT severity_category, COUNT(*) FROM images WHERE status = 'accepted' GROUP BY severity_category;")
    severity_stats = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("SELECT skin_tone_category, COUNT(*) FROM images WHERE status = 'accepted' GROUP BY skin_tone_category;")
    skin_stats = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("SELECT lighting_condition, COUNT(*) FROM images WHERE status = 'accepted' GROUP BY lighting_condition;")
    lighting_stats = {row[0]: row[1] for row in cursor.fetchall()}

    # 3. Annotation averages (accepted images only)
    cursor.execute("""
        SELECT COUNT(a.annotation_id), COUNT(DISTINCT i.image_id)
        FROM annotations a
        JOIN images i ON a.image_id = i.image_id
        WHERE i.status = 'accepted' AND a.is_original = 0;
    """)
    ann_row = cursor.fetchone()
    total_accepted_anns = ann_row[0] or 0
    total_accepted_imgs = ann_row[1] or 0
    avg_lesions = total_accepted_anns / total_accepted_imgs if total_accepted_imgs > 0 else 0.0

    # 4. Average Bbox Size (JSON extraction)
    cursor.execute("""
        SELECT data FROM annotations a
        JOIN images i ON a.image_id = i.image_id
        WHERE i.status = 'accepted' AND a.is_original = 0 AND a.is_valid = 1;
    """)
    box_data_rows = cursor.fetchall()
    
    avg_box_area = 0.0
    if box_data_rows:
        total_area = 0.0
        for row in box_data_rows:
            ann_data = json.loads(row[0])
            if "bbox" in ann_data:
                w, h = ann_data["bbox"][2:4]
                total_area += w * h
        avg_box_area = total_area / len(box_data_rows)

    # 5. Extract scores array for histograms
    cursor.execute("SELECT overall_score FROM images WHERE status = 'accepted';")
    overall_scores = [row[0] for row in cursor.fetchall() if row[0] is not None]

    cursor.execute("SELECT width FROM images WHERE status = 'accepted';")
    resolutions_w = [row[0] for row in cursor.fetchall() if row[0] is not None]

    # Generate ASCII histograms
    quality_hist = generate_ascii_histogram(overall_scores, bins=5, title="Overall Curated Score Spread")
    res_hist = generate_ascii_histogram(resolutions_w, bins=5, title="Resolution Dimensions (Width)")

    # 6. Generate Markdown Dashboard File
    dashboard_file = os.path.join(config['paths']['reports_dir'], "dataset_dashboard.md")
    
    # Ingestions Table Markdown
    sources_table_lines = [
        "| Dataset ID | Name | Ingested Images | Accepted for Curated Pool | Yield Rate |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    for row in sources_stats:
        ds_id, name, count = row[0], row[1], row[2]
        accepted = accepted_by_source.get(ds_id, 0)
        yield_rate = (accepted / count * 100.0) if count > 0 else 0.0
        sources_table_lines.append(f"| `{ds_id}` | {name} | {count} | {accepted} | {yield_rate:.1f}% |")

    # Metrics section
    summary_text = f"""## 📊 General Metrics Summary
- **Total Ingested Images**: {db_stats['total_images']}
  - **Accepted (Curated Pool)**: {total_accepted_imgs}
  - **Rejected (Filter/Audit Rejections)**: {db_stats['status_counts'].get('rejected', 0)}
  - **Duplicates Resolved**: {db_stats['status_counts'].get('duplicate', 0)}
  - **Awaiting Human Triage (Review Queue)**: {db_stats['status_counts'].get('review', 0)}
- **Yield Rate (Accepted/Total)**: { (total_accepted_imgs/db_stats['total_images']*100.0) if db_stats['total_images'] > 0 else 0.0:.1f}%
- **Curated Bounding Boxes**: {total_accepted_anns}
- **Average Lesions per Image**: {avg_lesions:.1f}
- **Average Bounding Box Area**: {avg_box_area*100:.2f}% of image frame
- **Average Quality Score**: {db_stats['avg_overall']:.2f}/10
- **Average YOLO prediction agreement**: {db_stats['avg_yolo_agreement']:.2f}/10
"""

    diversity_text = f"""## 🧬 Dataset Diversity Index (Curated Images Only)

### Acne Severity Distribution
- **Mild (< 5 lesions)**: {severity_stats.get('mild', 0)} images ({severity_stats.get('mild', 0)/total_accepted_imgs*100 if total_accepted_imgs>0 else 0:.1f}%)
- **Moderate (5 to 20 lesions)**: {severity_stats.get('moderate', 0)} images ({severity_stats.get('moderate', 0)/total_accepted_imgs*100 if total_accepted_imgs>0 else 0:.1f}%)
- **Severe (> 20 lesions)**: {severity_stats.get('severe', 0)} images ({severity_stats.get('severe', 0)/total_accepted_imgs*100 if total_accepted_imgs>0 else 0:.1f}%)

### Lighting Conditions Category
- **Optimal Exposure**: {lighting_stats.get('optimal', 0)} images
- **Low-Light / Dark**: {lighting_stats.get('low_light', 0)} images
- **High-Exposure / Flash**: {lighting_stats.get('high_exposure', 0)} images

### Skin Tone Lightness proxy (Fitzpatrick-like proxy)
- **L1 (Fair / Light Skin tone proxy)**: {skin_stats.get('L1_Light', 0)} images
- **L2 (Medium / Tan Skin tone proxy)**: {skin_stats.get('L2_Medium', 0)} images
- **L3 (Dark / Deep Skin tone proxy)**: {skin_stats.get('L3_Dark', 0)} images
"""

    histograms_text = f"""## 📈 Metric Distribution Histograms (Curated Images Only)

### Overall Curated Score Distribution (Composite Quality)
```text
{quality_hist}
```

### Resolution Dimensions (Original Width Distribution)
```text
{res_hist}
```
"""

    sections = {
        "Source Ingestion Metrics": "\n".join(sources_table_lines),
        "Dashboard Metrics": summary_text,
        "Diversity Indicators": diversity_text,
        "Distribution Charts": histograms_text
    }
    
    create_markdown_report(dashboard_file, "Dataset Health Dashboard & Report Card", "Unified status analysis of the ingested acne datasets.", sections)
    logger.info(f"Report card dashboard written to: {dashboard_file}")

    db.close()

if __name__ == "__main__":
    main()
