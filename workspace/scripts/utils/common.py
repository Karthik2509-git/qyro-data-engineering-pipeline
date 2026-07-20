import os
import sys
import logging
import hashlib
import yaml
from typing import Dict, Any

def setup_logger(name: str, log_dir: str = "workspace/reports") -> logging.Logger:
    """Configures and returns a logger that outputs to console and a file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str = "workspace/configs/default_dataset_policy.yaml") -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def calculate_file_hash(filepath: str, algorithm: str = "md5") -> str:
    """Computes the cryptographic hash of a file for integrity checks."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found for hashing: {filepath}")
        
    hash_func = hashlib.md5() if algorithm.lower() == "md5" else hashlib.sha256()
    
    # Read in chunks to handle large files
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_func.update(chunk)
            
    return hash_func.hexdigest()

def create_markdown_report(report_path: str, title: str, summary: str, sections: Dict[str, str]) -> None:
    """Generates a standardized Markdown report for pipeline steps."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    content = []
    content.append(f"# {title}")
    content.append(f"\n*Generated on: {logging.Formatter().formatTime(logging.LogRecord('name', 0, 'fn', 0, 'msg', (), None), '%Y-%m-%d %H:%M:%S')}*\n")
    content.append("## Executive Summary")
    content.append(summary)
    
    for section_title, section_text in sections.items():
        content.append(f"\n## {section_title}")
        content.append(section_text)
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))
