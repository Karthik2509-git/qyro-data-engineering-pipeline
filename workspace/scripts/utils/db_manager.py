import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

class DatabaseManager:
    """Manages SQLite operations for the Acne Dataset Engineering Pipeline."""
    
    def __init__(self, db_path: str = "workspace/database/dataset_index.sqlite"):
        self.db_path = db_path
        # Ensure database parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = None
        self.connect()
        self.initialize_schema()
        
    def connect(self):
        """Establishes connection to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        # Enable foreign key support
        self.conn.execute("PRAGMA foreign_keys = ON;")
        
    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            
    def initialize_schema(self):
        """Initializes tables for datasets, images, and annotations with the production-grade schema."""
        cursor = self.conn.cursor()
        
        # 1. Create datasets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source_url TEXT,
                license_type TEXT,
                citation TEXT,
                attribution_required INTEGER DEFAULT 0,
                commercial_use_allowed INTEGER DEFAULT 1,
                license_validation_status TEXT DEFAULT 'review',
                imported_at TEXT NOT NULL
            );
        """)
        
        # 2. Create images table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                dataset_id TEXT,
                original_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                perceptual_hash TEXT,
                width INTEGER,
                height INTEGER,
                blur_score REAL,
                exposure_score REAL,
                duplicate_risk REAL,
                annotation_quality_score REAL,
                lesion_visibility_score REAL,
                cluster_quality_score REAL,
                yolo_agreement_score REAL,
                overall_score REAL,
                yolo_box_count INTEGER,
                mean_iou REAL,
                missing_lesions INTEGER,
                extra_annotations INTEGER,
                severity_category TEXT,
                skin_tone_category TEXT,
                profile_view TEXT,
                lighting_condition TEXT,
                status TEXT NOT NULL CHECK (status IN ('accepted', 'rejected', 'ignored', 'review', 'duplicate')),
                rejection_reason TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
            );
        """)
        
        # 3. Create annotations table (supports bboxes and segmentation polygons)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                annotation_id TEXT PRIMARY KEY,
                image_id TEXT,
                class_label TEXT NOT NULL,
                annotation_type TEXT NOT NULL CHECK (annotation_type IN ('bbox', 'segmentation')),
                data TEXT NOT NULL, -- JSON formatted bounding box [x,y,w,h] or polygon vertices
                is_original INTEGER DEFAULT 1, -- 1=original, 0=standardized/converted
                is_valid INTEGER DEFAULT 1,
                audit_reason TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (image_id) REFERENCES images(image_id)
            );
        """)
        
        self.conn.commit()

    def insert_dataset(self, dataset_id: str, name: str, source_url: str = None, 
                       license_type: str = None, citation: str = None,
                       attribution_required: int = 0, commercial_use_allowed: int = 1,
                       license_validation_status: str = "review") -> bool:
        """Inserts or updates a dataset record with licensing metadata."""
        cursor = self.conn.cursor()
        now_str = datetime.now().isoformat()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO datasets (
                    dataset_id, name, source_url, license_type, citation, 
                    attribution_required, commercial_use_allowed, license_validation_status, imported_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (dataset_id, name, source_url, license_type, citation,
                  attribution_required, commercial_use_allowed, license_validation_status, now_str))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database error in insert_dataset: {e}")
            return False

    def insert_image(self, image_id: str, dataset_id: str, original_filename: str, 
                     file_path: str, file_hash: str, status: str = "review", 
                     width: int = None, height: int = None, perceptual_hash: str = None,
                     rejection_reason: str = None) -> bool:
        """Inserts a new image record with default placeholder metrics."""
        cursor = self.conn.cursor()
        now_str = datetime.now().isoformat()
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO images (
                    image_id, dataset_id, original_filename, file_path, file_hash, 
                    perceptual_hash, width, height, status, rejection_reason, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (image_id, dataset_id, original_filename, file_path, file_hash, 
                  perceptual_hash, width, height, status, rejection_reason, now_str))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database error in insert_image: {e}")
            return False

    def update_image_scores(self, image_id: str, scores: Dict[str, float]) -> bool:
        """Updates metrics, quality scores, and overall scores for a specific image."""
        cursor = self.conn.cursor()
        now_str = datetime.now().isoformat()
        try:
            cursor.execute("""
                UPDATE images
                SET blur_score = ?,
                    exposure_score = ?,
                    duplicate_risk = ?,
                    annotation_quality_score = ?,
                    lesion_visibility_score = ?,
                    cluster_quality_score = ?,
                    yolo_agreement_score = ?,
                    overall_score = ?,
                    updated_at = ?
                WHERE image_id = ?;
            """, (
                scores.get('blur_score'),
                scores.get('exposure_score'),
                scores.get('duplicate_risk'),
                scores.get('annotation_quality_score'),
                scores.get('lesion_visibility_score'),
                scores.get('cluster_quality_score'),
                scores.get('yolo_agreement_score'),
                scores.get('overall_score'),
                now_str,
                image_id
            ))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database error in update_image_scores: {e}")
            return False

    def update_image_yolo_agreement(self, image_id: str, yolo_box_count: int, 
                                    mean_iou: float, missing_lesions: int, 
                                    extra_annotations: int, yolo_agreement_score: float) -> bool:
        """Saves prediction comparison statistics and updates YOLO agreement score."""
        cursor = self.conn.cursor()
        now_str = datetime.now().isoformat()
        try:
            cursor.execute("""
                UPDATE images
                SET yolo_box_count = ?,
                    mean_iou = ?,
                    missing_lesions = ?,
                    extra_annotations = ?,
                    yolo_agreement_score = ?,
                    updated_at = ?
                WHERE image_id = ?;
            """, (yolo_box_count, mean_iou, missing_lesions, extra_annotations, yolo_agreement_score, now_str, image_id))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database error in update_image_yolo_agreement: {e}")
            return False

    def update_image_diversity(self, image_id: str, severity_category: str, 
                               skin_tone_category: str, profile_view: str, 
                               lighting_condition: str) -> bool:
        """Updates metadata fields relating to diversity tracking and dataset fairness."""
        cursor = self.conn.cursor()
        now_str = datetime.now().isoformat()
        try:
            cursor.execute("""
                UPDATE images
                SET severity_category = ?,
                    skin_tone_category = ?,
                    profile_view = ?,
                    lighting_condition = ?,
                    updated_at = ?
                WHERE image_id = ?;
            """, (severity_category, skin_tone_category, profile_view, lighting_condition, now_str, image_id))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database error in update_image_diversity: {e}")
            return False

    def update_image_status(self, image_id: str, status: str, rejection_reason: str = None) -> bool:
        """Updates image execution status."""
        cursor = self.conn.cursor()
        now_str = datetime.now().isoformat()
        try:
            cursor.execute("""
                UPDATE images
                SET status = ?, rejection_reason = ?, updated_at = ?
                WHERE image_id = ?;
            """, (status, rejection_reason, now_str, image_id))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database error in update_image_status: {e}")
            return False

    def insert_annotation(self, annotation_id: str, image_id: str, class_label: str,
                          annotation_type: str, data: Any, is_original: int = 1,
                          is_valid: int = 1, audit_reason: str = None) -> bool:
        """Inserts a new annotation record (converts data structure to JSON string)."""
        cursor = self.conn.cursor()
        now_str = datetime.now().isoformat()
        data_json = json.dumps(data)
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO annotations (
                    annotation_id, image_id, class_label, annotation_type, data, 
                    is_original, is_valid, audit_reason, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (annotation_id, image_id, class_label, annotation_type, data_json,
                  is_original, is_valid, audit_reason, now_str))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database error in insert_annotation: {e}")
            return False

    def query_stats(self) -> Dict[str, Any]:
        """Queries and returns aggregate stats from the database."""
        cursor = self.conn.cursor()
        stats = {}
        
        # Dataset count
        cursor.execute("SELECT COUNT(*) FROM datasets;")
        stats['dataset_count'] = cursor.fetchone()[0]
        
        # Image counts by status
        cursor.execute("SELECT status, COUNT(*) FROM images GROUP BY status;")
        stats['status_counts'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Total image count
        cursor.execute("SELECT COUNT(*) FROM images;")
        stats['total_images'] = cursor.fetchone()[0]
        
        # Annotation counts
        cursor.execute("SELECT COUNT(*), SUM(is_valid) FROM annotations;")
        row = cursor.fetchone()
        stats['total_annotations'] = row[0] if row else 0
        stats['valid_annotations'] = row[1] if row and row[1] is not None else 0
        
        # Average image scores
        cursor.execute("SELECT AVG(blur_score), AVG(exposure_score), AVG(overall_score), AVG(yolo_agreement_score) FROM images;")
        row = cursor.fetchone()
        stats['avg_blur'] = round(row[0], 2) if row and row[0] is not None else 0.0
        stats['avg_exposure'] = round(row[1], 2) if row and row[1] is not None else 0.0
        stats['avg_overall'] = round(row[2], 2) if row and row[2] is not None else 0.0
        stats['avg_yolo_agreement'] = round(row[3], 2) if row and row[3] is not None else 0.0
        
        return stats
