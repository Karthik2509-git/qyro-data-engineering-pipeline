import os
import sys

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

def main():
    dest_dir = "mock_dataset_download"
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Generating blank mock images and YOLO text annotations in '{dest_dir}'...")
    
    # 1. Create images
    for i in range(1, 6):
        img_path = os.path.join(dest_dir, f"image{i}.jpg")
        
        # Save a real JPEG image
        if PIL_AVAILABLE:
            try:
                # Differing colors so they are not exact pixel duplicates (for dHash checks)
                color = (i * 40, i * 40, i * 40)
                img = Image.new("RGB", (640, 640), color)
                img.save(img_path)
            except Exception as e:
                print(f"Failed to save PIL image: {e}")
                # fallback empty file
                with open(img_path, "wb") as f:
                    f.write(b"\x00" * 1000)
        else:
            with open(img_path, "wb") as f:
                f.write(b"\x00" * 1000)
                
        # 2. Save text label annotations
        lbl_path = os.path.join(dest_dir, f"image{i}.txt")
        with open(lbl_path, "w", encoding="utf-8") as f:
            if i == 1:
                # Normal papule box
                f.write("0 0.35 0.40 0.08 0.08\n")
            elif i == 2:
                # Normal pustule box
                f.write("1 0.60 0.55 0.12 0.10\n")
            elif i == 3:
                # Giant box (should trigger cluster box warning)
                f.write("0 0.50 0.50 0.85 0.85\n")
            elif i == 4:
                # Overlapping boxes
                f.write("0 0.30 0.30 0.10 0.10\n")
                f.write("1 0.32 0.32 0.10 0.10\n")
            elif i == 5:
                # Clean multiple boxes
                f.write("0 0.20 0.20 0.05 0.05\n")
                f.write("1 0.70 0.70 0.05 0.05\n")
                
    # Also write a classes.txt
    with open(os.path.join(dest_dir, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("papule\npustule\n")
        
    print("Mock dataset generation completed successfully.")

if __name__ == "__main__":
    main()
