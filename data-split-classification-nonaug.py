import os
import shutil
import random
from pathlib import Path
import glob

# กำหนดเส้นทาง
source_dir = "Classification"
target_dir = "augmented_dataset/Classification"

# กำหนดสัดส่วนการแบ่งข้อมูล
train_ratio = 0.6
test_ratio = 0.2
val_ratio = 0.2

# ตรวจสอบว่าสัดส่วนรวมกันได้ 1
assert train_ratio + test_ratio + val_ratio == 1.0, "สัดส่วนการแบ่งข้อมูลต้องรวมกันได้ 1"

# รายชื่อคลาสทั้งหมด
classes = ["Ameloblastoma", "Dentigerous cyst", "Normal jaw", "OKC"]

# ตรวจสอบและสร้างโฟลเดอร์ปลายทาง
def ensure_directory(directory):
    os.makedirs(directory, exist_ok=True)

# สร้างโฟลเดอร์สำหรับแต่ละโฟลเดอร์ปลายทาง
for split in ["train", "test", "val"]:
    for class_name in classes:
        ensure_directory(os.path.join(target_dir, split, class_name))

# ฟังก์ชันสำหรับคัดลอกไฟล์
def copy_file(src, dst):
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการคัดลอกไฟล์ {src} ไปยัง {dst}: {e}")
        return False

# ประมวลผลแต่ละคลาส
for class_name in classes:
    print(f"กำลังประมวลผลคลาส: {class_name}")
    
    # ค้นหาไฟล์ภาพทั้งหมดในคลาส
    image_files = glob.glob(os.path.join(source_dir, class_name, "*"))
    
    print(f"พบไฟล์ภาพทั้งหมด {len(image_files)} ไฟล์")
    
    # สับเปลี่ยนข้อมูลเพื่อความสุ่ม
    random.shuffle(image_files)
    
    # คำนวณจำนวนไฟล์สำหรับแต่ละชุดข้อมูล
    total_files = len(image_files)
    train_count = int(total_files * train_ratio)
    test_count = int(total_files * test_ratio)
    val_count = total_files - train_count - test_count
    
    print(f"แบ่งเป็น: train={train_count}, test={test_count}, val={val_count}")
    
    # แบ่งข้อมูล
    train_files = image_files[:train_count]
    test_files = image_files[train_count:train_count + test_count]
    val_files = image_files[train_count + test_count:]
    
    # คัดลอกไฟล์ไปยังโฟลเดอร์ที่กำหนด
    for split_name, file_list in [("train", train_files), ("test", test_files), ("val", val_files)]:
        for img_path in file_list:
            img_basename = os.path.basename(img_path)
            dst_img = os.path.join(target_dir, split_name, class_name, img_basename)
            copy_file(img_path, dst_img)

print("เสร็จสิ้นการแบ่งข้อมูล!")

# แสดงสรุปจำนวนไฟล์ในแต่ละโฟลเดอร์
print("\nสรุปจำนวนไฟล์:")
for split in ["train", "test", "val"]:
    for class_name in classes:
        img_count = len(glob.glob(os.path.join(target_dir, split, class_name, "*")))
        print(f"{split}/{class_name}: {img_count} ภาพ")
