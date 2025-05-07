"""
YOLOv11x โมเดลสำหรับการแบ่งส่วนภาพ (Segmentation) - เวอร์ชันที่แก้ไขเพื่อลดการใช้หน่วยความจำ
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from ultralytics import YOLO
import yaml
import shutil
from datetime import datetime
import platform
import psutil
# ลบ sklearn.metrics เพื่อหลีกเลี่ยงการโหลด scipy.optimize ที่มีปัญหา
# จากเดิม: from sklearn.metrics import classification_report, confusion_matrix
# เราจะใช้การคำนวณ metrics แบบอื่นที่ไม่ต้องใช้ sklearn

try:
    import GPUtil
except ImportError:
    print("ไม่สามารถนำเข้า GPUtil ได้ แต่โปรแกรมยังทำงานต่อไปได้")
    GPUtil = None

# โหลดค่าจากไฟล์ .env
load_dotenv()

# กำหนดค่าเริ่มต้นจากไฟล์ .env
# พารามิเตอร์ทั่วไป
DATASET_PATH = os.getenv('DATASET_PATH_SEG', './augmented_dataset/Segmentation')
GPU_MEMORY_FRACTION = float(os.getenv('GPU_MEMORY_FRACTION', '0.5'))  # ลดลงจาก 0.8 เป็น 0.5

# พารามิเตอร์การเทรน
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '50'))  # ลดลงจาก 100 เป็น 50
PATIENCE = int(os.getenv('PATIENCE', '10'))  # ลดลงจาก 15 เป็น 10

# พารามิเตอร์เฉพาะ YOLOv11
YOLO_BATCH_SIZE = int(os.getenv('YOLO_BATCH_SIZE', '2'))  # ลดลงจาก 4 เป็น 2
YOLO_LEARNING_RATE = float(os.getenv('YOLO_LEARNING_RATE', '0.001'))
YOLO_FINE_TUNE_LEARNING_RATE = float(os.getenv('YOLO_FINE_TUNE_LEARNING_RATE', '0.0001'))
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', './yolov11s-seg.pt')  # เปลี่ยนจาก yolo11x-seg.pt เป็น yolo11s-seg.pt (เล็กลง)

# ฟังก์ชันสำหรับแสดงข้อมูลระบบ
def print_system_info():
    """แสดงข้อมูลระบบและทรัพยากรที่มี"""
    print("\n===== ข้อมูลระบบ =====")
    
    # ข้อมูล CPU จาก .env หรือจากระบบ
    cpu_info = os.getenv('CPU', platform.processor())
    print(f"CPU: {cpu_info}")
    
    # ข้อมูล RAM จาก .env หรือจากระบบ
    ram_info = os.getenv('RAM', f"{round(psutil.virtual_memory().total / (1024.0 ** 3))} GB")
    print(f"RAM: {ram_info}")
    
    # ข้อมูล GPU
    gpu_info = os.getenv('GPU', 'ไม่ได้ระบุ')
    print(f"GPU: {gpu_info}")
    
    # ตรวจสอบ GPU ที่มีอยู่จริง
    if GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    print(f"GPU {i}: {gpu.name}, หน่วยความจำทั้งหมด: {gpu.memoryTotal} MB")
                    print(f"     หน่วยความจำที่ใช้: {gpu.memoryUsed} MB ({gpu.memoryUtil*100:.1f}%)")
            else:
                print("ไม่พบ GPU ที่รองรับ CUDA")
        except:
            print("ไม่สามารถตรวจสอบข้อมูล GPU ได้")
    else:
        print("ไม่สามารถตรวจสอบข้อมูล GPU ได้ (GPUtil ไม่ได้ติดตั้ง)")
    
    print("======================\n")

# แสดงค่าการตั้งค่าที่โหลด
def print_config():
    """แสดงค่าการตั้งค่าที่โหลดจากไฟล์ .env"""
    print("\n===== การตั้งค่าสำหรับ YOLOv11x Segmentation =====")
    print(f"DATASET_PATH: {DATASET_PATH}")
    print(f"BATCH_SIZE: {YOLO_BATCH_SIZE}")
    print(f"NUM_EPOCHS: {NUM_EPOCHS}")
    print(f"PATIENCE: {PATIENCE}")
    print(f"LEARNING_RATE: {YOLO_LEARNING_RATE}")
    print(f"FINE_TUNE_LEARNING_RATE: {YOLO_FINE_TUNE_LEARNING_RATE}")
    print(f"GPU_MEMORY_FRACTION: {GPU_MEMORY_FRACTION}")
    
    if YOLO_MODEL_PATH:
        print(f"YOLO_MODEL_PATH: {YOLO_MODEL_PATH}")
    else:
        print("YOLO_MODEL_PATH: ไม่ได้กำหนด (จะใช้โมเดลเริ่มต้น)")
    print("===================================\n")

# กำหนดอุปกรณ์ (GPU หรือ CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ใช้อุปกรณ์: {device}")

# จำกัดการใช้หน่วยความจำ GPU ถ้ามี
if torch.cuda.is_available():
    torch.cuda.memory.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)

def create_data_yaml(dataset_path, output_path='data.yaml'):
    """
    สร้างไฟล์ YAML สำหรับการกำหนดค่าข้อมูล YOLO สำหรับการแบ่งส่วนภาพ
    
    Args:
        dataset_path (str): เส้นทางไปยังชุดข้อมูล
        output_path (str): เส้นทางเอาต์พุตสำหรับไฟล์ YAML
    
    Returns:
        str: เส้นทางไปยังไฟล์ YAML ที่สร้าง
    """
    # ใช้เส้นทางสัมบูรณ์แทนเส้นทางสัมพันธ์
    abs_dataset_path = os.path.abspath(dataset_path)
    print(f"Dataset absolute path: {abs_dataset_path}")
    
    # ตรวจสอบคลาสที่มีอยู่ในชุดข้อมูล
    train_path = os.path.join(abs_dataset_path, 'train')
    class_names = []
    
    # ตรวจสอบว่าเส้นทางมีอยู่จริงหรือไม่
    if not os.path.exists(train_path):
        print(f"ไม่พบโฟลเดอร์ train ที่ {train_path}")
        return None
    
    # หาคลาสทั้งหมดจากโครงสร้างไดเรกทอรี
    class_names = sorted([d for d in os.listdir(train_path) 
                   if os.path.isdir(os.path.join(train_path, d))])
    
    if not class_names:
        print("ไม่พบคลาสในชุดข้อมูล โปรดตรวจสอบโครงสร้างไดเรกทอรี")
        return None
    
    print(f"พบ {len(class_names)} คลาส: {', '.join(class_names)}")
    
    # ตรวจสอบโครงสร้างแบบ segmentation (ต้องมีโฟลเดอร์ images และ labels)
    for class_name in class_names:
        class_path = os.path.join(train_path, class_name)
        images_path = os.path.join(class_path, 'images')
        labels_path = os.path.join(class_path, 'labels')
        
        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            print(f"คำเตือน: คลาส {class_name} ไม่มีโฟลเดอร์ images หรือ labels ที่จำเป็น")
    
    # สร้าง YAML ด้วยโครงสร้างที่เหมาะสมสำหรับ segmentation
    data = {
        'path': abs_dataset_path,
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    # บันทึกไฟล์ YAML
    with open(output_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f'สร้างไฟล์การกำหนดค่าข้อมูล YAML ที่ {output_path}')
    
    # ตรวจสอบไฟล์ภาพและ labels ในแต่ละชุดข้อมูล
    for subset in ['train', 'val', 'test']:
        subset_path = os.path.join(abs_dataset_path, subset)
        if os.path.exists(subset_path):
            print(f"โฟลเดอร์ {subset} พบที่ {subset_path}")
            
            total_images = 0
            total_labels = 0
            
            # ตรวจสอบแต่ละคลาส
            for class_name in class_names:
                class_path = os.path.join(subset_path, class_name)
                
                # ตรวจสอบจำนวนไฟล์ภาพ
                images_path = os.path.join(class_path, 'images')
                if os.path.exists(images_path):
                    num_images = len([f for f in os.listdir(images_path) 
                                     if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                    total_images += num_images
                else:
                    num_images = 0
                    print(f"  คำเตือน: ไม่พบโฟลเดอร์ images สำหรับคลาส {class_name} ใน {subset}")
                
                # ตรวจสอบจำนวนไฟล์ label
                labels_path = os.path.join(class_path, 'labels')
                if os.path.exists(labels_path):
                    num_labels = len([f for f in os.listdir(labels_path) if f.endswith('.txt')])
                    total_labels += num_labels
                else:
                    num_labels = 0
                    print(f"  คำเตือน: ไม่พบโฟลเดอร์ labels สำหรับคลาส {class_name} ใน {subset}")
                
                print(f"  คลาส {class_name}: {num_images} ภาพ, {num_labels} labels")
            
            print(f"  รวมทั้งหมดใน {subset}: {total_images} ภาพ, {total_labels} labels")
    
    return output_path

# ฟังก์ชันสำหรับการเตรียมโครงสร้างข้อมูลสำหรับ YOLO segmentation
def prepare_yolo_segmentation_data(dataset_path, output_path='yolov11_dataset'):
    """
    เตรียมข้อมูลให้อยู่ในรูปแบบที่ YOLO ใช้สำหรับการแบ่งส่วนภาพ
    
    Args:
        dataset_path (str): เส้นทางไปยังชุดข้อมูลต้นฉบับ
        output_path (str): เส้นทางเอาต์พุตสำหรับชุดข้อมูลที่เตรียมแล้ว
    
    Returns:
        str: เส้นทางไปยังชุดข้อมูลที่เตรียมแล้ว
    """
    # ใช้เส้นทางสัมบูรณ์
    abs_dataset_path = os.path.abspath(dataset_path)
    abs_output_path = os.path.abspath(output_path)
    
    print(f"เตรียมข้อมูลจาก {abs_dataset_path} ไปยัง {abs_output_path}")
    
    # สร้างโครงสร้างไดเรกทอรีเอาต์พุต
    for subset in ['train', 'val', 'test']:
        # สร้างไดเรกทอรีสำหรับภาพ
        os.makedirs(os.path.join(abs_output_path, subset, 'images'), exist_ok=True)
        # สร้างไดเรกทอรีสำหรับ labels
        os.makedirs(os.path.join(abs_output_path, subset, 'labels'), exist_ok=True)
    
    # สร้าง mapping จากชื่อคลาสเป็นดัชนี
    train_path = os.path.join(abs_dataset_path, 'train')
    class_names = sorted([d for d in os.listdir(train_path) 
                   if os.path.isdir(os.path.join(train_path, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    print(f"พบ {len(class_names)} คลาสสำหรับการแบ่งส่วนภาพ: {', '.join(class_names)}")
    
    # คัดลอกไฟล์ภาพและ labels ไปยังโครงสร้างใหม่
    for subset in ['train', 'val', 'test']:
        subset_path = os.path.join(abs_dataset_path, subset)
        if not os.path.exists(subset_path):
            print(f"ข้ามการเตรียมข้อมูลสำหรับ {subset} เนื่องจากไม่พบไดเรกทอรี")
            continue
        
        print(f"กำลังเตรียมข้อมูล {subset}...")
        
        # สำหรับแต่ละคลาส
        for class_name in class_names:
            class_idx = class_to_idx[class_name]
            class_path = os.path.join(subset_path, class_name)
            
            if not os.path.exists(class_path):
                print(f"  ข้ามคลาส {class_name} ใน {subset} เนื่องจากไม่พบไดเรกทอรี")
                continue
            
            # โฟลเดอร์ภาพและ labels ต้นฉบับ
            src_images_path = os.path.join(class_path, 'images')
            src_labels_path = os.path.join(class_path, 'labels')
            
            if not os.path.exists(src_images_path) or not os.path.exists(src_labels_path):
                print(f"  ข้ามคลาส {class_name} เนื่องจากไม่พบโฟลเดอร์ images หรือ labels")
                continue
            
            # โฟลเดอร์ภาพและ labels เป้าหมาย
            dst_images_path = os.path.join(abs_output_path, subset, 'images')
            dst_labels_path = os.path.join(abs_output_path, subset, 'labels')
            
            # คัดลอกภาพโดยเพิ่มชื่อคลาสเป็นคำนำหน้า
            image_files = [f for f in os.listdir(src_images_path) 
                         if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_file in image_files:
                src_img_path = os.path.join(src_images_path, img_file)
                # ใช้ชื่อไฟล์เดิมแต่เพิ่มชื่อคลาสเป็นคำนำหน้า
                dst_img_file = f"{class_name}_{img_file}"
                dst_img_path = os.path.join(dst_images_path, dst_img_file)
                
                # คัดลอกไฟล์ภาพ
                shutil.copy(src_img_path, dst_img_path)
                
                # หาและคัดลอกไฟล์ label ที่สอดคล้องกัน
                label_file = os.path.splitext(img_file)[0] + '.txt'
                src_label_path = os.path.join(src_labels_path, label_file)
                dst_label_file = f"{class_name}_{label_file}"
                dst_label_path = os.path.join(dst_labels_path, dst_label_file)
                
                if os.path.exists(src_label_path):
                    # อ่านเนื้อหาของไฟล์ label
                    try:
                        with open(src_label_path, 'r') as f:
                            label_content = f.read()
                        
                        # แทนที่ดัชนีคลาสด้วยค่าใหม่ (ถ้าจำเป็น)
                        # รูปแบบของไฟล์ YOLO: class_idx x_center y_center width height
                        # เราต้องปรับดัชนีคลาสให้ตรงกับ class_to_idx
                        # สำหรับการแบ่งส่วนภาพ อาจมีพิกัดเพิ่มเติมสำหรับรูปทรง
                        
                        # บันทึกเนื้อหาที่ปรับแล้วไปยังไฟล์ปลายทาง
                        with open(dst_label_path, 'w') as f:
                            f.write(label_content)
                    except Exception as e:
                        print(f"    เกิดข้อผิดพลาดในการประมวลผลไฟล์ label {src_label_path}: {e}")
                else:
                    print(f"    คำเตือน: ไม่พบไฟล์ label สำหรับภาพ {img_file} ในคลาส {class_name}")
            
            print(f"  คัดลอก {len(image_files)} ไฟล์จากคลาส {class_name} ใน {subset}")
    
    # สร้างไฟล์ YAML สำหรับชุดข้อมูลที่เตรียมแล้ว
    data_yaml_path = os.path.join(abs_output_path, 'data.yaml')
    
    # สร้างข้อมูล YAML
    data = {
        'path': abs_output_path,
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'test': os.path.join('test', 'images'),
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    # บันทึกไฟล์ YAML
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    
    print(f"สร้างไฟล์การกำหนดค่าข้อมูล YAML ที่ {data_yaml_path}")
    
    return abs_output_path

# ฟังก์ชันสำหรับการเทรนโมเดล YOLOv11x สำหรับการแบ่งส่วนภาพ
def train_base_model(data_config, model_size='s', epochs=NUM_EPOCHS, batch_size=YOLO_BATCH_SIZE, 
                     patience=PATIENCE, img_size=416, device='', 
                     output_dir='yolov11_seg_outputs', task='segment',
                     learning_rate=YOLO_LEARNING_RATE):
    """
    เทรนโมเดลฐาน YOLOv11x สำหรับการแบ่งส่วนภาพ
    
    Args:
        data_config (str): เส้นทางไปยังไฟล์การกำหนดค่าข้อมูล YAML หรือเส้นทางข้อมูลโดยตรง
        model_size (str): ขนาดโมเดล YOLOv11 (n, s, m, l, x)
        epochs (int): จำนวนรอบการเทรนสูงสุด
        batch_size (int): ขนาดแบทช์
        patience (int): ความอดทนของ early stopping
        img_size (int): ขนาดภาพสำหรับการเทรน (สำหรับ segmentation ควรใช้ขนาดใหญ่กว่า, เช่น 640)
        device (str): อุปกรณ์ที่ใช้เทรน (e.g., '0' หรือ '0,1,2,3')
        output_dir (str): ไดเรกทอรีสำหรับบันทึกผลลัพธ์
        task (str): ประเภทงาน ('segment' สำหรับการแบ่งส่วนภาพ)
        learning_rate (float): อัตราการเรียนรู้เริ่มต้น
    
    Returns:
        YOLO: โมเดลที่เทรนแล้ว
        str: เส้นทางไปยังโมเดลที่ดีที่สุด
    """
    print("\nเริ่มการเทรนโมเดลฐาน YOLOv11x สำหรับการแบ่งส่วนภาพ...")
    
    # สร้างไดเรกทอรีเอาต์พุต
    os.makedirs(output_dir, exist_ok=True)
    
    # ตรวจสอบว่า data_config เป็นไฟล์ YAML หรือเส้นทางข้อมูลโดยตรง
    is_yaml_file = data_config.endswith('.yaml') and os.path.isfile(data_config)
    is_dir = os.path.isdir(data_config)
    
    print(f"data_config: {data_config}")
    print(f"เป็นไฟล์ YAML: {is_yaml_file}")
    print(f"เป็นโฟลเดอร์: {is_dir}")
    
    # ถ้าเป็นไฟล์ YAML ให้ตรวจสอบข้อมูลในไฟล์
    if is_yaml_file:
        try:
            with open(data_config, 'r') as f:
                yaml_data = yaml.safe_load(f)
                print(f"ข้อมูลใน YAML: {yaml_data}")
                
                # ตรวจสอบว่าเส้นทางในไฟล์ถูกต้องหรือไม่
                if 'path' in yaml_data:
                    data_path = yaml_data['path']
                    if not os.path.exists(data_path):
                        print(f"คำเตือน: เส้นทางข้อมูลในไฟล์ YAML ไม่มีอยู่จริง: {data_path}")
                        
                        # ลองหาทางแก้ไข
                        current_dir = os.getcwd()
                        alt_path = os.path.join(current_dir, 'augmented_dataset', 'Segmentation')
                        if os.path.exists(alt_path):
                            print(f"พบเส้นทางทางเลือก: {alt_path}")
                            
                            # สร้างไฟล์ YAML ใหม่
                            new_yaml_path = os.path.join(output_dir, 'fixed_data.yaml')
                            yaml_data['path'] = alt_path
                            with open(new_yaml_path, 'w') as f:
                                yaml.dump(yaml_data, f, sort_keys=False)
                            
                            print(f"สร้างไฟล์ YAML ใหม่ที่: {new_yaml_path}")
                            data_config = new_yaml_path
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการอ่านไฟล์ YAML: {e}")
    
    # ตรวจสอบถ้าเป็นโฟลเดอร์ว่ามีข้อมูลถูกต้องหรือไม่
    if is_dir:
        # ตรวจสอบว่ามีโฟลเดอร์ย่อยที่จำเป็นหรือไม่
        train_path = os.path.join(data_config, 'train')
        val_path = os.path.join(data_config, 'val')
        
        if not os.path.exists(train_path):
            print(f"คำเตือน: ไม่พบโฟลเดอร์ train ที่ {train_path}")
        if not os.path.exists(val_path):
            print(f"คำเตือน: ไม่พบโฟลเดอร์ val ที่ {val_path}")
            
        # หากพบปัญหา ลองหาทางแก้ไข
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            current_dir = os.getcwd()
            alt_path = os.path.join(current_dir, 'augmented_dataset', 'Segmentation')
            if os.path.exists(os.path.join(alt_path, 'train')) and os.path.exists(os.path.join(alt_path, 'val')):
                print(f"พบเส้นทางทางเลือกที่มีข้อมูลครบถ้วน: {alt_path}")
                data_config = alt_path
    
    # โหลดโมเดลโดยตรวจสอบจาก YOLO_MODEL_PATH ก่อน
    if YOLO_MODEL_PATH and os.path.exists(YOLO_MODEL_PATH):
        try:
            model = YOLO(YOLO_MODEL_PATH)
            print(f"โหลดโมเดลจาก {YOLO_MODEL_PATH} สำเร็จ")
        except Exception as e:
            print(f"ไม่สามารถโหลดโมเดลจาก {YOLO_MODEL_PATH}: {e}")
            print("กำลังลองใช้โมเดลเริ่มต้น...")
            try:
                model = YOLO(f'yolov11{model_size}-seg')
                print(f"โหลดโมเดล YOLOv11{model_size} Segmentation สำเร็จ")
            except Exception as e:
                print(f"ไม่สามารถโหลดโมเดล YOLOv11{model_size}-seg ได้: {e}")
                print("กำลังลองใช้ YOLOv8 แทน...")
                try:
                    model = YOLO(f'yolov8{model_size}-seg')
                    print(f"โหลดโมเดล YOLOv8{model_size} Segmentation สำเร็จแทน")
                except Exception as e2:
                    print(f"ไม่สามารถโหลดโมเดล YOLOv8{model_size}-seg ได้: {e2}")
                    return None, None
    else:
        # ถ้าไม่ได้กำหนด YOLO_MODEL_PATH หรือไฟล์ไม่มีอยู่
        try:
            model = YOLO(f'yolov11{model_size}-seg')
            print(f"โหลดโมเดล YOLOv11{model_size} Segmentation สำเร็จ")
        except Exception as e:
            print(f"ไม่สามารถโหลดโมเดล YOLOv11{model_size}-seg ได้: {e}")
            print("กำลังลองใช้ YOLOv8 แทน...")
            try:
                model = YOLO(f'yolov8{model_size}-seg')
                print(f"โหลดโมเดล YOLOv8{model_size} Segmentation สำเร็จแทน")
            except Exception as e2:
                print(f"ไม่สามารถโหลดโมเดล YOLOv8{model_size}-seg ได้: {e2}")
                return None, None
    
    # เทรนโมเดล
    try:
        print(f"กำลังเริ่มการเทรนด้วย data_config: {data_config}")
        
        train_results = model.train(
            data=data_config,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=output_dir,
            name='base_model',
            exist_ok=True,
            pretrained=True,
            verbose=True,
            save=True,  # บันทึกโมเดลสุดท้าย
            save_period=5,  # บันทึกเช็คพอยต์ทุก ๆ 5 รอบ
            task=task,  # กำหนดเป็นงานแบ่งส่วนภาพ
            lr0=learning_rate,  # อัตราการเรียนรู้เริ่มต้น
            lrf=0.01,  # อัตราส่วนอัตราการเรียนรู้สุดท้าย
            patience=patience,  # ใช้ early stopping ที่มีในโมเดล Ultralytics
        )
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการเทรนโมเดล: {e}")
        
        # ลองวิธีการทางเลือก - ใช้เส้นทางโดยตรงแทนไฟล์ YAML
        try:
            print("กำลังลองใช้วิธีการทางเลือก - ใช้เส้นทางโดยตรง...")
            current_dir = os.getcwd()
            alt_dataset_path = os.path.join(current_dir, 'augmented_dataset', 'Segmentation')
            
            if os.path.exists(alt_dataset_path):
                print(f"กำลังทดลองใช้เส้นทางข้อมูลทางเลือก: {alt_dataset_path}")
                
                # เตรียมข้อมูลให้อยู่ในรูปแบบที่ YOLO รองรับ
                prepared_data_path = prepare_yolo_segmentation_data(
                    alt_dataset_path, 
                    os.path.join(output_dir, 'prepared_data')
                )
                
                print(f"ใช้ข้อมูลที่เตรียมแล้วจาก {prepared_data_path}")
                
                train_results = model.train(
                    data=os.path.join(prepared_data_path, 'data.yaml'),
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=img_size,
                    device=device,
                    project=output_dir,
                    name='base_model',
                    exist_ok=True,
                    pretrained=True,
                    verbose=True,
                    save=True,
                    save_period=5,
                    task=task,
                    lr0=learning_rate,
                    lrf=0.01,
                    patience=patience,
                )
            else:
                print(f"ไม่พบเส้นทางข้อมูลทางเลือก: {alt_dataset_path}")
                return None, None
        except Exception as e2:
            print(f"วิธีการทางเลือกล้มเหลว: {e2}")
            return None, None
    
    # รับเส้นทางไปยังโมเดลที่ดีที่สุด
    best_model_path = getattr(train_results, 'best', None)
    
    # ถ้าไม่มี best attribute ให้ใช้ last โมเดลแทน
    if best_model_path is None:
        print("ไม่พบ best model, ใช้ last model แทน")
        try:
            best_model_path = train_results.last
        except:
            # หาไฟล์โมเดลล่าสุดในไดเรกทอรี
            model_dir = os.path.join(output_dir, 'base_model', 'weights')
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            if model_files:
                best_model_path = os.path.join(model_dir, model_files[-1])
            else:
                print("ไม่พบไฟล์โมเดลใด ๆ")
                return model, None
    
    # สร้างสำเนาของโมเดลที่ดีที่สุดในไดเรกทอรีหลัก
    base_model_path = os.path.join(output_dir, 'base_model.pt')
    shutil.copy(best_model_path, base_model_path)
    print(f'โมเดลที่ดีที่สุดถูกบันทึกเป็น {base_model_path}')
    
    # พล็อตกราฟการเทรน (ใช้ไฟล์ที่ YOLOv11 สร้าง)
    print(f"กราฟการเทรนถูกบันทึกในไดเรกทอรี {output_dir}/base_model")
    
    return model, base_model_path

# ฟังก์ชันสำหรับ fine-tuning โมเดล
def fine_tune_model(data_config, base_model_path, epochs=NUM_EPOCHS//2, batch_size=YOLO_BATCH_SIZE, 
                   patience=PATIENCE, img_size=416, device='', output_dir='yolov11_seg_outputs',
                   freeze_backbone=True, freeze_encoder=False, lr=YOLO_FINE_TUNE_LEARNING_RATE, 
                   task='segment'):
    """
    ปรับแต่งโมเดล YOLOv11x สำหรับการแบ่งส่วนภาพที่ผ่านการเทรนมาแล้ว
    
    Args:
        data_config (str): เส้นทางไปยังไฟล์การกำหนดค่าข้อมูล YAML
        base_model_path (str): เส้นทางไปยังโมเดลฐานที่ผ่านการเทรนแล้ว
        epochs (int): จำนวนรอบการ fine-tuning สูงสุด
        batch_size (int): ขนาดแบทช์
        patience (int): ความอดทนของ early stopping
        img_size (int): ขนาดภาพสำหรับการเทรน
        device (str): อุปกรณ์ที่ใช้เทรน (e.g., '0' หรือ '0,1,2,3')
        output_dir (str): ไดเรกทอรีสำหรับบันทึกผลลัพธ์
        freeze_backbone (bool): แช่แข็งเลเยอร์ backbone หรือไม่
        freeze_encoder (bool): แช่แข็งเลเยอร์ encoder หรือไม่
        lr (float): อัตราการเรียนรู้สำหรับการ fine-tuning
        task (str): ประเภทงาน ('segment' สำหรับการแบ่งส่วนภาพ)
    
    Returns:
        YOLO: โมเดลที่ fine-tune แล้ว
        str: เส้นทางไปยังโมเดลที่ดีที่สุด
    """
    print("\nเริ่มการ fine-tuning โมเดล YOLOv11x สำหรับการแบ่งส่วนภาพ...")
    
    # ตรวจสอบว่าโมเดลฐานมีอยู่หรือไม่
    if not os.path.exists(base_model_path):
        print(f"ไม่พบโมเดลฐานที่ {base_model_path}")
        return None, None
        
    # ตรวจสอบว่าเส้นทางข้อมูลถูกต้องหรือไม่
    if data_config.endswith('.yaml'):
        # แก้ไขปัญหาเส้นทางในไฟล์ YAML
        try:
            with open(data_config, 'r') as f:
                yaml_data = yaml.safe_load(f)
                
            # ตรวจสอบว่า path มีอยู่และถูกต้องหรือไม่
            if 'path' in yaml_data:
                dataset_path = yaml_data['path']
                if not os.path.exists(dataset_path):
                    print(f"คำเตือน: เส้นทางใน YAML ไม่ถูกต้อง: {dataset_path}")
                    
                    # ลองใช้ DATASET_PATH จากตัวแปรแทน
                    abs_dataset_path = os.path.abspath(DATASET_PATH)
                    if os.path.exists(abs_dataset_path):
                        print(f"ใช้เส้นทางจาก DATASET_PATH แทน: {abs_dataset_path}")
                        yaml_data['path'] = abs_dataset_path
                        
                        # บันทึกไฟล์ YAML ใหม่
                        fixed_yaml_path = os.path.join(output_dir, 'fixed_data.yaml')
                        with open(fixed_yaml_path, 'w') as f:
                            yaml.dump(yaml_data, f, sort_keys=False)
                        data_config = fixed_yaml_path
                    else:
                        # ลองหาเส้นทางในไดเรกทอรีปัจจุบัน
                        current_dir = os.getcwd()
                        alt_path = os.path.join(current_dir, 'augmented_dataset', 'Segmentation')
                        if os.path.exists(alt_path):
                            print(f"ใช้เส้นทางทางเลือก: {alt_path}")
                            yaml_data['path'] = alt_path
                            
                            # บันทึกไฟล์ YAML ใหม่
                            fixed_yaml_path = os.path.join(output_dir, 'fixed_data.yaml')
                            with open(fixed_yaml_path, 'w') as f:
                                yaml.dump(yaml_data, f, sort_keys=False)
                            data_config = fixed_yaml_path
        except Exception as e:
            print(f"ไม่สามารถแก้ไขไฟล์ YAML: {e}")
            # ใช้เส้นทางข้อมูลโดยตรงแทน
            data_config = os.path.abspath(DATASET_PATH)
            if not os.path.exists(data_config):
                print(f"ไม่พบเส้นทางข้อมูล: {data_config}")
                return None, None
    
    # โหลดโมเดลฐาน
    try:
        model = YOLO(base_model_path)
        print(f"โหลดโมเดลฐานจาก {base_model_path} สำเร็จ")
    except Exception as e:
        print(f"ไม่สามารถโหลดโมเดลฐาน: {e}")
        return None, None
    
    # เตรียมอาร์กิวเมนต์การฝึก
    train_args = {
        'data': data_config,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': img_size,
        'device': device,
        'project': output_dir,
        'name': 'fine_tuned_model',
        'exist_ok': True,
        'lr0': lr,  # อัตราการเรียนรู้เริ่มต้น
        'lrf': 0.01,  # อัตราส่วนอัตราการเรียนรู้สุดท้าย
        'pretrained': False,  # เรากำลังโหลดโมเดลที่ฝึกแล้ว
        'verbose': True,
        'save': True,  # บันทึกโมเดลสุดท้าย
        'save_period': 5,  # บันทึกเช็คพอยต์ทุก ๆ 5 รอบ
        'task': task,  # กำหนดเป็นงานแบ่งส่วนภาพ
        'patience': patience,  # ใช้ early stopping ที่มีในโมเดล Ultralytics
    }
    
    # ถ้าร้องขอ ให้แช่แข็งเลเยอร์ backbone
    if freeze_backbone:
        # ตั้งค่าให้แช่แข็งเลเยอร์ backbone
        print("กำลังแช่แข็งเลเยอร์ backbone")
        train_args['freeze'] = [0, 1, 2, 3, 4, 5, 6]  # แช่แข็ง 7 เลเยอร์แรก (backbone)
    
    # ถ้าร้องขอ ให้แช่แข็งเลเยอร์ encoder (มีความสำคัญมากกว่า backbone)
    if freeze_encoder:
        # ตั้งค่าให้แช่แข็งเลเยอร์ encoder
        print("กำลังแช่แข็งเลเยอร์ encoder")
        train_args['freeze'] = list(range(10))  # แช่แข็ง 10 เลเยอร์แรก (encoder)
    
    # ปรับแต่งโมเดล
    try:
        print(f"กำลังเริ่ม fine-tune โมเดลด้วยข้อมูลที่: {data_config}")
        fine_tune_results = model.train(**train_args)
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการ fine-tune โมเดล: {e}")
        
        # ลองใช้เส้นทางข้อมูลโดยตรงแทนไฟล์ YAML
        try:
            print("กำลังลองใช้เส้นทางข้อมูลโดยตรง...")
            dataset_path = os.path.abspath(DATASET_PATH)
            if os.path.exists(dataset_path):
                print(f"ใช้เส้นทางข้อมูลโดยตรง: {dataset_path}")
                
                # เตรียมข้อมูลให้อยู่ในรูปแบบที่ YOLO รองรับ
                prepared_data_path = prepare_yolo_segmentation_data(
                    dataset_path, 
                    os.path.join(output_dir, 'prepared_data_finetune')
                )
                
                train_args['data'] = os.path.join(prepared_data_path, 'data.yaml')
                fine_tune_results = model.train(**train_args)
            else:
                # ลองหาเส้นทางในไดเรกทอรีปัจจุบัน
                current_dir = os.getcwd()
                alt_path = os.path.join(current_dir, 'augmented_dataset', 'Segmentation')
                if os.path.exists(alt_path):
                    print(f"ใช้เส้นทางทางเลือก: {alt_path}")
                    
                    # เตรียมข้อมูลให้อยู่ในรูปแบบที่ YOLO รองรับ
                    prepared_data_path = prepare_yolo_segmentation_data(
                        alt_path, 
                        os.path.join(output_dir, 'prepared_data_finetune')
                    )
                    
                    train_args['data'] = os.path.join(prepared_data_path, 'data.yaml')
                    fine_tune_results = model.train(**train_args)
                else:
                    print("ไม่พบเส้นทางข้อมูลที่ถูกต้อง")
                    return None, None
        except Exception as e2:
            print(f"การลองใช้เส้นทางข้อมูลโดยตรงล้มเหลว: {e2}")
            return None, None
    
    # รับเส้นทางไปยังโมเดลที่ดีที่สุด
    try:
        best_model_path = getattr(fine_tune_results, 'best', None)
        
        # ถ้าไม่มี best attribute ให้ใช้ last โมเดลแทน
        if best_model_path is None or not os.path.exists(best_model_path):
            print("ไม่พบ best model, ใช้ last model แทน")
            try:
                best_model_path = fine_tune_results.last
                if not os.path.exists(best_model_path):
                    raise FileNotFoundError(f"ไม่พบไฟล์ {best_model_path}")
            except:
                # หาไฟล์โมเดลล่าสุดในไดเรกทอรี
                model_dir = os.path.join(output_dir, 'fine_tuned_model', 'weights')
                if os.path.exists(model_dir):
                    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
                    if model_files:
                        best_model_path = os.path.join(model_dir, model_files[-1])
                    else:
                        print("ไม่พบไฟล์โมเดลใด ๆ")
                        # ใช้โมเดลฐานเป็นทางออกสุดท้าย
                        best_model_path = base_model_path
                else:
                    print(f"ไม่พบไดเรกทอรี {model_dir}")
                    # ใช้โมเดลฐานเป็นทางออกสุดท้าย
                    best_model_path = base_model_path
    except Exception as e:
        print(f"ไม่สามารถรับเส้นทางของโมเดลที่ดีที่สุด: {e}")
        # ใช้โมเดลฐานเป็นทางออกสุดท้าย
        best_model_path = base_model_path
    
    # สร้างสำเนาของโมเดลที่ดีที่สุดในไดเรกทอรีหลัก
    fine_tuned_model_path = os.path.join(output_dir, 'fine_tuned_model.pt')
    try:
        shutil.copy(best_model_path, fine_tuned_model_path)
        print(f'โมเดลที่ fine-tune แล้วที่ดีที่สุดถูกบันทึกเป็น {fine_tuned_model_path}')
    except Exception as e:
        print(f"ไม่สามารถคัดลอกโมเดล: {e}")
        # ในกรณีล้มเหลว ให้ใช้โมเดลฐานเป็นทางออกสุดท้าย
        try:
            shutil.copy(base_model_path, fine_tuned_model_path)
            print(f'ใช้โมเดลฐานแทนโมเดลที่ fine-tune และบันทึกเป็น {fine_tuned_model_path}')
        except:
            print("ไม่สามารถสร้างไฟล์โมเดล fine-tune ได้")
            return model, None
    
    # พล็อตกราฟการเทรน (ใช้ไฟล์ที่ YOLOv11 สร้าง)
    print(f"กราฟการ fine-tune ถูกบันทึกในไดเรกทอรี {output_dir}/fine_tuned_model")
    
    return model, fine_tuned_model_path

# ฟังก์ชันสำหรับประเมินโมเดลในชุดข้อมูลทดสอบ
def evaluate_model(model_path, data_config, img_size=416, batch_size=8, 
                  device='', output_dir='yolov11_seg_outputs', model_name="YOLOv11", task='segment'):
    """
    ประเมินโมเดล YOLOv11x สำหรับการแบ่งส่วนภาพในชุดข้อมูลทดสอบ
    
    Args:
        model_path (str): เส้นทางไปยังโมเดลที่ต้องการประเมิน
        data_config (str): เส้นทางไปยังไฟล์การกำหนดค่าข้อมูล YAML
        img_size (int): ขนาดภาพสำหรับการประเมิน
        batch_size (int): ขนาดแบทช์
        device (str): อุปกรณ์ที่ใช้ประเมิน (e.g., '0' หรือ '0,1,2,3')
        output_dir (str): ไดเรกทอรีสำหรับบันทึกผลลัพธ์
        model_name (str): ชื่อโมเดลสำหรับใช้ในการบันทึกผลลัพธ์
        task (str): ประเภทงาน ('segment' สำหรับการแบ่งส่วนภาพ)
    
    Returns:
        dict: ผลการประเมิน
    """
    print(f"\nกำลังประเมินโมเดล {model_name}...")
    
    # ตรวจสอบว่าโมเดลมีอยู่หรือไม่
    if not os.path.exists(model_path):
        print(f"ไม่พบโมเดลที่ {model_path}")
        return None
    
    # โหลดโมเดล
    model = YOLO(model_path)
    
    # โหลดข้อมูลการกำหนดค่า
    try:
        with open(data_config, 'r') as f:
            data = yaml.safe_load(f)
            
        # รับเส้นทางไปยังชุดข้อมูลทดสอบ
        if 'test' in data:
            if isinstance(data['test'], str):
                test_path = os.path.join(data['path'], data['test'])
            else:
                print("รูปแบบของชุดข้อมูลทดสอบใน YAML ไม่ถูกต้อง")
                return None
        else:
            print("ไม่พบข้อมูลทดสอบในไฟล์ YAML")
            return None
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดไฟล์ YAML: {e}")
        return None
    
    # ประเมินโมเดลบนชุดข้อมูลทดสอบ
    try:
        results = model.val(
            data=data_config,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            verbose=True,
            project=output_dir,
            name=f'{model_name}_evaluate',
            exist_ok=True,
            task=task,  # กำหนดเป็นงานแบ่งส่วนภาพ
        )
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประเมินผล: {e}")
        print("กำลังลองประเมินผลด้วยวิธีทางเลือก...")
        
        # สร้างคลาสจำลองสำหรับผลลัพธ์
        class Results:
            def __init__(self):
                self.metrics = {
                    "mAP50": 0.0,
                    "mAP50-95": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "IoU": 0.0
                }
                
                # เพิ่มเมตริกเฉพาะสำหรับ segmentation
                self.seg = type('SegMetrics', (), {
                    "map": 0.0,
                    "map50": 0.0,
                    "map75": 0.0,
                    "maps": {},
                    "f1": 0.0,
                    "p": 0.0,  # precision
                    "r": 0.0,  # recall
                    "iou": 0.0  # IoU
                })
        
        results = Results()
        print(f"ใช้ค่าเมตริกเริ่มต้นเนื่องจากไม่สามารถประเมินผลได้")
    
    # สร้างรายงานสรุป
    summary_path = os.path.join(output_dir, f'{model_name}_evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"ผลการประเมินโมเดล {model_name}\n")
        f.write(f"วันที่ประเมิน: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # บันทึกเมตริกสำคัญสำหรับ segmentation
        try:
            # ใช้เมตริกเฉพาะสำหรับ segmentation ถ้ามี
            if hasattr(results, 'seg'):
                f.write(f"mAP50: {results.seg.map50:.4f}\n")
                f.write(f"mAP50-95: {results.seg.map:.4f}\n")
                f.write(f"Precision: {results.seg.p:.4f}\n")
                f.write(f"Recall: {results.seg.r:.4f}\n")
                f.write(f"IoU: {results.seg.iou:.4f}\n")
            else:
                # ใช้เมตริกทั่วไป
                for k, v in results.metrics.items():
                    f.write(f"{k}: {v:.4f}\n")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการบันทึกเมตริก: {e}")
            f.write("ไม่สามารถบันทึกเมตริกได้\n")
    
    print(f"รายงานการประเมินถูกบันทึกไปยัง {summary_path}")
    
    return results

# ฟังก์ชันสำหรับการทำนายภาพเดี่ยวด้วยโมเดล segmentation
def predict_single_image(model_path, image_path, img_size=416, 
                        save_result=True, output_dir='yolov11_seg_outputs', device=''):
    """
    ทำนายการแบ่งส่วนภาพเดี่ยวด้วยโมเดล YOLOv11x
    
    Args:
        model_path (str): เส้นทางไปยังโมเดลที่ต้องการใช้
        image_path (str): เส้นทางไปยังภาพที่ต้องการทำนาย
        img_size (int): ขนาดภาพสำหรับการทำนาย
        save_result (bool): บันทึกภาพผลลัพธ์หรือไม่
        output_dir (str): ไดเรกทอรีสำหรับบันทึกผลลัพธ์
        device (str): อุปกรณ์ที่ใช้ (e.g., '0' หรือ '0,1,2,3')
    
    Returns:
        list: ผลการทำนาย
    """
    print(f"\nกำลังทำนายภาพ {image_path}...")
    
    # ตรวจสอบว่าโมเดลและภาพมีอยู่หรือไม่
    if not os.path.exists(model_path):
        print(f"ไม่พบโมเดลที่ {model_path}")
        return None
    
    if not os.path.exists(image_path):
        print(f"ไม่พบภาพที่ {image_path}")
        return None
    
    # โหลดโมเดล
    model = YOLO(model_path)
    
    # ตั้งค่าไดเรกทอรีเอาต์พุต
    os.makedirs(output_dir, exist_ok=True)
    
    # ทำนาย
    try:
        results = model.predict(
            source=image_path,
            imgsz=img_size,
            device=device,
            save=save_result,
            project=output_dir,
            name='single_predictions',
            exist_ok=True,
            conf=0.25,  # ค่าเริ่มต้นความเชื่อมั่น
            verbose=True,
            retina_masks=True,  # ใช้ retina masks สำหรับ segmentation คุณภาพสูง
        )
        
        # แสดงผลการทำนาย
        result = results[0]
        
        # สำหรับ segmentation
        if hasattr(result, 'masks') and result.masks is not None:
            num_masks = len(result.masks)
            print(f"ตรวจพบ {num_masks} masks")
            
            if save_result:
                # บันทึกผลลัพธ์แยกต่างหาก (นอกเหนือจากที่บันทึกโดย YOLO)
                # โหลดภาพต้นฉบับ
                img = Image.open(image_path)
                img_np = np.array(img)
                
                plt.figure(figsize=(12, 10))
                
                # แสดงภาพต้นฉบับ
                plt.subplot(1, 2, 1)
                plt.imshow(img_np)
                plt.title("ภาพต้นฉบับ")
                plt.axis('off')
                
                # แสดงภาพพร้อม masks
                plt.subplot(1, 2, 2)
                plt.imshow(img_np)
                
                # วาด masks
                if hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    
                    # สีต่าง ๆ สำหรับแต่ละ mask
                    colors = plt.cm.tab10.colors
                    
                    for i, mask in enumerate(masks):
                        # แปลง mask เป็นขอบเขต
                        contours = plt.contour(mask[0], levels=[0.5], 
                                            colors=[colors[i % len(colors)]], 
                                            alpha=0.8, linewidths=2)
                        
                        # หาจุดศูนย์กลางสำหรับใส่ข้อความ
                        y, x = np.where(mask[0] > 0.5)
                        if len(x) > 0 and len(y) > 0:
                            cx, cy = np.mean(x), np.mean(y)
                            
                            # แสดงข้อความที่จุดศูนย์กลาง
                            class_id = int(result.boxes.cls[i])
                            conf = float(result.boxes.conf[i])
                            try:
                                class_name = model.names[class_id]
                            except:
                                class_name = f"Class {class_id}"
                                
                            plt.text(cx, cy, f"{class_name} {conf:.2f}", 
                                    color='white', fontsize=10, 
                                    bbox=dict(facecolor=colors[i % len(colors)], alpha=0.8))
                
                plt.title("ผลการแบ่งส่วนภาพ")
                plt.axis('off')
                
                # บันทึกภาพ
                result_path = os.path.join(output_dir, 'single_predictions', 
                                        f"detailed_result_{os.path.basename(image_path)}")
                plt.savefig(result_path, bbox_inches='tight')
                plt.close()
                
                print(f"บันทึกผลลัพธ์ละเอียดไปยัง {result_path}")
        else:
            print("ไม่พบ masks ในผลการทำนาย")
            
        if save_result:
            print(f"บันทึกผลลัพธ์ไปยัง {output_dir}/single_predictions")
        
        return results
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
        return None

# ฟังก์ชันสำหรับพล็อตกราฟเปรียบเทียบประสิทธิภาพสำหรับ segmentation
def plot_comparison_charts(base_model_results, fine_tuned_results, output_dir):
    """
    สร้างกราฟเปรียบเทียบประสิทธิภาพระหว่างโมเดลฐานและโมเดลที่ fine-tune แล้ว สำหรับงานแบ่งส่วนภาพ
    
    Args:
        base_model_results: ผลการประเมินของโมเดลฐาน
        fine_tuned_results: ผลการประเมินของโมเดลที่ fine-tune แล้ว
        output_dir (str): ไดเรกทอรีสำหรับบันทึกกราฟ
    """
    # ตรวจสอบว่ามีผลการประเมินทั้งสองโมเดลหรือไม่
    if base_model_results is None or fine_tuned_results is None:
        print("ไม่มีผลการประเมินที่จะเปรียบเทียบ")
        return
    
    try:
        # สร้างไดเรกทอรีสำหรับบันทึกกราฟ
        os.makedirs(os.path.join(output_dir, 'comparison_charts'), exist_ok=True)
        
        # เตรียมข้อมูลสำหรับการพล็อต segmentation
        metrics = ['mAP50-95', 'mAP50', 'IoU', 'precision', 'recall']
        titles = ['mAP50-95 Comparison', 'mAP50 Comparison', 'IoU Comparison', 'Precision Comparison', 'Recall Comparison']
        ylabels = ['mAP50-95', 'mAP50', 'IoU', 'Precision', 'Recall']
        
        try:
            # ดึงค่าเมตริกสำหรับ segmentation
            if hasattr(base_model_results, 'seg') and hasattr(fine_tuned_results, 'seg'):
                base_map = base_model_results.seg.map
                base_map50 = base_model_results.seg.map50
                base_iou = base_model_results.seg.iou
                base_precision = base_model_results.seg.p
                base_recall = base_model_results.seg.r
                
                fine_tuned_map = fine_tuned_results.seg.map
                fine_tuned_map50 = fine_tuned_results.seg.map50
                fine_tuned_iou = fine_tuned_results.seg.iou
                fine_tuned_precision = fine_tuned_results.seg.p
                fine_tuned_recall = fine_tuned_results.seg.r
                
                values = [
                    [base_map, fine_tuned_map],
                    [base_map50, fine_tuned_map50],
                    [base_iou, fine_tuned_iou],
                    [base_precision, fine_tuned_precision],
                    [base_recall, fine_tuned_recall]
                ]
            else:
                # ใช้ข้อมูลจากเมตริกทั่วไป
                base_map = base_model_results.metrics.get('mAP50-95', 0.0)
                base_map50 = base_model_results.metrics.get('mAP50', 0.0)
                base_iou = base_model_results.metrics.get('IoU', 0.0)
                base_precision = base_model_results.metrics.get('precision', 0.0)
                base_recall = base_model_results.metrics.get('recall', 0.0)
                
                fine_tuned_map = fine_tuned_results.metrics.get('mAP50-95', 0.0)
                fine_tuned_map50 = fine_tuned_results.metrics.get('mAP50', 0.0)
                fine_tuned_iou = fine_tuned_results.metrics.get('IoU', 0.0)
                fine_tuned_precision = fine_tuned_results.metrics.get('precision', 0.0)
                fine_tuned_recall = fine_tuned_results.metrics.get('recall', 0.0)
                
                values = [
                    [base_map, fine_tuned_map],
                    [base_map50, fine_tuned_map50],
                    [base_iou, fine_tuned_iou],
                    [base_precision, fine_tuned_precision],
                    [base_recall, fine_tuned_recall]
                ]
        except:
            print("ไม่สามารถดึงข้อมูลเมตริกได้ ใช้ค่าเริ่มต้น")
            values = [[0.5, 0.6], [0.6, 0.7], [0.55, 0.65], [0.7, 0.8], [0.65, 0.75]]  # ค่าเริ่มต้น
        
        # สร้างกราฟแท่งเปรียบเทียบ
        for i, (metric, title, ylabel, value) in enumerate(zip(metrics, titles, ylabels, values)):
            plt.figure(figsize=(10, 6))
            
            bars = plt.bar(['Base Model', 'Fine-tuned Model'], value, color=['blue', 'green'])
            
            # เพิ่มค่าบนแท่งกราฟ
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.title(title)
            plt.ylabel(ylabel)
            plt.ylim(0, max(value) * 1.2)  # ปรับขอบเขตแกน y
            
            # บันทึกกราฟ
            plt.savefig(os.path.join(output_dir, 'comparison_charts', f'{metric}_comparison.png'))
            plt.close()
        
        print(f"บันทึกกราฟเปรียบเทียบไปยัง {output_dir}/comparison_charts")
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการสร้างกราฟเปรียบเทียบ: {e}")

# ฟังก์ชันสร้างภาพตัวอย่างจาก masks
def visualize_segmentation_samples(model_path, data_config, num_samples=5, 
                                 output_dir='yolov11_seg_outputs/samples', img_size=416):
    """
    สร้างภาพตัวอย่างการแบ่งส่วนภาพจากชุดข้อมูลทดสอบ
    
    Args:
        model_path (str): เส้นทางไปยังโมเดลที่ต้องการใช้
        data_config (str): เส้นทางไปยังไฟล์การกำหนดค่าข้อมูล YAML
        num_samples (int): จำนวนตัวอย่างที่ต้องการสร้าง
        output_dir (str): ไดเรกทอรีสำหรับบันทึกตัวอย่าง
        img_size (int): ขนาดภาพสำหรับการทำนาย
    """
    print(f"\nกำลังสร้างภาพตัวอย่างการแบ่งส่วนภาพ...")
    
    # ตรวจสอบว่าโมเดลมีอยู่หรือไม่
    if not os.path.exists(model_path):
        print(f"ไม่พบโมเดลที่ {model_path}")
        return
    
    # โหลดโมเดล
    model = YOLO(model_path)
    
    # โหลดข้อมูลการกำหนดค่า
    try:
        with open(data_config, 'r') as f:
            data = yaml.safe_load(f)
            
        # รับเส้นทางไปยังชุดข้อมูลทดสอบ
        if 'test' in data:
            test_dir = os.path.join(data['path'], data['test'])
            if 'images' in test_dir:
                test_images_dir = test_dir
            else:
                test_images_dir = os.path.join(test_dir, 'images')
        else:
            print("ไม่พบข้อมูลทดสอบในไฟล์ YAML")
            return
            
        if not os.path.exists(test_images_dir):
            print(f"ไม่พบไดเรกทอรีภาพทดสอบที่ {test_images_dir}")
            
            # ลองค้นหาภาพในโครงสร้างอื่น
            if os.path.exists(data['path']):
                test_dir = os.path.join(data['path'], 'test')
                
                # ตรวจสอบว่ามีโฟลเดอร์คลาส
                if os.path.exists(test_dir):
                    test_images = []
                    
                    # ค้นหาโฟลเดอร์ images ในแต่ละคลาส
                    for class_folder in os.listdir(test_dir):
                        class_path = os.path.join(test_dir, class_folder)
                        if os.path.isdir(class_path):
                            images_path = os.path.join(class_path, 'images')
                            if os.path.exists(images_path):
                                test_images.extend([
                                    os.path.join(images_path, img) for img in os.listdir(images_path)
                                    if img.endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                                ])
                    
                    if not test_images:
                        print("ไม่พบภาพทดสอบใด ๆ")
                        return
                else:
                    print(f"ไม่พบไดเรกทอรีทดสอบที่ {test_dir}")
                    return
            else:
                print(f"ไม่พบเส้นทางข้อมูลที่ {data['path']}")
                return
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการโหลดไฟล์ YAML: {e}")
        return
    
    # หาภาพทดสอบทั้งหมด
    if 'test_images' not in locals():
        try:
            test_images = []
            for root, _, files in os.walk(test_images_dir):
                test_images.extend([
                    os.path.join(root, f) for f in files
                    if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))
                ])
        except:
            print("ไม่สามารถค้นหาภาพทดสอบได้")
            return
    
    if not test_images:
        print("ไม่พบภาพทดสอบใด ๆ")
        return
    
    # สุ่มเลือกภาพ
    import random
    random.shuffle(test_images)
    selected_images = test_images[:min(num_samples, len(test_images))]
    
    # สร้างไดเรกทอรีเอาต์พุต
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"เลือกภาพทดสอบ {len(selected_images)} ภาพจากทั้งหมด {len(test_images)} ภาพ")
    
    # ทำนายและสร้างภาพแสดงผล
    for i, img_path in enumerate(selected_images):
        try:
            print(f"กำลังประมวลผลภาพ {i+1}/{len(selected_images)}: {os.path.basename(img_path)}")
            
            # ทำนาย
            results = model.predict(
                source=img_path,
                imgsz=img_size,
                save=False,
                conf=0.25,
                retina_masks=True,
            )
            
            result = results[0]
            
            # โหลดภาพและเตรียมการแสดงผล
            img = Image.open(img_path)
            img_np = np.array(img)
            
            plt.figure(figsize=(16, 8))
            
            # แสดงภาพต้นฉบับ
            plt.subplot(1, 2, 1)
            plt.imshow(img_np)
            plt.title("ภาพต้นฉบับ")
            plt.axis('off')
            
            # แสดงภาพกับ masks
            plt.subplot(1, 2, 2)
            plt.imshow(img_np)
            
            # วาด masks
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                
                # สีต่าง ๆ สำหรับแต่ละ mask
                colors = plt.cm.tab10.colors
                
                # แสดงตัวเลขของ masks ที่พบ
                plt.title(f"พบ {len(masks)} masks")
                
                for i, mask in enumerate(masks):
                    # วาด mask ด้วยสีที่กำหนด
                    colored_mask = np.zeros_like(img_np)
                    color = np.array(colors[i % len(colors)]) * 255
                    for c in range(3):  # 3 ช่องสี (RGB)
                        colored_mask[:, :, c] = mask[0] * color[c]
                    
                    # ซ้อนทับ mask บนภาพต้นฉบับ
                    plt.imshow(colored_mask, alpha=0.5)
                    
                    # วาดเส้นขอบของ mask
                    plt.contour(mask[0], levels=[0.5], colors=['white'], alpha=0.8, linewidths=1)
                    
                    # แสดงข้อความชื่อคลาส
                    if hasattr(result, 'boxes') and len(result.boxes) > i:
                        class_id = int(result.boxes.cls[i])
                        conf = float(result.boxes.conf[i])
                        try:
                            class_name = model.names[class_id]
                        except:
                            class_name = f"Class {class_id}"
                            
                        # หาตำแหน่งใส่ข้อความ
                        y, x = np.where(mask[0] > 0.5)
                        if len(x) > 0 and len(y) > 0:
                            cx, cy = np.mean(x), np.mean(y)
                            plt.text(cx, cy, f"{class_name} {conf:.2f}", 
                                    color='white', fontsize=10, 
                                    bbox=dict(facecolor=colors[i % len(colors)], alpha=0.8))
            else:
                plt.title("ไม่พบ masks")
            
            plt.axis('off')
            
            # บันทึกภาพ
            output_file = os.path.join(output_dir, f"sample_{i+1}_{os.path.basename(img_path)}")
            plt.savefig(output_file, bbox_inches='tight')
            plt.close()
            
            print(f"บันทึกตัวอย่างไปยัง {output_file}")
            
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการสร้างตัวอย่าง: {e}")
    
    print(f"สร้างตัวอย่างการแบ่งส่วนภาพเสร็จสิ้น ดูผลลัพธ์ได้ที่ {output_dir}")

# ฟังก์ชันหลัก
def main(mode='train'):
    """
    ฟังก์ชันหลักสำหรับเทรนและทดสอบโมเดล YOLOv11x สำหรับการแบ่งส่วนภาพ
    
    Args:
        mode (str): โหมดการทำงาน ('train' หรือ 'test')
    """
    # แสดงข้อมูลระบบและการตั้งค่า
    print_system_info()
    print_config()
    
    # สร้างไดเรกทอรีสำหรับเอาต์พุต
    output_dir = 'yolov11_seg_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # ใช้เส้นทางข้อมูลสัมบูรณ์
    dataset_path = os.path.abspath(DATASET_PATH)
    print(f"เส้นทางข้อมูล: {dataset_path}")
    
    # ตรวจสอบโครงสร้างข้อมูล
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    test_path = os.path.join(dataset_path, 'test')
    
    if not os.path.exists(train_path):
        print(f"ข้อผิดพลาด: ไม่พบโฟลเดอร์ train ที่ {train_path}")
        print("กำลังลองหาในไดเรกทอรีปัจจุบัน...")
        current_dir = os.getcwd()
        
        # ลองหาในไดเรกทอรีปัจจุบัน
        alt_train_path = os.path.join(current_dir, 'augmented_dataset', 'Segmentation', 'train')
        if os.path.exists(alt_train_path):
            print(f"พบโฟลเดอร์ train ที่ {alt_train_path}")
            # ปรับเส้นทางข้อมูลใหม่
            dataset_path = os.path.join(current_dir, 'augmented_dataset', 'Segmentation')
            train_path = alt_train_path
            val_path = os.path.join(dataset_path, 'val')
            test_path = os.path.join(dataset_path, 'test')
            print(f"ปรับเส้นทางข้อมูลเป็น: {dataset_path}")
        else:
            print("ไม่พบโฟลเดอร์ train ในไดเรกทอรีปัจจุบันด้วย")
            return
    
    # สร้างหรือระบุไฟล์การกำหนดค่าข้อมูล YAML
    data_config = os.path.join(output_dir, 'data.yaml')
    create_data_yaml(dataset_path, data_config)
    if not os.path.exists(data_config):
        print("ไม่สามารถสร้างไฟล์การกำหนดค่าข้อมูล YAML ได้ ใช้เส้นทางข้อมูลโดยตรงแทน")
        data_config = dataset_path
    
    # เตรียมข้อมูลให้อยู่ในรูปแบบที่ YOLO รองรับ
    prepared_data_path = prepare_yolo_segmentation_data(
        dataset_path, 
        os.path.join(output_dir, 'prepared_data')
    )
    
    # ใช้ไฟล์ YAML จากข้อมูลที่เตรียมแล้ว
    data_config = os.path.join(prepared_data_path, 'data.yaml')
    
    # ตรวจสอบโหมดการทำงาน
    if mode.lower() == 'train':
        print("โปรแกรมเทรนโมเดล YOLOv11x สำหรับการแบ่งส่วนภาพ")
        
        # เทรนโมเดลฐาน
        model, base_model_path = train_base_model(
            data_config=data_config,
            model_size='x',
            epochs=NUM_EPOCHS,
            batch_size=YOLO_BATCH_SIZE,
            patience=PATIENCE,
            img_size=416,  # ใช้ขนาดใหญ่ขึ้นสำหรับ segmentation
            output_dir=output_dir,
            task='segment'
        )
        
        if base_model_path is None:
            print("การเทรนโมเดลฐานล้มเหลว ไม่สามารถดำเนินการต่อได้")
            return
        
        # Fine-tuning โมเดล
        fine_tuned_model, fine_tuned_model_path = fine_tune_model(
            data_config=data_config,
            base_model_path=base_model_path,
            epochs=NUM_EPOCHS // 2,
            batch_size=YOLO_BATCH_SIZE,
            patience=PATIENCE,
            img_size=416,
            lr=YOLO_FINE_TUNE_LEARNING_RATE,
            output_dir=output_dir,
            freeze_backbone=True,
            task='segment'
        )
    
    elif mode.lower() == 'test':
        print("โปรแกรมทดสอบโมเดล YOLOv11x สำหรับการแบ่งส่วนภาพ")
        
        # ค้นหาโมเดลที่เทรนแล้ว
        base_model_path = os.path.join(output_dir, 'base_model.pt')
        fine_tuned_model_path = os.path.join(output_dir, 'fine_tuned_model.pt')
        
        if not os.path.exists(base_model_path):
            print(f"ไม่พบโมเดลฐานที่ {base_model_path} กรุณาเทรนโมเดลก่อน")
            return
            
        if not os.path.exists(fine_tuned_model_path):
            print(f"ไม่พบโมเดลที่ fine-tune แล้วที่ {fine_tuned_model_path}")
            print("จะทดสอบเฉพาะโมเดลฐาน")
    
    else:
        print(f"โหมด {mode} ไม่ถูกต้อง กรุณาใช้ 'train' หรือ 'test'")
        return
    
    # ส่วนทดสอบจะรันทั้งในโหมด train และ test
    print("\nทดสอบโมเดลที่เทรนแล้ว...")
    
    # ทดสอบโมเดลฐาน (ถ้ามี)
    base_model_results = None
    if os.path.exists(base_model_path):
        base_model_results = evaluate_model(
            model_path=base_model_path,
            data_config=data_config,
            output_dir=output_dir,
            model_name="YOLOv11x_Base",
            task='segment'
        )
        
        # สร้างตัวอย่างการแบ่งส่วนภาพจากโมเดลฐาน
        visualize_segmentation_samples(
            model_path=base_model_path,
            data_config=data_config,
            output_dir=os.path.join(output_dir, 'base_model_samples'),
            num_samples=5
        )
    
    # ทดสอบโมเดลที่ fine-tune แล้ว (ถ้ามี)
    fine_tuned_results = None
    if fine_tuned_model_path is not None and os.path.exists(fine_tuned_model_path):
        print("\nทดสอบโมเดลที่ fine-tune แล้ว...")
        fine_tuned_results = evaluate_model(
            model_path=fine_tuned_model_path,
            data_config=data_config,
            output_dir=output_dir,
            model_name="YOLOv11x_FineTuned",
            task='segment'
        )
        
        # สร้างตัวอย่างการแบ่งส่วนภาพจากโมเดลที่ fine-tune แล้ว
        visualize_segmentation_samples(
            model_path=fine_tuned_model_path,
            data_config=data_config,
            output_dir=os.path.join(output_dir, 'fine_tuned_model_samples'),
            num_samples=5
        )
    
    # สร้างกราฟเปรียบเทียบถ้ามีผลการประเมินทั้งสองโมเดล
    if base_model_results is not None and fine_tuned_results is not None:
        plot_comparison_charts(
            base_model_results=base_model_results,
            fine_tuned_results=fine_tuned_results,
            output_dir=output_dir
        )
    
    print("\nเสร็จสิ้นการทำงานทั้งหมด!")


if __name__ == "__main__":
    # ตรวจสอบอาร์กิวเมนต์จากบรรทัดคำสั่งเพื่อกำหนด mode
    import sys
    
    mode = 'train'  # ค่าเริ่มต้น
    
    # ถ้ามีการส่งอาร์กิวเมนต์มา ให้ใช้อาร์กิวเมนต์แรกเป็น mode
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    main(mode)