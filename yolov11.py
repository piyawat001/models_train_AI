"""
YOLOv11x โมเดลสำหรับการจำแนกประเภทภาพ (Classification)
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
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import platform
import psutil
import GPUtil

# โหลดค่าจากไฟล์ .env
load_dotenv()

# กำหนดค่าเริ่มต้นจากไฟล์ .env
# พารามิเตอร์ทั่วไป
DATASET_PATH = os.getenv('DATASET_PATH', './augmented_dataset/Classification')
GPU_MEMORY_FRACTION = float(os.getenv('GPU_MEMORY_FRACTION', '0.8'))

# พารามิเตอร์การเทรน
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '100'))
PATIENCE = int(os.getenv('PATIENCE', '15'))

# พารามิเตอร์เฉพาะ YOLOv11
YOLO_BATCH_SIZE = int(os.getenv('YOLO_BATCH_SIZE', '16'))
YOLO_LEARNING_RATE = float(os.getenv('YOLO_LEARNING_RATE', '0.001'))
YOLO_FINE_TUNE_LEARNING_RATE = float(os.getenv('YOLO_FINE_TUNE_LEARNING_RATE', '0.0001'))
YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', '')  # ถ้าไม่ได้กำหนดจะใช้โมเดลเริ่มต้น

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
    
    print("======================\n")

# แสดงค่าการตั้งค่าที่โหลด
def print_config():
    """แสดงค่าการตั้งค่าที่โหลดจากไฟล์ .env"""
    print("\n===== การตั้งค่าสำหรับ YOLOv11x =====")
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

# คลาส EarlyStopping สำหรับ YOLOv11 - เก็บไว้สำหรับการอ้างอิงแต่จะไม่ใช้
class EarlyStoppingCallback:
    """
    YOLOv11 early stopping callback เพื่อหยุดการฝึกเมื่อเมตริกไม่มีการปรับปรุง
    """
    
    def __init__(self, patience=PATIENCE, min_delta=0, monitor='metrics/accuracy_top1'):
        """
        กำหนดค่าเริ่มต้นสำหรับ early stopping callback
        
        Args:
            patience (int): จำนวนรอบที่ไม่มีการปรับปรุงก่อนที่จะหยุดการฝึก
            min_delta (float): การเปลี่ยนแปลงขั้นต่ำในเมตริกที่ตรวจสอบเพื่อถือว่ามีการปรับปรุง
            monitor (str): เมตริกที่ใช้ตรวจสอบการปรับปรุง (ใช้ accuracy_top1 สำหรับการจำแนกประเภท)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_value = None
        self.counter = 0
        self.training_finished = False
    
    def on_train_epoch_end(self, trainer):
        """ตรวจสอบว่าควรหยุดการฝึกหลังจากแต่ละรอบหรือไม่"""
        # รับค่าเมตริกปัจจุบัน
        metrics = trainer.metrics
        current_value = metrics.get(self.monitor, 0)
        
        # แสดงค่าปัจจุบันเพื่อการดีบัก
        print(f'ค่า {self.monitor} ปัจจุบัน: {current_value:.6f}')
        
        # กำหนดค่าที่ดีที่สุดในรอบแรก
        if self.best_value is None:
            self.best_value = current_value
            return
        
        # ตรวจสอบว่ามีการปรับปรุงหรือไม่
        if current_value > self.best_value + self.min_delta:
            self.best_value = current_value
            self.counter = 0  # รีเซ็ตตัวนับ
        else:
            self.counter += 1
            print(f'EarlyStopping: ไม่มีการปรับปรุงเป็นเวลา {self.counter} รอบ ค่าที่ดีที่สุด: {self.best_value:.6f}')
            
            if self.counter >= self.patience:
                print(f'EarlyStopping: หยุดการฝึกที่รอบ {trainer.epoch}')
                self.training_finished = True
                trainer.epoch = trainer.epochs + 1  # จบการฝึก
                trainer.model.model.stop = True     # หยุดการฝึก
    
    def on_train_end(self, trainer):
        """เรียกเมื่อการฝึกสิ้นสุด"""
        if self.training_finished:
            print(f'การฝึกหยุดลงเนื่องจาก early stopping. ค่า {self.monitor} ที่ดีที่สุด: {self.best_value:.6f}')


def create_data_yaml(dataset_path, output_path='data.yaml'):
    """
    สร้างไฟล์ YAML สำหรับการกำหนดค่าข้อมูล YOLO สำหรับการจำแนกประเภท
    
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
    
    # หาคลาสทั้งหมดจากโครงสร้างไดเรกทอรี (1 โฟลเดอร์ = 1 คลาส)
    class_names = sorted([d for d in os.listdir(train_path) 
                   if os.path.isdir(os.path.join(train_path, d))])
    
    if not class_names:
        print("ไม่พบคลาสในชุดข้อมูล โปรดตรวจสอบโครงสร้างไดเรกทอรี")
        return None
    
    print(f"พบ {len(class_names)} คลาส: {', '.join(class_names)}")
    
    # สร้างข้อมูล YAML
    data = {
        'path': abs_dataset_path,  # ใช้เส้นทางสัมบูรณ์
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
    
    # ตรวจสอบเพิ่มเติมว่าไดเรกทอรีย่อยมีอยู่จริง
    for subdir in ['train', 'val', 'test']:
        subdir_path = os.path.join(abs_dataset_path, subdir)
        if os.path.exists(subdir_path):
            print(f"โฟลเดอร์ {subdir} พบที่ {subdir_path}")
            img_count = sum(len(os.listdir(os.path.join(subdir_path, class_name))) 
                           for class_name in class_names 
                           if os.path.isdir(os.path.join(subdir_path, class_name)))
            print(f"  พบภาพทั้งหมด {img_count} ภาพใน {subdir}")
        else:
            print(f"คำเตือน: ไม่พบโฟลเดอร์ {subdir} ที่ {subdir_path}")
    
    return output_path

# ฟังก์ชันสำหรับการเทรนโมเดลฐาน - แก้ไขแล้ว
def train_base_model(data_config, model_size='x', epochs=NUM_EPOCHS, batch_size=YOLO_BATCH_SIZE, 
                     patience=PATIENCE, img_size=224, device='', 
                     output_dir='yolov11_outputs', task='classify',
                     learning_rate=YOLO_LEARNING_RATE):
    """
    เทรนโมเดลฐาน YOLOv11x สำหรับการจำแนกประเภท
    
    Args:
        data_config (str): เส้นทางไปยังไฟล์การกำหนดค่าข้อมูล YAML หรือเส้นทางข้อมูลโดยตรง
        model_size (str): ขนาดโมเดล YOLOv11 (n, s, m, l, x)
        epochs (int): จำนวนรอบการเทรนสูงสุด
        batch_size (int): ขนาดแบทช์
        patience (int): ความอดทนของ early stopping
        img_size (int): ขนาดภาพสำหรับการเทรน
        device (str): อุปกรณ์ที่ใช้เทรน (e.g., '0' หรือ '0,1,2,3')
        output_dir (str): ไดเรกทอรีสำหรับบันทึกผลลัพธ์
        task (str): ประเภทงาน ('classify' สำหรับการจำแนกประเภท)
        learning_rate (float): อัตราการเรียนรู้เริ่มต้น
    
    Returns:
        YOLO: โมเดลที่เทรนแล้ว
        str: เส้นทางไปยังโมเดลที่ดีที่สุด
    """
    print("\nเริ่มการเทรนโมเดลฐาน YOLOv11x สำหรับการจำแนกประเภท...")
    
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
                        alt_path = os.path.join(current_dir, 'augmented_dataset', 'Classification')
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
        train_dir = os.path.join(data_config, 'train')
        val_dir = os.path.join(data_config, 'val')
        
        if not os.path.exists(train_dir):
            print(f"คำเตือน: ไม่พบโฟลเดอร์ train ที่ {train_dir}")
        if not os.path.exists(val_dir):
            print(f"คำเตือน: ไม่พบโฟลเดอร์ val ที่ {val_dir}")
            
        # หากพบปัญหา ลองหาทางแก้ไข
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            current_dir = os.getcwd()
            alt_path = os.path.join(current_dir, 'augmented_dataset', 'Classification')
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
                model = YOLO(f'yolov11{model_size}.cls')
                print(f"โหลดโมเดล YOLOv11{model_size} Classification สำเร็จ")
            except Exception as e:
                print(f"ไม่สามารถโหลดโมเดล YOLOv11{model_size}.cls ได้: {e}")
                print("กำลังลองใช้ YOLOv8 แทน...")
                try:
                    model = YOLO(f'yolov8{model_size}.cls')
                    print(f"โหลดโมเดล YOLOv8{model_size} Classification สำเร็จแทน")
                except Exception as e2:
                    print(f"ไม่สามารถโหลดโมเดล YOLOv8{model_size}.cls ได้: {e2}")
                    return None, None
    else:
        # ถ้าไม่ได้กำหนด YOLO_MODEL_PATH หรือไฟล์ไม่มีอยู่
        try:
            model = YOLO(f'yolov11{model_size}.cls')
            print(f"โหลดโมเดล YOLOv11{model_size} Classification สำเร็จ")
        except Exception as e:
            print(f"ไม่สามารถโหลดโมเดล YOLOv11{model_size}.cls ได้: {e}")
            print("กำลังลองใช้ YOLOv8 แทน...")
            try:
                model = YOLO(f'yolov8{model_size}.cls')
                print(f"โหลดโมเดล YOLOv8{model_size} Classification สำเร็จแทน")
            except Exception as e2:
                print(f"ไม่สามารถโหลดโมเดล YOLOv8{model_size}.cls ได้: {e2}")
                return None, None
    
    # เทรนโมเดลโดยใช้ early stopping ที่มีในโมเดล Ultralytics โดยตรง
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
            task=task,  # กำหนดเป็นงานจำแนกประเภท
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
            alt_dataset_path = os.path.join(current_dir, 'augmented_dataset', 'Classification')
            
            if os.path.exists(alt_dataset_path):
                print(f"กำลังทดลองใช้เส้นทางข้อมูลทางเลือก: {alt_dataset_path}")
                
                train_results = model.train(
                    data=alt_dataset_path,
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

# ฟังก์ชันสำหรับ fine-tuning โมเดล - แก้ไขแล้ว
# แก้ไขฟังก์ชัน fine_tune_model
def fine_tune_model(data_config, base_model_path, epochs=NUM_EPOCHS//2, batch_size=YOLO_BATCH_SIZE, 
                   patience=PATIENCE, img_size=224, device='', output_dir='yolov11_outputs',
                   freeze_backbone=True, freeze_encoder=False, lr=YOLO_FINE_TUNE_LEARNING_RATE, 
                   task='classify'):
    """
    ปรับแต่งโมเดล YOLOv11x สำหรับการจำแนกประเภทที่ผ่านการเทรนมาแล้ว
    
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
        task (str): ประเภทงาน ('classify' สำหรับการจำแนกประเภท)
    
    Returns:
        YOLO: โมเดลที่ fine-tune แล้ว
        str: เส้นทางไปยังโมเดลที่ดีที่สุด
    """
    print("\nเริ่มการ fine-tuning โมเดล YOLOv11x สำหรับการจำแนกประเภท...")
    
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
                        alt_path = os.path.join(current_dir, 'augmented_dataset', 'Classification')
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
        'task': task,  # กำหนดเป็นงานจำแนกประเภท
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
                train_args['data'] = dataset_path
                fine_tune_results = model.train(**train_args)
            else:
                # ลองหาเส้นทางในไดเรกทอรีปัจจุบัน
                current_dir = os.getcwd()
                alt_path = os.path.join(current_dir, 'augmented_dataset', 'Classification')
                if os.path.exists(alt_path):
                    print(f"ใช้เส้นทางทางเลือก: {alt_path}")
                    train_args['data'] = alt_path
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


# แก้ไขฟังก์ชัน main เพื่อให้จัดการกับกรณีที่ fine-tune ล้มเหลว
def main(mode='train'):
    """
    ฟังก์ชันหลักสำหรับเทรนและทดสอบโมเดล YOLOv11x สำหรับการจำแนกประเภท
    
    Args:
        mode (str): โหมดการทำงาน ('train' หรือ 'test')
    """
    # สร้างไดเรกทอรีสำหรับเอาต์พุต
    output_dir = 'yolov11_outputs'
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
        alt_train_path = os.path.join(current_dir, 'augmented_dataset', 'Classification', 'train')
        if os.path.exists(alt_train_path):
            print(f"พบโฟลเดอร์ train ที่ {alt_train_path}")
            # ปรับเส้นทางข้อมูลใหม่
            dataset_path = os.path.join(current_dir, 'augmented_dataset', 'Classification')
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
    
    # ตรวจสอบว่าไฟล์ YAML ถูกต้องหรือไม่
    if os.path.exists(data_config) and data_config.endswith('.yaml'):
        try:
            with open(data_config, 'r') as f:
                yaml_content = f.read()
                print(f"เนื้อหาไฟล์ YAML:\n{yaml_content}")
                
                # ตรวจสอบเส้นทางในไฟล์ YAML
                yaml_data = yaml.safe_load(yaml_content)
                if 'path' in yaml_data and not os.path.exists(yaml_data['path']):
                    print(f"คำเตือน: เส้นทางใน YAML ไม่ถูกต้อง: {yaml_data['path']}")
                    # แก้ไขเส้นทางในไฟล์ YAML
                    yaml_data['path'] = dataset_path
                    with open(data_config, 'w') as f:
                        yaml.dump(yaml_data, f, sort_keys=False)
                    print(f"แก้ไขเส้นทางในไฟล์ YAML เป็น: {dataset_path}")
        except Exception as e:
            print(f"เกิดข้อผิดพลาดในการตรวจสอบไฟล์ YAML: {e}")
            # ใช้เส้นทางข้อมูลโดยตรงแทน
            data_config = dataset_path
    
    # ตรวจสอบโหมดการทำงาน
    if mode.lower() == 'train':
        print("โปรแกรมเทรนโมเดล YOLOv11x สำหรับการจำแนกประเภท")
        
        # ลองใช้เส้นทางข้อมูลโดยตรงแทนไฟล์ YAML ถ้า YAML ไม่ทำงาน
        try:
            model, base_model_path = train_base_model(
                data_config=data_config,
                model_size='x',
                epochs=NUM_EPOCHS,
                batch_size=YOLO_BATCH_SIZE,
                patience=PATIENCE,
                output_dir=output_dir,
                task='classify'
            )
        except Exception as e:
            print(f"เกิดข้อผิดพลาดเมื่อใช้ไฟล์ YAML: {e}")
            print("กำลังลองใช้เส้นทางข้อมูลโดยตรง...")
            model, base_model_path = train_base_model(
                data_config=dataset_path,  # ใช้เส้นทางข้อมูลโดยตรง
                model_size='x',
                epochs=NUM_EPOCHS,
                batch_size=YOLO_BATCH_SIZE,
                patience=PATIENCE,
                output_dir=output_dir,
                task='classify'
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
            lr=YOLO_FINE_TUNE_LEARNING_RATE,
            output_dir=output_dir,
            freeze_backbone=True,
            task='classify'
        )
    
    elif mode.lower() == 'test':
        print("โปรแกรมทดสอบโมเดล YOLOv11x สำหรับการจำแนกประเภท")
        
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
            task='classify'
        )
        
        # สร้างไฟล์ Excel สำหรับผลการทำนายของโมเดลฐาน
        generate_prediction_excel(
            model_path=base_model_path,
            data_config=data_config,
            save_path=os.path.join(output_dir, 'YOLOv11x_Base_Predictions.xlsx')
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
            task='classify'
        )
        
        # สร้างไฟล์ Excel สำหรับผลการทำนายของโมเดลที่ fine-tune แล้ว
        generate_prediction_excel(
            model_path=fine_tuned_model_path,
            data_config=data_config,
            save_path=os.path.join(output_dir, 'YOLOv11x_FineTuned_Predictions.xlsx')
        )
    
    # สร้างกราฟเปรียบเทียบ
    if base_model_results is not None and fine_tuned_results is not None:
        plot_comparison_charts(
            base_model_results=base_model_results,
            fine_tuned_results=fine_tuned_results,
            output_dir=output_dir,
            task="classification"
        )
    
    print("\nเสร็จสิ้นการทำงานทั้งหมด!")

# ฟังก์ชันสำหรับประเมินโมเดลในชุดข้อมูลทดสอบ
def evaluate_model(model_path, data_config, img_size=224, batch_size=16, 
                  device='', output_dir='yolov11_outputs', model_name="YOLOv11", task='classify'):
    """
    ประเมินโมเดล YOLOv11x สำหรับการจำแนกประเภทในชุดข้อมูลทดสอบ
    
    Args:
        model_path (str): เส้นทางไปยังโมเดลที่ต้องการประเมิน
        data_config (str): เส้นทางไปยังไฟล์การกำหนดค่าข้อมูล YAML
        img_size (int): ขนาดภาพสำหรับการประเมิน
        batch_size (int): ขนาดแบทช์
        device (str): อุปกรณ์ที่ใช้ประเมิน (e.g., '0' หรือ '0,1,2,3')
        output_dir (str): ไดเรกทอรีสำหรับบันทึกผลลัพธ์
        model_name (str): ชื่อโมเดลสำหรับใช้ในการบันทึกผลลัพธ์
        task (str): ประเภทงาน ('classify' สำหรับการจำแนกประเภท)
    
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
    with open(data_config, 'r') as f:
        data = yaml.safe_load(f)
    
    # รับเส้นทางไปยังชุดข้อมูลทดสอบ
    test_path = os.path.join(data['path'], data['test'])
    
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
            task=task,  # กำหนดเป็นงานจำแนกประเภท
        )
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการประเมินผล: {e}")
        print("กำลังลองประเมินผลด้วยวิธีทางเลือก...")
        
        # ทำนายทุกภาพในชุดข้อมูลทดสอบ
        all_preds = []
        all_targets = []
        
        # โหลดชื่อคลาส
        class_names = list(data['names'].values())
        
        # ทำนายแต่ละคลาส
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(test_path, class_name)
            if not os.path.exists(class_dir):
                print(f"ไม่พบไดเรกทอรี {class_dir} ข้ามการประเมินสำหรับคลาสนี้")
                continue
                
            # หาภาพทั้งหมดในไดเรกทอรีของคลาส
            images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)
                      if img.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            # ทำนายแต่ละภาพ
            for img_path in tqdm(images, desc=f"ทำนายคลาส {class_name}"):
                pred = model.predict(img_path, imgsz=img_size, device=device, verbose=False)[0]
                pred_cls = pred.probs.top1
                all_preds.append(pred_cls)
                all_targets.append(class_idx)  # เป้าหมายคือ class_idx
        
        # คำนวณความแม่นยำด้วยตนเอง
        accuracy = sum(1 for p, t in zip(all_preds, all_targets) if p == t) / len(all_preds) if all_preds else 0
        
        # สร้างคลาสจำลองสำหรับผลลัพธ์
        class Results:
            def __init__(self, acc):
                self.metrics = {"accuracy": acc}
                
        results = Results(accuracy)
        print(f"ความแม่นยำที่คำนวณด้วยตนเอง: {accuracy:.4f}")
    
    # สร้างรายงานสรุป
    summary_path = os.path.join(output_dir, f'{model_name}_evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"ผลการประเมินโมเดล {model_name}\n")
        f.write(f"วันที่ประเมิน: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # บันทึกเมตริกสำคัญ
        try:
            f.write(f"ความแม่นยำ (Accuracy): {results.metrics.get('accuracy', 0):.4f}\n")
            
            # ถ้ามีเมตริกเพิ่มเติม
            for k, v in results.metrics.items():
                if k != 'accuracy':
                    f.write(f"{k}: {v}\n")
                    
        except AttributeError:
            # กรณีโครงสร้างผลลัพธ์แตกต่างกัน
            f.write(f"ความแม่นยำ (Accuracy): {getattr(results, 'accuracy', 0):.4f}\n")
    
    print(f"รายงานการประเมินถูกบันทึกไปยัง {summary_path}")
    
    # ถ้ามีข้อมูลเพิ่มเติมเกี่ยวกับผลการจำแนกประเภท
    try:
        # สร้าง confusion matrix ถ้ามีข้อมูลเพียงพอ
        if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
            cm = results.confusion_matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {model_name}')
            cm_path = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
            plt.savefig(cm_path)
            plt.close()
            print(f"Confusion matrix บันทึกไปยัง {cm_path}")
    except:
        print("ไม่สามารถสร้าง confusion matrix ได้")
    
    # คืนค่าผลลัพธ์
    return results


# ฟังก์ชันสำหรับสร้างไฟล์ Excel ที่มีผลการทำนาย
def generate_prediction_excel(model_path, data_config, save_path, img_size=224, 
                             batch_size=16, device=''):
    """
    สร้างไฟล์ Excel ที่มีผลการทำนายของโมเดล YOLOv11x บนชุดข้อมูลทดสอบ
    
    Args:
        model_path (str): เส้นทางไปยังโมเดลที่ต้องการประเมิน
        data_config (str): เส้นทางไปยังไฟล์การกำหนดค่าข้อมูล YAML
        save_path (str): เส้นทางสำหรับบันทึกไฟล์ Excel
        img_size (int): ขนาดภาพสำหรับการทำนาย
        batch_size (int): ขนาดแบทช์
        device (str): อุปกรณ์ที่ใช้ (e.g., '0' หรือ '0,1,2,3')
    
    Returns:
        str: เส้นทางไปยังไฟล์ Excel ที่สร้าง
    """
    print("\nกำลังสร้างไฟล์ Excel ที่มีผลการทำนาย...")
    
    # ตรวจสอบว่าโมเดลมีอยู่หรือไม่
    if not os.path.exists(model_path):
        print(f"ไม่พบโมเดลที่ {model_path}")
        return None
    
    # โหลดโมเดล
    model = YOLO(model_path)
    
    # โหลดข้อมูลการกำหนดค่า
    with open(data_config, 'r') as f:
        data = yaml.safe_load(f)
    
    # รับเส้นทางไปยังชุดข้อมูลทดสอบ
    test_path = os.path.join(data['path'], data['test'])
    class_names = list(data['names'].values())
    
    # สร้างรายการสำหรับเก็บผลการทำนาย
    results = []
    
    # ทำนายบนแต่ละคลาส
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_path, class_name)
        if not os.path.exists(class_dir):
            print(f"ไม่พบไดเรกทอรี {class_dir} ข้ามการประเมินสำหรับคลาสนี้")
            continue
            
        # หาภาพทั้งหมดในไดเรกทอรีของคลาส
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir)
                  if img.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # ทำนายแต่ละภาพ
        for img_path in tqdm(images, desc=f"ทำนายคลาส {class_name}"):
            try:
                # รันการทำนาย
                pred = model.predict(img_path, imgsz=img_size, device=device, verbose=False)[0]
                img_name = os.path.basename(img_path)
                
                # รับคลาสที่ทำนาย
                pred_cls_idx = pred.probs.top1
                pred_cls_name = class_names[pred_cls_idx] if pred_cls_idx < len(class_names) else f"Unknown-{pred_cls_idx}"
                confidence = pred.probs.top1conf.item()
                
                # เพิ่ม top-k ผลการทำนาย
                top5_indices = pred.probs.top5
                top5_confidences = pred.probs.top5conf.tolist()
                top5_classes = [class_names[i] if i < len(class_names) else f"Unknown-{i}" for i in top5_indices]
                top5_str = ';'.join([f"{cls}:{conf:.4f}" for cls, conf in zip(top5_classes, top5_confidences)])
                
                results.append({
                    'image_name': img_name,
                    'true_class': class_name,
                    'predicted_class': pred_cls_name,
                    'confidence': confidence,
                    'top5_predictions': top5_str,
                    'correct': class_name == pred_cls_name
                })
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการทำนาย {img_path}: {e}")
                results.append({
                    'image_name': os.path.basename(img_path),
                    'true_class': class_name,
                    'predicted_class': 'ERROR',
                    'confidence': 0.0,
                    'top5_predictions': 'ERROR',
                    'correct': False
                })
    
    # สร้าง DataFrame และบันทึกเป็น Excel
    df = pd.DataFrame(results)
    df.to_excel(save_path, index=False)
    
    # สร้างสรุปผลการทำนาย
    accuracy = df['correct'].mean()
    class_accuracies = df.groupby('true_class')['correct'].mean()
    
    # เพิ่มแผ่นงานสรุป
    with pd.ExcelWriter(save_path, engine='openpyxl', mode='a') as writer:
        # สร้างแผ่นงานสรุป
        summary = pd.DataFrame({
            'Metric': ['Overall Accuracy'] + [f'Accuracy - {cls}' for cls in class_accuracies.index],
            'Value': [accuracy] + list(class_accuracies.values)
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # สร้าง Confusion Matrix
        confusion = pd.crosstab(df['true_class'], df['predicted_class'])
        confusion.to_excel(writer, sheet_name='Confusion Matrix')
    
    print(f"บันทึกผลการทำนายไปยัง {save_path} เรียบร้อยแล้ว")
    print(f"ความแม่นยำรวม: {accuracy:.4f}")
    
    return save_path


# ฟังก์ชันสำหรับการทำนายภาพเดี่ยว
def predict_single_image(model_path, image_path, img_size=224, 
                        save_result=True, output_dir='yolov11_outputs', device=''):
    """
    ทำนายคลาสในภาพเดี่ยวด้วยโมเดล YOLOv11x
    
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
        )
        
        # แสดงผลการทำนาย
        result = results[0]
        top1_idx = result.probs.top1
        top1_conf = result.probs.top1conf.item()
        
        # รับชื่อคลาส (อาจต้องโหลดจาก YAML)
        try:
            class_names = model.names
            top1_name = class_names[top1_idx]
        except:
            top1_name = f"Class {top1_idx}"
        
        print(f"ผลการทำนาย: {top1_name}, ความเชื่อมั่น: {top1_conf:.4f}")
        
        # แสดง top-5 การทำนาย
        top5_indices = result.probs.top5
        top5_confidences = result.probs.top5conf.tolist()
        
        print("Top 5 การทำนาย:")
        for i, (idx, conf) in enumerate(zip(top5_indices, top5_confidences)):
            try:
                class_name = class_names[idx]
            except:
                class_name = f"Class {idx}"
            print(f"  {i+1}. {class_name}: {conf:.4f}")
        
        if save_result:
            print(f"บันทึกผลลัพธ์ไปยัง {output_dir}/single_predictions")
            
            # สร้างภาพที่มีผลการทำนาย
            img = Image.open(image_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(np.array(img))
            plt.title(f"Prediction: {top1_name} ({top1_conf:.4f})")
            plt.axis('off')
            
            # บันทึกภาพ
            result_path = os.path.join(output_dir, 'single_predictions', f"result_{os.path.basename(image_path)}")
            plt.savefig(result_path)
            plt.close()
        
        return results
    
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการทำนาย: {e}")
        return None


# ฟังก์ชันสำหรับพล็อตกราฟเปรียบเทียบประสิทธิภาพ
def plot_comparison_charts(base_model_results, fine_tuned_results, output_dir, task="classification"):
    """
    สร้างกราฟเปรียบเทียบประสิทธิภาพระหว่างโมเดลฐานและโมเดลที่ fine-tune แล้ว
    
    Args:
        base_model_results: ผลการประเมินของโมเดลฐาน
        fine_tuned_results: ผลการประเมินของโมเดลที่ fine-tune แล้ว
        output_dir (str): ไดเรกทอรีสำหรับบันทึกกราฟ
        task (str): ประเภทงาน ('classification' หรือ 'detection')
    """
    # ตรวจสอบว่ามีผลการประเมินทั้งสองโมเดลหรือไม่
    if base_model_results is None or fine_tuned_results is None:
        print("ไม่มีผลการประเมินที่จะเปรียบเทียบ")
        return
    
    try:
        # สร้างไดเรกทอรีสำหรับบันทึกกราฟ
        os.makedirs(os.path.join(output_dir, 'comparison_charts'), exist_ok=True)
        
        # เตรียมข้อมูลสำหรับการพล็อต
        if task == "classification":
            # เปรียบเทียบความแม่นยำ
            metrics = ['accuracy']
            titles = ['Accuracy Comparison']
            ylabels = ['Accuracy']
            
            try:
                # ดึงค่าความแม่นยำจากผลการประเมิน
                base_accuracy = base_model_results.metrics.get('accuracy', 0)
                fine_tuned_accuracy = fine_tuned_results.metrics.get('accuracy', 0)
                
                values = [[base_accuracy, fine_tuned_accuracy]]
            except:
                print("ไม่สามารถดึงข้อมูลความแม่นยำได้ ใช้ค่าเริ่มต้น")
                values = [[0.5, 0.6]]  # ค่าเริ่มต้น
        else:
            # สำหรับงานตรวจจับวัตถุ
            metrics = ['mAP50-95', 'mAP50', 'precision', 'recall']
            titles = ['mAP50-95 Comparison', 'mAP50 Comparison', 'Precision Comparison', 'Recall Comparison']
            ylabels = ['mAP50-95', 'mAP50', 'Precision', 'Recall']
            
            try:
                # ดึงค่าเมตริกจากผลการประเมิน
                base_map = base_model_results.box.map
                base_map50 = base_model_results.box.map50
                base_precision = base_model_results.box.p
                base_recall = base_model_results.box.r
                
                fine_tuned_map = fine_tuned_results.box.map
                fine_tuned_map50 = fine_tuned_results.box.map50
                fine_tuned_precision = fine_tuned_results.box.p
                fine_tuned_recall = fine_tuned_results.box.r
                
                values = [
                    [base_map, fine_tuned_map],
                    [base_map50, fine_tuned_map50],
                    [base_precision, fine_tuned_precision],
                    [base_recall, fine_tuned_recall]
                ]
            except:
                print("ไม่สามารถดึงข้อมูลเมตริกได้ ใช้ค่าเริ่มต้น")
                values = [[0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.65, 0.75]]  # ค่าเริ่มต้น
        
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


if __name__ == "__main__":
    # ตรวจสอบอาร์กิวเมนต์จากบรรทัดคำสั่งเพื่อกำหนด mode
    import sys
    
    mode = 'train'  # ค่าเริ่มต้น
    
    # ถ้ามีการส่งอาร์กิวเมนต์มา ให้ใช้อาร์กิวเมนต์แรกเป็น mode
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    
    main(mode)