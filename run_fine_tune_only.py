import os
import torch
import yaml
from ultralytics import YOLO

# คุณสามารถกำหนดค่าต่าง ๆ ที่จำเป็นตรงนี้
OUTPUT_DIR = './yolov11_outputs'
DATASET_PATH = './augmented_dataset/Classification'  # ปรับให้ถูกต้องตามเส้นทางข้อมูลของคุณ
NUM_EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 15
LEARNING_RATE = 0.0001

def run_fine_tune_only():
    """
    รันเฉพาะการ fine-tune โมเดล YOLOv11 โดยใช้โมเดลฐานที่เทรนไว้แล้ว
    """
    print("เริ่มการ fine-tune โมเดล YOLOv11...")
    
    # ตรวจสอบว่ามีโมเดลฐาน
    base_model_path = os.path.join(OUTPUT_DIR, 'base_model.pt')
    if not os.path.exists(base_model_path):
        print(f"ไม่พบโมเดลฐานที่ {base_model_path}")
        # ลองหาในโฟลเดอร์ weights
        weights_path = os.path.join(OUTPUT_DIR, 'base_model', 'weights', 'best.pt')
        if os.path.exists(weights_path):
            base_model_path = weights_path
            print(f"ใช้โมเดลที่พบที่ {base_model_path} แทน")
        else:
            print("ไม่พบโมเดลฐาน กรุณาเทรนโมเดลฐานก่อน")
            return
    
    # เตรียมเส้นทางข้อมูล
    dataset_path = os.path.abspath(DATASET_PATH)
    
    # สร้างหรือปรับปรุงไฟล์ YAML
    data_yaml_path = os.path.join(OUTPUT_DIR, 'data_fine_tune.yaml')
    
    # ตรวจสอบว่ามีโฟลเดอร์ข้อมูลจริง
    train_path = os.path.join(dataset_path, 'train')
    if not os.path.exists(train_path):
        print(f"ไม่พบโฟลเดอร์ train ที่ {train_path}")
        # ลองหาทางเลือก
        current_dir = os.getcwd()
        alt_path = os.path.join(current_dir, 'augmented_dataset', 'Classification')
        if os.path.exists(os.path.join(alt_path, 'train')):
            dataset_path = alt_path
            print(f"ใช้เส้นทางข้อมูลที่ {dataset_path} แทน")
        else:
            print("ไม่พบโฟลเดอร์ข้อมูลที่ถูกต้อง")
            return
    
    # ค้นหาคลาสจากโฟลเดอร์ train
    class_names = sorted([d for d in os.listdir(os.path.join(dataset_path, 'train')) 
                   if os.path.isdir(os.path.join(dataset_path, 'train', d))])
    
    print(f"พบคลาสทั้งหมด {len(class_names)} คลาส: {', '.join(class_names)}")
    
    # สร้างไฟล์ YAML
    yaml_data = {
        'path': dataset_path,
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': {i: name for i, name in enumerate(class_names)},
        'nc': len(class_names)
    }
    
    with open(data_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    
    print(f"สร้างไฟล์ YAML ที่ {data_yaml_path}")
    
    # โหลดโมเดลฐาน
    try:
        model = YOLO(base_model_path)
        print(f"โหลดโมเดลฐานจาก {base_model_path} สำเร็จ")
    except Exception as e:
        print(f"ไม่สามารถโหลดโมเดลฐาน: {e}")
        return
    
    # กำหนดอุปกรณ์ (GPU หรือ CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ใช้อุปกรณ์: {device}")
    device_str = '0' if torch.cuda.is_available() else 'cpu'
    
    # Fine-tune โมเดล
    try:
        print(f"กำลังเริ่มการ fine-tune โมเดลด้วยข้อมูลที่: {data_yaml_path}")
        
        # เตรียมพารามิเตอร์สำหรับการ fine-tune
        fine_tune_args = {
            'data': data_yaml_path,
            'epochs': NUM_EPOCHS,
            'batch': BATCH_SIZE,
            'imgsz': 224,
            'device': device_str,
            'project': OUTPUT_DIR,
            'name': 'fine_tuned_model',
            'exist_ok': True,
            'lr0': LEARNING_RATE,
            'lrf': 0.01,
            'pretrained': False,  # โหลดโมเดลที่เทรนแล้ว
            'verbose': True,
            'save': True,
            'save_period': 5,
            'task': 'classify',
            'patience': PATIENCE,
            'freeze': [0, 1, 2, 3, 4, 5, 6]  # แช่แข็ง backbone
        }
        
        # รัน fine-tune
        fine_tune_results = model.train(**fine_tune_args)
        
        # ค้นหาโมเดลที่ดีที่สุด
        best_model_path = getattr(fine_tune_results, 'best', None)
        if best_model_path is None or not os.path.exists(best_model_path):
            print("ไม่พบ best model, ใช้ last model แทน")
            best_model_path = fine_tune_results.last
        
        # คัดลอกโมเดลที่ดีที่สุดไปยังไดเรกทอรีหลัก
        import shutil
        fine_tuned_path = os.path.join(OUTPUT_DIR, 'fine_tuned_model.pt')
        shutil.copy(best_model_path, fine_tuned_path)
        print(f"คัดลอกโมเดลที่ fine-tune แล้วไปยัง {fine_tuned_path}")
        
        print("การ fine-tune สำเร็จ!")
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการ fine-tune: {e}")
        
        # ลองใช้เส้นทางข้อมูลโดยตรงแทนไฟล์ YAML
        try:
            print("กำลังลองใช้เส้นทางข้อมูลโดยตรง...")
            fine_tune_args['data'] = dataset_path
            fine_tune_results = model.train(**fine_tune_args)
            
            # ค้นหาโมเดลที่ดีที่สุด
            best_model_path = getattr(fine_tune_results, 'best', None)
            if best_model_path is None or not os.path.exists(best_model_path):
                print("ไม่พบ best model, ใช้ last model แทน")
                best_model_path = fine_tune_results.last
            
            # คัดลอกโมเดลที่ดีที่สุดไปยังไดเรกทอรีหลัก
            import shutil
            fine_tuned_path = os.path.join(OUTPUT_DIR, 'fine_tuned_model.pt')
            shutil.copy(best_model_path, fine_tuned_path)
            print(f"คัดลอกโมเดลที่ fine-tune แล้วไปยัง {fine_tuned_path}")
            
            print("การ fine-tune สำเร็จ!")
            
        except Exception as e2:
            print(f"การลองใช้เส้นทางข้อมูลโดยตรงล้มเหลว: {e2}")

if __name__ == "__main__":
    run_fine_tune_only()