"""
ไฟล์รันหลักสำหรับเทรนและทดสอบทุกโมเดล (EfficientNetV2, InceptionV3, VGG16, RegNet, ResNet50)
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

# ฟังก์ชันสำหรับรันโมเดล
def run_model(model_name, mode):
    print(f"\n{'=' * 50}")
    print(f"กำลังรัน {model_name} ในโหมด {mode}")
    print(f"{'=' * 50}\n")
    
    start_time = time.time()
    
    try:
        # รันโมเดลตามชื่อที่กำหนด
        subprocess.run(['python', f'{model_name.lower()}.py'], check=True)
        
        end_time = time.time()
        duration = end_time - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n{'=' * 50}")
        print(f"รัน {model_name} เสร็จสิ้น")
        print(f"ใช้เวลา: {int(hours)}:{int(minutes):02d}:{seconds:.2f}")
        print(f"{'=' * 50}\n")
        
        return True
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการรัน {model_name}: {e}")
        return False

def main():
    # สร้าง parser สำหรับรับพารามิเตอร์
    parser = argparse.ArgumentParser(description='รันโมเดลทั้งหมดหรือเฉพาะโมเดลที่เลือก')
    parser.add_argument('--models', nargs='+', 
                      choices=['efficientnetv2', 'inceptionv3', 'vgg16', 'regnet', 'resnet50', 'all'],
                      default=['all'],
                      help='ระบุโมเดลที่ต้องการรัน (efficientnetv2, inceptionv3, vgg16, regnet, resnet50, all)')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                      help='โหมดการทำงาน (train, test)')
    
    args = parser.parse_args()
    
    # รายการโมเดลทั้งหมด
    available_models = ['efficientnetv2', 'inceptionv3', 'vgg16', 'regnet', 'resnet50']
    
    # กำหนดโมเดลที่จะรัน
    if 'all' in args.models:
        models_to_run = available_models
    else:
        models_to_run = args.models
    
    # จัดการโฟลเดอร์ผลลัพธ์
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    # รันแต่ละโมเดล
    start_time = time.time()
    successful_models = []
    failed_models = []
    
    for model in models_to_run:
        success = run_model(model, args.mode)
        if success:
            successful_models.append(model)
        else:
            failed_models.append(model)
    
    # สรุปผล
    end_time = time.time()
    total_duration = end_time - start_time
    hours, remainder = divmod(total_duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 50)
    print("สรุปผลการทำงาน")
    print("=" * 50)
    print(f"รันโมเดลเสร็จสิ้น: {len(successful_models)} โมเดล")
    if successful_models:
        print("โมเดลที่รันสำเร็จ:")
        for model in successful_models:
            print(f"- {model}")
    
    if failed_models:
        print("\nโมเดลที่รันไม่สำเร็จ:")
        for model in failed_models:
            print(f"- {model}")
    
    print(f"\nเวลาทั้งหมดที่ใช้: {int(hours)}:{int(minutes):02d}:{seconds:.2f}")
    
    # บันทึกผลการรัน
    with open(os.path.join(result_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("สรุปผลการทำงาน\n")
        f.write("=" * 30 + "\n")
        f.write(f"รันโมเดลเสร็จสิ้น: {len(successful_models)} โมเดล\n")
        if successful_models:
            f.write("โมเดลที่รันสำเร็จ:\n")
            for model in successful_models:
                f.write(f"- {model}\n")
        
        if failed_models:
            f.write("\nโมเดลที่รันไม่สำเร็จ:\n")
            for model in failed_models:
                f.write(f"- {model}\n")
        
        f.write(f"\nเวลาทั้งหมดที่ใช้: {int(hours)}:{int(minutes):02d}:{seconds:.2f}\n")
    
    print(f"\nบันทึกสรุปผลไว้ที่: {os.path.join(result_dir, 'summary.txt')}")

if __name__ == "__main__":
    main()
