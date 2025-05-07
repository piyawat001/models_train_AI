"""
ไฟล์สำหรับทำนายภาพใหม่ด้วยโมเดลที่เทรนแล้ว
สามารถทำนายภาพเดี่ยวหรือทั้งโฟลเดอร์
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# โหลดค่าจากไฟล์ .env
load_dotenv()

# กำหนดอุปกรณ์ (GPU หรือ CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ใช้อุปกรณ์: {device}")

# จำกัดการใช้หน่วยความจำ GPU ถ้ามี
if torch.cuda.is_available() and hasattr(torch.cuda, 'memory'):
    GPU_MEMORY_FRACTION = float(os.getenv('GPU_MEMORY_FRACTION', '0.7'))
    torch.cuda.memory.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)

# กำหนดชื่อคลาส
CLASS_NAMES = ['ameloblastoma', 'dentigerous cyst', 'normal jaw', 'okc']

# ฟังก์ชันสำหรับโหลดโมเดล EfficientNetV2
def load_efficientnetv2(model_path, num_classes=4):
    try:
        # สำหรับ PyTorch รุ่นใหม่
        model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
    except:
        # สำหรับ PyTorch รุ่นเก่า
        model = models.efficientnet_v2_s(pretrained=True)
    
    # ปรับชั้นสุดท้าย
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    # โหลดพารามิเตอร์
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

# ฟังก์ชันสำหรับโหลดโมเดล InceptionV3
def load_inceptionv3(model_path, num_classes=4):
    try:
        # สำหรับ PyTorch รุ่นใหม่
        model = models.inception_v3(weights='IMAGENET1K_V1')
    except:
        try:
            # สำหรับ PyTorch รุ่นกลาง
            from torchvision.models import Inception_V3_Weights
            model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        except:
            # สำหรับ PyTorch รุ่นเก่า
            model = models.inception_v3(pretrained=True)
    
    # ปรับชั้นสุดท้าย
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.aux_logits = False  # ปิดการใช้งาน auxiliary classifier
    
    # โหลดพารามิเตอร์
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

# ฟังก์ชันสำหรับโหลดโมเดล VGG16
def load_vgg16(model_path, num_classes=4):
    try:
        # สำหรับ PyTorch รุ่นใหม่
        model = models.vgg16(weights='IMAGENET1K_V1')
    except:
        # สำหรับ PyTorch รุ่นเก่า
        model = models.vgg16(pretrained=True)
    
    # ปรับชั้นสุดท้าย
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, num_classes)
    
    # โหลดพารามิเตอร์
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

# ฟังก์ชันสำหรับโหลดโมเดล RegNet
def load_regnet(model_path, num_classes=4):
    try:
        # สำหรับ PyTorch รุ่นใหม่
        model = models.regnet_y_32gf(weights='IMAGENET1K_SWAG_E2E_V1')
    except:
        # สำหรับ PyTorch รุ่นเก่า
        model = models.regnet_y_32gf(pretrained=True)
    
    # ปรับชั้นสุดท้าย
    num_features_in = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features_in, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, num_classes)
    )
    
    # โหลดพารามิเตอร์
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

# ฟังก์ชันสำหรับโหลดโมเดล ResNet50
def load_resnet50(model_path, num_classes=4):
    try:
        # สำหรับ PyTorch รุ่นใหม่
        model = models.resnet50(weights='IMAGENET1K_V2')
    except:
        # สำหรับ PyTorch รุ่นเก่า
        model = models.resnet50(pretrained=True)
    
    # ปรับชั้นสุดท้าย
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # โหลดพารามิเตอร์
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

# ฟังก์ชันสำหรับทำนายภาพเดี่ยว
def predict_image(model, image_path, transform, class_names, output_dir=None, model_name=None):
    # โหลดภาพ
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"ไม่สามารถโหลดภาพ {image_path}: {e}")
        return None
    
    # แปลงภาพ
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # ทำนาย
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = class_names[pred_idx]
    
    # แสดงผลการทำนาย
    print(f"\nผลการทำนายสำหรับภาพ {os.path.basename(image_path)}:")
    print(f"ทำนายเป็น: {pred_class} (ความมั่นใจ: {probs[pred_idx]:.2%})")
    print("ความน่าจะเป็นของแต่ละคลาส:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {probs[i]:.2%}")
    
    # บันทึกภาพพร้อมผลการทำนาย
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # สร้างภาพพร้อมผลการทำนาย
        plt.figure(figsize=(10, 6))
        
        # แสดงภาพ
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"ภาพ {os.path.basename(image_path)}")
        plt.axis('off')
        
        # สร้างแผนภูมิแท่ง
        plt.subplot(1, 2, 2)
        bars = plt.bar(range(len(class_names)), probs.cpu().numpy())
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.title("ความน่าจะเป็นของแต่ละคลาส")
        plt.xlabel("คลาส")
        plt.ylabel("ความน่าจะเป็น")
        
        # เน้นแท่งที่มีค่าสูงสุด
        bars[pred_idx].set_color('red')
        
        # เพิ่มค่าความน่าจะเป็นเหนือแต่ละแท่ง
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{probs[i]:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # บันทึกภาพ
        filename = os.path.splitext(os.path.basename(image_path))[0]
        model_suffix = f"_{model_name}" if model_name else ""
        save_path = os.path.join(output_dir, f"{filename}_prediction{model_suffix}.png")
        plt.savefig(save_path)
        plt.close()
        
        print(f"บันทึกภาพผลการทำนายไปยัง: {save_path}")
    
    return {
        'image_path': image_path,
        'predicted_class': pred_class,
        'probabilities': {class_name: float(probs[i]) for i, class_name in enumerate(class_names)}
    }

# ฟังก์ชันสำหรับทำนายทั้งโฟลเดอร์
def predict_folder(model, folder_path, transform, class_names, output_dir, model_name=None):
    # ตรวจสอบว่าโฟลเดอร์มีอยู่จริง
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"ไม่พบโฟลเดอร์ {folder_path}")
        return None
    
    # หาไฟล์ภาพทั้งหมดในโฟลเดอร์
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
                  if os.path.isfile(os.path.join(folder_path, f)) and 
                  f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"ไม่พบไฟล์ภาพในโฟลเดอร์ {folder_path}")
        return None
    
    print(f"พบไฟล์ภาพ {len(image_files)} ไฟล์ในโฟลเดอร์ {folder_path}")
    
    # สร้างโฟลเดอร์สำหรับบันทึกผลลัพธ์
    os.makedirs(output_dir, exist_ok=True)
    
    # ทำนายแต่ละภาพ
    results = []
    
    for image_path in tqdm(image_files, desc="กำลังทำนายภาพ"):
        result = predict_image(model, image_path, transform, class_names, output_dir, model_name)
        if result:
            results.append(result)
    
    # สร้างรายงานสรุป
    if results:
        # จำนวนของการทำนายแต่ละคลาส
        class_counts = {}
        for class_name in class_names:
            class_counts[class_name] = sum(1 for r in results if r['predicted_class'] == class_name)
        
        print("\nสรุปผลการทำนาย:")
        for class_name, count in class_counts.items():
            percentage = count / len(results) * 100
            print(f"{class_name}: {count} ภาพ ({percentage:.1f}%)")
        
        # สร้างไฟล์ Excel สรุปผล
        df_results = []
        for result in results:
            row = {
                'image_name': os.path.basename(result['image_path']),
                'predicted_class': result['predicted_class']
            }
            # เพิ่มความน่าจะเป็นของแต่ละคลาส
            for class_name in class_names:
                row[f'{class_name}_probability'] = result['probabilities'][class_name]
            
            df_results.append(row)
        
        # บันทึกเป็น Excel
        model_suffix = f"_{model_name}" if model_name else ""
        excel_path = os.path.join(output_dir, f"folder_predictions{model_suffix}.xlsx")
        pd.DataFrame(df_results).to_excel(excel_path, index=False)
        print(f"บันทึกผลการทำนายไปยัง {excel_path}")
        
        # สร้างกราฟสรุปผล
        plt.figure(figsize=(10, 6))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title("จำนวนภาพที่ทำนายแต่ละคลาส")
        plt.xlabel("คลาส")
        plt.ylabel("จำนวนภาพ")
        plt.xticks(rotation=45)
        
        # เพิ่มจำนวนภาพและเปอร์เซ็นต์เหนือแต่ละแท่ง
        for class_name, count in class_counts.items():
            percentage = count / len(results) * 100
            plt.text(class_name, count + 0.5, f"{count} ({percentage:.1f}%)", 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"summary_chart{model_suffix}.png"))
        plt.close()
    
    return results

# ฟังก์ชัน main
def main():
    parser = argparse.ArgumentParser(description="ทำนายภาพด้วยโมเดลที่เทรนแล้ว")
    parser.add_argument('--model', choices=['efficientnetv2', 'inceptionv3', 'vgg16', 'regnet', 'resnet50', 'ensemble'], 
                        default='efficientnetv2', help='เลือกโมเดลที่ต้องการใช้')
    parser.add_argument('--model_path', type=str, 
                        help='พาธของโมเดลที่ต้องการใช้ (ถ้าไม่ระบุจะใช้ค่าเริ่มต้น)')
    parser.add_argument('--input', type=str, required=True,
                        help='พาธของไฟล์ภาพหรือโฟลเดอร์ที่ต้องการทำนาย')
    parser.add_argument('--output_dir', type=str, default='prediction_results',
                        help='โฟลเดอร์สำหรับบันทึกผลลัพธ์')
    
    args = parser.parse_args()
    
    # กำหนดการแปลงข้อมูลสำหรับแต่ละโมเดล
    transforms_dict = {
        'efficientnetv2': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'inceptionv3': transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'vgg16': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'regnet': transforms.Compose([
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'resnet50': transforms.Compose([
            transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    # กำหนดพาธเริ่มต้นของโมเดลต่างๆ
    default_model_paths = {
        'efficientnetv2': 'efficientnet_outputs/fine_tuned_model_acc.pth',
        'inceptionv3': 'inception_outputs/fine_tuned_model_acc.pth',
        'vgg16': 'vgg16_outputs/VGG16_Fine-Tune.pt',
        'regnet': 'regnet_outputs/RegNet_Fine-Tune.pt',
        'resnet50': 'resnet50_outputs/Resnet50_Fine-Tune.pt'
    }
    
    # ถ้าเป็นการทำ ensemble ให้โหลดทุกโมเดล
    if args.model == 'ensemble':
        models_to_load = ['efficientnetv2', 'inceptionv3', 'vgg16', 'resnet50']  # ไม่รวม RegNet เนื่องจากใช้หน่วยความจำมาก
        print("กำลังโหลดโมเดลทั้งหมดสำหรับการทำ ensemble...")
        
        loaded_models = {}
        for model_name in models_to_load:
            try:
                model_path = default_model_paths[model_name]
                if model_name == 'efficientnetv2':
                    loaded_models[model_name] = load_efficientnetv2(model_path)
                elif model_name == 'inceptionv3':
                    loaded_models[model_name] = load_inceptionv3(model_path)
                elif model_name == 'vgg16':
                    loaded_models[model_name] = load_vgg16(model_path)
                elif model_name == 'resnet50':
                    loaded_models[model_name] = load_resnet50(model_path)
                
                print(f"โหลดโมเดล {model_name} สำเร็จ")
            except Exception as e:
                print(f"ไม่สามารถโหลดโมเดล {model_name}: {e}")
        
        # ตรวจสอบว่ามีโมเดลที่โหลดสำเร็จอย่างน้อย 1 โมเดล
        if not loaded_models:
            print("ไม่สามารถโหลดโมเดลใดๆ สำหรับการทำ ensemble")
            return
        
        # ทำนายโดยใช้ ensemble
        if os.path.isdir(args.input):
            # กรณีเป็นโฟลเดอร์ ให้ทำนายแต่ละไฟล์ด้วยทุกโมเดล และใช้การโหวต
            image_files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                          if os.path.isfile(os.path.join(args.input, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
            
            if not image_files:
                print(f"ไม่พบไฟล์ภาพในโฟลเดอร์ {args.input}")
                return
            
            print(f"พบไฟล์ภาพ {len(image_files)} ไฟล์ในโฟลเดอร์ {args.input}")
            
            # สร้างโฟลเดอร์สำหรับบันทึกผลลัพธ์
            os.makedirs(args.output_dir, exist_ok=True)
            
            # ทำนายแต่ละภาพด้วยทุกโมเดล
            ensemble_results = []
            
            for image_path in tqdm(image_files, desc="กำลังทำนายภาพด้วย ensemble"):
                image_results = {}
                
                for model_name, model in loaded_models.items():
                    transform = transforms_dict[model_name]
                    result = predict_image(model, image_path, transform, CLASS_NAMES, 
                                          args.output_dir, model_name)
                    if result:
                        image_results[model_name] = result
                
                # ถ้ามีผลการทำนายจากอย่างน้อยหนึ่งโมเดล
                if image_results:
                    # หาผลการทำนายโดยการรวมความน่าจะเป็น
                    combined_probs = {class_name: 0.0 for class_name in CLASS_NAMES}
                    
                    for model_result in image_results.values():
                        for class_name, prob in model_result['probabilities'].items():
                            combined_probs[class_name] += prob
                    
                    # หารด้วยจำนวนโมเดลเพื่อให้ได้ค่าเฉลี่ย
                    for class_name in combined_probs:
                        combined_probs[class_name] /= len(image_results)
                    
                    # หาคลาสที่มีความน่าจะเป็นเฉลี่ยสูงสุด
                    predicted_class = max(combined_probs, key=combined_probs.get)
                    
                    # สร้างผลลัพธ์ ensemble
                    ensemble_result = {
                        'image_path': image_path,
                        'predicted_class': predicted_class,
                        'probabilities': combined_probs,
                        'individual_predictions': {model_name: result['predicted_class'] 
                                                 for model_name, result in image_results.items()}
                    }
                    
                    ensemble_results.append(ensemble_result)
                    
                    # แสดงผลการทำนาย ensemble
                    print(f"\nผลการทำนาย ensemble สำหรับภาพ {os.path.basename(image_path)}:")
                    print(f"ทำนายเป็น: {predicted_class} (ความน่าจะเป็นเฉลี่ย: {combined_probs[predicted_class]:.2%})")
                    
                    # สร้างภาพสรุปผล ensemble
                    plt.figure(figsize=(12, 6))
                    
                    # แสดงภาพ
                    plt.subplot(1, 2, 1)
                    img = Image.open(image_path).convert('RGB')
                    plt.imshow(img)
                    plt.title(f"ภาพ {os.path.basename(image_path)}")
                    plt.axis('off')
                    
                    # สร้างแผนภูมิแท่งสำหรับผลลัพธ์ ensemble
                    plt.subplot(1, 2, 2)
                    bars = plt.bar(range(len(CLASS_NAMES)), 
                                  [combined_probs[class_name] for class_name in CLASS_NAMES])
                    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45)
                    plt.title("ความน่าจะเป็นเฉลี่ยจาก Ensemble")
                    plt.xlabel("คลาส")
                    plt.ylabel("ความน่าจะเป็นเฉลี่ย")
                    
                    # เน้นแท่งที่มีค่าสูงสุด
                    max_idx = list(CLASS_NAMES).index(predicted_class)
                    bars[max_idx].set_color('red')
                    
                    # เพิ่มค่าความน่าจะเป็นเหนือแต่ละแท่ง
                    for i, class_name in enumerate(CLASS_NAMES):
                        height = combined_probs[class_name]
                        plt.text(i, height + 0.01, f'{height:.2%}', 
                                ha='center', va='bottom')
                    
                    plt.tight_layout()
                    
                    # บันทึกภาพ
                    filename = os.path.splitext(os.path.basename(image_path))[0]
                    save_path = os.path.join(args.output_dir, f"{filename}_ensemble.png")
                    plt.savefig(save_path)
                    plt.close()
            
            # สร้างไฟล์ Excel สรุปผล ensemble
            if ensemble_results:
                df_results = []
                for result in ensemble_results:
                    row = {
                        'image_name': os.path.basename(result['image_path']),
                        'ensemble_prediction': result['predicted_class']
                    }
                    # เพิ่มความน่าจะเป็นของแต่ละคลาส
                    for class_name in CLASS_NAMES:
                        row[f'{class_name}_probability'] = result['probabilities'][class_name]
                    
                    # เพิ่มผลการทำนายจากแต่ละโมเดล
                    for model_name, prediction in result['individual_predictions'].items():
                        row[f'{model_name}_prediction'] = prediction
                    
                    df_results.append(row)
                
                # บันทึกเป็น Excel
                excel_path = os.path.join(args.output_dir, "ensemble_predictions.xlsx")
                pd.DataFrame(df_results).to_excel(excel_path, index=False)
                print(f"บันทึกผลการทำนาย ensemble ไปยัง {excel_path}")
                
                # สร้างกราฟสรุปผล
                class_counts = {}
                for class_name in CLASS_NAMES:
                    class_counts[class_name] = sum(1 for r in ensemble_results if r['predicted_class'] == class_name)
                
                plt.figure(figsize=(10, 6))
                plt.bar(class_counts.keys(), class_counts.values())
                plt.title("จำนวนภาพที่ทำนายแต่ละคลาสโดย Ensemble")
                plt.xlabel("คลาส")
                plt.ylabel("จำนวนภาพ")
                plt.xticks(rotation=45)
                
                # เพิ่มจำนวนภาพและเปอร์เซ็นต์เหนือแต่ละแท่ง
                for class_name, count in class_counts.items():
                    percentage = count / len(ensemble_results) * 100
                    plt.text(class_name, count + 0.5, f"{count} ({percentage:.1f}%)", 
                            ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(args.output_dir, "ensemble_summary_chart.png"))
                plt.close()
        else:
            # กรณีเป็นไฟล์เดียว ให้ทำนายด้วยทุกโมเดลและแสดงผลเปรียบเทียบ
            image_path = args.input
            image_results = {}
            
            for model_name, model in loaded_models.items():
                transform = transforms_dict[model_name]
                result = predict_image(model, image_path, transform, CLASS_NAMES, 
                                      args.output_dir, model_name)
                if result:
                    image_results[model_name] = result
            
            # ถ้ามีผลการทำนายจากอย่างน้อยหนึ่งโมเดล
            if image_results:
                # หาผลการทำนายโดยการรวมความน่าจะเป็น
                combined_probs = {class_name: 0.0 for class_name in CLASS_NAMES}
                
                for model_result in image_results.values():
                    for class_name, prob in model_result['probabilities'].items():
                        combined_probs[class_name] += prob
                
                # หารด้วยจำนวนโมเดลเพื่อให้ได้ค่าเฉลี่ย
                for class_name in combined_probs:
                    combined_probs[class_name] /= len(image_results)
                
                # หาคลาสที่มีความน่าจะเป็นเฉลี่ยสูงสุด
                predicted_class = max(combined_probs, key=combined_probs.get)
                
                # แสดงผลการทำนายจากทุกโมเดลและ ensemble
                print(f"\nสรุปผลการทำนายสำหรับภาพ {os.path.basename(image_path)}:")
                for model_name, result in image_results.items():
                    print(f"{model_name}: {result['predicted_class']} " 
                          f"(ความมั่นใจ: {result['probabilities'][result['predicted_class']]:.2%})")
                
                print(f"Ensemble: {predicted_class} " 
                      f"(ความน่าจะเป็นเฉลี่ย: {combined_probs[predicted_class]:.2%})")
                
                # สร้างภาพเปรียบเทียบผลการทำนาย
                plt.figure(figsize=(15, 6))
                
                # แสดงภาพ
                plt.subplot(1, len(image_results) + 1, 1)
                img = Image.open(image_path).convert('RGB')
                plt.imshow(img)
                plt.title(f"ภาพ {os.path.basename(image_path)}")
                plt.axis('off')
                
                # แสดงผลการทำนายจากแต่ละโมเดล
                for i, (model_name, result) in enumerate(image_results.items(), 1):
                    plt.subplot(1, len(image_results) + 1, i + 1)
                    bars = plt.bar(range(len(CLASS_NAMES)), 
                                  [result['probabilities'][class_name] for class_name in CLASS_NAMES])
                    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=90)
                    plt.title(f"{model_name}: {result['predicted_class']}")
                    
                    # เน้นแท่งที่มีค่าสูงสุด
                    max_idx = list(CLASS_NAMES).index(result['predicted_class'])
                    bars[max_idx].set_color('red')
                
                plt.tight_layout()
                
                # บันทึกภาพ
                os.makedirs(args.output_dir, exist_ok=True)
                filename = os.path.splitext(os.path.basename(image_path))[0]
                save_path = os.path.join(args.output_dir, f"{filename}_comparison.png")
                plt.savefig(save_path)
                plt.close()
                
                print(f"บันทึกภาพเปรียบเทียบผลการทำนายไปยัง: {save_path}")
        
        return
    
    # โหลดโมเดลที่ต้องการใช้
    try:
        # กำหนดพาธของโมเดล
        model_path = args.model_path if args.model_path else default_model_paths[args.model]
        
        print(f"กำลังโหลดโมเดล {args.model} จาก {model_path}...")
        
        # โหลดโมเดลตามที่เลือก
        if args.model == 'efficientnetv2':
            model = load_efficientnetv2(model_path)
        elif args.model == 'inceptionv3':
            model = load_inceptionv3(model_path)
        elif args.model == 'vgg16':
            model = load_vgg16(model_path)
        elif args.model == 'regnet':
            model = load_regnet(model_path)
        elif args.model == 'resnet50':
            model = load_resnet50(model_path)
        
        # กำหนดการแปลงข้อมูลตามโมเดลที่เลือก
        transform = transforms_dict[args.model]
        
        # ตรวจสอบว่า input เป็นไฟล์หรือโฟลเดอร์
        if os.path.isdir(args.input):
            # ทำนายทั้งโฟลเดอร์
            predict_folder(model, args.input, transform, CLASS_NAMES, args.output_dir, args.model)
        else:
            # ทำนายไฟล์เดียว
            predict_image(model, args.input, transform, CLASS_NAMES, args.output_dir, args.model)
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()