"""
InceptionV3 โมเดลสำหรับการจำแนกประเภทภาพรังสี
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv

# โหลดค่าจากไฟล์ .env
load_dotenv()

# กำหนดค่าเริ่มต้นจากไฟล์ .env
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))  # ค่าเริ่มต้นให้เหมาะสมกับ GPU 3080 ที่มี VRAM 10GB
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '100'))  # ลด epochs ลงเพื่อให้ทดสอบได้เร็วขึ้น
PATIENCE = int(os.getenv('PATIENCE', '10'))  # early stopping patience
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.0001'))
DATASET_PATH = os.getenv('DATASET_PATH', './augmented_dataset/Classification')
GPU_MEMORY_FRACTION = float(os.getenv('GPU_MEMORY_FRACTION', '0.7'))  # เพิ่มการใช้ GPU จาก 0.5 เป็น 0.7 เนื่องจากเป็น 3080

# กำหนดอุปกรณ์ (GPU หรือ CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ใช้อุปกรณ์: {device}")

# จำกัดการใช้หน่วยความจำ GPU ถ้ามี
if torch.cuda.is_available():
    torch.cuda.memory.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)

# กำหนดการแปลงข้อมูลสำหรับการเทรนและการตรวจสอบ
train_transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 ต้องการขนาดภาพเป็น 299x299
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # เพิ่มการหมุนภาพเพื่อเพิ่ม data augmentation
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ปรับความสว่างและความคมชัด
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((299, 299)),  # InceptionV3 ต้องการขนาดภาพเป็น 299x299
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# โหลดชุดข้อมูล
def load_datasets(dataset_path):
    """โหลดชุดข้อมูลจากพาธที่กำหนด"""
    train_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(dataset_path, 'val'), transform=val_transform)
    test_dataset = ImageFolder(os.path.join(dataset_path, 'test'), transform=val_transform)
    
    print(f"จำนวนภาพในชุดข้อมูลเทรน: {len(train_dataset)}")
    print(f"จำนวนภาพในชุดข้อมูลตรวจสอบ: {len(val_dataset)}")
    print(f"จำนวนภาพในชุดข้อมูลทดสอบ: {len(test_dataset)}")
    print(f"คลาสทั้งหมด: {train_dataset.classes}")
    
    return train_dataset, val_dataset, test_dataset

# สร้าง DataLoader
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """สร้าง DataLoader สำหรับชุดข้อมูลที่กำหนด"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }
    
    return dataloaders, dataset_sizes

# สร้างโมเดล InceptionV3
def create_model(num_classes):
    """สร้างโมเดล InceptionV3 และปรับแต่งให้เหมาะสมกับงานที่ต้องการ"""
    try:
        # สำหรับ PyTorch รุ่นใหม่
        inception = models.inception_v3(weights='IMAGENET1K_V1')
    except:
        try:
            # สำหรับ PyTorch รุ่นกลาง
            from torchvision.models import Inception_V3_Weights
            inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        except:
            # สำหรับ PyTorch รุ่นเก่า
            inception = models.inception_v3(pretrained=True)
    
    # ปรับชั้นเชื่อมต่อสุดท้ายให้เข้ากับจำนวนคลาสในชุดข้อมูลของเรา
    inception.fc = nn.Linear(inception.fc.in_features, num_classes)
    
    # InceptionV3 จำเป็นต้องกำหนด aux_logits=False ถ้าคุณไม่ต้องการใช้ auxiliary classifier
    # ถ้าต้องการใช้ aux_logits, ต้องปรับ AuxLogits.fc เช่นกัน
    if hasattr(inception, 'AuxLogits'):
        inception.AuxLogits.fc = nn.Linear(inception.AuxLogits.fc.in_features, num_classes)
    
    return inception.to(device)

# ฟังก์ชันการฝึกอบรมหลัก
def train_model(model, optimizer, dataloaders, dataset_sizes, num_epochs, patience, save_dir):
    """ฝึกอบรมโมเดลด้วยการใช้ early stopping"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_val_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}', '-' * 10)
        
        # แต่ละ epoch มีช่วงเทรนและตรวจสอบ
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # โหมดเทรน
            else:
                model.eval()   # โหมดประเมิน
                
            running_loss = 0.0
            running_corrects = 0
            
            # วนลูปผ่านชุดข้อมูล
            pbar = tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch+1}')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # ล้างค่าเกรเดียนต์
                optimizer.zero_grad()
                
                # ส่งข้อมูลเข้าโมเดลและคำนวณเกรเดียนต์ในช่วงเทรนเท่านั้น
                with torch.set_grad_enabled(phase == 'train'):
                    # InceptionV3 เป็นพิเศษตรงที่มี aux_logits
                    if phase == 'train' and hasattr(model, 'aux_logits') and model.aux_logits:
                        outputs, aux_outputs = model(inputs)
                        loss1 = nn.CrossEntropyLoss()(outputs, labels)
                        loss2 = nn.CrossEntropyLoss()(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2  # ค่าถ่วงน้ำหนักของ aux_logits
                    else:
                        outputs = model(inputs)
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    # ย้อนกลับและอัปเดตพารามิเตอร์ในช่วงเทรนเท่านั้น
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # สถิติ
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                pbar.set_postfix({'loss': loss.item()})
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # บันทึกค่าสำหรับพล็อต
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
            
            # บันทึกโมเดลที่ดีที่สุดตามความแม่นยำของชุดตรวจสอบ
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                    torch.save(best_model_wts, os.path.join(save_dir, 'best_model_acc.pth'))
                
                # บันทึกโมเดลที่ดีที่สุดตามค่าความสูญเสียของชุดตรวจสอบ
                if epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss
                    counter = 0
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_model_loss.pth'))
                else:
                    counter += 1
                    
                # Early stopping
                if counter >= patience:
                    print(f'Early stopping หลังจาก {patience} epochs ที่ไม่มีการปรับปรุง')
                    break
        
        # หยุดการเทรนหากถึงเงื่อนไข early stopping
        if counter >= patience:
            break
    
    # โหลดน้ำหนักที่ดีที่สุด
    model.load_state_dict(best_model_wts)
    
    # พล็อตกราฟการเทรน
    plot_and_save_training_results(train_losses, val_losses, train_accs, val_accs, save_dir)
    
    return model

# ฟังก์ชันการ fine-tuning โมเดล
def fine_tune_model(model, optimizer, dataloaders, dataset_sizes, num_epochs, patience, save_dir):
    """ฟังก์ชันสำหรับ fine-tuning โมเดลที่เทรนแล้ว"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_val_loss = float('inf')
    counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # ปลดล็อคทุกชั้นสำหรับการปรับแต่ง
    for param in model.parameters():
        param.requires_grad = True
    
    # ลดอัตราการเรียนรู้สำหรับการปรับแต่ง
    for param_group in optimizer.param_groups:
        param_group['lr'] = LEARNING_RATE / 10
    
    for epoch in range(num_epochs):
        print(f'Fine-tuning Epoch {epoch+1}/{num_epochs}', '-' * 10)
        
        # แต่ละ epoch มีช่วงเทรนและตรวจสอบ
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # โหมดเทรน
            else:
                model.eval()   # โหมดประเมิน
                
            running_loss = 0.0
            running_corrects = 0
            
            # วนลูปผ่านชุดข้อมูล
            pbar = tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch+1}')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # ล้างค่าเกรเดียนต์
                optimizer.zero_grad()
                
                # ส่งข้อมูลเข้าโมเดลและคำนวณเกรเดียนต์ในช่วงเทรนเท่านั้น
                with torch.set_grad_enabled(phase == 'train'):
                    # InceptionV3 เป็นพิเศษตรงที่มี aux_logits
                    if phase == 'train' and hasattr(model, 'aux_logits') and model.aux_logits:
                        outputs, aux_outputs = model(inputs)
                        loss1 = nn.CrossEntropyLoss()(outputs, labels)
                        loss2 = nn.CrossEntropyLoss()(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2  # ค่าถ่วงน้ำหนักของ aux_logits
                    else:
                        outputs = model(inputs)
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    # ย้อนกลับและอัปเดตพารามิเตอร์ในช่วงเทรนเท่านั้น
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # สถิติ
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                pbar.set_postfix({'loss': loss.item()})
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # บันทึกค่าสำหรับพล็อต
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc)
            
            # บันทึกโมเดลที่ดีที่สุดตามความแม่นยำของชุดตรวจสอบ
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    counter = 0
                    torch.save(best_model_wts, os.path.join(save_dir, 'fine_tuned_model_acc.pth'))
                
                # บันทึกโมเดลที่ดีที่สุดตามค่าความสูญเสียของชุดตรวจสอบ
                if epoch_loss < min_val_loss:
                    min_val_loss = epoch_loss
                    counter = 0
                    torch.save(model.state_dict(), os.path.join(save_dir, 'fine_tuned_model_loss.pth'))
                else:
                    counter += 1
                    
                # Early stopping
                if counter >= patience:
                    print(f'Early stopping หลังจาก {patience} epochs ที่ไม่มีการปรับปรุง')
                    break
        
        # หยุดการเทรนหากถึงเงื่อนไข early stopping
        if counter >= patience:
            break
    
    # โหลดน้ำหนักที่ดีที่สุด
    model.load_state_dict(best_model_wts)
    
    # พล็อตกราฟการเทรน
    plot_and_save_training_results(train_losses, val_losses, train_accs, val_accs, save_dir, prefix='fine_tuned_')
    
    return model

# ฟังก์ชันสำหรับพล็อตผลลัพธ์การเทรน
def plot_and_save_training_results(train_losses, val_losses, train_accs, val_accs, save_dir, prefix=''):
    """บันทึกกราฟการเทรน"""
    # พล็อตค่าสูญเสีย
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # พล็อตความแม่นยำ
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}training_results.png'))
    plt.close()

# ฟังก์ชันสำหรับประเมินโมเดลในชุดข้อมูลทดสอบ
def evaluate_model(model, dataloader, class_labels, save_dir):
    """ประเมินโมเดลและสร้าง confusion matrix"""
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # คำนวณความแม่นยำโดยรวม
    accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
    print(f'ความแม่นยำในชุดข้อมูลทดสอบ: {accuracy * 100:.2f}%')
    
    # สร้างและบันทึก confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    
    # เพิ่มค่าตัวเลขใน confusion matrix
    thresh = conf_matrix_norm.max() / 2.
    for i in range(conf_matrix_norm.shape[0]):
        for j in range(conf_matrix_norm.shape[1]):
            plt.text(j, i, f'{conf_matrix_norm[i, j]:.2f}',
                     ha="center", va="center",
                     color="white" if conf_matrix_norm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    return accuracy, conf_matrix

# ฟังก์ชันสำหรับสร้างไฟล์ Excel ที่มีผลการทำนาย
def generate_prediction_excel(model, dataloader, dataset, class_labels, save_path):
    """สร้างไฟล์ Excel ที่มีผลการทำนาย"""
    model.eval()
    result_df = pd.DataFrame(columns=["file_name", "true_class", "predicted_class"] + class_labels)
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloader, desc='Generating predictions')):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            for j in range(inputs.size(0)):
                # คำนวณ index ที่ถูกต้อง
                idx = i * dataloader.batch_size + j
                if idx < len(dataset):
                    file_path = dataset.samples[idx][0]
                    true_class = class_labels[labels[j]]
                    predicted_class = class_labels[preds[j]]
                    
                    row = {
                        "file_name": file_path,
                        "true_class": true_class,
                        "predicted_class": predicted_class
                    }
                    
                    # เพิ่มความน่าจะเป็นสำหรับแต่ละคลาส
                    for k, class_name in enumerate(class_labels):
                        row[class_name] = probs[j][k].item()
                    
                    result_df = pd.concat([result_df, pd.DataFrame([row])], ignore_index=True)
    
    # บันทึก DataFrame เป็นไฟล์ Excel
    result_df.to_excel(save_path, index=False)
    print(f'บันทึกผลการทำนายไปยัง {save_path}')

# ฟังก์ชันสำหรับทำนายภาพเดี่ยว
def predict_single_image(model, image_path, class_labels):
    """ทำนายคลาสของภาพเดี่ยว"""
    # โหลดและแปลงภาพ
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(299),  # InceptionV3 ต้องการขนาดภาพเป็น 299x299
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # ทำนาย
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = class_labels[pred_idx]
        
    # แสดงผลลัพธ์
    print(f"ทำนายคลาส: {pred_class} (ความน่าจะเป็น: {probs[pred_idx]:.2%})")
    
    # แสดงความน่าจะเป็นสำหรับทุกคลาส
    for i, class_name in enumerate(class_labels):
        print(f"{class_name}: {probs[i]:.2%}")
    
    return pred_class, probs

# ฟังก์ชัน Main
def main():
    # สร้างโฟลเดอร์สำหรับบันทึกผลลัพธ์
    output_dir = 'inception_outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # โหลดชุดข้อมูล
    train_dataset, val_dataset, test_dataset = load_datasets(DATASET_PATH)
    dataloaders, dataset_sizes = create_dataloaders(train_dataset, val_dataset, test_dataset, BATCH_SIZE)
    
    # สร้างโมเดล
    num_classes = len(train_dataset.classes)
    model = create_model(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ฝึกอบรมโมเดล
    print("เริ่มการฝึกอบรมโมเดล...")
    trained_model = train_model(
        model,
        optimizer,
        dataloaders,
        dataset_sizes,
        NUM_EPOCHS,
        PATIENCE,
        output_dir
    )
    
    # Fine-tuning โมเดล
    print("เริ่มการ fine-tuning โมเดล...")
    fine_tuned_model = fine_tune_model(
        trained_model,
        optimizer,
        dataloaders,
        dataset_sizes,
        NUM_EPOCHS,
        PATIENCE,
        output_dir
    )
    
    # ประเมินโมเดลในชุดข้อมูลทดสอบ
    print("ประเมินโมเดลที่ฝึกอบรมแล้ว (Base Model)...")
    trained_model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model_acc.pth')))
    _, _ = evaluate_model(trained_model, dataloaders['test'], train_dataset.classes, output_dir)
    
    # สร้างไฟล์ Excel สำหรับผลการทำนายของ Base Model
    print("สร้างไฟล์ Excel สำหรับผลการทำนายของ Base Model...")
    generate_prediction_excel(
        trained_model,
        dataloaders['test'],
        test_dataset,
        train_dataset.classes,
        os.path.join(output_dir, 'base_model_predictions.xlsx')
    )
    
    # ประเมินโมเดลที่ fine-tuned ในชุดข้อมูลทดสอบ
    print("ประเมินโมเดลที่ fine-tuned...")
    fine_tuned_model.load_state_dict(torch.load(os.path.join(output_dir, 'fine_tuned_model_acc.pth')))
    _, _ = evaluate_model(fine_tuned_model, dataloaders['test'], train_dataset.classes, output_dir)
    
    # สร้างไฟล์ Excel สำหรับผลการทำนายของ Fine-tuned Model
    print("สร้างไฟล์ Excel สำหรับผลการทำนายของ Fine-tuned Model...")
    generate_prediction_excel(
        fine_tuned_model,
        dataloaders['test'],
        test_dataset,
        train_dataset.classes,
        os.path.join(output_dir, 'fine_tuned_model_predictions.xlsx')
    )
    
    print("การประเมินเสร็จสิ้น")

if __name__ == "__main__":
    main()