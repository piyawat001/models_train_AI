"""
ResNet50 โมเดลสำหรับการจำแนกประเภทภาพรังสี
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
import random
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from dotenv import load_dotenv

# โหลดค่าจากไฟล์ .env
load_dotenv()

# กำหนดค่าเริ่มต้นจากไฟล์ .env
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))  # ค่าเริ่มต้นให้เหมาะสมกับ GPU 3080 ที่มี VRAM 10GB
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '100'))  # ลด epochs ลงเพื่อให้ทดสอบได้เร็วขึ้น
PATIENCE = int(os.getenv('PATIENCE', '10'))  # early stopping patience
LEARNING_RATE = float(os.getenv('LEARNING_RATE', '0.0001'))
FINE_TUNE_LEARNING_RATE = float(os.getenv('FINE_TUNE_LEARNING_RATE', '0.00001'))
DATASET_PATH = os.getenv('DATASET_PATH', './augmented_dataset/Classification')
GPU_MEMORY_FRACTION = float(os.getenv('GPU_MEMORY_FRACTION', '0.7'))  # เพิ่มการใช้ GPU จาก 0.5 เป็น 0.7 เนื่องจากเป็น 3080

# กำหนดอุปกรณ์ (GPU หรือ CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ใช้อุปกรณ์: {device}")

# จำกัดการใช้หน่วยความจำ GPU ถ้ามี
if torch.cuda.is_available():
    torch.cuda.memory.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)

# กำหนดคลาสสำหรับชุดข้อมูล
class ChestXRayDataset(Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')]
            print(f'พบ {len(images)} ตัวอย่างของคลาส {class_name}')
            return images
        
        self.images = {}
        self.class_names = ['ameloblastoma', 'dentigerous cyst', 'normal jaw', 'okc']
        
        for c in self.class_names:
            self.images[c] = get_images(c)
            
        self.image_dirs = image_dirs
        self.transform = transform
        
    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])
    
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)

# กำหนดคลาสสำหรับชุดข้อมูลทดสอบ
class ChestXRayDatasetTest(Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')]
            print(f'พบ {len(images)} ตัวอย่างของคลาส {class_name}')
            return images
        
        self.images = {}
        self.class_names = ['ameloblastoma', 'dentigerous cyst', 'normal jaw', 'okc']
        
        for c in self.class_names:
            self.images[c] = get_images(c)
            
        self.image_dirs = image_dirs
        self.transform = transform
        
    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])
    
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name), image_name

# กำหนดการแปลงข้อมูลสำหรับการเทรนและการทดสอบ (ตามที่ใช้ใน notebook)
train_transform = transforms.Compose([
    transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),  # เพิ่มการหมุนภาพเพื่อเพิ่ม data augmentation
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # ปรับความสว่างและความคมชัด
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ฟังก์ชันการแสดงภาพ
def show_images(images, labels, preds, class_names):
    plt.figure(figsize=(16, 9))
    for i in range(min(6, len(images))):
        plt.subplot(1, 6, i + 1, xticks=[], yticks=[])
        image = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green'
        if preds[i] != labels[i]:
            col = 'red'
        plt.xlabel(f'{class_names[int(labels[i].cpu().numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].cpu().numpy())]}', color=col)
    plt.tight_layout()
    plt.savefig('prediction_examples_resnet50.png')
    plt.close()

# คลาส EarlyStopping สำหรับบันทึกโมเดลที่ดีที่สุด
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path='Resnet50_Best-Model.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# คลาส EarlyStopping สำหรับ Fine-tuning
class EarlyStoppingFineTune:
    def __init__(self, patience=5, delta=0, verbose=False, path='Resnet50_Fine-Tune.pt'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ฟังก์ชันสำหรับการเทรนโมเดล
def train(model, optimizer, loss_fn, dl_train, dl_test, test_dataset, epochs, patience, save_path, device):
    print('เริ่มการฝึกอบรม...')
    # EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=save_path)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for e in range(0, epochs):
        print('\n')
        print('=' * 20)
        print(f'เริ่ม epoch {e + 1}/{epochs}')
        print('=' * 20)

        train_loss = 0.
        val_loss = 0.

        model.train()  # ตั้งโมเดลให้อยู่ในโหมดเทรน

        # ลูปเทรน
        pbar_train = tqdm(dl_train, desc=f'Epoch {e+1}/{epochs} [Train]')
        for train_step, (images, labels) in enumerate(pbar_train):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar_train.set_postfix({'loss': loss.item()})

        train_loss /= (train_step + 1)
        history['train_loss'].append(train_loss)

        print(f'Training Loss: {train_loss:.4f}')

        # ลูปตรวจสอบ
        accuracy = 0
        model.eval()  # ตั้งโมเดลให้อยู่ในโหมดประเมิน

        pbar_val = tqdm(dl_test, desc=f'Epoch {e+1}/{epochs} [Val]')
        for val_step, (images, labels, _) in enumerate(pbar_val):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                accuracy += sum((preds.cpu() == labels.cpu()).numpy())
            pbar_val.set_postfix({'loss': loss.item()})

        val_loss /= (val_step + 1)
        accuracy = accuracy / len(test_dataset)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(accuracy)

        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
        model.train()

        # ตรวจสอบ early stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("\nEarly stopping")
            break

    print('การฝึกอบรมเสร็จสิ้น')
    
    # พล็อตกราฟการเทรน
    plt.figure(figsize=(12, 5))
    
    # กราฟ Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # กราฟ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history_resnet50.png')
    plt.close()
    
    return model

# ฟังก์ชันสำหรับ fine-tuning โมเดล
def train_fine_tune(model, optimizer, loss_fn, dl_train, dl_test, test_dataset, epochs, patience, save_path, device, threshold_percent=50):
    print('เริ่มการ fine-tuning...')
    
    # EarlyStopping
    early_stopping = EarlyStoppingFineTune(patience=patience, verbose=True, path=save_path)
    
    # Freeze layers based on threshold
    total_layers = len(list(model.parameters()))
    threshold_layer = total_layers - (total_layers * threshold_percent // 100)
    count = 0
    for name, param in model.named_parameters():
        if count >= threshold_layer:
            param.requires_grad = True  # Enable gradients from here onwards
        else:
            param.requires_grad = False  # Freeze earlier layers
        count += 1
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for e in range(0, epochs):
        print('\n')
        print('=' * 20)
        print(f'เริ่ม fine-tuning epoch {e + 1}/{epochs}')
        print('=' * 20)

        train_loss = 0.
        val_loss = 0.

        model.train()  # ตั้งโมเดลให้อยู่ในโหมดเทรน

        # ลูปเทรน
        pbar_train = tqdm(dl_train, desc=f'Epoch {e+1}/{epochs} [Train]')
        for train_step, (images, labels) in enumerate(pbar_train):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar_train.set_postfix({'loss': loss.item()})

        train_loss /= (train_step + 1)
        history['train_loss'].append(train_loss)

        print(f'Training Loss: {train_loss:.4f}')

        # ลูปตรวจสอบ
        accuracy = 0
        model.eval()  # ตั้งโมเดลให้อยู่ในโหมดประเมิน

        pbar_val = tqdm(dl_test, desc=f'Epoch {e+1}/{epochs} [Val]')
        for val_step, (images, labels, _) in enumerate(pbar_val):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                accuracy += sum((preds.cpu() == labels.cpu()).numpy())
            pbar_val.set_postfix({'loss': loss.item()})

        val_loss /= (val_step + 1)
        accuracy = accuracy / len(test_dataset)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(accuracy)

        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
        model.train()

        # ตรวจสอบ early stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("\nEarly stopping")
            break

    print('Fine-tuning เสร็จสิ้น')
    
    # พล็อตกราฟ fine-tuning
    plt.figure(figsize=(12, 5))
    
    # กราฟ Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Fine-tuning - Training and Validation Loss')
    
    # กราฟ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Fine-tuning - Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('fine_tuning_history_resnet50.png')
    plt.close()
    
    return model

# ฟังก์ชันสำหรับทดสอบโมเดล
def test_model(model, dl_test, class_names, device, model_name="ResNet50"):
    model.eval()
    y_true = []
    y_pred = []
    
    print(f"ทดสอบโมเดล {model_name}...")
    with torch.no_grad():
        for images, labels, image_names in tqdm(dl_test, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
            # แสดงตัวอย่างการทำนาย (เฉพาะแบทช์แรก)
            if len(y_true) <= len(labels):
                show_images(images, labels, predicted, class_names)
    
    # คำนวณความแม่นยำ
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    total = len(y_true)
    accuracy = correct / total
    
    print(f'ความแม่นยำของโมเดล {model_name}: {accuracy:.4f} ({correct}/{total})')
    
    # สร้าง confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)
    
    # เพิ่มค่าตัวเลขใน confusion matrix
    thresh = conf_matrix_norm.max() / 2.
    for i in range(conf_matrix_norm.shape[0]):
        for j in range(conf_matrix_norm.shape[1]):
            plt.text(j, i, f'{conf_matrix_norm[i, j]:.2f}',
                     ha="center", va="center",
                     color="white" if conf_matrix_norm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()
    
    # สร้างรายงานการจำแนก
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("รายงานการจำแนก:")
    print(report)
    
    with open(f'classification_report_{model_name}.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    return accuracy

# ฟังก์ชันสำหรับสร้างไฟล์ Excel ที่มีผลการทำนาย
def generate_prediction_excel(model, dl_test, class_names, device, save_path):
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for images, labels, image_names in tqdm(dl_test, desc="Generating predictions"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            for i in range(len(images)):
                true_label = labels[i].item()
                pred_label = preds[i].item()
                
                row = {
                    'image_name': image_names[i],
                    'true_class': class_names[true_label],
                    'predicted_class': class_names[pred_label],
                    'correct': true_label == pred_label
                }
                
                # เพิ่มความน่าจะเป็นสำหรับแต่ละคลาส
                for j, class_name in enumerate(class_names):
                    row[f'{class_name}_probability'] = probs[i][j].item()
                
                results.append(row)
    
    # สร้าง DataFrame และบันทึกเป็น Excel
    df = pd.DataFrame(results)
    df.to_excel(save_path, index=False)
    print(f"บันทึกผลการทำนายไปยัง {save_path} เรียบร้อยแล้ว")

# ฟังก์ชันหลัก
def main():
    print("โปรแกรมเทรนโมเดล ResNet50 สำหรับจำแนกประเภทภาพรังสี")
    
    # สร้างโฟลเดอร์สำหรับผลลัพธ์
    output_dir = 'resnet50_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # กำหนดพาธของชุดข้อมูล
    train_dirs = {
        'ameloblastoma': os.path.join(DATASET_PATH, 'train/Ameloblastoma'),
        'dentigerous cyst': os.path.join(DATASET_PATH, 'train/Dentigerous cyst'),
        'normal jaw': os.path.join(DATASET_PATH, 'train/Normal jaw'),
        'okc': os.path.join(DATASET_PATH, 'train/OKC'),
    }
    
    test_dirs = {
        'ameloblastoma': os.path.join(DATASET_PATH, 'test/Ameloblastoma'),
        'dentigerous cyst': os.path.join(DATASET_PATH, 'test/Dentigerous cyst'),
        'normal jaw': os.path.join(DATASET_PATH, 'test/Normal jaw'),
        'okc': os.path.join(DATASET_PATH, 'test/OKC'),
    }
    
    # โหลดชุดข้อมูล
    print("กำลังโหลดชุดข้อมูลฝึกอบรม...")
    train_dataset = ChestXRayDataset(train_dirs, train_transform)
    
    print("\nกำลังโหลดชุดข้อมูลทดสอบ...")
    test_dataset = ChestXRayDatasetTest(test_dirs, test_transform)
    
    # สร้าง DataLoader
    dl_train = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    dl_test = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    
    print(f"จำนวนแบทช์สำหรับฝึกอบรม: {len(dl_train)}")
    print(f"จำนวนแบทช์สำหรับทดสอบ: {len(dl_test)}")
    
    # สร้างโมเดล ResNet50
    try:
        # สำหรับ PyTorch รุ่นใหม่
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    except:
        # สำหรับ PyTorch รุ่นเก่า
        model = models.resnet50(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # ปรับชั้นสุดท้าย (fc) สำหรับการจำแนก 4 คลาส
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model = model.to(device)
    
    # กำหนด loss function และ optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # เทรนโมเดล
    print("\nเริ่มการเทรนโมเดล...")
    trained_model = train(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dl_train=dl_train,
        dl_test=dl_test,
        test_dataset=test_dataset,
        epochs=NUM_EPOCHS,
        patience=PATIENCE,
        save_path=os.path.join(output_dir, 'Resnet50_Best-Model.pt'),
        device=device
    )
    
    # ทดสอบโมเดลที่เทรนแล้ว
    print("\nทดสอบโมเดลที่เทรนแล้ว...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'Resnet50_Best-Model.pt')))
    test_model(model, dl_test, train_dataset.class_names, device, "ResNet50_Base")
    
    # สร้างไฟล์ Excel สำหรับผลการทำนายของโมเดลที่เทรนแล้ว
    generate_prediction_excel(
        model=model,
        dl_test=dl_test,
        class_names=train_dataset.class_names,
        device=device,
        save_path=os.path.join(output_dir, 'ResNet50_Base_Predictions.xlsx')
    )
    
    # Fine-tuning โมเดล
    print("\nเริ่มการ fine-tuning โมเดล...")
    optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LEARNING_RATE)
    fine_tuned_model = train_fine_tune(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        dl_train=dl_train,
        dl_test=dl_test,
        test_dataset=test_dataset,
        epochs=NUM_EPOCHS,
        patience=PATIENCE,
        save_path=os.path.join(output_dir, 'Resnet50_Fine-Tune.pt'),
        device=device,
        threshold_percent=50
    )
    
    # ทดสอบโมเดลที่ fine-tune แล้ว
    print("\nทดสอบโมเดลที่ fine-tune แล้ว...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'Resnet50_Fine-Tune.pt')))
    test_model(model, dl_test, train_dataset.class_names, device, "ResNet50_FineTuned")
    
    # สร้างไฟล์ Excel สำหรับผลการทำนายของโมเดลที่ fine-tune แล้ว
    generate_prediction_excel(
        model=model,
        dl_test=dl_test,
        class_names=train_dataset.class_names,
        device=device,
        save_path=os.path.join(output_dir, 'ResNet50_FineTuned_Predictions.xlsx')
    )
    
    print("\nเสร็จสิ้นการทำงานทั้งหมด!")

if __name__ == "__main__":
    main()