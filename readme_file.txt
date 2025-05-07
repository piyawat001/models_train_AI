# โครงการจำแนกประเภทภาพรังสีกรามด้วย Deep Learning

โปรเจคนี้ใช้โมเดล Deep Learning ที่หลากหลายเพื่อจำแนกประเภทของภาพรังสีกรามเป็น 4 ประเภท:
1. Ameloblastoma
2. Dentigerous cyst
3. Normal jaw
4. OKC (Odontogenic Keratocyst)

## โมเดลที่สนับสนุน

- **EfficientNetV2** - ประสิทธิภาพสูงและขนาดเล็ก
- **InceptionV3** - ความสามารถในการจดจำรูปแบบหลากหลายขนาด
- **VGG16** - โครงสร้างเรียบง่ายและเป็นที่นิยม
- **RegNet** - โมเดลสมัยใหม่ที่ออกแบบมาเพื่อประสิทธิภาพสูง
- **ResNet50** - โมเดล Residual Network ที่มีประสิทธิภาพดี

## ความต้องการของระบบ

- Python 3.8 หรือใหม่กว่า
- PyTorch 1.10 หรือใหม่กว่า
- CUDA Toolkit 11.3 หรือใหม่กว่า (สำหรับการประมวลผลด้วย GPU)
- GPU ที่มีหน่วยความจำอย่างน้อย 8GB (แนะนำ 10GB หรือมากกว่า)
- RAM อย่างน้อย 16GB

## การติดตั้ง

1. โคลนรีโพสิตอรี:
```bash
git clone https://github.com/yourusername/dental-xray-classification.git
cd dental-xray-classification
```

2. สร้างสภาพแวดล้อมเสมือนและติดตั้ง dependencies:
```bash
python -m venv venv
source venv/bin/activate  # บน Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. เตรียมข้อมูล:
   - จัดโครงสร้างโฟลเดอร์ข้อมูลของคุณเป็น:
   ```
   augmented_dataset/
   └── Classification/
       ├── train/
       │   ├── Ameloblastoma/
       │   ├── Dentigerous cyst/
       │   ├── Normal jaw/
       │   └── OKC/
       ├── test/
       │   ├── Ameloblastoma/
       │   ├── Dentigerous cyst/
       │   ├── Normal jaw/
       │   └── OKC/
       └── val/
           ├── Ameloblastoma/
           ├── Dentigerous cyst/
           ├── Normal jaw/
           └── OKC/
   ```

## การใช้งาน

### การฝึกอบรมโมเดลหนึ่งโมเดล

ใช้คำสั่งพื้นฐานเพื่อฝึกอบรมโมเดล:

```bash
python efficientnetv2.py  # สำหรับ EfficientNetV2
python inceptionv3.py     # สำหรับ InceptionV3
python vgg16.py           # สำหรับ VGG16
python regnet.py          # สำหรับ RegNet
python resnet50.py        # สำหรับ ResNet50
```

### การฝึกอบรมหลายโมเดล

หากต้องการฝึกอบรมทุกโมเดล:

```bash
python run_all.py --models all --mode train
```

หรือระบุเฉพาะโมเดลบางตัว:

```bash
python run_all.py --models efficientnetv2 resnet50 --mode train
```

### การปรับแต่งพารามิเตอร์

คุณสามารถปรับแต่งพารามิเตอร์ได้ในไฟล์ `.env` สำหรับแต่ละโมเดล เช่น:
- ขนาดแบทช์
- อัตราการเรียนรู้
- จำนวนรอบการฝึกอบรม
- ค่า patience สำหรับ early stopping

## โครงสร้างของโปรเจค

```
.
├── .env                     # ไฟล์ตั้งค่าพารามิเตอร์
├── README.md                # ไฟล์คำอธิบายโปรเจค (ที่คุณกำลังอ่านอยู่)
├── efficientnetv2.py        # โมเดล EfficientNetV2
├── inceptionv3.py           # โมเดล InceptionV3
├── vgg16.py                 # โมเดล VGG16
├── regnet.py                # โมเดล RegNet
├── resnet50.py              # โมเดล ResNet50
├── run_all.py               # สคริปต์สำหรับรันหลายโมเดล
└── augmented_dataset/       # โฟลเดอร์ข้อมูล
    └── Classification/      # ข้อมูลภาพจำแนกประเภท
        ├── train/           # ชุดข้อมูลฝึกอบรม
        ├── test/            # ชุดข้อมูลทดสอบ
        └── val/             # ชุดข้อมูลตรวจสอบ
```

## ผลลัพธ์

หลังจากการฝึกอบรม แต่ละโมเดลจะสร้างโฟลเดอร์ผลลัพธ์ที่มี:
- โมเดลที่บันทึกไว้ (.pt)
- กราฟการฝึกอบรม (loss และ accuracy)
- Confusion matrix
- ไฟล์ Excel ที่มีผลการทำนาย
- รายงานการจำแนกประเภท

## หมายเหตุ

- การฝึกอบรมโมเดลขนาดใหญ่อย่าง RegNet อาจต้องใช้หน่วยความจำ GPU มากกว่า 10GB ให้ปรับขนาดแบทช์ลงในไฟล์ `.env` หากพบปัญหาเรื่องหน่วยความจำไม่พอ
- ทุกโมเดลรองรับการ fine-tuning หลังการฝึกอบรมเบื้องต้น ซึ่งช่วยปรับปรุงประสิทธิภาพของโมเดล
- ข้อมูลควรมีการแบ่งอย่างสม่ำเสมอระหว่างคลาสต่างๆ เพื่อผลลัพธ์ที่ดีที่สุด

## การแก้ไขปัญหา

หากพบข้อผิดพลาด "CUDA out of memory" ให้ลองวิธีต่อไปนี้:
1. ลดขนาดแบทช์ใน `.env`
2. ลด `GPU_MEMORY_FRACTION` ใน `.env`
3. ใช้โมเดลขนาดเล็กกว่า เช่น VGG16 หรือ ResNet50 แทน RegNet
