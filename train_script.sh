#!/bin/bash
# สคริปต์สำหรับฝึกอบรมโมเดลทั้งหมดหรือเฉพาะที่เลือก

# สร้างสภาพแวดล้อมเสมือน (หากยังไม่ได้สร้าง)
if [ ! -d "venv" ]; then
    echo "กำลังสร้างสภาพแวดล้อมเสมือน..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# ตรวจสอบพารามิเตอร์
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "วิธีใช้: ./train.sh [options]"
    echo "  --all              ฝึกอบรมทุกโมเดล (ค่าเริ่มต้น)"
    echo "  --efficientnetv2   ฝึกอบรมเฉพาะโมเดล EfficientNetV2"
    echo "  --inceptionv3      ฝึกอบรมเฉพาะโมเดล InceptionV3"
    echo "  --vgg16            ฝึกอบรมเฉพาะโมเดล VGG16"
    echo "  --regnet           ฝึกอบรมเฉพาะโมเดล RegNet"
    echo "  --resnet50         ฝึกอบรมเฉพาะโมเดล ResNet50"
    exit 0
fi

# กำหนดโมเดลที่จะฝึกอบรม
MODELS="all"
if [ $# -gt 0 ]; then
    if [ "$1" == "--efficientnetv2" ]; then
        MODELS="efficientnetv2"
    elif [ "$1" == "--inceptionv3" ]; then
        MODELS="inceptionv3"
    elif [ "$1" == "--vgg16" ]; then
        MODELS="vgg16"
    elif [ "$1" == "--regnet" ]; then
        MODELS="regnet"
    elif [ "$1" == "--resnet50" ]; then
        MODELS="resnet50"
    elif [ "$1" == "--all" ]; then
        MODELS="all"
    else
        echo "ไม่รู้จักตัวเลือก: $1"
        echo "ใช้คำสั่ง './train.sh --help' เพื่อดูวิธีใช้"
        exit 1
    fi
fi

# เริ่มการฝึกอบรม
echo "กำลังเริ่มการฝึกอบรมโมเดล: $MODELS"
python run_all.py --models $MODELS --mode train

echo "เสร็จสิ้นการฝึกอบรม"