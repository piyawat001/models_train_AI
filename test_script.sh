#!/bin/bash
# สคริปต์สำหรับทดสอบโมเดลที่ฝึกอบรมแล้ว

# เรียกใช้สภาพแวดล้อมเสมือน
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "ไม่พบสภาพแวดล้อมเสมือน กรุณาเรียกใช้ train.sh ก่อน"
    exit 1
fi

# ตรวจสอบพารามิเตอร์
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "วิธีใช้: ./test.sh [options]"
    echo "  --all              ทดสอบทุกโมเดล (ค่าเริ่มต้น)"
    echo "  --efficientnetv2   ทดสอบเฉพาะโมเดล EfficientNetV2"
    echo "  --inceptionv3      ทดสอบเฉพาะโมเดล InceptionV3"
    echo "  --vgg16            ทดสอบเฉพาะโมเดล VGG16"
    echo "  --regnet           ทดสอบเฉพาะโมเดล RegNet"
    echo "  --resnet50         ทดสอบเฉพาะโมเดล ResNet50"
    echo "  --model_path PATH  ระบุพาธของโมเดลที่ต้องการทดสอบโดยตรง"
    echo "  --output_dir DIR   ระบุโฟลเดอร์สำหรับบันทึกผลลัพธ์ (ค่าเริ่มต้น: test_results)"
    exit 0
fi

# กำหนดค่าเริ่มต้น
MODEL="all"
MODEL_PATH=""
OUTPUT_DIR="test_results"

# ประมวลผลพารามิเตอร์
while [[ $# -gt 0 ]]; do
    case $1 in
        --efficientnetv2)
            MODEL="efficientnetv2"
            shift
            ;;
        --inceptionv3)
            MODEL="inceptionv3"
            shift
            ;;
        --vgg16)
            MODEL="vgg16"
            shift
            ;;
        --regnet)
            MODEL="regnet"
            shift
            ;;
        --resnet50)
            MODEL="resnet50"
            shift
            ;;
        --all)
            MODEL="all"
            shift
            ;;
        --model_path)
            MODEL_PATH="--model_path $2"
            shift
            shift
            ;;
        --output_dir)
            OUTPUT_DIR=$2
            shift
            shift
            ;;
        *)
            echo "ไม่รู้จักตัวเลือก: $1"
            echo "ใช้คำสั่ง './test.sh --help' เพื่อดูวิธีใช้"
            exit 1
            ;;
    esac
done

# เริ่มการทดสอบ
echo "กำลังเริ่มการทดสอบโมเดล: $MODEL"
python test_models.py --model $MODEL $MODEL_PATH --output_dir $OUTPUT_DIR

echo "เสร็จสิ้นการทดสอบ ผลลัพธ์ถูกบันทึกไว้ที่: $OUTPUT_DIR"
