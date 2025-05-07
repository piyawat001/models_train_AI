#!/bin/bash
# สคริปต์สำหรับทำนายภาพด้วยโมเดลที่ฝึกอบรมแล้ว

# เรียกใช้สภาพแวดล้อมเสมือน
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "ไม่พบสภาพแวดล้อมเสมือน กรุณาเรียกใช้ train.sh ก่อน"
    exit 1
fi

# ตรวจสอบพารามิเตอร์
if [ "$1" == "--help" ] || [ "$1" == "-h" ] || [ $# -lt 1 ]; then
    echo "วิธีใช้: ./predict.sh <image_path> [options]"
    echo "  <image_path>       พาธของไฟล์ภาพหรือโฟลเดอร์ที่ต้องการทำนาย (จำเป็น)"
    echo "  --efficientnetv2   ใช้โมเดล EfficientNetV2 (ค่าเริ่มต้น)"
    echo "  --inceptionv3      ใช้โมเดล InceptionV3"
    echo "  --vgg16            ใช้โมเดล VGG16"
    echo "  --regnet           ใช้โมเดล RegNet"
    echo "  --resnet50         ใช้โมเดล ResNet50"
    echo "  --ensemble         ใช้ทุกโมเดลและรวมผลการทำนาย"
    echo "  --model_path PATH  ระบุพาธของโมเดลที่ต้องการใช้โดยตรง"
    echo "  --output_dir DIR   ระบุโฟลเดอร์สำหรับบันทึกผลลัพธ์ (ค่าเริ่มต้น: prediction_results)"
    exit 0
fi

# กำหนดค่าเริ่มต้น
INPUT_PATH=$1
shift  # ลบพารามิเตอร์แรกออก (พาธของภาพ)
MODEL="efficientnetv2"
MODEL_PATH=""
OUTPUT_DIR="prediction_results"

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
        --ensemble)
            MODEL="ensemble"
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
            echo "ใช้คำสั่ง './predict.sh --help' เพื่อดูวิธีใช้"
            exit 1
            ;;
    esac
done

# ตรวจสอบว่า input path มีอยู่จริง
if [ ! -e "$INPUT_PATH" ]; then
    echo "ไม่พบไฟล์หรือโฟลเดอร์: $INPUT_PATH"
    exit 1
fi

# เริ่มการทำนาย
echo "กำลังทำนาย $INPUT_PATH ด้วยโมเดล $MODEL"
python predict_image.py --model $MODEL $MODEL_PATH --input "$INPUT_PATH" --output_dir $OUTPUT_DIR

echo "เสร็จสิ้นการทำนาย ผลลัพธ์ถูกบันทึกไว้ที่: $OUTPUT_DIR"
