输电线路数据集，已打好标签，包含三个种类：绝缘子，金具，螺栓。一共3375张，大小7个g![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/29b1597638664c4daa7369a8374dbd24.jpeg)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/55cbbda0815046b58a36d4ae273ab6ec.jpeg)
2
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e2e2180d3c454711b8b32327b8cba2d3.jpeg)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/693268d301a143a9abacbbad6749639a.jpeg)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5301a8bc52964bb7986ca439bec95fc1.jpeg)
感谢您提供的详细信息！以下是针对 **输电线路检测数据集** 的完整训练与推理方案，包含：

- 数据集结构说明  
- `data.yaml` 配置文件  
- **YOLOv8 详细训练代码（支持3类：绝缘子、金具、螺栓）**  
- **验证与推理代码**  
- 模型评估建议  

> ✅ 您的数据集已按标准 YOLO 格式组织（含 `train.txt`, `val.txt`, `test.txt`），可直接用于训练。

---

## 📊 输电线路检测数据集概览

| 项目 | 说明 |
| :--- | :--- |
| **任务类型** | 多类别目标检测（电力设备缺陷识别） |
| **图像总数** | 3,375 张（约 7GB，高分辨率） |
| **标注格式** | YOLO `.txt`（每图一个文件，归一化边界框） |
| **类别数量** | 3 类：<br>• `insulator`（绝缘子）<br>• `fittings`（金具）<br>• `bolt`（螺栓） |
| **应用场景** | 无人机巡检、智能电网、电力运维、AI质检 |

---

## 📂 数据集目录结构（已确认）

```bash
datasets/
│
├── images/
│   ├── train/          # 训练图像
│   ├── val/            # 验证图像
│   └── test/           # 测试图像
│
├── labels/
│   ├── train/          # YOLO .txt 标注文件
│   ├── val/
│   └── test/
│
├── train.txt           # 列出所有训练图像路径
├── val.txt             # 验证集图像路径
├── test.txt            # 测试集图像路径
└── data.yaml           # 配置文件（需创建）
```

> ✅ 您的结构完全符合 YOLO 格式要求！

---

## 📄 `data.yaml` 配置文件

```yaml
# data.yaml
train: ./datasets/images/train
val: ./datasets/images/val
test: ./datasets/images/test

nc: 3
names: ['insulator', 'fittings', 'bolt']
```

> 💡 建议使用英文命名，便于模型训练与日志输出。

---

## 🚀 YOLOv8 训练代码（完整版）

```python
# train_power_line.py
from ultralytics import YOLO
import torch
import os

# ==================== 配置参数 ====================
MODEL_TYPE = 'yolov8s.pt'           # 推荐：yolov8s（平衡速度与精度）
DATA_CONFIG = 'datasets/data.yaml'
EPOCHS = 100                        # 可根据收敛情况调整
BATCH_SIZE = 16                     # 根据 GPU 显存调整（16/32）
IMG_SIZE = 640                      # 输入尺寸
PROJECT_NAME = 'power_line_detection'
EXPERIMENT_NAME = 'yolov8s_640_power'

os.makedirs(f"runs/train/{EXPERIMENT_NAME}", exist_ok=True)

# ==================== 设备检查 ====================
DEVICE = '0' if torch.cuda.is_available() else 'cpu'
print(f"✅ 使用设备: {DEVICE}")

# ==================== 加载模型 ====================
model = YOLO(MODEL_TYPE)

# ==================== 开始训练 ====================
results = model.train(
    data=DATA_CONFIG,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    save=True,
    save_period=10,
    device=DEVICE,
    workers=4,
    optimizer='AdamW',
    lr0=1e-3,
    lrf=1e-2,
    momentum=0.937,
    weight_decay=1e-4,
    warmup_epochs=3.0,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    close_mosaic=10,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.2,
    scale=0.5,
    shear=2.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    patience=20,
)
```

---

## 🔍 模型验证与评估

```python
# validate.py
from ultralytics import YOLO

model = YOLO(f'runs/train/{EXPERIMENT_NAME}/weights/best.pt')
metrics = model.val(
    data=DATA_CONFIG,
    split='val',
    imgsz=IMG_SIZE,
    conf=0.001,
    iou=0.5,
    device=DEVICE,
    save_json=True,
)

print(f"📊 mAP@0.5: {metrics.box.map:.4f}")
print(f"📊 mAP@0.5:0.95: {metrics.box.map50:.4f}")
print(f"📊 Precision: {metrics.box.p:.4f}")
print(f"📊 Recall: {metrics.box.r:.4f}")

# 按类别输出 AP
class_names = ['insulator', 'fittings', 'bolt']
for i, name in enumerate(class_names):
    print(f"  ├─ {name}: AP@0.5 = {metrics.box.ap[i]:.4f}")
```

---

## 🔦 推理代码（支持图片/视频）

```python
# predict.py
from ultralytics import YOLO
import cv2

model = YOLO(f'runs/train/{EXPERIMENT_NAME}/weights/best.pt')

source = 'datasets/images/test'  # 或 'test_video.mp4' 或 '0'（摄像头）

results = model.predict(
    source=source,
    imgsz=640,
    conf=0.3,
    iou=0.5,
    device=DEVICE,
    show=True,
    save=True,
    project='runs/predict/power_line_result',
    name='result',
    line_width=2,
    verbose=True,
)

for r in results:
    boxes = r.boxes
    print(f"🖼️ 图像: {r.path}")
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        print(f"  ├─ {label}: {conf:.2f}")
```

---

## 🎯 优化建议（提升精度）

| 技巧 | 说明 |
|------|------|
| **使用 YOLOv8m / v8l** | 更大模型可提升小部件（如螺栓）检出率 |
| **高分辨率训练（1280）** | 若图像清晰，可尝试 `imgsz=1280` |
| **TTA（Test-Time Augmentation）** | 推理时多尺度+翻转，提升召回率 |
| **后处理过滤** | 设置最小面积阈值，避免误检噪声 |
| **部署到边缘设备** | 导出 ONNX/TensorRT，用于无人机或边缘服务器 |

---

## 📦 模型导出（ONNX / TensorRT）

```python
# export_model.py
from ultralytics import YOLO

model = YOLO('runs/train/yolov8s_640_power/weights/best.pt')

# 导出为 ONNX
model.export(format='onnx', opset=12)

# 导出为 TensorRT（NVIDIA）
model.export(format='torchscript')  # 用于 TensorRT
```

---


