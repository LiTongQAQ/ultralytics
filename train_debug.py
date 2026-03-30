from ultralytics import YOLO

# 1. 加载预训练模型（也可以用 yolov8n.yaml 从头训练）
# model = YOLO("yolov8n.pt")
# model = YOLO("yolo26.yaml")
model = YOLO("yolo26n.pt")


# 2. 开始训练（使用内置小数据集 coco8，方便调试）
results = model.train(
    data="coco8.yaml",   # 数据集配置
    epochs=3,            # 调试用，少跑几轮
    imgsz=640,           # 输入图片尺寸
    batch=2,             # batch size，根据显存调整
    device="0",        # 没有GPU时用cpu，有GPU改为 0
    workers=0,           # 调试时设为0，避免多进程干扰断点
    verbose=True,
)

print(results)
