from ultralytics import YOLO
import os

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))

# 加载模型
model_path = os.path.join(script_dir, '2-24.pt')
model = YOLO(model_path)

# 输出类别信息
print(f"模型: {model_path}")
print(f"类别数量: {len(model.names)}")
print("\n类别列表:")
for class_id, class_name in model.names.items():
    print(f"  ID {class_id}: {class_name}")
