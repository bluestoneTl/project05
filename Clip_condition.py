import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 定义图片目录
image_dir = "datasets/ZZCX_01_20/train_RGB/HQ"  

# 定义保存特征的目录
feature_dir = "datasets/ZZCX_01_20/train/condition_RGB"
os.makedirs(feature_dir, exist_ok=True)

# 定义图片转换
transform = transforms.Compose([
    transforms.Resize((550, 550)),  # 确保图片尺寸为550x550
    transforms.ToTensor()
])

# 遍历图片目录
for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        # 读取图片
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0).to(device)  # 添加批次维度并移动到设备上

        # 使用CLIP模型提取特征
        with torch.no_grad():
            image_features = model.encode_image(image)

        # 保存特征
        feature_filename = os.path.splitext(filename)[0] + ".pt"
        feature_path = os.path.join(feature_dir, feature_filename)
        torch.save(image_features.cpu(), feature_path)

print("图片特征提取并保存完成。")