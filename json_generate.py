import os
import json
from PIL import Image
import requests
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
model.to(device)

PROMPT = "Question: Describe the picture with as much as possible detailed information. Answer:" 

# 自定义函数，用于生成prompt
def generate_prompt(img_pth):
    image = Image.open(img_pth)
    inputs = processor(images=image, text=None , return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    text = generated_text+"And ensure the person with real skin and figure."
    return text

# 指定文件夹路径
source_folder_path = "datasets/makeup_data/instogram"
target_folder_path = "datasets/makeup_data/original_400_400"

# 获取source文件夹内所有图片的路径
file_names = os.listdir(source_folder_path)
image_names = [f for f in file_names if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")]
image_names.sort()

data = []
for image_name in tqdm(image_names):
    path_1 = os.path.join(source_folder_path, image_name)
    path_2 = os.path.join(target_folder_path, image_name)
    sample = {
        "source": path_1,
        "target": path_2,
        "prompt": generate_prompt(path_1)
    }
    data.append(sample)

with open('data.json', 'w') as f:
    json.dump(data, f)
    