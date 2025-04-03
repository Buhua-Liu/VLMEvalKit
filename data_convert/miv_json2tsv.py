import json
import os
import base64
from datasets import Dataset
import pandas as pd

def encode_image_file_to_base64(image_path):
    """读取图片文件，并返回 base64 编码后的字符串"""
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode('utf-8')

# 图片文件的完整路径前缀
base_path = "/date0/xxy/VLMEvalKit-main/FoodieQA"

# 读取 JSON 文件
with open('/date0/xxy/VLMEvalKit-main/FoodieQA/mivqa_tidy.json', 'r', encoding='utf-8') as fin:
    data = json.load(fin)

out_data = []

for idx, record in enumerate(data):
    index = idx
    # 从记录中获取图片路径列表（字段名为 "images"），将列表转换为以逗号分隔的字符串保存原始路径
    image_paths = record.get("images", [])
    
    # 保留方括号格式，路径用单引号包裹
    original_paths = "[" + ",".join([f"'{path}'" for path in image_paths]) + "]"

    base64_images = []
    # 遍历图片路径列表，转换成完整路径并进行 base64 编码
    for path in image_paths:
        full_path = os.path.join(base_path, path)
        try:
            encoded = encode_image_file_to_base64(full_path)
        except Exception as e:
            encoded = ""
            print(f"Error encoding image {full_path}: {e}")
        base64_images.append(encoded)
    
    # 保证有 4 个图片编码：不足补空，多余只取前 4 个
    if len(base64_images) < 4:
        base64_images.extend([""] * (4 - len(base64_images)))
    else:
        base64_images = base64_images[:4]
    
    # 将4个图片的 base64 编码用逗号连接，生成 image 列内容
    # 对每个 base64 编码用单引号包裹并保留方括号
    formatted_base64 = "[" + ",".join([f"'{b64}'" for b64 in base64_images]) + "]"
    
    # 构造输出记录
    item = {
        "index": index,
        "image": formatted_base64,         # 保留格式化后的 base64 编码
        "image_path": original_paths,      # 保留格式化后的原始图片路径
        "question": record.get("question", ""),
        "answer": record.get("answer", ""),
        "A": "The first image",
        "B": "The second image",
        "C": "The third image",
        "D": "The fourth image"
    }
    out_data.append(item)

df = pd.DataFrame(out_data)
dataset = Dataset.from_pandas(df)
output_path = '/date0/xxy/VLMEvalKit-main/FoodieQA/foodieqa_miv_buhua.tsv' # 本地路径
dataset.to_csv(output_path, sep='\t', index=False)

df = pd.read_csv(output_path, sep='\t')
# 打印前五行
print(df.head())
