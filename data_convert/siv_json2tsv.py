import json
import csv
import os
import base64
from datasets import Dataset
import pandas as pd

def encode_image_file_to_base64(image_path):
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode('utf-8')

# 请将 base_path 修改为图片文件的完整路径前缀（不改变原有的 image_file 路径）
base_path = "/date0/xxy/VLMEvalKit-main/FoodieQA"  # 图片文件目录


with open('/date0/xxy/VLMEvalKit-main/FoodieQA/sivqa_tidy.json', 'r', encoding='utf-8') as fin:
    data = json.load(fin)

out_data = []


for idx, record in enumerate(data):
    index = idx
    image_file = record.get("food_meta", {}).get("food_file", "")
    # 拼接完整图片路径
    full_image_path = os.path.join(base_path, image_file)
    try:
        image_base64 = encode_image_file_to_base64(full_image_path)
    except Exception as e:
        image_base64 = ""
        print(f"Error encoding image {full_image_path}: {e}")

    choices = record.get("choices", [])

    # 保证选项有4个，不足的补空字符串，多余的截断
    if len(choices) < 4:
        choices.extend([""] * (4 - len(choices)))
    else:
        choices = choices[:4]
    item = {
            'index': idx,
            'image': image_base64,
            'question': record.get("question", ""),
            'answer': record.get("answer", []),
            # 'difficulty': json_data.get('difficulty', ''),
            # 'question_type': json_data.get('question_type', '')
        }
        
    # 添加选项
    item['A'] = choices[0]
    item['B'] = choices[1]
    item['C'] = choices[2]
    item['D'] = choices[3]
    
    out_data.append(item)

# 创建Dataset对象
dataset = Dataset.from_pandas(pd.DataFrame(out_data))

# 保存为TSV文件
output_path = '/date0/xxy/VLMEvalKit-main/FoodieQA/foodieqa_siv_buhua.tsv' # 本地路径
dataset.to_csv(output_path, sep='\t', index=False)
        
