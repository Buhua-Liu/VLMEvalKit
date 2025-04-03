import json
from datasets import Dataset
import pandas as pd

# 读取 JSON 文件
with open('/date0/xxy/VLMEvalKit-main/FoodieQA/textqa_tidy.json', 'r', encoding='utf-8') as fin: # 本地路径
    data = json.load(fin)

out_data = []

# 遍历每条记录，并生成输出项
for idx, record in enumerate(data):
    # 获取选项列表，并保证有 4 个选项，不足则补空字符串，多余则截断
    choices = record.get("choices", [])
    if len(choices) < 4:
        choices.extend([""] * (4 - len(choices)))
    else:
        choices = choices[:4]
    
    item = {
        'index': idx,
        'question': record.get("question", ""),
        'food_name': record.get("food_name", ""),
        'cuisine_type': record.get("cuisine_type", ""),
        'question_type': record.get("question_type", ""),
        'A': choices[0],
        'B': choices[1],
        'C': choices[2],
        'D': choices[3],
        'answer': record.get("answer", "")
    }
    out_data.append(item)

# 利用 pandas 构造 DataFrame，再利用 datasets 库保存为 TSV 文件
df = pd.DataFrame(out_data)
dataset = Dataset.from_pandas(df)
output_path = '/date0/xxy/VLMEvalKit-main/FoodieQA/foodieqa_text_buhua.tsv' # 本地路径
dataset.to_csv(output_path, sep='\t', index=False)
