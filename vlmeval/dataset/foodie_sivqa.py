import sys
import warnings
import pandas as pd
import string
from pathlib import Path
from .image_base import ImageBaseDataset
from .image_mcq import ImageMCQDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *

class Foodie_sivqa(ImageBaseDataset):
    TYPE = 'MCQ'
    # 本地路径
    DATASET_PATHS = {
        'Foodie_mivqa': '/date0/xxy/VLMEvalKit-main/FoodieQA/foodieqa_siv_buhua.tsv' # 本地路径
    }

    OPTION_LABELS = ['A', 'B', 'C', 'D'] 

    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name, **kwargs)
        if dataset_name in self.DATASET_PATHS:
            self.data = self.load_from_local(self.DATASET_PATHS[dataset_name])
            self.validate_data()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # def load_from_local(self, file_path):
    #     """加载TSV文件并解析image列"""
    #     df = pd.read_csv(file_path, sep='\t')
    #     # 拆分image列为列表
    #     df['image_list'] = df['image'].apply(lambda x: [p.strip() for p in x.split(',')])
    #     return df
    
    # def validate_data(self):
    #     """验证数据完整性"""
    #     for idx, row in self.data.iterrows():
    #         # 检查图片数量是否为4
    #         if len(row['image_list']) != 4:
    #             raise ValueError(f"Row {idx} has {len(row['image_list'])} images, expected 4")
    #         # 检查答案是否合法
    #         if row['answer'].upper() not in self.OPTION_LABELS:
    #             raise ValueError(f"Invalid answer {row['answer']} in row {idx}")

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)
        
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        
        options_prompt = '选项:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        
        prompt = ''
        prompt += f'问题: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += "请直接回答选项中的字母。\n"
        
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.vmcbench import get_mc_score, report_vmc_acc
        suffix = eval_file.split('.')[-1]
        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        data['hit'] = data.apply(get_mc_score, axis=1)
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')
        dump(data, result_file)
        acc = report_vmc_acc(data)
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(acc, score_file)

        return acc