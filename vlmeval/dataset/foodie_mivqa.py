import sys
import warnings
import pandas as pd
import string
from pathlib import Path
from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *


class Foodie_mivqa(ImageBaseDataset):
    TYPE = 'MCQ'
    # 本地路径
    DATASET_PATHS = {
        'Foodie_mivqa': '/date0/xxy/VLMEvalKit-main/FoodieQA/foodieqa_miv_buhua.tsv' # 本地路径
    }

    OPTION_LABELS = ['A', 'B', 'C', 'D'] 

    def __init__(self, dataset_name, **kwargs):
        super().__init__(dataset_name, **kwargs)
        if dataset_name in self.DATASET_PATHS:
            self.data = self.load_from_local(self.DATASET_PATHS[dataset_name])
            self.validate_data()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")



    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]
            
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)  # 假设返回单个路径或路径列表

        # 将多图合并为单图（通过 concat_images_vlmeval）
        if isinstance(tgt_path, list):
            # 合并图片并获取新路径
            merged_image_path = concat_images_vlmeval(
                image_paths=tgt_path,
                target_size=512
            )
            tgt_path = merged_image_path  # 最终只保留合并后的图片路径
        else:
            # 单图直接使用原路径
            pass

        # 准备问题和选项
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = '选项:\n' + '\n'.join([f'{k}. {v}' for k, v in options.items()]) if options else ''
        
        # 添加 Hint
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = f"Hint: {hint}\n" if hint else ""
        prompt += f"问题: {question}\n{options_prompt}"
        if options:
            prompt += "\n请根据选项提示，在A、B、C、D中选择一个答案。"

        # 生成图像部分的内容（每个图像都是独立的项）
        image_msgs = []
        if isinstance(tgt_path, list):
            # 如果是多图，依次添加每张图片
            for path in tgt_path:
                image_msgs.append(dict(type='image', value=path))
        else:
            # 单张图像
            image_msgs.append(dict(type='image', value=tgt_path))

        # 添加文本内容
        full_prompt = f"Question: {question}\n{options_prompt}\n请根据选项提示，在A、B、C、D中选择一个答案。\n"
        full_prompt += f"Hint: {hint}\n" if hint else ""
        full_prompt += f"\n{prompt}"

        # 将图像和文本信息合并
        result = image_msgs  # 先加图像部分
        result.append(dict(type='text', value=full_prompt))  # 最后加文本部分

        return result



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