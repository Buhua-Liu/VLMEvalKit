from .text_base import TextBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *


class Foodie_textqa(TextBaseDataset):
    TYPE = 'MCQ'

    # 本地路径
    DATASET_PATHS = {
        'Foodie_mivqa': '/date0/xxy/VLMEvalKit-main/FoodieQA/foodieqa_text_buhua.tsv' # 本地路径
    }

    OPTION_LABELS = ['A', 'B', 'C', 'D'] 

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += '请在A、B、C、D中直接选择答案，尽量简洁的回答理由。 \n'

        msgs = []

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