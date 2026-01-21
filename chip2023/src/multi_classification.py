#!/usr/bin/python
# -*- coding: UTF-8 -*-

from openai import OpenAI
import sys
import os
import json
from tqdm.auto import tqdm 

from dotenv import load_dotenv

load_dotenv()

sys.path.append("/Users/monacui/about_src/study/llm/common_module")

from llms import chat_openai


print(os.getcwd())

DATA_SET = '../data/dev.txt'
CLASS_SET = '../data/class.txt'
PRED_FILE = "../data/pred_new.txt"


def multi_classification(user_prompt, max_tokens): 
    sys_prompt = f"""
            你是一名中文糖尿病问题分类专家。请严格按照以下规则进行分类：
            分类标准(互斥且全面)：
            0-诊断相关(Diagnosis): 仅涉及诊断过程、检查方法、确认标准、症状判断。如"餐后血糖10.5正常吗？","空腹78血糖算糖尿病吗？"
            1-治疗相关(Treatment): 仅涉及药物治疗、手术、疗法、治疗方案。如"孕妇得了糖尿病怎么办","肥胖型糖尿病怎么办"
            2-常识知识(Common Knowledge): 糖尿病基础知识、概念解释、病例机制。如"吃糖会不会增加糖尿病的危险","糖尿病患者为什么一到晚上就感觉尿部出来"
            3-健康生活(Healthy Lifestyle): 饮食、运动、生活习惯、日常管理。如"妊娠糖尿病能不能吃玉米","糖尿病肌酐高能吃鸡蛋吗"
            4-流行病学(Epidemiology): 发病率、统计数据、风险因素、人群研究。如"糖尿病会引起肾早衰的症状","糖尿病能引起四肢麻木吗"
            5-其他(Other): 费用、设备、机构、非医疗问题。如"糖尿病人敢和爱人同房吗","儿童糖尿病蜜月期能延长多久"

            重要区分规则：
            - 症状原因、并发症风险 -> 流行病学（4）
            - 能吃某种食物、运动建议 -> 健康生活（3）
            - 疾病基础知识、机制原理 -> 常识知识（2）
            - 具体治疗方法、药物使用 -> 治疗相关（1）
            - 诊断标准、检查方法 -> 诊断相关（0）

            输出要求：
            - 只输出一个数字：0、1、2、3、4、5
            - 不要输出任何其他文字、标点或解释
            - 如果不确定，选择最可能的类别

            示例：
            输入："空腹78血糖算糖尿病吗？"
            输出："0"

            输入："胆结石又有糖尿病怎么办"
            输出："1"

            输入："压力大会引起糖尿病吗"
            输出："2"

            输入："糖尿病患者为什么一到晚上就感觉尿部出来"
            输出："2"

            输入："i型糖尿病与肥胖有关吗" 
            输入："2"

            输入："糖尿病胃轻瘫会致命吗"
            输出："2"

            输入："糖尿病人可以吃紫薯和南瓜吗"
            输出："3"

            输入："糖尿病会疲惫吗"
            输出："4"

            输入："糖尿病人可以无痛人流"
            输出："5"

            输入："胰岛素一支需要多少钱"
            输出："5"

        """
    llm = chat_openai.ChatOpenAI()
    messages = llm.build_messages(sys_prompt=sys_prompt, user_prompt = user_prompt)    
    result = llm.chat_completions(model_name = "qwen/qwen3-8b", messages = messages, max_tokens = max_tokens, temperature = 0)
    return result


def read_pred(file_name):
    try:
        with open(file_name, 'r', encoding = 'utf-8') as file: 
            lines = file.readlines()

        data_set = []
        for line in lines: 
            text_2_label = line.rstrip('\n').split('\t')
            text = text_2_label[0]
            label = text_2_label[1]
            data_set.append({
                'text': text,
                'label': label
            })
        return data_set

    except Exception as e: 
        print(e)
        return []
    
def prediction(data): 
    result_set = []
    pbar = tqdm(data)
    n_total = 0
    n_correct = 0
    for item in pbar: 
        n_total += 1
        text = item['text']
        label = item['label']
        user_prompt = f"""
            文本：{text}
            类别编码：/no_think
        """
        result = multi_classification(user_prompt, 10).split('</think>')[1].strip()
        
        result_set.append({
            'text': text,
            'label': label,
            'pre': result
        })
        if (label == result): 
            n_correct += 1
        pbar.set_description(f'total: {n_total}, correct: {n_correct}, accuracy: {n_correct / n_total * 100:.2f}%')
    return result_set


def calculate_accuracy(pre_result): 
    n_wrong = 0
    wrong_list = []
    for item in pre_result:
        label = item['label']
        pre = item['pre']
        if (pre != label): 
            n_wrong += 1
            wrong_list.append(item)
    n_total = len(pre_result)
    accuracy = f"{n_wrong / n_total * 100:.2f}%"
    acc_result = {
        'total': n_total,
        'wrong_cnt': n_wrong,
        'accuracy': accuracy,
        'wrong_list': wrong_list
    }
    return acc_result

def analysis(acc_result): 
    ana_ret = {}
    for item in acc_result['wrong_list']:
        label = item['label']
        text = item['text']
        pre = item['pre']

        if label not in ana_ret.keys(): 
            ana_ret[label] = {}
        if pre not in ana_ret[label].keys(): 
            ana_ret[label][pre] = []
        ana_ret[label][pre].append(text)
    return ana_ret

def save_file(filename, data, indent = 4): 
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)



if __name__ == "__main__":
    data = read_pred(DATA_SET)
    pre_result = prediction(data[1:100])
    save_file(PRED_FILE, pre_result)
    acc_result = calculate_accuracy(pre_result)
    ana_result = analysis(acc_result)

    for label in ana_result.keys(): 
        for pre in ana_result[label].keys(): 
            print(f'{label} -> {pre}: {len(ana_result[label][pre])}, {ana_result[label][pre]}')
