from openai import OpenAI
import os
import json
import time
from datetime import datetime
import pprint
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.append("/Users/monacui/about_src/study/llm/common_module")
from llms import chat_openai


data_set = '/Users/monacui/about_src/study/LLM/sentiment_analysis/data/sentiment_analysis.json'


def read_from_json(path): 
    with open(path, 'r') as f: 
        dict = json.load(f)
    return dict


def analyze_sentiment(sys_prompt, user_prompt, max_tokens): 
    llm = chat_openai.ChatOpenAI()
    messages = llm.build_messages(sys_prompt=sys_prompt, user_prompt = user_prompt)    
    result = llm.chat_completions(model_name = "qwen/qwen3-8b", messages = messages, max_tokens = max_tokens, temperature = 0)
    

##----------------------- 版本1: 批量预测 & 准确率评估 -----------------------## 
# 1. 分别把训练集和groundtruth组织成prompt送给模型做预测和准确率评估   ----------- #
# 2. 存在问题：                                                  ----------- #
# a)  prompt过于复杂，模型解析难度，易导致输出结果混乱；               ----------- #
# b)  数据集大时，可能达到模型上下文限制，增加成本                     ----------- #
# c)  缺乏对模型输出的结构化解析，返回的自由文本结果难以被程序自动化处理   ----------- #
##------------------------------------------------------------------------- #
def version_1(base_url, api_key):
    def parser_data_set(path): 
        dict = read_from_json(path)
        trainning_data = []
        ground_truth = []
        for item in dict['sentiment_analysis_dataset']:
            id = item['id']
            label = item['label']
            text = item['text']
            trainning_data.append(f"{id}:{text}")
            ground_truth.append(f"{id}:{label}")
        trainning_data = "\n".join(trainning_data)
        ground_truth = "\n".join(ground_truth)
        return trainning_data, ground_truth
    (trainning_data, ground_truth) = parser_data_set(data_set)

    print(f"-------------------------------training data----------------------------\n {trainning_data}" )
    print(f"-------------------------------ground truth----------------------------\n {ground_truth}" )
    prompt = f"""
        请帮我进行情感分析，拆分两个步骤：
        1. 请判断以下文本的情感倾向，【文本分析数据集】中包含多个编号和文本，对于每个文本都用正向、负向和中性回答，以表格的方式输出，表头为：编号、文本、模型判断、真实结果
        2. 步骤1的判断结果和【情感倾向结果集】正确结果做比对，给出整体准确率，如有不一致请列出不一致的文本编号和文本
        文本分析数据集：{trainning_data}
        情感倾向结果集：{ground_truth}

        输出结果如下：
        1. 情感分析结果
        2. 准确率
        """
    start = time.time()
    result = analyze_sentiment(base_url, api_key, prompt)
    end = time.time()
    print(f"-------------------------------result----------------------------\n ${result}")
    print("耗时：%.2fs" % (end - start))
    
def version_2():
    def calculate_accuracy(data_with_result):
        right_num = 0
        wrong_num = 0
        wrong_detail = []
        for item in data_with_result: 
            if (item['label'] == item['result']): 
                right_num += 1
            else: 
                wrong_num += 1
                wrong_detail.append(item)
        total_num = right_num + wrong_num
        accuracy_rate = right_num / total_num * 100
        print(f'total: {total_num}, wrong: {wrong_num}, right: {right_num}, accuracy_rate: {accuracy_rate}%')
        if (wrong_num > 0):
            for item in wrong_detail:
                print(item)
    
    dict = read_from_json(data_set)
    data_with_result = []
    for item in dict['sentiment_analysis_dataset']: 
        id = item['id']
        text = item['text']
        label = item['label']

        # prompt原则：角色 + 思维链（CoT）+ 输出格式限定
        sys_prompt = f"""
            你是一名中文情感分析专家，请按照以下分析要求进行情感分析，仅用一个词“正向”、“负向”或者“中性”回答
            1. 如果是表扬、积极、满意、乐观的内容，回答“正向”
            2. 如果是批评、消极、不满、悲观的内容，回答“负向”
            3. 仅为客观描述、商业评价、技术参数，或只有“还不错”“尚可”“比较合理”等弱修饰，无明显情绪色彩，回答“中性”，包括：
                - 客观描述事实；
                - 理性评价（即使有轻微褒贬，但整体平衡、克制、无情绪化语言）；
                - 技术性、说明性、总结性语句。
        """

        user_prompt = f"""
                文本：{text}
                情感倾向：
        """

        result = analyze_sentiment(sys_prompt, user_prompt, 10)
        data_with_result.append({
            'id': id,
            'text': text,
            'label': label,
            'result': result
        })
        print(f'id: {id} | text: {text} | label: {label} | result: {result}')
    calculate_accuracy(data_with_result)

if __name__ == "__main__":
    # version_1()
          
    version_2()
   
    
    