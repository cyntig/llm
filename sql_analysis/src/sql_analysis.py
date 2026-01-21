#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys 
from dotenv import load_dotenv
import pandas
from tqdm.auto import tqdm 
import json
import os
import time
import jsonlines
from checkpoint import Checkpoint
from checkpoint import STORAGE_LEVEL

load_dotenv()

sys.path.append("/Users/monacui/about_src/study/llm/common_module/")

from llms import chat_openai
from db import postgres_utils


pg_utils = postgres_utils.PostgresUtils(os.environ.get("POSTGRES_HOST"), os.environ.get("POSTGRES_PORT"), os.environ.get("POSTGRES_DATABASE"), os.environ.get("POSTGRES_USER"), os.environ.get("POSTGRES_PWD"))
table_schema = "llm"

dataset_2_table = {"SuperStore": {"table": "tbl_super_store", "short": "SuperStore"}, 
                   "厦门工资拖欠2023年.xlsx": {"table": "tbl_salary_xm", "short": "SalaryXm"}
                    }


def get_table_structure(dataset):
    table_name = dataset_2_table[dataset]["table"]
    return pg_utils.get_schema(table_schema, table_name)

    

def get_table_structure_str(dataset):
    table_sturcture = get_table_structure(dataset)
    return ', '.join([item['column_name'] + "-" + item['data_type'] for item in table_sturcture])



def predict_sql(model, row, sys_prompt): 
    question = row['问题']
    user_prompt = f"""
        分析文本: {question}
        sql代码: 
    """

    llm = chat_openai.ChatOpenAI()
    messages = llm.build_messages(sys_prompt=sys_prompt, user_prompt=user_prompt)
    result = llm.chat_completions(model_name=model, messages=messages, max_tokens=500, temperature = 0, extra_body={'enable_thinking': False})

    print("1. 预测sql")
    print(f"{question}: {result}")
    return result

def get_prompt_for_sql(dataset): 
    table_structure = get_table_structure_str(dataset)
    table_name = dataset_2_table[dataset]["table"]
    prompt = f"""
        你是一名BI数据分析专家, 我这有表结构信息和用户问题，严格按照PostgreSQL语法规范生成SQL查询语句

        分析步骤：
        1. 理解llm.tbl_super_store（超市订单表）定义的内容
        2. 请按照如下步骤理解和分析用户问题：
            - 如果不相关，请直接回答"0"
            - 如果相关，请严格按照PostgreSQL语法规范生成SQL查询语句
        
        输出要求：
            - 字段名匹配：生成的SQL语句中，所有字段名必须严格使用表结构信息中的中文字段名称
            - 特殊字段名：如果字段名包含特殊字符（如-、空格），请使用双引号括起来
            - 代码格式：只输出SQL代码，不要额外解释
            - 代码优化：如果有筛选条件，尽量在WHERE子句中使用，而不是在FROM子句中使用子查询
        
        表结构信息：
            - 数据库类型：PostgreSQL
            - 表名：{table_schema}.{table_name}
            - 字段列表：名称-类型 \n {table_structure}


        例如：
        输入：平均每年的运费是多少？
        输出：select avg(运费) from (select sum(运费) as 运费 from llm.tbl_super_store group by 订单年份) t
    """
    return prompt
    
def get_prompt_for_answer():
    prompt = f"""
        你是一个优秀的数据分析师，你的任务是根据问题和问题答案生成标准文本。

        输出要求：
        - 输出文本必须严格按照问题和问题结果生成，不能对问题答案有任何改动，输出内容仅包含标准文本，不能包含原始问题和答案
    """
    return prompt

def get_prompt_for_accuracy():
    prompt = f"""
        你是一个优秀的数据分析师，我现在有一个问题、参考答案，以及数据标注人员生成的答案，你需要阅读问题、参考答案，评判标注人员生成的答案是否也可以作为正确答案。
        不需要人工标注答案与参考答案完全一致才算正确，只要标注的答案包含参考答案中的内容，能够用于回答问题即可算作正确；涉及到数值时，不需要严格要求每位小数都一致，可以四舍五入。
        如果人工标注答案可以作为正确答案，返回1，否则，返回0，只返回0或者1即可，不要包含其他任何描述性内容。
    """
    return prompt

## 分析SQL查询结果并生成标准文本，主要包含以下几个步骤
## 1. 预测sql查询语句
## 2. 执行sql语句
## 3. 生成标准文本
def sql_analysis(dataset, predictSModel, evaluationSModel, cp: Checkpoint = None): 
    evaluationModel = short_2_model[evaluationSModel]
    predictModel = short_2_model[predictSModel]
    df = pandas.read_excel('../data/v12_20250326.xlsx', 'Sheet1')
    ss_df = df[df['数据来源（表格名称）'] == dataset].reset_index(drop=True)
    start_index = cp.get_continuous_index()
    for i, row in tqdm(ss_df.iterrows(), total=len(ss_df)):
        index = i + 1
        if index <= start_index:
        # if row['问题'] != "封存时间为2023年6月份处理来源于小程序的案件有多少？":
            print(f"跳过 {index} - {row['问题']}")
            continue 
        else:
            sql = predict_sql(predictModel, row, get_prompt_for_sql(dataset))
            sql_ret = "很抱歉，无法查询到您提问的相关数据。"
            answer_ret = "很抱歉，无法查询到您提问的相关数据"
            if sql != "0": 
                sql_ret = execute_sql(sql)
                print("2. 执行sql")
                print(f"{sql}: {sql_ret}")
                answer_ret = predict_answer(predictModel, row['问题'], sql_ret, get_prompt_for_answer())
            accuracy = evaluate_accuracy(evaluationModel, row['问题'], row['标准答案'], answer_ret, get_prompt_for_accuracy())
            cp.checkpoint({
                'index': index,
                'uuid': row['UUID'],
                '难度': row['难度'],
                '问题': row['问题'],
                '标准答案': row['标准答案'],
                '问题类型': row['问题类型'],
                '意图类型Label': row['意图类型Label'],
                'sql': sql,
                'sql_ret': sql_ret,
                '标注人员生成的答案': answer_ret,
                'accuracy': accuracy
            })
    return cp.get_data()

def evaluate_accuracy(model, question, ref_answer, answer, sys_prompt): 
    user_prompt = f"""
        问题：{question}
        参考答案：{ref_answer}
        标注人员生成的答案：{answer}

        请回答：
    """

    llm = chat_openai.ChatOpenAI()
    messages = llm.build_messages(sys_prompt=sys_prompt, user_prompt=user_prompt)
    result = llm.chat_completions(model_name=model, messages=messages, max_tokens=500, temperature = 0)
    print(f"{question} - {ref_answer} vs {answer}: {result}")
    return result

def predict_answer(model, question, sql_ret, sys_prompt): 
    user_prompt = f"""
        原始问题: {question}
        问题答案: {sql_ret}

        请回答:
    """

    llm = chat_openai.ChatOpenAI()
    messages = llm.build_messages(sys_prompt=sys_prompt, user_prompt=user_prompt)
    result = llm.chat_completions(model_name=model, messages=messages, max_tokens=500, temperature = 0, extra_body={'enable_thinking': False})

    print("3. 生成标准文本")
    print(f"{question}: {result}")
    return result

def save_excel(results, dataset, model, version):
    file_path = f"../out/{version}_{dataset_2_table[dataset]["short"]}.xlsx"
    print(f"保存文件: {file_path}")
    df = pandas.DataFrame(results)
    if not os.path.exists(file_path):
        with pandas.ExcelWriter(file_path) as writer:
            df.to_excel(writer, sheet_name=model, index=False)
    else:
        with pandas.ExcelWriter(file_path, mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=model, index=False)

def execute_sql(sql):
    conn = pg_utils.create_connection()
    return pg_utils.execute_sql(conn, sql)

def calculate_accuracy_rate(result): 
    total = len(result)
    correct = sum([1 if item['accuracy'] == '1' else 0 for item in result])
    accuracy_rate = correct / total
    print(f"Total: {total}, Correct: {correct}, Accuracy Rate: {accuracy_rate}")

if __name__ == "__main__":
    # dataset: SuperStore, 厦门工资拖欠2023年.xlsx

    SMODEL_8B = '8b'
    SMODEL_30B = '30b'
    SMODEL_235B = '235b'
    short_2_model = {'8b': 'Qwen/Qwen3-8B', '30b': 'Qwen/Qwen3-30B-A3B-Instruct-2507', '235b': 'Qwen/Qwen3-235B-A22B-Instruct-2507'}

    usedModel = SMODEL_30B

    dataset = sys.argv[1]

    print(f"使用模型: {usedModel}")  
    version = "v2"

    
    checkpoint_dir = "../out/checkpoint/"
    file_name = f"cp_{version}_{usedModel}_{dataset_2_table[dataset]["short"]}.jsonl" 
    
    ck_pt = Checkpoint(checkpoint_dir, file_name, STORAGE_LEVEL.DISK)
    ck_pt.initialize()

    results = sql_analysis(dataset, usedModel, SMODEL_235B, ck_pt)
    calculate_accuracy_rate(results)
    save_excel(results, dataset, usedModel, version)



