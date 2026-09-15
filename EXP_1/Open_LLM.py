import os
import subprocess
import requests
import pandas as pd
from pathlib import Path
import time
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers import AutoProcessor, AutoModelForCausalLM
import json


def pick_best_gpu(allowed_gpus):
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits"
        ],
        encoding="utf-8"
    )

    gpu_info = {}
    for line in result.strip().split("\n"):
        idx, mem_free = line.split(",")
        gpu_info[int(idx.strip())] = int(mem_free.strip())

    candidates = {gpu: gpu_info[gpu] for gpu in allowed_gpus if gpu in gpu_info}
    if not candidates:
        raise ValueError(f"指定的 GPU 不存在或不可用: {allowed_gpus}")

    best_gpu = max(candidates, key=candidates.get)
    return best_gpu

# 只允许从这四张卡里选
ALLOWED_GPUS = [2, 3, 6, 7]

# 先选出空闲显存最多的一张
best_gpu = pick_best_gpu(ALLOWED_GPUS)
print(f"将使用物理 GPU: {best_gpu}")

# 必须在 import torch / transformers 之前设置
os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("未找到环境变量 HF_TOKEN")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
END_POINT=200

def convertAnswerKey2Number(answerKey):
    if answerKey=="A":
        return 0
    elif answerKey=='B':
        return 1
    elif answerKey=='C':
        return 2
    elif answerKey=='D':
        return 3
    else:
        return int(answerKey)-1

def convertChoices2String(choices):
    ans=''
    for i in range(len(choices['text'])):
        # print(choices['label'][i])
        if i == len(choices['text'])-1:
            ans+=(choices['label'][i] +': ' + choices['text'][i]+'.')
        else:
            ans+=(choices['label'][i] +': ' + choices['text'][i]+', ')
    return ans


def load_model(model_name):
    print("model_name =", repr(model_name))
    print("HF_TOKEN =", repr(HF_TOKEN))
    print("type(model_name) =", type(model_name))
    print("type(HF_TOKEN) =", type(HF_TOKEN))

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
        trust_remote_code=True,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype="auto",
        trust_remote_code=True
    ).to(DEVICE)

    model.eval()
    return tokenizer, model


def get_llm_response(tokenizer, model, content, max_new_tokens=1024):
    messages = [
        {"role": "user", "content": content}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )

    result = tokenizer.decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True
    )

    return result

def exp_1(file,save_dir,model,tokenizer):

    df = pd.read_parquet('./{0}/train-00000-of-00001.parquet'.format(file))
    # print(df['question'].iloc[1])

    output_csv = save_dir/'{0}_initial_answer.csv'.format(file)
    output_path=Path(output_csv)
    #恢复之前被断掉的问题
    if output_path.exists():
        rdf=pd.read_csv(output_csv)
        for i in range(len(df)):
            if i>=len(rdf):
                start=i
                break
    else:
        start=0

    for i in range(start, len(df)):
        choice_string = convertChoices2String(df.iloc[i]['choices'])
        question = df.iloc[i]['question']
        contents = (
            'Please answer the question: ' + question +
            ' The choices are in the format of "{label}: {choice}": ' +
            choice_string +
            ' Only answer the label of each choice, such as A, B, C, and D or 1, 2, 3, and 4. '
            'Do not provide any explanations or reasoning; provide only the final answer.'
        )
        ans_pos = convertAnswerKey2Number(df.iloc[i]['answerKey'])
        answer = df.iloc[i]['answerKey'] + ': ' + df.iloc[i]['choices']['text'][ans_pos]

        print('----------------------------Q id: {0}----------------------------'.format(i))
        print(contents)
        print(answer)
        
        #下面的函数是有以下功能
        #如果没有“choice”就说明是超出了rate limit，那么就等三分钟然后重新请求
        #如果代码卡住了，那么就等三分钟然后重新再request
        #这样就可以不同手动的重复执行代码
        max_retry = 5
        retry_count = 0
        while True:
            try:
                response = get_llm_response(tokenizer,model,contents)
                print(response)

                row_df = pd.DataFrame(
                    [[question, response, df.iloc[i]['answerKey']]],
                    columns=["question", "answer", "ground truth"]
                )
                row_df.to_csv(
                    output_csv,
                    mode='a',
                    header=not os.path.exists(output_csv),
                    index=False,
                    encoding='utf-8-sig'
                )

                print('----------------------------Q id: {0} finished----------------------------'.format(i))
                break

            except requests.exceptions.Timeout:
                retry_count += 1
                print(f"Q id {i} 请求超时，第 {retry_count} 次重试，3分钟后继续...")
                if retry_count >= max_retry:
                    print(f"Q id {i} 超过最大重试次数，跳过")
                    break
                time.sleep(180)
                print("三分钟结束，开始重新请求")

            except (requests.exceptions.RequestException, ValueError, KeyError, IndexError) as e:
                retry_count += 1
                print(f"Q id {i} 请求失败: {e}，第 {retry_count} 次重试，3分钟后继续...")
                if retry_count >= max_retry:
                    print(f"Q id {i} 超过最大重试次数，跳过")
                    break
                time.sleep(180)
                print("三分钟结束，开始重新请求")
        if i >= END_POINT - 1:
            break
        
def exp_nature_question(file,save_dir,model,tokenizer):
    output_csv = save_dir/'{0}_initial_answer.csv'.format(file)
    output_path=Path(output_csv)
    print("----------------------------Conducting NQ-open on {0}----------------------------".format(model))
    with open("{0}.dev.jsonl".format(file), "r", encoding="utf-8") as f:
        start=0
        if output_path.exists():
            rdf=pd.read_csv(output_csv)
            start=len(rdf)
        for i, line in enumerate(f):
            if i< start:
                continue
            print('----------------------------Q id: {0}----------------------------'.format(i))
            line = line.strip()
            if not line:
                continue
            row_dict = json.loads(line)
            question=row_dict['question']
            contents = (
                'Please answer the question: {0}\n'
                'Do not provide any explanations or reasoning; provide only the final answer.'
            ).format(question)
            print(contents)
            response = get_llm_response(tokenizer,model,contents)
            print(response)
            
            row_df = pd.DataFrame(
                [[question, response, row_dict['answer']]],
                columns=["question", "answer", "ground truth"]
            )
            row_df.to_csv(
                output_csv,
                mode='a',
                header=not os.path.exists(output_csv),
                index=False,
                encoding='utf-8-sig'
            )
            if i >= END_POINT:
                break
            print('----------------------------Q id: {0} finished----------------------------'.format(i))
    print("----------------------------Conducting NQ-open on {0} finished----------------------------".format(model))
        
def exp_3_NQ_open(file,save_dir,model,tokenizer):
    df = pd.read_csv("./{0}_background.csv".format(file), encoding="utf-8-sig")
    # print(df)
    output_csv = save_dir/'{0}_misleading.csv'.format(file)
    output_path=Path(output_csv)
    print("----------------------------Conducting exp_3 {1} on {0}----------------------------".format(model,file))
    # with open("{0}.jsonl".format(file), "r", encoding="utf-8") as f:
    start=0
    if output_path.exists():
        rdf=pd.read_csv(output_csv)
        start=len(rdf)
    if start>=END_POINT:
        return
    for i in range(start,len(df)):
        background = df.iloc[i]['background']
        question = df.iloc[i]['question']
        contents=(
            'Answer the following questions based on the background information provided.\n'
            'Background: {0}\n'
            'Question: {1}\n'
            'Do not provide any explanations or reasoning; provide only the final answer.'
        ).format(background,question)
        print('----------------------------Q id: {0}----------------------------'.format(i))
        print(contents)
        response = get_llm_response(tokenizer,model,contents)
        print(response)
        row_df = pd.DataFrame(
            [[question, response, df.iloc[i]['ground truth']]],
            columns=["question", "answer", "ground truth"]
        )

        row_df.to_csv(
            output_csv,
            mode='a',
            header=not os.path.exists(output_csv),
            index=False,
            encoding='utf-8-sig'
        )
        print('----------------------------Q id: {0} finished----------------------------'.format(i))
        # time.sleep(time_wait)
    print("----------------------------Conducting exp_3 {1} on {0} finished----------------------------".format(model,file))
        
LLMS = ["qwen/qwen3-8B"]#"meta-llama/Llama-2-7b-hf"
if __name__ == '__main__':
    for llm_id in range(len(LLMS)):
        tokenizer, model=load_model(LLMS[llm_id])
        save_dir=Path('{0}'.format(LLMS[llm_id].replace(':','_')))
        save_dir.mkdir(parents=True, exist_ok=True)
        exp_1('ARC-challenge',save_dir,model,tokenizer)
        # exp_nature_question("NQ-open",save_dir,model,tokenizer)
    # print(list(CONFIG_MAPPING.keys()))