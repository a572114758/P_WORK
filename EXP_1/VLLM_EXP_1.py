import pandas as pd
from pathlib import Path
import os
import requests
import time
import json



from vllm_utils import get_gpu_free_memory,convertChoices2String,convertAnswerKey2Number
from vllm_utils import extract_answer_math_json



END_POINT=200
# 指定要使用的物理 GPU
gpu_list = [2, 3]

gpu_memory = get_gpu_free_memory(gpu_list)

print("Selected GPUs:")
for gpu_id in gpu_list:
    print(f"GPU {gpu_id}, free memory: {gpu_memory[gpu_id]} MiB")

# 让当前程序只看到 gpu_list 中的 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))

print("CUDA_VISIBLE_DEVICES =", os.environ["CUDA_VISIBLE_DEVICES"])


from vllm import LLM, SamplingParams

MODEL_LIST= [
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "Qwen/Qwen2.5-32B-Instruct",
    # "NousResearch/Meta-Llama-3.1-70B-Instruct",
    # "google/gemma-2-27b-it",
]
sampling_params = SamplingParams(
    temperature=1,
    max_tokens=512,
)

# du -sh ~/.cache/huggingface， du -sh ~/.cache/huggingface/hub 查看缓存占了多少 
# rm -rf ~/.cache/huggingface/hub 删除现有缓存
# mkdir -p ~/.cache/huggingface/hub #创建存放模型的文件

def exp_1_multi_choice(file,save_dir,model):
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
        
    llm = LLM(
        model=model,
        dtype="auto",
        trust_remote_code=True,
        tensor_parallel_size=len(gpu_list),
    )
        
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
                outputs = llm.generate([contents], sampling_params)

                raw_response = outputs[0].outputs[0].text.strip()
                print("Raw response:", raw_response)

                response = raw_response.split()[0].strip().strip(".:,;")
                print("Parsed response:", response)

                # row_df = pd.DataFrame(
                #     [[question, response, df.iloc[i]['answerKey']]],
                #     columns=["question", "answer", "ground truth"]
                # )
                # row_df.to_csv(
                #     output_csv,
                #     mode='a',
                #     header=not os.path.exists(output_csv),
                #     index=False,
                #     encoding='utf-8-sig'
                # )
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
        if i >= END_POINT:
            break
    return

if __name__ == "__main__":
    # model_path = MODEL_LIST[1]
    for llm_id in range(len(MODEL_LIST)):
        save_dir=Path('{0}'.format(MODEL_LIST[llm_id].replace(':','_')))
        save_dir.mkdir(parents=True, exist_ok=True)
        exp_1_multi_choice('ARC-challenge',save_dir,MODEL_LIST[llm_id])