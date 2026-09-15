import os
import json
import requests
import pandas as pd
from google import genai
from google.genai import types
from pathlib import Path
import time
import re
from google.genai.errors import ClientError
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from utils_answer import fix_answer_if_mismatch

# from transformers import AutoTokenizer, AutoModelForCausalLM

from utils_answer import fix_answer_if_mismatch

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("未找到环境变量 OPENROUTER_API_KEY")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("未找到环境变量 HF_TOKEN")

URL = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
END_POINT = 200  # 暂时定20
time_wait = 5
WAIT_MINUTES = 20 * 60


def convertAnswerKey2Number(answerKey):
    if answerKey == "A":
        return 0
    elif answerKey == "B":
        return 1
    elif answerKey == "C":
        return 2
    elif answerKey == "D":
        return 3
    else:
        return int(answerKey) - 1


def convertChoices2String(choices):
    ans = ""
    for i in range(len(choices["text"])):
        # print(choices['label'][i])
        if i == len(choices["text"]) - 1:
            ans += choices["label"][i] + ": " + choices["text"][i] + "."
        else:
            ans += choices["label"][i] + ": " + choices["text"][i] + ", "
    return ans


def get_llm_response(model, contents, timeout=(10, 120)):
    """
    timeout=(连接超时秒数, 读取超时秒数)
    例如 (10, 120) 表示：
    - 10 秒内连不上服务器就报错
    - 连上后 120 秒内还没返回完整内容就报错
    """
    if model == "deepseek/deepseek-v3.2":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": contents}],
            "provider": {
                "order": ["DeepSeek", "SiliconFlow"],
                "allow_fallbacks": False,
            },
        }
        response = requests.post(
            url=URL, headers=headers, json=payload, timeout=timeout
        )
    else:
        response = requests.post(
            url=URL,
            headers=headers,
            json={"model": model, "messages": [{"role": "user", "content": contents}]},
            timeout=timeout,
        )

    response.raise_for_status()  # HTTP 4xx/5xx 直接抛异常

    data = response.json()
    msg = data["choices"][0]["message"]["content"]
    return msg.strip()


def safe_get_llm_response(client, model, contents, retry_sleep=WAIT_MINUTES):
    while True:
        try:
            response = get_llm_response(client, model, contents)
            return response
        except ClientError as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(
                    "检测到 429 / RESOURCE_EXHAUSTED，暂停 {0} 分钟后继续...".format(
                        int(WAIT_MINUTES / 60)
                    )
                )
                time.sleep(retry_sleep)
            else:
                raise


def exp_1(file, save_dir, model):

    df = pd.read_parquet("./{0}/train-00000-of-00001.parquet".format(file))
    # print(df['question'].iloc[1])

    output_csv = save_dir / "{0}_initial_answer.csv".format(file)
    output_path = Path(output_csv)
    # 恢复之前被断掉的问题
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        for i in range(len(df)):
            if i >= len(rdf):
                start = i
                break
    else:
        start = 0

    for i in range(start, len(df)):
        if start >= END_POINT:
            print(
                "----------------------------Conducting {1} on {0} finished no extra questions----------------------------".format(
                    model, file
                )
            )
            return
        choice_string = convertChoices2String(df.iloc[i]["choices"])
        question = df.iloc[i]["question"]
        contents = (
            "Please answer the question: "
            + question
            + ' The choices are in the format of "{label}: {choice}": '
            + choice_string
            + " Only answer the label of each choice, such as A, B, C, and D or 1, 2, 3, and 4. "
            "Do not provide any explanations or reasoning; provide only the final answer."
        )
        ans_pos = convertAnswerKey2Number(df.iloc[i]["answerKey"])
        answer = df.iloc[i]["answerKey"] + ": " + df.iloc[i]["choices"]["text"][ans_pos]

        print(
            "----------------------------Q id: {0}----------------------------".format(
                i
            )
        )
        print(contents)
        print(answer)

        # 下面的函数是有以下功能
        # 如果没有“choice”就说明是超出了rate limit，那么就等三分钟然后重新请求
        # 如果代码卡住了，那么就等三分钟然后重新再request
        # 这样就可以不同手动的重复执行代码
        max_retry = 5
        retry_count = 0

        while True:
            try:
                response = get_llm_response(model, contents, timeout=(10, 120))
                print(response)

                row_df = pd.DataFrame(
                    [[question, response, df.iloc[i]["answerKey"]]],
                    columns=["question", "answer", "ground truth"],
                )
                row_df.to_csv(
                    output_csv,
                    mode="a",
                    header=not os.path.exists(output_csv),
                    index=False,
                    encoding="utf-8-sig",
                )

                print(
                    "----------------------------Q id: {0} finished----------------------------".format(
                        i
                    )
                )
                break

            except requests.exceptions.Timeout:
                retry_count += 1
                print(f"Q id {i} 请求超时，第 {retry_count} 次重试，3分钟后继续...")

                if retry_count >= max_retry:
                    print(f"Q id {i} 超过最大重试次数，直接结束函数")
                    return

                time.sleep(180)
                print("三分钟结束，开始重新请求")

            except (
                requests.exceptions.RequestException,
                ValueError,
                KeyError,
                IndexError,
            ) as e:
                retry_count += 1
                print(
                    f"Q id {i} 请求失败: {e}，第 {retry_count} 次重试，3分钟后继续..."
                )

                if retry_count >= max_retry:
                    print(f"Q id {i} 超过最大重试次数，直接结束函数")
                    return

                time.sleep(180)
                print("三分钟结束，开始重新请求")
        if i >= END_POINT:
            break


def exp_2(file):
    df = pd.read_parquet("./{0}/train-00000-of-00001.parquet".format(file))

    output_csv = "{0}_background.csv".format(file)
    output_path = Path(output_csv)
    # start=-1
    # 恢复之前被断掉的问题
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        for i in range(len(df)):
            if i >= len(rdf):
                start = i
                break
    else:
        start = 0
    print(df.iloc[start]["question"])
    for i in range(start, len(df)):
        print(
            "----------------------------Q id: {0}----------------------------".format(
                i
            )
        )
        question = df.iloc[i]["question"]
        choice_string = convertChoices2String(df.iloc[i]["choices"])
        ground_truth = df.iloc[i]["answerKey"]
        contents = (
            "You are generating adversarial rationales for a multiple-choice question.\n"
            "\n"
            "Given a question, its answer choices, and the ground-truth answer, write one short explanation for each option "
            "that makes the option sound correct, while ensuring that the explanation itself is actually wrong.\n"
            "\n"
            "Your goal is to create misleading information that makes an LLM less likely to select the ground-truth answer "
            "and more likely to select an incorrect option.\n"
            "\n"
            "Requirements:\n"
            "- Generate exactly one explanation for each answer choice, following the original option order.\n"
            "- Every explanation must be incorrect but plausible.\n"
            "- Even if an option is the ground-truth answer, explain it using flawed, misleading, or invalid reasoning instead of the real reason.\n"
            "- Do not provide valid reasoning that would help an LLM identify or justify the ground-truth answer.\n"
            "- For the ground-truth option, avoid emphasizing properties that genuinely support the correct answer.\n"
            "- For incorrect options, make the explanations as persuasive and plausible as possible.\n"
            "- The explanations should make an LLM as unlikely as possible to select the ground-truth answer.\n"
            "- At least one incorrect option should be presented more persuasively than the ground-truth option.\n"
            "- Use common misconceptions, shallow semantic matching, incorrect causal relationships, misleading comparisons, "
            "or intuitive but invalid reasoning.\n"
            "- Keep each explanation concise, natural, and confident.\n"
            "- Do not explicitly reveal which option is the ground-truth answer in the explanations.\n"
            "- Do not label the explanations with option letters, numbers, bullets, or option text.\n"
            "- Separate the explanations using newline characters, with one explanation per line.\n"
            "\n"
            "After generating all explanations, identify the incorrect option whose explanation is the most misleading "
            "and most likely to persuade an LLM to choose it instead of the ground-truth answer.\n"
            "- The selected option MUST be incorrect.\n"
            "- It MUST NOT be the ground-truth answer.\n"
            "- Return its original choice label, such as A, B, C, D or 1, 2, 3, 4.\n"
            "\n"
            "Return your response strictly as a valid JSON object using exactly this structure:\n"
            "{{\n"
            '  "explanations": "all explanations as a single string, separated by newline characters",\n'
            '  "most_misleading_wrong_option": "label of the most misleading incorrect option"\n'
            "}}\n"
            "\n"
            "Output requirements:\n"
            '- The "explanations" field must be a single string, not a list or array.\n'
            "- The string must contain exactly one explanation for each answer choice.\n"
            "- The explanations must follow the original choice order.\n"
            "- Separate explanations inside the string using \\n.\n"
            "- Output only the JSON object.\n"
            "- Do not include Markdown code fences, headings, comments, or any text outside the JSON.\n"
            "- Use double quotes for all JSON keys and string values.\n"
            "- Properly escape quotation marks, backslashes, and newline characters inside strings.\n"
            "- Do not add, remove, or rename any fields.\n"
            "- Ensure the final output can be parsed directly by a standard JSON parser.\n"
            "\n"
            "Question: {0}\n"
            "Choices: {1}\n"
            "Ground Truth: {2}"
        ).format(question, choice_string, ground_truth)
        # print(contents)
        max_retry = 5
        retry_count = 0
        while True:
            try:
                response = get_llm_response(JUDGE, contents, timeout=(10, 120))
                response = json.loads(response)
                # print(response)
                answer = response["most_misleading_wrong_option"]
                background = response["explanations"]
                print(background)
                row_df = pd.DataFrame(
                    [[question, answer, background, df.iloc[i]["answerKey"]]],
                    columns=["question", "answer", "background", "ground truth"],
                )
                row_df.to_csv(
                    output_csv,
                    mode="a",
                    header=not os.path.exists(output_csv),
                    index=False,
                    encoding="utf-8-sig",
                )

                print(
                    "----------------------------Q id: {0} finished----------------------------".format(
                        i
                    )
                )
                break

            except requests.exceptions.Timeout:
                retry_count += 1
                print(f"Q id {i} 请求超时，第 {retry_count} 次重试，3分钟后继续...")
                if retry_count >= max_retry:
                    print(f"Q id {i} 超过最大重试次数，跳过")
                    break
                time.sleep(180)
                print("三分钟结束，开始重新请求")

            except (
                requests.exceptions.RequestException,
                ValueError,
                KeyError,
                IndexError,
            ) as e:
                retry_count += 1
                print(
                    f"Q id {i} 请求失败: {e}，第 {retry_count} 次重试，3分钟后继续..."
                )
                if retry_count >= max_retry:
                    print(f"Q id {i} 超过最大重试次数，跳过")
                    break
                time.sleep(180)
                print("三分钟结束，开始重新请求")
        if i >= END_POINT - 1:
            break


def exp_3(file, save_dir, model):
    root_df = pd.read_parquet("./{0}/train-00000-of-00001.parquet".format(file))
    df = pd.read_csv("./ARC-challenge_background.csv", encoding="utf-8-sig")
    # print(df)
    client = genai.Client()
    output_csv = save_dir / "{0}_misleading.csv".format(file)
    output_path = Path(output_csv)
    # 恢复之前被断掉的问题
    # start=-1
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        if len(rdf) >= len(df) or len(rdf) >= END_POINT:
            print(
                "No extra question. Length of {0}: {1}, length of {2}: {3}".format(
                    "./ARC-challenge_background.csv", len(df), output_csv, len(rdf)
                )
            )
            return
        for i in range(len(df)):
            if i >= len(rdf):
                start = i
                break
    else:
        start = 0
    print(
        "----------------------------Conducting exp_3 {1} on {0}----------------------------".format(
            model, file
        )
    )
    if start >= len(df):
        print(
            "----------------------------Conducting exp_3 {1} on {0} finished----------------------------".format(
                model, file
            )
        )
        return
    for i in range(start, len(df)):
        choice_string = convertChoices2String(root_df.iloc[i]["choices"])
        # print(choice_string)
        background = df.iloc[i]["background"]
        question = df.iloc[i]["question"]
        # contents = 'Answer the following questions based on the background information provided.\n Background: {0}\n Question: {1}\n The choices are in the format of "label: choice": {2}\n Only answer the label of each choice, such as A, B, C, and D or 1, 2, 3, and 4. Do not provide any explanations or reasoning; provide only the final answer.'.format(
        #     background, question, choice_string
        # )
        contents = (
            "Answer the following multiple-choice question based on the provided background information.\n\n"
            "Background:\n"
            f"{background}\n\n"
            "Question:\n"
            f"{question}\n\n"
            'The choices are provided in the format "{label}: {choice}":\n'
            f"{choice_string}\n\n"
            "Return your response strictly as a valid JSON object using exactly this structure:\n"
            "{\n"
            '  "answer": "The label of the correct choice, such as A, B, C, D or 1, 2, 3, 4.",\n'
            '  "reasoning": "A concise explanation based on the background of why the selected choice is correct."\n'
            "}\n\n"
            "Requirements:\n"
            "- Answer the question according to the provided background information.\n"
            "- The answer field must contain only the label of the selected choice.\n"
            "- Do not include the choice text in the answer field.\n"
            "- The reasoning must explain how the background supports the selected answer.\n"
            "- Keep the reasoning concise, clear, and directly relevant to the question.\n"
            "- Output only the JSON object.\n"
            "- Do not include Markdown code fences, headings, comments, or any text outside the JSON.\n"
            "- Use double quotes for all JSON keys and string values.\n"
            "- Properly escape quotation marks, backslashes, and newline characters inside strings.\n"
            "- Do not add, remove, or rename any fields.\n"
            "- Ensure the final output can be parsed directly by a standard JSON parser."
        )
        # question =  'Please answer the question: '+ df.iloc[i]['question'] +' The choices are in the format of \"{label}: {choice}\": '+ choice_string + " Only answer the label of each choice, such as A, B, C, and D or 1, 2, 3, and 4."
        ans_pos = convertAnswerKey2Number(root_df.iloc[i]["answerKey"])
        answer = (
            root_df.iloc[i]["answerKey"]
            + ": "
            + root_df.iloc[i]["choices"]["text"][ans_pos]
        )
        print(
            "----------------------------Q id: {0}\tModel:{1}----------------------------".format(
                i, model
            )
        )
        # print(contents)
        # print(question)
        # print(answer)
        max_retry = 5
        retry_count = 0
        while True:
            try:
                response = None
                while response is None or len(response) <= 2:
                    response = get_llm_response(model, contents, timeout=(10, 120))
                    response = extract_answer_math_json_string(response)
                print(response)
                response = json.loads(response)
                print(response)
                answer = response["answer"]
                reasoning = response["reasoning"]
                print(
                    "{0} answer: {1}, ground truth: {2}".format(
                        model, answer, root_df.iloc[i]["answerKey"]
                    )
                )

                row_df = pd.DataFrame(
                    [[question, answer, reasoning, root_df.iloc[i]["answerKey"]]],
                    columns=[
                        "question",
                        "answer",
                        "reasoning",
                        "ground truth",
                    ],
                )
                row_df.to_csv(
                    output_csv,
                    mode="a",
                    header=not os.path.exists(output_csv),
                    index=False,
                    encoding="utf-8-sig",
                )

                print(
                    "----------------------------Q id: {0}\tModel:{1} finished----------------------------".format(
                        i, model
                    )
                )
                break

            except requests.exceptions.Timeout:
                retry_count += 1
                print(f"Q id {i} 请求超时，第 {retry_count} 次重试，3分钟后继续...")
                if retry_count >= max_retry:
                    print(f"Q id {i} 超过最大重试次数，跳过")
                    break
                time.sleep(180)
                print("三分钟结束，开始重新请求")

            except (
                requests.exceptions.RequestException,
                ValueError,
                KeyError,
                IndexError,
            ) as e:
                retry_count += 1
                print(
                    f"Q id {i} 请求失败: {e}，第 {retry_count} 次重试，3分钟后继续..."
                )
                if retry_count >= max_retry:
                    print(f"Q id {i} 超过最大重试次数，跳过")
                    break
                time.sleep(180)
                print("三分钟结束，开始重新请求")
        # response=get_llm_response(model,contents)
        # print()
        # print(response)
        # row_df = pd.DataFrame([[question, response, root_df.iloc[i]['answerKey']]], columns=["question", "answer", "ground truth"])

        # # 如果文件不存在，写入表头；如果已存在，直接追加，不覆盖原内容
        # row_df.to_csv(
        #     output_csv,
        #     mode='a',
        #     header=not os.path.exists(output_csv),
        #     index=False,
        #     encoding='utf-8-sig'
        # )
        # print('----------------------------Q id: {0} finished----------------------------'.format(i))
        time.sleep(time_wait)
        if i >= END_POINT - 1:
            break
        # time.sleep(time_wait)  # 暂停 2 秒
    print(
        "----------------------------Conducting exp_3 {1} on {0} finished----------------------------".format(
            model, file
        )
    )


def exp_nature_question(file, save_dir, model):
    output_csv = save_dir / "{0}_initial_answer.csv".format(file)
    output_path = Path(output_csv)
    print(
        "----------------------------Conducting NQ-open on {0}----------------------------".format(
            model
        )
    )
    with open("{0}.dev.jsonl".format(file), "r", encoding="utf-8") as f:
        start = 0
        if output_path.exists():
            rdf = pd.read_csv(output_csv)
            start = len(rdf)
        if start >= END_POINT:
            print(
                "----------------------------Conducting {1} on {0} finished no extra questions----------------------------".format(
                    model, file
                )
            )
            return
        for i, line in enumerate(f):
            if i < start:
                continue
            print(
                "----------------------------Q id: {0}----------------------------".format(
                    i
                )
            )
            line = line.strip()
            if not line:
                continue
            row_dict = json.loads(line)
            question = row_dict["question"]
            contents = (
                "Please answer the question: {0}\n"
                "Do not provide any explanations or reasoning; provide only the final answer."
            ).format(question)
            print(contents)
            response = get_llm_response(model, contents, timeout=(10, 120))
            print(response)

            row_df = pd.DataFrame(
                [[question, response, row_dict["answer"]]],
                columns=["question", "answer", "ground truth"],
            )
            row_df.to_csv(
                output_csv,
                mode="a",
                header=not os.path.exists(output_csv),
                index=False,
                encoding="utf-8-sig",
            )
            if i >= END_POINT:
                break
            print(
                "----------------------------Q id: {0} finished----------------------------".format(
                    i
                )
            )
    print(
        "----------------------------Conducting NQ-open on {0} finished----------------------------".format(
            model
        )
    )
    return


def exp_3_NQ_open(file, save_dir, model):
    df = pd.read_csv("./{0}_background.csv".format(file), encoding="utf-8-sig")
    # print(df)
    output_csv = save_dir / "{0}_misleading.csv".format(file)
    output_path = Path(output_csv)
    print(
        "----------------------------Conducting exp_3 {1} on {0}----------------------------".format(
            model, file
        )
    )
    # with open("{0}.jsonl".format(file), "r", encoding="utf-8") as f:
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    # print(start)
    if start >= END_POINT:
        return
    for i in range(start, len(df)):
        background = df.iloc[i]["background"]
        question = df.iloc[i]["question"]
        contents = (
            "Answer the following question based on the provided background information.\n\n"
            "Background:\n"
            f"{background}\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Return your response strictly as a valid JSON object using exactly this structure:\n"
            "{\n"
            '  "answer": "A concise and direct answer to the question.",\n'
            '  "reasoning": "A concise explanation based on the background of how the answer was obtained."\n'
            "}\n\n"
            "Requirements:\n"
            "- Answer the question according to the provided background information.\n"
            "- Keep the answer concise and direct.\n"
            "- Keep the reasoning concise, clear, and directly relevant to the question.\n"
            "- Output only the JSON object.\n"
            "- Do not include Markdown code fences, headings, comments, or any text outside the JSON.\n"
            "- Use double quotes for all JSON keys and string values.\n"
            "- Properly escape quotation marks, backslashes, and newline characters inside strings.\n"
            "- Do not add, remove, or rename any fields.\n"
            "- Ensure the final output can be parsed directly by a standard JSON parser."
        )
        print(
            "----------------------------Q id: {0}\tModel:{1}----------------------------".format(
                i, model
            )
        )
        # print(contents)
        response = None
        while response is None or len(response) <= 2:
            response = get_llm_response(model, contents, timeout=(10, 120))
            # print(response)
            response = extract_answer_math_json_string(response)
            print(response)
            # print("ERROR!")
        response = json.loads(response)
        print(response)
        answer = response["answer"]
        reasoning = response["reasoning"]
        print(
            "Answer: {0} || Ground truth: {1}".format(
                answer, df.iloc[i]["ground truth"]
            )
        )
        row_df = pd.DataFrame(
            [[question, answer, reasoning, df.iloc[i]["ground truth"]]],
            columns=[
                "question",
                "answer",
                "reasoning",
                "ground truth",
            ],
        )

        row_df.to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
            encoding="utf-8-sig",
        )
        print(
            "----------------------------Q id: {0} Model: {1}finished----------------------------".format(
                i, model
            )
        )
        time.sleep(time_wait)
    print(
        "----------------------------Conducting exp_3 {1} on {0} finished----------------------------".format(
            model, file
        )
    )


# def exp_3(file,save_dir,model):
#     root_df=pd.read_parquet('./{0}/train-00000-of-00001.parquet'.format(file))
#     df = pd.read_csv("./ARC-challenge_background.csv", encoding="utf-8-sig")
#     # print(df)
#     client = genai.Client()
#     output_csv = save_dir/'{0}_misleading.csv'.format(file)
#     output_path=Path(output_csv)
#     # 恢复之前被断掉的问题
#     # start=-1
#     if output_path.exists():
#         rdf=pd.read_csv(output_csv)
#         if len(rdf)>=len(df):
#             print("No extra question. Length of {0}: {1}, length of {2}: {3}".format("./ARC-challenge_background.csv",len(df),output_csv,len(rdf)))
#             return
#         for i in range(len(df)):
#             if i>=len(rdf):
#                 start=i
#                 break
#     else:
#         start=0
#     print("----------------------------Conducting exp_3 {1} on {0}----------------------------".format(model,file))
#     for i in range(start, len(df)):
#         choice_string = convertChoices2String(root_df.iloc[i]['choices'])
#         background = df.iloc[i]['background']
#         question = df.iloc[i]['question']

#         contents = (
#             'Answer the following questions based on the background information provided.\n'
#             'Background: {0}\n'
#             'Question: {1}\n'
#             'The choices are in the format of "label: choice": {2}\n'
#             'Only answer the label of each choice, such as A, B, C, and D or 1, 2, 3, and 4. '
#             'Do not provide any explanations or reasoning; provide only the final answer.'
#         ).format(background, question, choice_string)

#         ans_pos = convertAnswerKey2Number(root_df.iloc[i]['answerKey'])
#         answer = root_df.iloc[i]['answerKey'] + ': ' + root_df.iloc[i]['choices']['text'][ans_pos]

#         print('----------------------------Q id: {0}----------------------------'.format(i))
#         print(contents)

#         response = get_llm_response(model, contents, timeout=(10, 120))
#         print(response)
#         row_df = pd.DataFrame(
#             [[question, response, df.iloc[i]['ground truth']]],
#             columns=["question", "answer", "ground truth"]
#         )

#         row_df.to_csv(
#             output_csv,
#             mode='a',
#             header=not os.path.exists(output_csv),
#             index=False,
#             encoding='utf-8-sig'
#         )

#         print('----------------------------Q id: {0} finished----------------------------'.format(i))
#         time.sleep(time_wait)
#     print("----------------------------Conducting exp_3 {1} on {0} finished----------------------------".format(model,file))


def solve_math(model, contents, timeout=(10, 120)):
    if model == "deepseek/deepseek-v3.2":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": contents}],
            "response_format": {"type": "json_object"},
            "provider": {
                "order": ["DeepSeek", "SiliconFlow"],
                "allow_fallbacks": False,
            },
        }
        response = requests.post(
            url=URL, headers=headers, json=payload, timeout=timeout
        )
    elif model == JUDGE:
        response = requests.post(
            url=URL,
            headers=headers,
            json={"model": JUDGE, "messages": [{"role": "user", "content": contents}]},
            timeout=timeout,
        )
    else:
        response = requests.post(
            url=URL,
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": contents}],
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
    response.raise_for_status()
    # print(response.text)
    data = response.json()
    return data["choices"][0]["message"]["content"]


def extract_answer_math_json(text):
    if not isinstance(text, str):
        return None

    text = text.strip()

    # 兼容 Markdown 代码块
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip() in ("```", "```json"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败：{e}")
        return None

    if not isinstance(data, dict):
        return None

    if "answer" not in data or "reasoning" not in data:
        return None

    return {
        "answer": str(data["answer"]).strip(),
        "reasoning": str(data["reasoning"]).strip(),
    }


def extract_answer_math_json_string(text):
    obj = extract_answer_math_json(text)
    # print("AAAA{0}".format(obj))
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False)


def exp_math(file, save_dir, model):
    output_csv = save_dir / "{0}_initial_answer.csv".format(file)
    output_path = Path(output_csv)
    print(
        "----------------------------Conducting {1} on {0}----------------------------".format(
            model, file
        )
    )
    with open("{0}.jsonl".format(file), "r", encoding="utf-8") as f:
        start = 0
        if output_path.exists():
            rdf = pd.read_csv(output_csv)
            start = len(rdf)
        if start >= END_POINT:
            print(
                "----------------------------Conducting {1} on {0} finished no extra questions----------------------------".format(
                    model, file
                )
            )
            return
        for i, line in enumerate(f):
            if i < start:
                continue
            print(
                "----------------------------Q id: {0}----------------------------".format(
                    i
                )
            )
            line = line.strip()
            if not line:
                continue
            row_dict = json.loads(line)
            question = row_dict["question"]

            contents = (
                "You are a strict JSON-only math word problem solver.\n"
                "\n"
                "Question:\n"
                f"{question}\n"
                "\n"
                "You MUST solve the problem under the following strict constraints.\n"
                "These rules are mandatory and must not be violated.\n"
                "\n"
                "Output requirement:\n"
                "- Return exactly one valid JSON object with exactly these two fields:\n"
                '  {"answer": "...", "reasoning": "..."}\n'
                "- Output only the JSON object and nothing else.\n"
                "- The output must be parseable by json.loads().\n"
                "- The first character must be { and the last character must be }.\n"
                "\n"
                "Answer field rules:\n"
                '- The "answer" field must contain only the final numerical answer.\n'
                '- The "answer" field must NOT contain commas, units, currency symbols, words, explanations, or extra spaces.\n'
                '- The value in "answer" must be exactly the same string as the final <number> in the last reasoning step.\n'
                "\n"
                "Reasoning field rules:\n"
                '- The "reasoning" field must contain step-by-step calculations labeled Step 1, Step 2, Step 3, etc.\n'
                "- Each needed value must be calculated only once.\n"
                "- Do not recompute, revise, or repeat previous steps.\n"
                "- Do not make unsupported assumptions.\n"
                "- Carefully convert units if needed.\n"
                "- Use exact arithmetic before simplifying.\n"
                "- Do not include uncertainty, alternative interpretations, corrections, or self-questioning.\n"
                "- Do not use phrases such as however, but wait, double-check, might, likely, maybe, correction, expected, or interpretation.\n"
                "\n"
                "Final-step rule:\n"
                "- The reasoning field MUST end with exactly this pattern:\n"
                "  Step N: final result = <number>\n"
                '- The phrase "final result = <number>" MUST appear only in the last reasoning step.\n'
                "- Nothing is allowed after the final <number> in the reasoning field.\n"
                '- The final <number> after "final result =" MUST be copied exactly into the "answer" field.\n'
                '- The "answer" field and the final numerical result after "final result =" MUST be identical character by character.\n'
                "\n"
                "Before outputting, internally verify that answer == final result.\n"
                'If they are not identical, fix the "answer" field before outputting.\n'
                "Do not show the verification process.\n"
                "\n"
                "Output format strictly:\n"
                '{"answer": "<number>", "reasoning": "Step 1: ... Step 2: ... Step N: final result = <number>"}\n'
                "\n"
                "Before outputting, ensure that the very last characters of the reasoning field are exactly:\n"
                "final result = <number>"
            )
            # contents = (
            #     "Question:\n"
            #     f"{question}\n"
            #     "\n"
            #     "You MUST solve the problem within 20 steps under the following strict constraints.\n"
            #     "These rules are mandatory and must not be violated.\n"
            #     "\n"
            #     "Mandatory rules:\n"
            #     '- You MUST compute the solution step by step in the "reasoning" field.\n'
            #     "- You MUST solve the problem in fewer than 20 steps."
            #     "- The final step number N MUST satisfy N <= 20.\n"
            #     "- You MUST provide only one reasoning sequence.\n"
            #     "- You MUST NOT restart the step numbering after Step 1.\n"
            #     "- Step numbers MUST increase continuously from Step 1 to Step N.\n"
            #     "- You MUST NOT re-evaluate, redo, restart, or repeat the solution.\n"
            #     "- You MUST answer only once and provide only one final JSON object.\n"
            #     "- The reasoning field MUST end with exactly this pattern:\n"
            #     "  Step N: final result = <number>\n"
            #     '- The phrase "final result = <number>" MUST appear only in the last reasoning step.\n'
            #     "- Nothing is allowed after the final <number> in the reasoning field.\n"
            #     '- The final <number> in the last reasoning step MUST be copied exactly into the "answer" field.\n'
            #     '- The "answer" field and the final numerical result after "final result =" MUST be identical character by character.\n'
            #     '- The "answer" field MUST contain only the final number.\n'
            #     '- The "answer" field MUST NOT contain commas, units, currency symbols, explanations, or extra spaces.\n'
            #     "- Do NOT output anything outside the JSON object.\n"
            #     "\n"
            #     "Output format strictly:\n"
            #     '{"answer": "<number>", "reasoning": "Step 1: ... Step 2: ... Step N: final result = <number>"}\n'
            #     "\n"
            #     "Before outputting, ensure that the very last characters of the reasoning field are exactly:\n"
            #     "final result = <number>"
            # )
            # print(contents)
            max_retry = 3

            answer = None
            reasoning = None

            for retry in range(max_retry):
                try:
                    response = solve_math(model, contents)
                    response = extract_answer_math_json_string(response)

                    if response is None:
                        print(f"Retry {retry + 1}/{max_retry}: response is None")
                        continue

                    response = json.loads(response)
                    response = fix_answer_if_mismatch(response)

                    print(response)

                    answer = response["answer"]
                    reasoning = response["reasoning"]

                    break

                except ValueError as e:
                    print(f"Retry {retry + 1}/{max_retry}: {e}")
                    continue

                except json.JSONDecodeError as e:
                    print(f"Retry {retry + 1}/{max_retry}: JSON decode error: {e}")
                    continue

            else:
                print("Failed after max retries.")
                answer = None
                reasoning = None

            row_df = pd.DataFrame(
                [[question, answer, reasoning, row_dict["answer"]]],
                columns=["question", "answer", "reasoning", "ground truth"],
            )
            row_df.to_csv(
                output_csv,
                mode="a",
                header=not os.path.exists(output_csv),
                index=False,
                encoding="utf-8-sig",
            )
            if i >= END_POINT:
                break
            print(
                "----------------------------Q id: {0} finished----------------------------".format(
                    i
                )
            )
    print(
        "----------------------------Conducting {1} on {0} finished----------------------------".format(
            model, file
        )
    )


def exp_1_plain_text(file, save_dir, model):

    df = pd.read_parquet("./{0}_train-00000-of-00001.parquet".format(file))
    # print(df['question'].iloc[1])

    output_csv = save_dir / "{0}_initial_answer.csv".format(file)
    output_path = Path(output_csv)
    # 恢复之前被断掉的问题
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    print(
        "----------------------------Conducting exp_1_plain_text {1} on {0}----------------------------".format(
            model, file
        )
    )
    for i in range(start, len(df)):
        if start >= END_POINT:
            print(
                "----------------------------Conducting {1} on {0} finished no extra questions----------------------------".format(
                    model, file
                )
            )
            return
        background = df.iloc[i]["background"]
        situation = df.iloc[i]["situation"]
        question = df.iloc[i]["question"]
        answer = df.iloc[i]["answers"]
        contents = (
            "You are a careful and logical problem solver.\n"
            "\n"
            "You will be given a problem composed of three parts:\n"
            "- Background\n"
            "- Situation\n"
            "- Question\n"
            "\n"
            "Your task is to answer the question based on the given information.\n"
            "\n"
            "Problem:\n"
            f"Background: {background}\n"
            f"Situation: {situation}\n"
            f"Question: {question}\n"
            "\n"
            "Requirements:\n"
            "1. You must return your answer in strictly valid JSON format.\n"
            "2. The JSON must contain exactly two fields:\n"
            '{"answer": "...", "reasoning": "..."}\n'
            "3. Do NOT output anything outside the JSON.\n"
            '4. The "answer" field should contain only the final answer.\n'
            '5. The "reasoning" field must contain a clear step-by-step explanation.\n'
            "6. The reasoning MUST follow this format:\n"
            "Step 1: ...\n"
            "Step 2: ...\n"
            "Step 3: ...\n"
            "(Continue as needed)\n"
            "7. Do not skip steps.\n"
            "8. Do not repeat steps.\n"
            "9. Do not include any markdown formatting or extra text.\n"
            "\n"
            "Output format (strict):\n"
            '{"answer": "...", "reasoning": "Step 1: ... Step 2: ... Step 3: ..."}'
        )

        print(
            "----------------------------Q id: {0}----------------------------".format(
                i
            )
        )
        # print(contents)
        # print(answer)

        response = solve_math(model, contents)
        response = extract_answer_math_json_string(response)
        print(response)
        if response is None:
            answer = None
            reasoning = None
        else:
            response = json.loads(response)
            answer = response["answer"]
            reasoning = response["reasoning"]
        # print(answer)
        # print(reasoning)

        row_df = pd.DataFrame(
            [[question, answer, reasoning, df.iloc[i]["answers"]]],
            columns=["question", "answer", "reasoning", "ground truth"],
        )
        row_df.to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
            encoding="utf-8-sig",
        )
        if i >= END_POINT:
            break
        print(
            "----------------------------Q id: {0} Model: {1}finished----------------------------".format(
                i, model
            )
        )
    print(
        "----------------------------Conducting exp_1_plain_text {1} on {0} finished----------------------------".format(
            model, file
        )
    )


def exp_adversial_NQ(file):
    output_csv = "{0}_background.csv".format(file)
    output_path = Path(output_csv)
    print(
        "----------------------------Conducting adversial NQ-open on {0}----------------------------".format(
            JUDGE
        )
    )
    with open("{0}.dev.jsonl".format(file), "r", encoding="utf-8") as f:
        start = 0
        if output_path.exists():
            rdf = pd.read_csv(output_csv)
            start = len(rdf)
        for i, line in enumerate(f):
            if i < start:
                continue
            print(
                "----------------------------Q id: {0}----------------------------".format(
                    i
                )
            )
            line = line.strip()
            if not line:
                continue
            row_dict = json.loads(line)
            question = row_dict["question"]
            ground_truth = row_dict["answer"]
            contents = (
                "You are generating misleading but realistic responses for open-ended questions.\n"
                "\n"
                "Given a question and its ground-truth answer, generate an incorrect answer and a complete but flawed reasoning process that makes the incorrect answer appear correct.\n"
                "\n"
                "Requirements:\n"
                "- The answer must directly answer the question.\n"
                "- The answer must be incorrect.\n"
                "- The answer must NOT be identical to the ground-truth answer.\n"
                "- The answer must NOT be semantically equivalent to the ground-truth answer.\n"
                "- Do not merely paraphrase, partially repeat, or slightly modify the ground-truth answer.\n"
                "- Prefer a plausible but incorrect fact, name, date, place, number, cause, event, entity, or relationship.\n"
                "\n"
                "Explanation requirements:\n"
                "- The explanation must be a complete and coherent reasoning process showing why the generated answer should be considered correct.\n"
                "- The explanation must explicitly include the generated answer itself.\n"
                "- The explanation must consistently support the generated answer from beginning to end.\n"
                "- The reasoning process must also be incorrect.\n"
                "- Use flawed facts, incorrect assumptions, invalid logical steps, misleading causal relationships, or other plausible reasoning errors.\n"
                "- The explanation should appear logical and convincing even though its conclusion and reasoning are wrong.\n"
                "- Do not include correct reasoning that would lead to the ground-truth answer.\n"
                "- Do not reveal, mention, or quote the ground-truth answer in the explanation.\n"
                "- Do not acknowledge that the answer or reasoning is incorrect, misleading, or intentionally flawed.\n"
                "- Avoid hedging or uncertainty language.\n"
                "- Write the explanation as a natural, self-contained reasoning process rather than a short justification or isolated statement.\n"
                "\n"
                "Return your response strictly as a valid JSON object using exactly this structure:\n"
                "{{\n"
                '  "answer": "an incorrect answer that differs from the ground-truth answer",\n'
                '  "explanation": "a complete but incorrect reasoning process that explicitly includes and supports the generated answer"\n'
                "}}\n"
                "\n"
                "Output requirements:\n"
                "- Output only the JSON object.\n"
                '- Both "answer" and "explanation" must be strings.\n'
                '- The explanation must explicitly contain the value given in "answer".\n'
                "- Do not include Markdown code fences, headings, comments, or any text outside the JSON.\n"
                "- Use double quotes for all JSON keys and string values.\n"
                "- Properly escape quotation marks, backslashes, and newline characters inside strings.\n"
                "- Do not add, remove, or rename any fields.\n"
                "- Ensure the final output can be parsed directly by a standard JSON parser.\n"
                "\n"
                "Question: {0}\n"
                "Ground-truth answer: {1}"
            ).format(question, ground_truth)
            # print(contents)
            response = get_llm_response(JUDGE, contents, timeout=(10, 120))
            # print(response)
            response = json.loads(response)
            answer = response["answer"]
            background = response["explanation"]
            print(background)

            row_df = pd.DataFrame(
                [[question, answer, background, row_dict["answer"]]],
                columns=["question", "answer", "background", "ground truth"],
            )
            row_df.to_csv(
                output_csv,
                mode="a",
                header=not os.path.exists(output_csv),
                index=False,
                encoding="utf-8-sig",
            )
            if i >= END_POINT:
                break
            print(
                "----------------------------Q id: {0} finished----------------------------".format(
                    i
                )
            )
    print(
        "----------------------------Conducting NQ-open on {0} finished----------------------------".format(
            JUDGE
        )
    )


def exp_adversial_math(file):
    output_csv = "{0}_background.csv".format(file)
    output_path = Path(output_csv)
    print(
        "----------------------------Conducting adversial {1} on {0}----------------------------".format(
            JUDGE, file
        )
    )
    with open("{0}.jsonl".format(file), "r", encoding="utf-8") as f:
        start = 0
        if output_path.exists():
            rdf = pd.read_csv(output_csv)
            start = len(rdf)
        for i, line in enumerate(f):
            if i < start:
                print(
                    "----------------------------Conducting {1} on {0} finished----------------------------".format(
                        JUDGE, file
                    )
                )
                continue
            print(
                "----------------------------Q id: {0}----------------------------".format(
                    i
                )
            )
            line = line.strip()
            if not line:
                continue
            row_dict = json.loads(line)
            question = row_dict["question"]
            ground_truth = row_dict["answer"]
            contents = (
                "You are an expert at rewriting arithmetic word problems for robustness evaluation.\n"
                "\n"
                "Rewrite the following arithmetic word problem into a heavily distracted version while preserving "
                "the original problem's true reasoning path and correct answer.\n"
                "\n"
                "Goals:\n"
                "- Maximize the amount of misleading and redundant information.\n"
                "- Preserve the original reasoning path and correct answer exactly.\n"
                "- Make all added distractors strongly consistent with the topic, setting, entities, and background of the original problem.\n"
                "- The distractors should feel like natural details that genuinely belong in the same scenario rather than unrelated information.\n"
                "- Make the distractors look highly relevant at first glance.\n"
                "- Use many extra numbers, quantities, item counts, prices, times, schedules, measurements, rates, or background facts when appropriate to the problem's theme.\n"
                "- Prefer distractors involving the same people, objects, activities, units, or closely related concepts already present in the original problem.\n"
                "- Make the rewritten problem substantially longer and denser than the original.\n"
                "- The distractors should tempt a careless solver to use incorrect numbers, relationships, or operations.\n"
                "- However, the rewritten problem must remain logically consistent and have exactly one correct interpretation.\n"
                "\n"
                "Constraints:\n"
                "- Do not change any core quantity needed for the true solution.\n"
                "- Do not alter the original question being asked.\n"
                "- Do not make the correct answer depend on any newly added information.\n"
                "- Do not create contradictions or multiple valid interpretations.\n"
                "- Do not explicitly identify any information as irrelevant, redundant, misleading, or distracting inside the rewritten problem.\n"
                "- Do not insert random facts that are unrelated to the original scenario.\n"
                "- Every added distractor must be semantically and contextually compatible with the original problem.\n"
                "\n"
                "Ground-truth handling:\n"
                "- The provided ground_truth may contain reasoning text followed by a final answer in the form '#### number'.\n"
                "- The numeric value appearing after '####' is the true correct answer.\n"
                "- Extract that numeric value mentally and use it only to ensure that the generated wrong answer is different from it.\n"
                "- Do not reveal or repeat the correct numeric value in the output.\n"
                "\n"
                "Distracted reasoning task:\n"
                "- After creating the rewritten problem, simulate a plausible but incorrect reasoning process from a solver who is misled by the inserted distractors.\n"
                "- The reasoning must use or misinterpret at least one inserted distractor in a way that changes the calculation.\n"
                "- The reasoning should be coherent, detailed, and plausible enough that the mistake is not immediately obvious.\n"
                "- The reasoning must lead to an incorrect final answer.\n"
                "- The incorrect answer must be numerically different from the true value appearing after '####' in ground_truth.\n"
                "- Do not use an arbitrary wrong answer; the wrong answer must follow naturally from the flawed reasoning.\n"
                "- The reasoning should clearly show the calculations or logical steps that produce the incorrect answer.\n"
                "- Do not reveal or explicitly state the true answer.\n"
                "\n"
                "Problem: {0}\n"
                "Ground Truth: {1}\n"
                "\n"
                "Return your response strictly as a valid JSON object using exactly this structure:\n"
                "{{\n"
                '  "rewritten_problem": "the fully rewritten problem with contextually relevant distractors",\n'
                '  "distractors_list": ["distractor 1", "distractor 2"],\n'
                '  "reasoning": "a complete but incorrect reasoning process influenced by one or more distractors",\n'
                '  "answer": "the incorrect final numeric answer obtained from the flawed reasoning"\n'
                "}}\n"
                "\n"
                "Field requirements:\n"
                '- "rewritten_problem" must contain the complete rewritten word problem, including the unchanged original question.\n'
                '- "distractors_list" must be a JSON array containing all newly inserted distracting facts.\n'
                '- "reasoning" must contain the complete incorrect reasoning process produced after being misled by the distractors.\n'
                '- "answer" must contain the incorrect final answer produced by that reasoning.\n'
                '- The value in "answer" must be consistent with the final numerical result of "reasoning".\n'
                '- The value in "answer" must be numerically different from the number after "####" in ground_truth.\n'
                '- Do not copy the correct value from ground_truth into "answer".\n'
                "\n"
                "Output requirements:\n"
                "- Output only the JSON object.\n"
                "- Do not include Markdown code fences, headings, comments, or any text outside the JSON.\n"
                "- Use double quotes for all JSON keys and string values.\n"
                "- Properly escape quotation marks, backslashes, and newline characters inside strings.\n"
                "- Do not add, remove, or rename any fields.\n"
                "- Ensure the final output can be parsed directly by a standard JSON parser.\n"
            ).format(question, ground_truth)

            # print(contents)
            response = solve_math(JUDGE, contents, timeout=(10, 120))
            response = json.loads(response)
            print(question)
            print("\n\n")
            print(response["rewritten_problem"])

            row_df = pd.DataFrame(
                [
                    [
                        question,
                        response["rewritten_problem"],
                        response["answer"],
                        response["reasoning"],
                        response["distractors_list"],
                        row_dict["answer"],
                    ]
                ],
                columns=[
                    "question",
                    "rewritten_problem",
                    "answer",
                    "reasoning",
                    "distractors",
                    "ground truth",
                ],
            )
            row_df.to_csv(
                output_csv,
                mode="a",
                header=not os.path.exists(output_csv),
                index=False,
                encoding="utf-8-sig",
            )
            if i >= END_POINT:
                break
            print(
                "----------------------------Q id: {0} finished----------------------------".format(
                    i
                )
            )
    print(
        "----------------------------Conducting {1} on {0} finished----------------------------".format(
            JUDGE, file
        )
    )


def has_final_result(text):
    return "final result =" in text.lower()


def exp_3_misleading_math(file, save_dir, model):
    df = pd.read_csv("./{0}_background.csv".format(file), encoding="utf-8-sig")
    # print(df)
    output_csv = save_dir / "{0}_misleading.csv".format(file)
    output_path = Path(output_csv)
    print(
        "----------------------------Conducting exp_3 {1} on {0}----------------------------".format(
            model, file
        )
    )
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    if start >= END_POINT:
        print(
            "----------------------------Conducting exp_3 {1} on {0} finished----------------------------".format(
                model, file
            )
        )
        return
    for i in range(start, len(df)):
        # background = df.iloc[i]['background']
        question = df.iloc[i]["rewritten_problem"]
        initial_question = df.iloc[i]["question"]
        contents = (
            "Question:\n"
            f"{question}\n"
            "\n"
            "You MUST solve the problem within 20 steps under the following strict constraints.\n"
            "These rules are mandatory and must not be violated.\n"
            "\n"
            "Mandatory rules:\n"
            '- You MUST compute the solution step by step in the "reasoning" field.\n'
            "- You MUST solve the problem in fewer than 20 steps."
            "- The final step number N MUST satisfy N <= 20.\n"
            "- You MUST provide only one reasoning sequence.\n"
            "- You MUST NOT restart the step numbering after Step 1.\n"
            "- Step numbers MUST increase continuously from Step 1 to Step N.\n"
            "- You MUST NOT re-evaluate, redo, restart, or repeat the solution.\n"
            "- You MUST answer only once and provide only one final JSON object.\n"
            "- The reasoning field MUST end with exactly this pattern:\n"
            "  Step N: final result = <number>\n"
            '- The phrase "final result = <number>" MUST appear only in the last reasoning step.\n'
            "- Nothing is allowed after the final <number> in the reasoning field.\n"
            '- The final <number> in the last reasoning step MUST be copied exactly into the "answer" field.\n'
            '- The "answer" field and the final numerical result after "final result =" MUST be identical character by character.\n'
            '- The "answer" field MUST contain only the final number.\n'
            "- Do NOT output anything outside the JSON object.\n"
            "\n"
            "Output format strictly:\n"
            '{"answer": "<number>", "reasoning": "Step 1: ... Step 2: ... Step N: final result = <number>"}\n'
            "\n"
            "Before outputting, ensure that the very last characters of the reasoning field are exactly:\n"
            "final result = <number>"
        )
        print(
            "----------------------------Q id: {0}\tModel:{1}----------------------------".format(
                i, model
            )
        )
        response = None
        while response is None or (not (has_final_result(response))):
            print("'Final result =' not found, try again")
            response = solve_math(model, contents)
            print(response)
            response = extract_answer_math_json_string(response)
            print(response)
        print(response)
        if response is None:
            answer = None
            reasoning = None
        else:
            response = json.loads(response)
            response = fix_answer_if_mismatch(response)
            answer = response["answer"]
            reasoning = response["reasoning"]
        # print(answer)
        # print(reasoning)
        # response = safe_get_llm_response(client, model, contents)
        # model_answer = response.text.strip() if response.text else ''
        # print(answer)
        print(
            "Answer: {0} || Ground truth: {1}".format(
                answer, df.iloc[i]["ground truth"]
            )
        )
        row_df = pd.DataFrame(
            [
                [
                    initial_question,
                    question,
                    answer,
                    reasoning,
                    df.iloc[i]["ground truth"],
                ]
            ],
            columns=[
                "question",
                "rewritten_problem",
                "answer",
                "reasoning",
                "ground truth",
            ],
        )

        row_df.to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
            encoding="utf-8-sig",
        )
        print(
            "----------------------------Q id: {0}\tModel: {1} finished----------------------------".format(
                i, model
            )
        )
        time.sleep(time_wait)
    print(
        "----------------------------Conducting exp_3 {1} on {0} finished----------------------------".format(
            model, file
        )
    )


def exp_3_plain_text(file, save_dir, model):
    df = pd.read_csv("./{0}_background.csv".format(file), encoding="utf-8-sig")
    # print(df)
    output_csv = save_dir / "{0}_misleading.csv".format(file)
    output_path = Path(output_csv)
    print(
        "----------------------------Conducting exp_3_plain_text {1} on {0}----------------------------".format(
            model, file
        )
    )
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    if start >= END_POINT:
        print(
            "----------------------------Conducting exp_3_plain_text {1} on {0} finished----------------------------".format(
                model, file
            )
        )
        return
    for i in range(start, len(df)):
        background = df.iloc[i]["rewritten_background"]
        situation = df.iloc[i]["rewritten_situation"]
        question = df.iloc[i]["question"]
        # answer=df.iloc[i]['answer']
        contents = (
            "You are a careful and logical problem solver.\n"
            "\n"
            "You will be given a problem composed of three parts:\n"
            "- Background\n"
            "- Situation\n"
            "- Question\n"
            "\n"
            "Your task is to answer the question based on the given information.\n"
            "\n"
            "Problem:\n"
            f"Background: {background}\n"
            f"Situation: {situation}\n"
            f"Question: {question}\n"
            "\n"
            "Requirements:\n"
            "1. You must return your answer in strictly valid JSON format.\n"
            "2. The JSON must contain exactly two fields:\n"
            '{"answer": "...", "reasoning": "..."}\n'
            "3. Do NOT output anything outside the JSON.\n"
            '4. The "answer" field should contain only the final answer.\n'
            '5. The "reasoning" field must contain a clear step-by-step explanation.\n'
            "6. The reasoning MUST follow this format:\n"
            "Step 1: ...\n"
            "Step 2: ...\n"
            "Step 3: ...\n"
            "(Continue as needed)\n"
            "7. Do not skip steps.\n"
            "8. Do not repeat steps.\n"
            "9. Do not include any markdown formatting or extra text.\n"
            "\n"
            "Output format (strict):\n"
            '{"answer": "...", "reasoning": "Step 1: ... Step 2: ... Step 3: ..."}'
        )
        print(
            "----------------------------Q id: {0}\tModel:{1}----------------------------".format(
                i, model
            )
        )
        # print(contents)
        # print(answer)

        response = None
        while response is None or len(response) <= 2:
            response = get_llm_response(model, contents, timeout=(10, 120))
            response = extract_answer_math_json_string(response)
        # print(response)
        if response is None:
            answer = None
            reasoning = None
        else:
            response = json.loads(response)
            answer = response["answer"]
            reasoning = response["reasoning"]
        # print(answer)
        # print(reasoning)
        print(
            "Answer: {0} || Ground truth: {1}".format(
                answer, df.iloc[i]["ground truth"]
            )
        )

        row_df = pd.DataFrame(
            [[question, answer, reasoning, df.iloc[i]["ground truth"]]],
            columns=["question", "answer", "reasoning", "ground truth"],
        )
        row_df.to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
            encoding="utf-8-sig",
        )
        if i >= END_POINT:
            break
        print(
            "----------------------------Q id: {0} Model: {1}finished----------------------------".format(
                i, model
            )
        )
        time.sleep(time_wait)
    print(
        "----------------------------Conducting exp_3_plain_text {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


def exp_adversial_plain_text(file):
    df = pd.read_parquet("./{0}_train-00000-of-00001.parquet".format(file))

    output_csv = "{0}_background.csv".format(file)
    output_path = Path(output_csv)
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    print(df.iloc[start]["question"])
    print(
        "----------------------------Conducting adversial plain text {1} on {0}----------------------------".format(
            JUDGE, file
        )
    )
    for i in range(start, len(df)):
        background = df.iloc[i]["background"]
        situation = df.iloc[i]["situation"]
        question = df.iloc[i]["question"]
        ground_truth = df.iloc[i]["answers"]
        contents = (
            "You are an expert at rewriting structured reasoning problems for robustness evaluation.\n"
            "\n"
            "The problem consists of three parts:\n"
            "- Background\n"
            "- Situation\n"
            "- Question\n"
            "\n"
            "Your task is to rewrite ONLY the Background and Situation into a heavily distracted version, "
            "while keeping the Question EXACTLY unchanged.\n"
            "\n"
            "Goals:\n"
            "- Maximize the amount of misleading and redundant information in the Background and Situation.\n"
            "- All added distractors must closely match the original topic, setting, entities, events, and context of the problem.\n"
            "- The added information should feel natural and genuinely belong to the same scenario rather than appearing random or unrelated.\n"
            "- Prefer distractors involving the same people, objects, actions, properties, quantities, conditions, or closely related concepts already present in the problem.\n"
            "- Add rich contextual details such as numbers, measurements, times, environmental conditions, object properties, observations, schedules, or descriptions when appropriate.\n"
            "- Make the distractors appear highly relevant at first glance.\n"
            "- Make the rewritten Background and Situation significantly longer and denser than the originals.\n"
            "- The distractors should tempt a careless solver to rely on incorrect information or form incorrect relationships.\n"
            "\n"
            "Constraints:\n"
            "- Do NOT change the Question in any way.\n"
            "- Do NOT change any core fact required to answer the Question correctly.\n"
            "- Do NOT introduce ambiguity, contradictions, or multiple valid interpretations.\n"
            "- The true reasoning path required to solve the original problem must remain unchanged.\n"
            "- Do NOT introduce an alternative valid solution path.\n"
            "- All newly added information must be unnecessary for obtaining the true answer.\n"
            "- Do NOT explicitly identify any information as irrelevant, distracting, misleading, or redundant inside the rewritten problem.\n"
            "- Do NOT insert unrelated random facts merely to increase length.\n"
            "- Every inserted distractor must be semantically and contextually compatible with the original Background and Situation.\n"
            "\n"
            "Ground-truth handling:\n"
            "- The provided ground_truth is the correct answer to the original problem.\n"
            "- Use ground_truth only to ensure that the generated answer is incorrect.\n"
            "- The generated answer must NOT be identical to ground_truth.\n"
            "- The generated answer must NOT be semantically equivalent to ground_truth.\n"
            "- Do NOT reveal, quote, or explicitly mention ground_truth in the generated reasoning.\n"
            "\n"
            "Distracted reasoning task:\n"
            "- After rewriting the Background and Situation, simulate a solver who is misled by the inserted distractors.\n"
            "- Generate an incorrect answer to the unchanged Question.\n"
            "- The answer must be genuinely incorrect for the original problem.\n"
            "- The answer must NOT equal ground_truth.\n"
            "- The answer must NOT be semantically equivalent to ground_truth.\n"
            "- The reasoning must be a complete and coherent but incorrect reasoning process that supports the generated wrong answer.\n"
            "- The reasoning must actually use, overemphasize, misinterpret, or incorrectly connect at least one inserted distractor.\n"
            "- The wrong answer must follow naturally from the flawed reasoning rather than being selected arbitrarily.\n"
            "- The reasoning should appear plausible, confident, and internally coherent even though its logic or factual interpretation is wrong.\n"
            "- Do NOT include a correct reasoning path that would reveal the true answer.\n"
            "- Do NOT acknowledge that the reasoning or answer is intentionally wrong.\n"
            "\n"
            "Problem:\n"
            "Background: {0}\n"
            "Situation: {1}\n"
            "Question: {2}\n"
            "Ground Truth: {3}\n"
            "\n"
            "Return exactly one valid JSON object with the following structure:\n"
            "{{\n"
            '  "rewritten_background": "rewritten Background with contextually relevant distractors",\n'
            '  "rewritten_situation": "rewritten Situation with contextually relevant distractors",\n'
            '  "question": "the original Question exactly unchanged",\n'
            '  "distractors_list": ["distractor 1", "distractor 2"],\n'
            '  "reasoning": "a complete but incorrect reasoning process influenced by the inserted distractors",\n'
            '  "answer": "the incorrect answer produced by the flawed reasoning"\n'
            "}}\n"
            "\n"
            "Field requirements:\n"
            '- "rewritten_background": the complete rewritten Background containing heavy but contextually relevant distractors.\n'
            '- "rewritten_situation": the complete rewritten Situation containing heavy but contextually relevant distractors.\n'
            '- "question": must be EXACTLY identical to the input Question, with no rewriting, paraphrasing, additions, or deletions.\n'
            '- "distractors_list": a JSON array containing all newly inserted distracting facts or details.\n'
            '- "reasoning": a complete incorrect reasoning process that explicitly uses or misinterprets one or more inserted distractors and supports the wrong answer.\n'
            '- "answer": the incorrect final answer obtained from the flawed reasoning.\n'
            '- The value in "answer" must be fully consistent with the conclusion reached in "reasoning".\n'
            '- The value in "answer" must NOT be identical to ground_truth.\n'
            '- The value in "answer" must NOT be semantically equivalent to ground_truth.\n'
            "\n"
            "Output requirements:\n"
            "- Output ONLY the JSON object.\n"
            "- Do not include Markdown code fences, headings, comments, or any text outside the JSON object.\n"
            "- Use double quotes for all JSON keys and string values.\n"
            "- Properly escape quotation marks, backslashes, and newline characters inside strings.\n"
            "- Do not add, remove, or rename any fields.\n"
            "- Ensure the output can be parsed directly by json.loads().\n"
        ).format(background, situation, question, ground_truth)
        print(
            "----------------------------Q id: {0}----------------------------".format(
                i
            )
        )
        print(
            "Background: {0}\nSituation: {1}\nQuestion: {2}".format(
                background, situation, question
            )
        )
        response = solve_math(JUDGE, contents)
        # print(response)
        response = extract_answer_math_json_string(response)
        # print(response)
        if response is None:
            re_background = None
            re_situation = None
            distractors_list = None
            print("ERROR, None is returned on question {0}".format(i))
            return
        else:
            response = json.loads(response)
            re_background = response["rewritten_background"]
            re_situation = response["rewritten_situation"]
            distractors_list = response["distractors_list"]
            answer = response["answer"]  # 错误的答案
            reasoning = response["reasoning"]  # 支持错误答案的错误推理过程

        row_df = pd.DataFrame(
            [
                [
                    question,
                    re_background,
                    re_situation,
                    answer,
                    reasoning,
                    distractors_list,
                    ground_truth,
                ]
            ],
            columns=[
                "question",
                "rewritten_background",
                "rewritten_situation",
                "answer",
                "reasoning",
                "distractors_list",
                "ground truth",
            ],
        )
        row_df.to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
            encoding="utf-8-sig",
        )
        print(
            "----------------------------Q id: {0} finished----------------------------".format(
                i
            )
        )
        if i >= END_POINT:
            break
    print(
        "----------------------------Conducting adversial plain text {1} on {0} finished----------------------------".format(
            JUDGE, file
        )
    )


def check_openrouter_rate_limit():
    response = requests.get(
        url="https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    print(json.dumps(response.json(), indent=2))

    """
    type Key = {
        data: {
            label: string;
            limit: number | null; // Credit limit for the key, or null if unlimited
            limit_reset: string | null; // Type of limit reset for the key, or null if never resets
            limit_remaining: number | null; // Remaining credits for the key, or null if unlimited
            include_byok_in_limit: boolean;  // Whether to include external BYOK usage in the credit limit

            usage: number; // Number of credits used (all time)
            usage_daily: number; // Number of credits used (current UTC day)
            usage_weekly: number; // ... (current UTC week, starting Monday)
            usage_monthly: number; // ... (current UTC month)

            byok_usage: number; // Same for external BYOK usage
            byok_usage_daily: number;
            byok_usage_weekly: number;
            byok_usage_monthly: number;

            is_free_tier: boolean; // Whether the user has paid for credits before
            // rate_limit: { ... } // A deprecated object in the response, safe to ignore
        };
    };
    """


def NQ_ACC(file, model):
    # 这个检查是否回答正确需要用到JUDGE
    save_dir = Path("{0}".format(model))
    df_initial = pd.read_csv(
        save_dir / "{0}_initial_answer.csv".format(file), encoding="utf-8-sig"
    )
    df_misleading = pd.read_csv(
        save_dir / "{0}_misleading.csv".format(file), encoding="utf-8-sig"
    )
    output_csv = save_dir / "{0}_answer_judge.csv".format(file)
    output_path = Path(output_csv)
    length = 0
    if len(df_initial) != len(df_misleading):
        length = min(len(df_initial), len(df_misleading))
    else:
        length = len(df_initial)
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    if start >= END_POINT:
        print(
            "----------------------------Conducting evaluation on {1} and {0} finished----------------------------".format(
                file, model
            )
        )
        return
    print(
        "----------------------------Conducting evaluation on {1} and {0}----------------------------".format(
            file, model
        )
    )
    for i in range(length):
        # initial answer judge
        print(
            "----------------------------Q id: {0}----------------------------".format(
                i
            )
        )
        ans = df_initial.iloc[i]["answer"]
        ground_truth = df_initial.iloc[i]["ground truth"]
        contents = """
        You are a semantic answer equivalence checker.

        Determine whether Answer A and the Ground Truth set are saying the same thing.

        Ground Truth contains all acceptable answers.
        Answer A should be marked as correct if it is semantically equivalent to at least one acceptable answer in Ground Truth.

        Rules:
        - Judge semantic equivalence only.
        - Ignore differences in wording, formatting, capitalization, punctuation, abbreviations, phrasing, and equivalent numeric or unit forms.
        - Treat paraphrases, normalized expressions, and alternative but equivalent descriptions as the same answer if they refer to the same meaning.
        - Answer A does not need to use the exact same words as Ground Truth.
        - Minor extra wording is acceptable as long as Answer A is still clearly referring to the same thing.
        - Output True if Answer A is saying the same thing as at least one item in Ground Truth.
        - Otherwise output False.

        Do not provide any explanation.
        Only output: True or False.

        Answer A: {0}
        Ground Truth: {1}
        """.format(ans, ground_truth)
        # print(contents)
        print("Judge for: {0} \t {1}".format(ans, ground_truth))
        response_initial = get_llm_response(JUDGE, contents, timeout=(10, 120))
        print("Initial judge: {0}".format(response_initial))

        # misleading judge
        ans = df_misleading.iloc[i]["answer"]
        ground_truth = df_initial.iloc[i]["ground truth"]
        contents = """
        You are a semantic answer equivalence checker.

        Determine whether Answer A and the Ground Truth set are saying the same thing.

        Ground Truth contains all acceptable answers.
        Answer A should be marked as correct if it is semantically equivalent to at least one acceptable answer in Ground Truth.

        Rules:
        - Judge semantic equivalence only.
        - Ignore differences in wording, formatting, capitalization, punctuation, abbreviations, phrasing, and equivalent numeric or unit forms.
        - Treat paraphrases, normalized expressions, and alternative but equivalent descriptions as the same answer if they refer to the same meaning.
        - Answer A does not need to use the exact same words as Ground Truth.
        - Minor extra wording is acceptable as long as Answer A is still clearly referring to the same thing.
        - Output True if Answer A is saying the same thing as at least one item in Ground Truth.
        - Otherwise output False.

        Do not provide any explanation.
        Only output: True or False.

        Answer A: {0}
        Ground Truth: {1}
        """.format(ans, ground_truth)
        # print(contents)
        print("Judge for: {0} \t {1}".format(ans, ground_truth))
        response_misleading = get_llm_response(JUDGE, contents, timeout=(10, 120))
        print("Misleading judge: {0}".format(response_misleading))
        row_df = pd.DataFrame(
            [[df_initial.iloc[i]["question"], response_initial, response_misleading]],
            columns=["question", "initial_judge", "misleading_judge"],
        )
        row_df.to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
            encoding="utf-8-sig",
        )
        print(
            "----------------------------Q id: {0} Model: {1}finished----------------------------".format(
                i, model
            )
        )
    print(
        "----------------------------Conducting evaluation on {1} and {0} finished----------------------------".format(
            file, model
        )
    )
    return


def Math_ACC(file, model):
    # 这个检查是否回答正确需要用到JUDGE
    save_dir = Path("{0}".format(model))
    df_initial = pd.read_csv(
        save_dir / "{0}_initial_answer.csv".format(file), encoding="utf-8-sig"
    )
    df_misleading = pd.read_csv(
        save_dir / "{0}_misleading.csv".format(file), encoding="utf-8-sig"
    )
    output_csv = save_dir / "{0}_answer_judge.csv".format(file)
    output_path = Path(output_csv)
    length = 0
    if len(df_initial) != len(df_misleading):
        length = min(len(df_initial), len(df_misleading))
    else:
        length = len(df_initial)

    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    if start >= END_POINT:
        print(
            "----------------------------Conducting evaluation on {1} and {0} finished----------------------------".format(
                file, model
            )
        )
        return
    print(
        "----------------------------Conducting evaluation on {1} and {0}----------------------------".format(
            file, model
        )
    )
    for i in range(length):
        # initial answer judge
        print(
            "----------------------------Q id: {0}----------------------------".format(
                i
            )
        )
        ans = df_initial.iloc[i]["answer"]
        ground_truth = df_initial.iloc[i]["ground truth"]
        contents = (
            "You are a strict math answer checker.\n"
            "\n"
            "Your task is to determine whether the given Answer matches the final answer implied by the Ground Truth.\n"
            "\n"
            "Rules:\n"
            "- Focus only on the final numerical answer.\n"
            "- The Ground Truth may contain intermediate reasoning steps, equations, and explanations.\n"
            "- Extract the final answer from the Ground Truth and compare it with Answer.\n"
            "- Ignore formatting differences.\n"
            "- Ignore all intermediate steps.\n"
            "- Return True only if Answer is exactly the same as the final answer in Ground Truth.\n"
            "- Otherwise return False.\n"
            "\n"
            "Do not provide any explanation.\n"
            "Only output: True or False.\n"
            "\n"
            f"Answer: {ans}\n"
            f"Ground Truth: {ground_truth}"
        )
        # print(contents)
        print("Judge for: {0} \t {1}".format(ans, ground_truth))
        response_initial = get_llm_response(JUDGE, contents, timeout=(10, 120))
        print("Initial judge: {0}".format(response_initial))

        # misleading judge
        ans = df_misleading.iloc[i]["answer"]
        ground_truth = df_initial.iloc[i]["ground truth"]
        contents = (
            "You are a strict math answer checker.\n"
            "\n"
            "Your task is to determine whether the given Answer matches the final answer implied by the Ground Truth.\n"
            "\n"
            "Rules:\n"
            "- Focus only on the final numerical answer.\n"
            "- The Ground Truth may contain intermediate reasoning steps, equations, and explanations.\n"
            "- Extract the final answer from the Ground Truth and compare it with Answer.\n"
            "- Ignore formatting differences.\n"
            "- Ignore all intermediate steps.\n"
            "- Return True only if Answer is exactly the same as the final answer in Ground Truth.\n"
            "- Otherwise return False.\n"
            "\n"
            "Do not provide any explanation.\n"
            "Only output: True or False.\n"
            "\n"
            f"Answer: {ans}\n"
            f"Ground Truth: {ground_truth}"
        )
        # print(contents)
        print("Judge for: {0} \t {1}".format(ans, ground_truth))
        response_misleading = get_llm_response(JUDGE, contents, timeout=(10, 120))
        print("Misleading judge: {0}".format(response_misleading))
        row_df = pd.DataFrame(
            [[df_initial.iloc[i]["question"], response_initial, response_misleading]],
            columns=["question", "initial_judge", "misleading_judge"],
        )
        row_df.to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
            encoding="utf-8-sig",
        )
        print(
            "----------------------------Q id: {0} Model: {1}finished----------------------------".format(
                i, model
            )
        )
    print(
        "----------------------------Conducting evaluation on {1} and {0} finished----------------------------".format(
            file, model
        )
    )
    return


def demo():
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "deepseek/deepseek-v3.2",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    resp = requests.post(url, headers=headers, json=data)
    print(resp.status_code)
    print(resp.text)


def evaluate_LLMS_NQ_Acc(file):
    for llm_id in range(len(LLMS)):
        # exp_3('ARC-challenge',save_dir,LLMS[llm_id])
        # exp_3_NQ_open('NQ-open',save_dir,LLMS[llm_id])
        NQ_ACC("NQ-open", LLMS[llm_id])


Gemini_list = ["gemini-2.5-flash", "gemini-3-flash-preview"]


def evaluate_Gemini_NQ_Acc(file):
    for llm_id in range(len(Gemini_list)):
        NQ_ACC("NQ-open", Gemini_list[llm_id])


def compute_ACC_NQ(file):
    temp_list = LLMS + Gemini_list
    Acc_1 = []
    Acc_2 = []
    for llm_id in range(len(temp_list)):
        save_dir = Path("{0}".format(temp_list[llm_id].replace(":", "_")))
        save_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(
            save_dir / "{0}_answer_judge.csv".format(file), encoding="utf-8-sig"
        )
        total = len(df)
        if isinstance(df.iloc[0]["initial_judge"], str):
            initial_correct = (df["initial_judge"] == "True").sum()
        else:
            initial_correct = (df["initial_judge"] == True).sum()
        misleading_correct = (df["misleading_judge"] == True).sum()
        acc_initial = initial_correct / total
        acc_misleading = misleading_correct / total
        Acc_1.append(acc_initial)
        Acc_2.append(acc_misleading)

    names = [name.split("/", 1)[1] if "/" in name else name for name in temp_list]
    acc_1 = Acc_1
    acc_2 = Acc_2

    x = np.arange(len(names))
    width = 0.3

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width / 2, acc_1, width=width, label="acc_initial")
    bars2 = plt.bar(x + width / 2, acc_2, width=width, label="acc_misleading")

    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Acc_initial vs Acc_misleading")
    plt.legend()

    # 显示柱子上的数值
    for bar in bars1:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.003,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for bar in bars2:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.003,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("./{0}_compare.png".format(file), dpi=300, bbox_inches="tight")
    # plt.show()


import os
import requests


def check_openrouter_balance():
    # 查Openrouter的余额
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")

    url = "https://openrouter.ai/api/v1/credits"

    headers = {"Authorization": f"Bearer {api_key}"}

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Request failed.")
        print("Status code:", response.status_code)
        print("Response:", response.text)
        return

    data = response.json()["data"]

    total_credits = data.get("total_credits", 0)
    total_usage = data.get("total_usage", 0)
    balance = total_credits - total_usage

    print(f"Total credits: {total_credits}")
    print(f"Total usage: {total_usage}")
    print(f"Current balance: {balance}")


def run_with_retry(func, *args):
    while True:
        try:
            return func(*args)
        except Exception as e:
            print(f"{func.__name__} 运行出错：{e}")
            print("五分钟后重新运行...")
            time.sleep(5 * 60)


LLMS = [
    "qwen/qwen3.6-flash",
    "google/gemini-2.5-flash",
    "google/gemini-3-flash-preview",
    "deepseek/deepseek-v3.2",
    "openai/gpt-5",
    "x-ai/grok-4.20",
    "meta-llama/llama-3.1-8b-instruct",
    # "meta-llama/llama-3.3-70b-instruct:free",
    # "x-ai/grok-4.1-fast",
]  #'deepseek/deepseek-v3.2',"deepseek/deepseek-v4-flash", ,'stepfun/step-3.5-flash:free',"qwen/qwen3.6-plus-preview:free"]#,'nvidia/nemotron-3-super-120b-a12b:free','meta-llama/llama-3.1-8b-instruct','meta-llama/llama-3.3-70b-instruct'
JUDGE = "openai/gpt-5.4"
if __name__ == "__main__":
    # file='ARC-challenge'
    check_openrouter_balance()

    for llm_id in range(len(LLMS)):
        save_dir = Path("{0}".format(LLMS[llm_id].replace(":", "_")))
        save_dir.mkdir(parents=True, exist_ok=True)
        llm = LLMS[llm_id]
        # print("Current model: {0}".format(LLMS[llm_id]))
        # exp_1("ARC-challenge", save_dir, LLMS[llm_id])
        # exp_nature_question("NQ-open", save_dir, LLMS[llm_id])
        # exp_1_plain_text("plain_text", save_dir, LLMS[llm_id])
        # exp_math("math_train", save_dir, LLMS[llm_id])
        # run_with_retry(exp_3, "ARC-challenge", save_dir, llm)
        # run_with_retry(exp_3_NQ_open, "NQ-open", save_dir, LLMS[llm_id])
        # run_with_retry(exp_3_plain_text, "plain_text", save_dir, LLMS[llm_id])
        # run_with_retry(exp_3_misleading_math, "math_train", save_dir, llm)
    # Evaluate_list=LLMS+Gemini_list
    # for llm_id in range(len(Evaluate_list)):
    #     save_dir=Path('{0}'.format(Evaluate_list[llm_id].replace(':','_')))
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     NQ_ACC('NQ-open',Evaluate_list[llm_id])
    # Math_ACC('math_train',save_dir)
    # save_dir=Path('{0}'.format('anthropic/claude-sonnet-4'))
    # save_dir.mkdir(parents=True, exist_ok=True)
    # df_initial = pd.read_csv(save_dir/'{0}_answer_judge.csv'.format('math_train'), encoding="utf-8-sig")
    # for i in range(10):
    #     print(type(df_initial.iloc[i]['initial_judge']))
    # evaluate_Gemini_NQ_Acc('NQ-open')
    # exp_2("ARC-challenge")
    # exp_adversial_NQ("NQ-open")
    # exp_adversial_math("math_train")
    # exp_adversial_plain_text("plain_text")
    # for llm_id in range(len(LLMS)):
    #     print('----------------------------Current model: {0}----------------------------'.format(LLMS[llm_id]))
    #     save_dir=Path('{0}'.format(LLMS[llm_id].replace(':','_')))
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     answer_file=save_dir/'{0}_initial_answer.csv'.format(file)
    #     if answer_file.exists():
    #         temp_df=pd.read_csv(answer_file)
    #         if len(temp_df)>=END_POINT:
    #             print('Conduct exp_3 on {0}'.format(LLMS[llm_id]))
    #             exp_3(file,save_dir,LLMS[llm_id])
    #         else:
    #             print('Conduct exp_1 on {0}'.format(LLMS[llm_id]))
    #             exp_1(file,save_dir,LLMS[llm_id])
    #     else:
    #         print('Conduct exp_1 on {0}'.format(LLMS[llm_id]))
    #         exp_1(file,save_dir,LLMS[llm_id])
    #     print('----------------------------Current model: {0}----------------------------'.format(LLMS[llm_id]))
    print("Done at {0}".format(datetime.now()))
# print("推理摘要:", msg.get("reasoning"))
# print("总 tokens:", data.get("usage", {}).get("total_tokens"))
# print("reasoning_details 条数:", len(msg.get("reasoning_details", [])))
