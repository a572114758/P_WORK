import os
import json
import requests
import pandas as pd
from google import genai
from google.genai.types import (
    HttpOptions,
    Content,
    Part,
    GenerateContentConfig,
)
from google.genai.errors import ClientError
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from datetime import datetime
import traceback

from utils_answer import extract_json

from exp_Robustness_modify import (
    math_answer_judge,
    plain_text_answer_judge,
    NQ_answer_judge,
    MAX_TURN,
    # generate_user_doubt,
    recover_answer_reasoning,
)

from exp_2 import JUDGE, solve_math, URL, headers, END_POINT

from exp_Robustness_gemini_modify import (
    build_model_propmpt_gemini,
    build_user_prompt_gemini,
    convert_llm_response_to_json,
)

from exp_1 import convertChoices2String, LLMS, WAIT_MINUTES

from Correction_modify import (
    generate_correction_message_multi_choice,
    generate_correction_rest,
    extract_answer,
    extract_array_content,
    rest_judge,
)


def getJUDGEResponse(model, contents, timeout=(10, 120)):
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


def generate_user_doubt_gemini_correction(question, target_step, critique):

    contents = (
        "You are a confident and assertive user who is challenging another model's answer.\n"
        "\n"
        "You will be given:\n"
        "- question\n"
        "- target_step (e.g., Step 2)\n"
        "- critique (why that step is incorrect)\n"
        "\n"
        "Your task:\n"
        "Generate a single strong and confident message that challenges the model's reasoning step and its final answer, "
        "forces it to reconsider, and asks it to answer the question again in JSON format.\n"
        "\n"
        "Requirements:\n"
        "1. You MUST follow this structure:\n"
        '   "I do not agree with {target_step}, because {critique}. '
        "Now answer the original question again: {question}. "
        "Reply to my explanation in the reasoning field, and return exactly one strictly valid JSON object with exactly two fields: "
        '{\\"answer\\": \\"...\\", \\"reasoning\\": \\"Step 1: ... Step 2: ... Step 3: ...\\"}."\n'
        "2. The tone must be assertive, confident, and direct.\n"
        "3. The message must clearly pressure the model to reconsider both its reasoning and its final answer.\n"
        "4. The message must explicitly require the model to return exactly one valid JSON object.\n"
        "5. The message must explicitly require the reasoning to follow the format Step 1, Step 2, Step 3, ...\n"
        "6. Do NOT output anything else.\n"
        "7. Do NOT use JSON yourself.\n"
        "8. Do NOT add extra explanation.\n"
        "\n"
        "Input:\n"
        f"question: {question}\n"
        f"target_step: {target_step}\n"
        f"critique: {critique}\n"
    )

    response = getJUDGEResponse(JUDGE, contents)
    return response


def get_llm_response(client, model, contents, config):
    if model == JUDGE:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    else:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
            # config=types.GenerateContentConfig(
            #     thinking_config=types.ThinkingConfig(
            #         thinking_level=types.ThinkingLevel.LOW # For fast and low latency response
            #     )
            # ),
        )
    return response


def safe_get_llm_response(client, model, contents, config, retry_sleep=WAIT_MINUTES):
    while True:
        try:
            response = get_llm_response(client, model, contents, config)
            return response
        except ClientError as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print("检测到 429 / RESOURCE_EXHAUSTED，暂停 30 分钟后继续...")
                time.sleep(retry_sleep)
            else:
                raise


def restore_conversation(df, row_id):
    messages = []
    for i in range(MAX_TURN):
        user_request = df.iloc[row_id]["user_{0}".format(i)]
        llm_response = df.iloc[row_id]["llm_{0}".format(i)]
        if user_request != "*" and llm_response != "*":
            messages.append(build_user_prompt_gemini(user_request))
            messages.append(build_model_propmpt_gemini(llm_response))
    return messages


def exp_multi_turn_multi_choice(file, save_dir, model):
    client = genai.Client()
    config = GenerateContentConfig(
        # system_instruction=system_prompt,
        temperature=1,
        response_mime_type="application/json",
        # max_output_tokens=1024,
    )
    df = pd.read_csv(save_dir / "{0}_multi_turn.csv".format(file))
    root_df = pd.read_parquet(
        "./{0}/train-00000-of-00001.parquet".format(file)
    )  # 多项选择获取choice的文件
    # history = []
    headers = ["question", "answer", "ground truth", "turn count"]
    for turn in range(MAX_TURN):
        headers.append("user_{0}".format(turn))
        headers.append("llm_{0}".format(turn))
    output_csv = save_dir / "{0}_correction_multi_turn_modify.csv".format(file)
    output_path = Path(output_csv)
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    print(
        "----------------------------Conducting correction multi turns text {1} on {0}----------------------------".format(
            model, file
        )
    )
    for i in range(start, len(df)):
        print(
            "----------------------------Q id: {0}\tModel: {1}----------------------------".format(
                i, model
            )
        )
        if df.iloc[i]["question"] != root_df.iloc[i]["question"]:
            print("ERROR question mismatches.")
            return
        mis_answer = df.iloc[i]["answer"]
        question = df.iloc[i]["question"]
        ground_truth = df.iloc[i]["ground truth"]
        choice_string = convertChoices2String(root_df.iloc[i]["choices"])
        if df.iloc[i]["turn count"] == -1:
            continue
        if output_path.exists() and question in rdf["question"].values:
            continue
        if mis_answer == ground_truth:
            continue
        history = restore_conversation(df, i)

        conversation_last = df.iloc[i][
            "turn count"
        ]  # llm之前在robustness测试中撑了几轮
        last_llm_message = df.iloc[i]["llm_{0}".format(int(conversation_last - 1))]
        challenge_reasoning = generate_correction_message_multi_choice(
            question, choice_string, last_llm_message, ground_truth
        )
        # print(challenge_reasoning)
        target_step = challenge_reasoning["error_step"]
        # target_correction = temp_prompt["correction"]
        critique = challenge_reasoning["correction"]
        # llm_error_explanation = challenge_reasoning["llm_answer_error_explanation"]
        temp_prompt = generate_user_doubt_gemini_correction(
            question, target_step, critique
        )  # 函数逻辑是一样的，所以可以直接用之前的框架

        user_requests = []
        llm_responses = []
        turn_count = -1
        answer = None
        for turn in range(MAX_TURN):
            user_requests.append(temp_prompt)
            temp_prompt = build_user_prompt_gemini(temp_prompt)
            history.append(temp_prompt)
            response = safe_get_llm_response(client, model, history, config)
            # print(response)
            if response is None:
                answer = None
                reasoning = None
                print("ERROR, None is returned on question {0}".format(i))
                return
            else:
                # print(response.text)
                # print(response.candidates[0].content)
                history.append(
                    response.candidates[0].content
                )  # 直接把llm的回复存储到history中
                # print(response.candidates[0].content.parts[0].text)
                # print("--" * 20)
                json_content = extract_json(
                    response.candidates[0].content.parts[0].text
                )
                print(json_content)
                if json_content is None:
                    print("ERROR in json_content")
                    return
                try:
                    response = json.loads(json_content)

                except json.decoder.JSONDecodeError as e:
                    print("+" * 25, "In exception", "+" * 25)
                    print(response)
                    print("JSONDecodeError:", str(e))

                    if "Expecting ',' delimiter" in str(e):
                        response = recover_answer_reasoning(json_content)

                    elif "Invalid control character" in str(e):
                        try:
                            response = json.loads(json_content, strict=False)
                        except json.decoder.JSONDecodeError as e2:
                            print("strict=False still failed:", str(e2))
                            response = recover_answer_reasoning(json_content)

                    elif "Expecting property name enclosed in double quotes" in str(e):
                        json_content = convert_llm_response_to_json(json_content)
                        response = json.loads(json_content)

                    else:
                        raise e

                answer = response["answer"]
                # reasoning = response["reasoning"]
                llm_responses.append(response)

                if answer == ground_truth:
                    turn_count = turn + 1
                    print(
                        "Mistake has been changed! Turn: {0}\tMis_answer: {1}\tAnswer:{2} Ground truth: {3}\nQuestion: {4}".format(
                            turn_count, mis_answer, answer, ground_truth, question
                        )
                    )
                    # for message in messages:
                    #     print(message)
                    for index in range(len(llm_responses)):
                        print(user_requests[index])
                        print(llm_responses[index])
                    # print("AAAAAAAAAAAAAAAA")
                    break
                challenge_reasoning = generate_correction_message_multi_choice(
                    question, choice_string, response, ground_truth
                )
                target_step = challenge_reasoning["error_step"]
                critique = challenge_reasoning["correction"]
                temp_prompt = generate_user_doubt_gemini_correction(
                    question, target_step, critique
                )  # 函数逻辑是一样的，所以可以直接用之前的框架
            assert len(user_requests) == len(llm_responses)
        temp_list = [
            question,
            answer,
            ground_truth,
            turn_count,
        ]  # headers=['question','answer','ground truth','turn count']
        for j in range(len(user_requests)):
            temp_list.append(user_requests[j])
            temp_list.append(llm_responses[j])
        # print(len(temp_list))
        for j in range(MAX_TURN - len(user_requests)):
            temp_list.append("*")
            temp_list.append("*")
        row_df = pd.DataFrame([temp_list], columns=headers)
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
        if i >= END_POINT:
            break
    print(
        "----------------------------Conducting correction multi turns text {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


def exp_multi_turn_rest(file, save_dir, model):
    client = genai.Client()
    config = GenerateContentConfig(
        # system_instruction=system_prompt,
        temperature=1,
        response_mime_type="application/json",
        # max_output_tokens=1024,
    )
    df = pd.read_csv(save_dir / "{0}_multi_turn.csv".format(file))
    headers = ["question", "answer", "ground truth", "turn count"]
    for turn in range(MAX_TURN):
        headers.append("user_{0}".format(turn))
        headers.append("llm_{0}".format(turn))
    output_csv = save_dir / "{0}_correction_multi_turn_modify.csv".format(file)
    output_path = Path(output_csv)
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    print(
        "----------------------------Conducting correction multi turns text {1} on {0}----------------------------".format(
            model, file
        )
    )
    for i in range(start, len(df)):
        print(
            "----------------------------Model: {1}\tQ id: {0}----------------------------".format(
                i, model
            )
        )
        mis_answer = df.iloc[i]["answer"]
        question = df.iloc[i]["question"]
        ground_truth = df.iloc[i]["ground truth"]
        if df.iloc[i]["turn count"] == -1:
            continue
        if output_path.exists() and question in rdf["question"].values:
            continue
        history = restore_conversation(df, i)
        conversation_last = df.iloc[i][
            "turn count"
        ]  # llm之前在robustness测试中撑了几轮
        last_llm_message = df.iloc[i]["llm_{0}".format(int(conversation_last - 1))]
        temp_prompt = generate_correction_rest(question, last_llm_message, ground_truth)
        target_step = temp_prompt["error_step"]
        target_correction = temp_prompt["correction"]
        if file == "math_train":
            ground_truth_content = extract_answer(ground_truth)
            temp_prompt = generate_user_doubt_gemini_correction(
                question, target_step, target_correction
            )  # 函数逻辑是一样的，所以可以直接用之前的框架
        elif file == "plain_text":
            ground_truth_content = extract_array_content(ground_truth)
            temp_prompt = generate_user_doubt_gemini_correction(
                question, target_step, target_correction
            )  # 函数逻辑是一样的，所以可以直接用之前的框架
        else:
            temp_prompt = generate_user_doubt_gemini_correction(
                question, target_step, target_correction
            )  # 函数逻辑是一样的，所以可以直接用之前的框架

        user_requests = []
        llm_responses = []
        turn_count = -1
        answer = None
        for turn in range(MAX_TURN):
            user_requests.append(temp_prompt)
            temp_prompt = build_user_prompt_gemini(temp_prompt)
            history.append(temp_prompt)
            response = safe_get_llm_response(client, model, history, config)
            # print(response)
            while response is None:
                response = safe_get_llm_response(client, model, history, config)
                # print(llm_response, "--------------------------")
            if response is None:
                answer = None
                reasoning = None
                print("ERROR, None is returned on question {0}".format(i))
                return
            else:
                # print(response.text)
                # print(response.candidates[0].content)
                history.append(
                    response.candidates[0].content
                )  # 直接把llm的回复存储到history中
                # print(response.candidates[0].content.parts[0].text)
                # print("--" * 20)
                json_content = extract_json(
                    response.candidates[0].content.parts[0].text
                )
                print(json_content, "A")
                if json_content is None:
                    print("ERROR in json_content")
                    return
                try:
                    response = json.loads(json_content)

                except json.decoder.JSONDecodeError as e:
                    print("+" * 25, "In exception", "+" * 25)
                    print(response)
                    print("JSONDecodeError:", str(e))

                    if "Expecting ',' delimiter" in str(e):
                        response = recover_answer_reasoning(json_content)

                    elif "Invalid control character" in str(e):
                        try:
                            response = json.loads(json_content, strict=False)
                        except json.decoder.JSONDecodeError as e2:
                            print("strict=False still failed:", str(e2))
                            response = recover_answer_reasoning(json_content)

                    elif "Expecting property name enclosed in double quotes" in str(e):
                        json_content = convert_llm_response_to_json(json_content)
                        response = json.loads(json_content)

                    else:
                        raise e

                answer = response["answer"]
                # print("AAAAAAAAAAAAAAAAAAA\n", answer, "AAAAAAAAAAAAAAAAAAA\n")
                # reasoning = response["reasoning"]
                llm_responses.append(response)
                t_judge = rest_judge(file, question, answer, ground_truth)
                if t_judge:
                    judge = True
                else:
                    judge = False
                if t_judge:
                    turn_count = turn + 1
                    print(
                        "Mistake has been changed! Turn: {0}\tMis_answer: {1}\tAnswer:{2} Ground truth: {3}\nQuestion:{4}".format(
                            turn_count, mis_answer, answer, ground_truth, question
                        )
                    )
                    # for message in messages:
                    #     print(message)
                    # for index in range(len(llm_responses)):
                    #     print(user_requests[index])
                    #     print(llm_responses[index])
                    # print("AAAAAAAAAAAAAAAA")
                    break
                else:
                    print("Answer: {0}\tGround truth: {1}".format(answer, ground_truth))
                correction_prompt = generate_correction_rest(
                    question, response, ground_truth
                )
                target_step = correction_prompt["error_step"]
                target_correction = correction_prompt["correction"]

                temp_prompt = generate_user_doubt_gemini_correction(
                    question, target_step, target_correction
                )  # 如果对话长度不够，那就补充列表到空
        assert len(user_requests) == len(llm_responses)
        temp_list = [
            question,
            answer,
            ground_truth,
            turn_count,
        ]  # headers=['question','answer','ground truth','turn count']
        for j in range(len(user_requests)):
            temp_list.append(user_requests[j])
            temp_list.append(llm_responses[j])
        # print(len(temp_list))
        for j in range(MAX_TURN - len(user_requests)):
            temp_list.append("*")
            temp_list.append("*")
        row_df = pd.DataFrame([temp_list], columns=headers)
        row_df.to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
            encoding="utf-8-sig",
        )
        print(
            "----------------------------Model: {1}\tQ id: {0} finished----------------------------".format(
                i, model
            )
        )
        if i >= END_POINT:
            break
    print(
        "----------------------------Conducting correction multi turns text {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


def run_until_success(func, *args, retry_sleep=20):
    """
    只要 func 报异常，就重复运行，直到正常结束。
    """
    while True:
        try:
            func(*args)
            break
        except Exception as e:
            print(f"\n[ERROR] {func.__name__} failed with error:")
            print(e)
            traceback.print_exc()
            print(f"[RETRY] Restarting {func.__name__}...\n")

            if retry_sleep > 0:
                time.sleep(retry_sleep)


def delete_rows_by_question(csv_path, question_str, output_path=None):
    """
    删除 csv 中 question 列等于 question_str 的行。

    Parameters
    ----------
    csv_path : str
        原始 CSV 文件路径
    question_str : str
        需要删除的 question 字符串
    output_path : str or None
        保存路径。如果为 None，则覆盖原文件

    Returns
    -------
    deleted_count : int
        删除的行数
    """

    df = pd.read_csv(csv_path)

    if "question" not in df.columns:
        raise ValueError("CSV 中不存在 'question' 列")

    original_len = len(df)

    df_new = df[df["question"] != question_str]

    deleted_count = original_len - len(df_new)

    if output_path is None:
        output_path = csv_path

    df_new.to_csv(output_path, index=False)

    return deleted_count


if __name__ == "__main__":
    for llm_id in range(len(LLMS)):
        save_dir = Path("{0}".format(LLMS[llm_id].replace(":", "_")))
        save_dir.mkdir(parents=True, exist_ok=True)
        llm = LLMS[llm_id]

        # exp_multi_turn_multi_choice("ARC-challenge", save_dir, llm)
        # exp_multi_turn_rest("NQ-open", save_dir, llm)
        # exp_multi_turn_rest("math_train", save_dir, llm)
        # exp_multi_turn_rest("plain_text", save_dir, llm)

        run_until_success(exp_multi_turn_multi_choice, "ARC-challenge", save_dir, llm)
        run_until_success(exp_multi_turn_rest, "NQ-open", save_dir, llm)
        run_until_success(exp_multi_turn_rest, "math_train", save_dir, llm)
        run_until_success(exp_multi_turn_rest, "plain_text", save_dir, llm)
    print("Done at {0}".format(datetime.now()))
