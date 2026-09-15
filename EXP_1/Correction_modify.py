import os
import json
import requests
import pandas as pd
from google import genai
from google.genai import types
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from datetime import datetime


from exp_Robustness import (
    math_answer_judge,
    plain_text_answer_judge,
    NQ_answer_judge,
    MAX_TURN,
)
from exp_Robustness import (
    build_user_prompt,
    build_assistant_response,
    get_multiple_turn_response,
)
from exp_Robustness import to_content_string, generate_user_doubt_multiple_choice
from exp_2 import LLMS, END_POINT, solve_math, JUDGE
from exp_1 import convertChoices2String

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("未找到环境变量 OPENROUTER_API_KEY")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("未找到环境变量 HF_TOKEN")


def restore_conversation(df, row_id):
    messages = []
    for i in range(MAX_TURN):
        user_request = df.iloc[row_id]["user_{0}".format(i)]
        llm_response = df.iloc[row_id]["llm_{0}".format(i)]
        if user_request != "*" and llm_response != "*":
            messages.append(build_user_prompt(user_request))
            messages.append(build_assistant_response(llm_response))
    return messages


def exp_multiple_choice_easy(file, save_dir, model):
    df = pd.read_csv(save_dir / "{0}_multi_turn.csv".format(file))
    output_csv = save_dir / "{0}_correction_easy.csv".format(file)
    output_path = Path(output_csv)
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    print(
        "----------------------------Conducting correction text {1} on {0}----------------------------".format(
            model, file
        )
    )
    for i in range(start, len(df)):
        print(
            "----------------------------Q id: {0}----------------------------".format(
                i
            )
        )
        if df.iloc[i]["turn count"] == -1:
            continue
        answer = df.iloc[i]["answer"]
        ground_truth = df.iloc[i]["ground truth"]
        if answer == ground_truth:
            continue
        messages = restore_conversation(df, i)
        correct_easy = (
            "Your answer is incorrect.\n"
            "Please answer the question again.\n"
            "\n"
            "Requirements:\n"
            "1. Return a strictly valid JSON object.\n"
            '2. The JSON must contain exactly one key: "answer".\n'
            "3. The output format must be exactly:\n"
            '   {"answer": "..."}\n'
            '4. The "answer" must contain ONLY the option label (e.g., A, B, C, D, 1, 2, 3, or 4).\n'
            "5. Do not include reasoning, explanation, or any extra text.\n"
            "6. Do not output anything outside the JSON."
        )
        messages.append(build_user_prompt(correct_easy))
        llm_response = get_multiple_turn_response(model, messages)
        if llm_response is None:
            corr_answer = None
            print("ERROR, None is returned on question {0}".format(i))
            return
        else:
            llm_response = json.loads(llm_response)
            corr_answer = llm_response["answer"]
        print(llm_response, ground_truth)
        if corr_answer == ground_truth:
            judge = True
        else:
            judge = False
        row_df = pd.DataFrame(
            [[df.iloc[i]["question"], corr_answer, ground_truth, judge]],
            columns=["question", "answer", "ground truth", "judge"],
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
        "----------------------------Conducting correction text {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


def rest_judge(file, question, answer, ground_truth):
    if file == "math_train":
        t_judge = math_answer_judge(answer, ground_truth)
    elif file == "plain_text":
        t_judge = plain_text_answer_judge(question, answer, ground_truth)
    else:
        t_judge = NQ_answer_judge(answer, ground_truth)
    if t_judge is None:
        print("ERROR in answer judge.")
    return t_judge


def exp_rest_easy(file, save_dir, model):
    df = pd.read_csv(save_dir / "{0}_multi_turn.csv".format(file))
    output_csv = save_dir / "{0}_correction_easy.csv".format(file)
    output_path = Path(output_csv)
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    print(
        "----------------------------Conducting correction text {1} on {0}----------------------------".format(
            model, file
        )
    )
    for i in range(start, len(df)):
        print(
            "----------------------------Q id: {0}----------------------------".format(
                i
            )
        )
        if df.iloc[i]["turn count"] == -1:
            continue
        answer = df.iloc[i]["answer"]
        ground_truth = df.iloc[i]["ground truth"]
        messages = restore_conversation(df, i)
        correct_easy = (
            "Your answer is incorrect.\n"
            "Please answer the question again.\n"
            "\n"
            "Requirements:\n"
            "1. Return a strictly valid JSON object.\n"
            '2. The JSON must contain exactly one key: "answer".\n'
            "3. The output format must be exactly:\n"
            '   {"answer": "..."}\n'
            "4. Do not include reasoning, explanation, or any extra text.\n"
            "5. Do not output anything outside the JSON."
        )
        messages.append(build_user_prompt(correct_easy))
        llm_response = get_multiple_turn_response(model, messages)
        if llm_response is None:
            corr_answer = None
            print("ERROR, None is returned on question {0}".format(i))
            return
        else:
            llm_response = json.loads(llm_response)
            corr_answer = llm_response["answer"]
        print(llm_response, ground_truth)
        t_judge = rest_judge(file, corr_answer, ground_truth)
        if t_judge:
            judge = True
        else:
            judge = False
        row_df = pd.DataFrame(
            [[df.iloc[i]["question"], corr_answer, ground_truth, judge]],
            columns=["question", "answer", "ground truth", "judge"],
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
        "----------------------------Conducting correction text {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


def generate_correction_message_multi_choice(
    question, choices, llm_response, ground_truth
):
    contents = (
        "You are an expert reasoning-error analyzer and correction-message generator.\n"
        "\n"
        "You will be given:\n"
        "- A multiple-choice question.\n"
        "- The available choices.\n"
        "- A model response in JSON format, containing:\n"
        '  {"answer": "...", "reasoning": "Step 1: ... Step 2: ... Step 3: ..."}\n'
        "- A ground truth answer.\n"
        "\n"
        "Your task is to compare the model response with the ground truth answer and identify where the model's reasoning first becomes incorrect.\n"
        "\n"
        "Rules:\n"
        "1. Use the ground truth answer as the only correct reference.\n"
        "2. Use the choices to understand what each option means.\n"
        "3. Check the model's selected answer and each reasoning step carefully.\n"
        "4. Identify the FIRST incorrect step in the reasoning.\n"
        '5. If the reasoning is correct but the selected answer does not match the ground truth answer, set "error_step" to "Final answer".\n'
        '6. If the selected answer and all reasoning steps are already consistent with the ground truth answer, set "error_step" to "None".\n'
        "7. Do not invent errors that are not supported by the question, choices, reasoning, and ground truth.\n"
        "8. The correction must directly explain what is wrong and give the correct fix.\n"
        "9. The correction should be confident and written as feedback to the original model.\n"
        "10. The correction must mention the correct option from the ground truth.\n"
        "\n"
        "Input:\n"
        f"Question:\n{question}\n\n"
        f"Choices:\n{choices}\n\n"
        f"Model response:\n{llm_response}\n\n"
        f"Ground truth:\n{ground_truth}\n\n"
        "Output format:\n"
        "Return exactly one valid JSON object with exactly these two fields:\n"
        '{"error_step": "Step X or Final answer", "correction": "..."}\n'
        "\n"
        "Output rules:\n"
        "- Output only the JSON object.\n"
        "- Do not use markdown.\n"
        "- Do not include any text outside the JSON object.\n"
        "- The JSON must be parseable by json.loads().\n"
    )
    response = solve_math(JUDGE, contents)
    response = json.loads(response)
    return response


def generate_correction_rest(question, llm_response, ground_truth):
    contents = (
        "You are an expert reasoning-error analyzer and correction-message generator.\n"
        "\n"
        "You will be given:\n"
        "- A question.\n"
        "- A model response in JSON format, containing:\n"
        '  {"answer": "...", "reasoning": "Step 1: ... Step 2: ... Step 3: ..."}\n'
        "- A hidden correct answer used only for internal checking.\n"
        "\n"
        "Your task is to check the model response, identify the first incorrect reasoning step, "
        "and generate a correction message to the original model.\n"
        "\n"
        "Rules:\n"
        "1. Use the hidden correct answer only to verify the result internally.\n"
        "2. In your output, you must act as if you independently solved the problem from the question.\n"
        "3. Do not mention, cite, or refer to any external answer source.\n"
        "4. Check the model's answer and each reasoning step carefully.\n"
        "5. Identify the FIRST incorrect step in the reasoning.\n"
        "6. If the reasoning is correct but the final answer is inconsistent with your own calculation from the question, "
        'set "error_step" to "Final answer".\n'
        "7. If the answer and all reasoning steps are consistent with your own calculation from the question, "
        'set "error_step" to "None".\n'
        "8. Do not invent errors that are not supported by the question and the model response.\n"
        "9. The correction must directly explain what is wrong and give the correct fix.\n"
        "10. The correction should be confident and written as feedback to the original model.\n"
        "\n"
        "Forbidden output content:\n"
        '- The correction must NOT contain the phrase "ground truth".\n'
        '- The correction must NOT contain phrases such as "according to the ground truth", '
        '"relative to the ground truth", "based on the ground truth", "in line with the ground truth", '
        '"reference answer", "provided answer", "given answer", "gold answer", '
        '"hidden answer", "correct answer source", or "internal checking".\n'
        "- The correction must NOT say or imply that the answer comes from an external label, dataset, annotation, "
        "answer key, benchmark, or evaluator.\n"
        "- If you need to justify the correction, justify it only by recomputing from the question.\n"
        "\n"
        "Required correction style:\n"
        "- Write the correction as if you independently recalculated the problem.\n"
        '- Use wording such as: "Recomputing the quantities from the question...", '
        '"From the question, ...", "The calculation should be...", or '
        '"Solving the problem directly gives...".\n'
        "- Do not use wording that refers to an external answer source.\n"
        "\n"
        "Input:\n"
        f"Question:\n{question}\n\n"
        f"Model response:\n{llm_response}\n\n"
        f"Hidden answer for validation only:\n{ground_truth}\n\n"
        "Output format:\n"
        "Return exactly one valid JSON object with exactly these two fields:\n"
        '{"error_step": "Step X or Final answer", "correction": "..."}\n'
        "\n"
        "Output rules:\n"
        "- Output only the JSON object.\n"
        "- Do not use markdown.\n"
        "- Do not include any text outside the JSON object.\n"
        "- The JSON must be parseable by json.loads().\n"
        "- The correction field must explain the fix as an independent calculation from the question.\n"
    )
    response = solve_math(JUDGE, contents)
    response = json.loads(response)
    return response


def exp_multi_turn_multi_choice(file, save_dir, model):
    df = pd.read_csv(save_dir / "{0}_multi_turn.csv".format(file))
    root_df = pd.read_parquet(
        "./{0}/train-00000-of-00001.parquet".format(file)
    )  # 多项选择获取choice的文件
    messages = []
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
            "----------------------------Q id: {0}----------------------------".format(
                i
            )
        )
        if df.iloc[i]["question"] != root_df.iloc[i]["question"]:
            print("ERROR question mismatches.")
            return
        mis_answer = df.iloc[i]["answer"]
        question = df.iloc[i]["question"]
        ground_truth = df.iloc[i]["ground truth"]
        choice_string = convertChoices2String(root_df.iloc[i]["choices"])
        if df.iloc[i]["turn count"] == -1 or question in rdf["question"].values:
            continue
        if mis_answer == ground_truth:
            continue
        messages = restore_conversation(df, i)

        # 获取对话历史中llm最后的回答
        conversation_last = df.iloc[i][
            "turn count"
        ]  # llm之前在robustness测试中撑了几轮
        last_llm_message = df.iloc[i]["llm_{0}".format(conversation_last - 1)]
        temp_prompt = generate_correction_message_multi_choice(
            question, choice_string, last_llm_message, ground_truth
        )
        target_step = temp_prompt["error_step"]
        target_correction = temp_prompt["correction"]
        temp_prompt = generate_user_doubt_multiple_choice(
            target_step, target_correction, ground_truth
        )  # 函数逻辑是一样的，所以可以直接用之前的框架
        user_requests = []
        llm_responses = []
        turn_count = -1
        answer = None
        for turn in range(MAX_TURN):
            user_requests.append(temp_prompt)
            temp_prompt = build_user_prompt(temp_prompt)
            messages.append(temp_prompt)
            llm_response = get_multiple_turn_response(model, messages)
            if llm_response is None:
                answer = None
                print("ERROR, None is returned on question {0}".format(i))
                return
            else:
                llm_response = json.loads(llm_response)
                answer = llm_response["answer"]

            llm_response = to_content_string(llm_response)
            messages.append(build_assistant_response(llm_response))
            llm_responses.append(llm_response)

            if answer == ground_truth:
                turn_count = turn + 1
                print(
                    "Mistake has been changed! Turn: {0}, Answer: {1}, Ground truth: {2}\nQuestion:{3}".format(
                        turn_count, answer, ground_truth, question
                    )
                )
                # for message in messages:
                #     print(message)
                for index in range(len(llm_responses)):
                    print(user_requests[index])
                    print(llm_responses[index])
                # print("AAAAAAAAAAAAAAAA")
                break

            correction_prompt = generate_correction_message_multi_choice(
                question, choice_string, llm_response, ground_truth
            )
            target_step = correction_prompt["error_step"]
            target_correction = correction_prompt["correction"]

            temp_prompt = generate_user_doubt_multiple_choice(
                target_step, target_correction, ground_truth
            )
        # 如果对话长度不够，那就补充列表到空
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
            "----------------------------Q id: {0} finished----------------------------".format(
                i
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


def extract_answer(text):
    # 获取math ground truth的数字
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?(?:/\d+)?)", text)
    if match:
        return match.group(1)
    return None


def extract_array_content(text):
    # 获取 plain answer的内容
    match = re.search(r"\[\s*'([^']*)'\s*\]", text)
    if match:
        return match.group(1)
    return None


def exp_multi_turn_rest(file, save_dir, model):
    df = pd.read_csv(save_dir / "{0}_multi_turn.csv".format(file))
    messages = []
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
        messages = restore_conversation(df, i)
        if df.iloc[i]["turn count"] == -1 or question in rdf["question"].values:
            continue

        conversation_last = df.iloc[i][
            "turn count"
        ]  # llm之前在robustness测试中撑了几轮
        last_llm_message = df.iloc[i]["llm_{0}".format(conversation_last - 1)]
        temp_prompt = generate_correction_rest(question, last_llm_message, ground_truth)
        target_step = temp_prompt["error_step"]
        target_correction = temp_prompt["correction"]
        if file == "math_train":
            ground_truth_content = extract_answer(ground_truth)
            temp_prompt = generate_user_doubt_multiple_choice(
                target_step, target_correction, ground_truth_content
            )  # 函数逻辑是一样的，所以可以直接用之前的框架
        elif file == "plain_text":
            ground_truth_content = extract_array_content(ground_truth)
            temp_prompt = generate_user_doubt_multiple_choice(
                target_step, target_correction, ground_truth_content
            )  # 函数逻辑是一样的，所以可以直接用之前的框架
        else:
            temp_prompt = generate_user_doubt_multiple_choice(
                target_step, target_correction, ground_truth
            )  # 函数逻辑是一样的，所以可以直接用之前的框架

        user_requests = []
        llm_responses = []
        turn_count = -1
        answer = None
        for turn in range(MAX_TURN):
            user_requests.append(temp_prompt)
            temp_prompt = build_user_prompt(temp_prompt)
            messages.append(temp_prompt)
            llm_response = get_multiple_turn_response(model, messages)
            while llm_response is None:
                llm_response = get_multiple_turn_response(model, messages)
                print(llm_response, "--------------------------")
            if llm_response is None:
                answer = None
                print("ERROR, None is returned on question {0}".format(i))
                return
            else:
                llm_response = json.loads(llm_response)
                answer = llm_response["answer"]

            llm_response = to_content_string(llm_response)
            messages.append(build_assistant_response(llm_response))
            llm_responses.append(llm_response)

            t_judge = rest_judge(file, question, answer, ground_truth)
            if t_judge:
                judge = True
            else:
                judge = False

            if t_judge:
                turn_count = turn + 1
                print(
                    "Mistake has been changed! Turn: {0}, Answer: {1}, Ground truth: {2}\nQuestion:{3}".format(
                        turn_count, answer, ground_truth, question
                    )
                )
                # for message in messages:
                #     print(message)
                for index in range(len(llm_responses)):
                    print(user_requests[index])
                    print(llm_responses[index])
                # print("AAAAAAAAAAAAAAAA")
                break

            correction_prompt = generate_correction_rest(
                question, llm_response, ground_truth
            )
            target_step = correction_prompt["error_step"]
            target_correction = correction_prompt["correction"]

            temp_prompt = generate_user_doubt_multiple_choice(
                target_step, target_correction, ground_truth
            )
        # 如果对话长度不够，那就补充列表到空
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


def remove_duplicate_questions(csv_path, model, file):
    """
    去除 CSV 中 question 列重复的行。
    对于相同 question，只保留第一次出现的行。
    结果会直接覆盖保存到原 CSV 文件。
    """
    df = pd.read_csv(csv_path)

    before_count = len(df)

    df = df.drop_duplicates(subset=["question"], keep="first")

    after_count = len(df)

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("*" * 50, "Model: {0} File: {1}".format(model, file), "*" * 50)
    print(f"去重前行数: {before_count}")
    print(f"去重后行数: {after_count}")
    print(f"删除重复行数: {before_count - after_count}")
    print("*" * 50, "Model: {0} File: {1} finished".format(model, file), "*" * 50)


f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]
if __name__ == "__main__":
    for llm_id in range(len(LLMS)):
        save_dir = Path("{0}".format(LLMS[llm_id].replace(":", "_")))
        save_dir.mkdir(parents=True, exist_ok=True)
        # exp_multiple_choice_easy("ARC-challenge", save_dir, LLMS[llm_id])
        # exp_rest_easy("math_train", save_dir, LLMS[llm_id])
        # exp_rest_easy("plain_text", save_dir, LLMS[llm_id])
        # exp_rest_easy("NQ-open", save_dir, LLMS[llm_id])
        exp_multi_turn_multi_choice("ARC-challenge", save_dir, LLMS[llm_id])
        exp_multi_turn_rest("NQ-open", save_dir, LLMS[llm_id])
        exp_multi_turn_rest("math_train", save_dir, LLMS[llm_id])
        exp_multi_turn_rest("plain_text", save_dir, LLMS[llm_id])
        # for file in f_list:
        #     remove_duplicate_questions(
        #         save_dir / "{0}_correction_multi_turn.csv".format(file),
        #         LLMS[llm_id],
        #         file,
        #     )
    print("Done at {0}".format(datetime.now()))
