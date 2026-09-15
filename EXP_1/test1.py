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

from exp_2 import JUDGE, get_llm_response, URL

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("未找到环境变量 OPENROUTER_API_KEY")


def NQ_answer_judge_explaination(answer, ground_truth, question):
    prompt = (
        "You are a strict but fair open-ended answer judge.\n"
        "\n"
        "Your task is to determine whether the Model Answer correctly answers the Question according to the Ground Truth.\n"
        "\n"
        "Input:\n"
        "Question:\n"
        f"{question}\n"
        "\n"
        "Ground Truth:\n"
        f"{ground_truth}\n"
        "\n"
        "Model Answer:\n"
        f"{answer}\n"
        "\n"
        "Ground Truth format:\n"
        "- The Ground Truth may be a list of acceptable answers, a JSON-like object, or plain text.\n"
        "- If the Ground Truth is a list, any item in the list can be treated as an acceptable correct answer.\n"
        '- If the Ground Truth is a JSON-like object and contains a field such as "answer", "answers", "text", or "label", use the value of that field as the acceptable answer.\n'
        "- The Ground Truth may contain different valid forms of the same answer, including dates, names, places, numbers, short phrases, or aliases.\n"
        "\n"
        "Judging rules:\n"
        "1. Use the Question to understand what kind of answer is required.\n"
        "2. Return True if the Model Answer correctly expresses the same meaning as any acceptable Ground Truth answer.\n"
        "3. Accept equivalent wording, capitalization differences, punctuation differences, abbreviations, aliases, and common date or number formats.\n"
        "4. Accept a more specific answer if it is consistent with the Ground Truth and does not introduce any contradiction.\n"
        "5. Accept a less specific answer only when it is still sufficient to answer the Question.\n"
        "6. Ignore extra explanation if the answer is correct and the extra content does not contradict the Ground Truth.\n"
        "7. Return False if the Model Answer gives the wrong entity, time, number, place, event, relation, or meaning.\n"
        "8. Return False if the Model Answer is too vague, incomplete, evasive, or only provides related background without answering the Question.\n"
        "9. Return False if the Model Answer contains a contradiction, even if it also contains the correct answer.\n"
        "10. Return False if the Model Answer lists multiple possible answers and at least one of them conflicts with the Ground Truth.\n"
        "11. Do not require exact string matching. Judge semantic correctness in the context of the Question.\n"
        "\n"
        "Output requirement:\n"
        "- Output exactly one valid JSON object.\n"
        "- The JSON object must contain exactly two fields:\n"
        '  - "judge": true or false\n'
        '  - "explaination": a brief explanation of why the answer is correct or incorrect\n'
        "- Do not output anything outside the JSON object.\n"
    )
    response_initial = get_llm_response(JUDGE, prompt, timeout=(10, 120))
    result = json.loads(response_initial)
    judge = result["judge"]
    explaination = result["explaination"]
    return judge, explaination


def plain_text_answer_judge_explaination(
    question, background, situation, answer, ground_truth
):
    prompt = (
        "You are a strict but fair judge for Reasoning Over Paragraph Effects in Situations.\n"
        "\n"
        "Your task is to determine whether the Model Answer correctly answers the Question according to the Ground Truth, "
        "by reasoning over the Background and applying it to the Situation.\n"
        "\n"
        "Input:\n"
        "Background:\n"
        f"{background}\n"
        "\n"
        "Situation:\n"
        f"{situation}\n"
        "\n"
        "Question:\n"
        f"{question}\n"
        "\n"
        "Ground Truth:\n"
        f"{ground_truth}\n"
        "\n"
        "Model Answer:\n"
        f"{answer}\n"
        "\n"
        "Task meaning:\n"
        "- The Background describes a general cause-effect relationship, comparison, rule, or scientific principle.\n"
        "- The Situation describes a specific scenario where that relationship must be applied.\n"
        "- The Question asks about the effect, outcome, comparison, cause, or entity in the Situation.\n"
        "- The Ground Truth contains the acceptable correct answer or answers.\n"
        "- The Model Answer is the answer that needs to be judged.\n"
        "\n"
        "Ground Truth format:\n"
        "- The Ground Truth may be a JSON-like object, a list, or plain text.\n"
        '- If the Ground Truth is a JSON-like object and contains a field named "text", use the values in Ground Truth["text"] as the acceptable correct answers.\n'
        '- If the Ground Truth contains fields such as "answer", "answers", or "label", use those values as acceptable correct answers.\n'
        "- If the Ground Truth is a list, any item in the list can be treated as an acceptable correct answer.\n"
        "- If the Ground Truth is plain text, use it directly as the correct answer.\n"
        "\n"
        "Judging rules:\n"
        "1. Use the Background to identify the relevant cause-effect relationship, rule, or comparison.\n"
        "2. Apply that relationship to the specific entities, events, or conditions described in the Situation.\n"
        "3. Use the Question to determine what type of answer is required.\n"
        "4. Return true if the Model Answer gives the same correct answer as the Ground Truth in this context.\n"
        "5. The Model Answer does not need to exactly match the Ground Truth wording.\n"
        "6. Accept equivalent names, references, descriptions, or complete sentences that clearly identify the correct answer.\n"
        "7. Accept an answer that includes correct reasoning, as long as it clearly gives the correct final answer.\n"
        "8. Ignore extra explanation if it does not contradict the Background, Situation, Question, or Ground Truth.\n"
        "9. Return false if the Model Answer applies the Background relationship incorrectly.\n"
        "10. Return false if the Model Answer identifies the wrong person, object, option, effect, cause, comparison, or conclusion.\n"
        "11. Return false if the Model Answer is too vague, incomplete, evasive, or only repeats the Background without answering the Question.\n"
        "12. Return false if the Model Answer contradicts the Background, Situation, Question, or Ground Truth, even if it also mentions the correct answer.\n"
        "13. Return false if the Model Answer gives multiple possible answers and at least one conflicts with the Ground Truth.\n"
        "14. Do not rely only on string matching. Judge whether the Model Answer is correct under the Background-Situation reasoning chain.\n"
        "\n"
        "Output requirement:\n"
        "- Output exactly one valid JSON object.\n"
        "- The JSON object must contain exactly two fields:\n"
        '  - "judge": true or false\n'
        '  - "explaination": a brief explanation of why the answer is correct or incorrect\n'
        '- The value of "judge" must be a JSON boolean, not a string.\n'
        "- Do not output anything outside the JSON object.\n"
    )
    response_initial = get_llm_response(JUDGE, prompt, timeout=(10, 120))
    result = json.loads(response_initial)
    judge = result["judge"]
    explaination = result["explaination"]
    return judge, explaination


def math_answer_judge_explaination(question, answer, ground_truth):
    prompt = (
        "You are a strict but fair math answer judge.\n"
        "\n"
        "Your task is to determine whether the Model Answer gives the same final answer as the Ground Truth for the given Question.\n"
        "\n"
        "Input:\n"
        "Question:\n"
        f"{question}\n"
        "\n"
        "Ground Truth:\n"
        f"{ground_truth}\n"
        "\n"
        "Model Answer:\n"
        f"{answer}\n"
        "\n"
        "Ground Truth format:\n"
        "- The Ground Truth may contain intermediate reasoning steps, calculations, explanations, and units.\n"
        "- The final answer in the Ground Truth is indicated by the marker ####.\n"
        "- Use only the value after #### as the correct final answer.\n"
        "- Infer the expected unit, currency, percentage, or measurement type from the Question and the full Ground Truth when needed.\n"
        "\n"
        "Judging rules:\n"
        "1. Focus on the final answer, not on whether the Model Answer uses the same reasoning steps as the Ground Truth.\n"
        "2. Return true if the Model Answer gives a final answer that is mathematically equivalent to the Ground Truth final answer.\n"
        "3. Accept equivalent numeric formats, including integers, decimals, fractions, percentages, currency symbols, commas, and written forms of numbers.\n"
        "4. Accept equivalent units or currencies if they clearly represent the same value required by the Question.\n"
        "5. If the Model Answer does not include a unit, focus on whether the numerical value is correct, unless the Question specifically requires a unit-sensitive answer.\n"
        "6. If the Model Answer contains multiple numbers, identify the number intended as the final answer.\n"
        "7. Return false if the Model Answer's final answer has the wrong numerical value.\n"
        "8. Return false if the Model Answer gives only intermediate calculations and does not provide a final answer.\n"
        "9. Return false if the Model Answer is ambiguous and it is unclear which value is the final answer.\n"
        "10. Return false if the Model Answer contradicts itself or gives multiple possible final answers that do not all match the Ground Truth.\n"
        "11. Do not require exact string matching. Judge mathematical equivalence of the final answer.\n"
        "\n"
        "Output requirement:\n"
        "- Output exactly one valid JSON object.\n"
        "- The JSON object must contain exactly two fields:\n"
        '  - "judge": true or false\n'
        '  - "explaination": a brief explanation of why the final answer is correct or incorrect\n'
        '- The value of "judge" must be a JSON boolean, not a string.\n'
        "- Do not output anything outside the JSON object.\n"
    )
    response_initial = get_llm_response(JUDGE, prompt, timeout=(10, 120))
    result = json.loads(response_initial)
    judge = result["judge"]
    explaination = result["explaination"]
    return judge, explaination


def rest_judge_explaination(
    file, answer, ground_truth, question=None, background=None, situation=None
):
    # if file == "math_train":
    #     t_judge = math_answer_judge(answer, ground_truth)
    # elif file == "plain_text":
    #     t_judge = plain_text_answer_judge(question, answer, ground_truth)
    # else:
    if file == "NQ-open":
        t_judge, explaination = NQ_answer_judge_explaination(
            answer, ground_truth, question
        )
    elif file == "plain_text":
        t_judge, explaination = plain_text_answer_judge_explaination(
            question, background, situation, answer, ground_truth
        )
    elif file == "math_train":
        t_judge, explaination = math_answer_judge_explaination(
            question, answer, ground_truth
        )
    # if t_judge is None:
    #     print("ERROR in answer judge.")
    return t_judge, explaination


def judge_reasoning_acc_explanation(file, save_dir, model):
    if (save_dir / "{0}_mis_reasoning_modify.csv".format(file)).exists() == False:
        print(
            "Model: {0}\t{1}_mis_reasoning_modify.csv does not exist.".format(
                model, file
            )
        )
        return
    if file == "plain_text":
        original_df = pd.read_parquet("./{0}_train-00000-of-00001.parquet".format(file))
    df = pd.read_csv(save_dir / "{0}_mis_reasoning_modify.csv".format(file))
    output_csv = save_dir / "{0}_mis_reasoning_modify_judge_explaination.csv".format(
        file
    )
    mislead_df = pd.read_csv(save_dir / "{0}_misleading.csv".format(file))
    output_csv_missleading = save_dir / "{0}_misleading_judge_explaination.csv".format(
        file
    )
    output_path = Path(output_csv)
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    print(
        "----------------------------Judge answer {1} on {0}----------------------------".format(
            model, file
        )
    )
    for i in range(start, len(df)):
        initial_answer = df.iloc[i]["initial_answer"]
        mis_reasoning_answer = df.iloc[i]["answer_under_mis_reasoning"]
        mislead_answer = mislead_df.iloc[i]["answer"]
        question = df.iloc[i]["question"]
        ground_truth = df.iloc[i]["ground truth"]
        if file == "plain_text":
            situation = original_df.iloc[i]["situation"]
            background = original_df.iloc[i]["background"]
        print(
            "----------------------------Q id: {0}\tModel: {1}----------------------------".format(
                i, model
            )
        )
        if file != "plain_text":
            initial_judge, initial_explaination = rest_judge_explaination(
                file, initial_answer, ground_truth, question
            )
            mis_reasoning_judge, mis_reasoning_explaination = rest_judge_explaination(
                file, mis_reasoning_answer, ground_truth, question
            )
            misleading_judge, misleading_explaination = rest_judge_explaination(
                file, mislead_answer, ground_truth=ground_truth, question=question
            )
        else:
            initial_judge, initial_explaination = rest_judge_explaination(
                file,
                initial_answer,
                ground_truth,
                question=question,
                background=background,
                situation=situation,
            )
            mis_reasoning_judge, mis_reasoning_explaination = rest_judge_explaination(
                file,
                mis_reasoning_answer,
                ground_truth,
                question=question,
                background=background,
                situation=situation,
            )
            misleading_judge, misleading_explaination = rest_judge_explaination(
                file,
                mislead_answer,
                ground_truth=ground_truth,
                question=question,
                background=background,
                situation=situation,
            )
        row_df = pd.DataFrame(
            [
                [
                    df.iloc[i]["question"],
                    initial_judge,
                    initial_explaination,
                    mis_reasoning_judge,
                    mis_reasoning_explaination,
                ]
            ],
            columns=[
                "question",
                "initial_judge",
                "initial_explaination",
                "mis_reasoning_judge",
                "mis_reasoning_explaination",
            ],
        )
        row_misleading_df = pd.DataFrame(
            [
                [
                    df.iloc[i]["question"],
                    initial_judge,
                    initial_explaination,
                    misleading_judge,
                    misleading_explaination,
                ]
            ],
            columns=[
                "question",
                "initial_judge",
                "initial_explaination",
                "misleading_judge",
                "misleading_explaination",
            ],
        )
        print(initial_answer, mis_reasoning_answer, mislead_answer)
        print(ground_truth)
        print(
            "Initial judge: {0}\t Mis_reasoning judge: {1}\t Misleading judge: {2}".format(
                initial_judge, mis_reasoning_judge, misleading_judge
            )
        )
        print(
            "Initial explaination: {0}\nMis_reasoning explaination: {1}\nMisleading explaination: {2}".format(
                initial_explaination,
                mis_reasoning_explaination,
                misleading_explaination,
            )
        )
        row_df.to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
            encoding="utf-8-sig",
        )
        row_misleading_df.to_csv(
            output_csv_missleading,
            mode="a",
            header=not os.path.exists(output_csv_missleading),
            index=False,
            encoding="utf-8-sig",
        )
        print(
            "----------------------------Q id: {0}\tModel: {1} finished----------------------------".format(
                i, model
            )
        )
    print(
        "----------------------------Judge answer {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


def build_judge_prompt(
    question,
    answer,
    ground_truth,
    file=None,
    background=None,
    situation=None,
):
    prompt = "You are a strict answer correctness judge.\n" "\n"

    if file == "plain_text":
        prompt += f"Background:\n{background}\n" "\n" f"Situation:\n{situation}\n" "\n"

    prompt += (
        f"Question:\n{question}\n"
        "\n"
        f"Answer:\n{answer}\n"
        "\n"
        f"Ground Truth:\n{ground_truth}\n"
        "\n"
        "Determine whether the Answer correctly answers the Question and is "
        "semantically consistent with the Ground Truth.\n"
        "\n"
        "Different wording is allowed as long as the meaning is equivalent. "
        "Additional information is allowed if it does not contradict the Ground Truth.\n"
        "\n"
        "If the Answer does not answer the Question, contradicts the Ground Truth, "
        "or gives a materially different answer, judge it as incorrect.\n"
        "\n"
        "Output requirements:\n"
        "- Return ONLY a valid JSON object.\n"
        "- The JSON object must contain exactly two fields:\n"
        '  - "judge": true or false\n'
        '  - "explaination": a brief explanation of why the answer is correct or incorrect\n'
        "- Do not output any other text."
    )

    return prompt


def test_answer_judge_explaination(
    question, answer, ground_truth, file=None, background=None, situation=None
):
    prompt = build_judge_prompt(question, answer, ground_truth)
    response_initial = get_llm_response(JUDGE, prompt, timeout=(10, 120))
    result = json.loads(response_initial)
    judge = result["judge"]
    explaination = result["explaination"]
    return judge, explaination


def text_judge(file, save_dir, model):
    if (save_dir / "{0}_mis_reasoning_modify.csv".format(file)).exists() == False:
        print(
            "Model: {0}\t{1}_mis_reasoning_modify.csv does not exist.".format(
                model, file
            )
        )
        return
    if file == "plain_text":
        original_df = pd.read_parquet("./{0}_train-00000-of-00001.parquet".format(file))
    df = pd.read_csv(save_dir / "{0}_mis_reasoning_modify.csv".format(file))
    output_csv = (
        save_dir / "{0}_test_mis_reasoning_modify_judge_explaination.csv".format(file)
    )
    mislead_df = pd.read_csv(save_dir / "{0}_misleading.csv".format(file))
    output_csv_missleading = (
        save_dir / "{0}_test_misleading_judge_explaination.csv".format(file)
    )
    output_path = Path(output_csv)
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    print(
        "----------------------------Judge answer {1} on {0}----------------------------".format(
            model, file
        )
    )
    for i in range(start, len(df)):
        initial_answer = df.iloc[i]["initial_answer"]
        mis_reasoning_answer = df.iloc[i]["answer_under_mis_reasoning"]
        mislead_answer = mislead_df.iloc[i]["answer"]
        question = df.iloc[i]["question"]
        ground_truth = df.iloc[i]["ground truth"]
        if file == "plain_text":
            situation = original_df.iloc[i]["situation"]
            background = original_df.iloc[i]["background"]
        print(
            "----------------------------Q id: {0}\tModel: {1}----------------------------".format(
                i, model
            )
        )
        if file != "plain_text":
            initial_judge, initial_explaination = test_answer_judge_explaination(
                question, initial_answer, ground_truth
            )
            mis_reasoning_judge, mis_reasoning_explaination = (
                test_answer_judge_explaination(
                    question, mis_reasoning_answer, ground_truth
                )
            )
            misleading_judge, misleading_explaination = test_answer_judge_explaination(
                question, mislead_answer, ground_truth
            )
        else:
            initial_judge, initial_explaination = test_answer_judge_explaination(
                file=file,
                answer=initial_answer,
                ground_truth=ground_truth,
                question=question,
                background=background,
                situation=situation,
            )
            mis_reasoning_judge, mis_reasoning_explaination = (
                test_answer_judge_explaination(
                    file=file,
                    answer=mis_reasoning_answer,
                    ground_truth=ground_truth,
                    question=question,
                    background=background,
                    situation=situation,
                )
            )
            misleading_judge, misleading_explaination = test_answer_judge_explaination(
                file=file,
                answer=mislead_answer,
                ground_truth=ground_truth,
                question=question,
                background=background,
                situation=situation,
            )
        row_df = pd.DataFrame(
            [
                [
                    df.iloc[i]["question"],
                    initial_judge,
                    initial_explaination,
                    mis_reasoning_judge,
                    mis_reasoning_explaination,
                ]
            ],
            columns=[
                "question",
                "initial_judge",
                "initial_explaination",
                "mis_reasoning_judge",
                "mis_reasoning_explaination",
            ],
        )
        row_misleading_df = pd.DataFrame(
            [
                [
                    df.iloc[i]["question"],
                    initial_judge,
                    initial_explaination,
                    misleading_judge,
                    misleading_explaination,
                ]
            ],
            columns=[
                "question",
                "initial_judge",
                "initial_explaination",
                "misleading_judge",
                "misleading_explaination",
            ],
        )
        print(initial_answer, mis_reasoning_answer, mislead_answer)
        print(ground_truth)
        print(
            "Initial judge: {0}\t Mis_reasoning judge: {1}\t Misleading judge: {2}".format(
                initial_judge, mis_reasoning_judge, misleading_judge
            )
        )
        print(
            "Initial explaination: {0}\nMis_reasoning explaination: {1}\nMisleading explaination: {2}".format(
                initial_explaination,
                mis_reasoning_explaination,
                misleading_explaination,
            )
        )
        row_df.to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
            encoding="utf-8-sig",
        )
        row_misleading_df.to_csv(
            output_csv_missleading,
            mode="a",
            header=not os.path.exists(output_csv_missleading),
            index=False,
            encoding="utf-8-sig",
        )
        print(
            "----------------------------Q id: {0}\tModel: {1} finished----------------------------".format(
                i, model
            )
        )
        if i >= 100:
            break
    print(
        "----------------------------Judge answer {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


def merge_and_compare_judge(
    csv_a,
    csv_b,
    csv_answer,
    output_file="common_questions.csv",
):
    # 读取三个CSV
    df_a = pd.read_csv(csv_a, dtype=str)
    df_b = pd.read_csv(csv_b, dtype=str)
    df_answer = pd.read_csv(csv_answer, dtype=str)

    # 清理question
    for df in [df_a, df_b, df_answer]:
        df["question"] = df["question"].fillna("").str.strip()

    # 清理initial_judge
    df_a["initial_judge"] = df_a["initial_judge"].fillna("").str.strip()

    df_b["initial_judge"] = df_b["initial_judge"].fillna("").str.strip()

    # 清理answer和ground_truth
    df_answer["answer"] = df_answer["answer"].fillna("").str.strip()

    df_answer["ground_truth"] = df_answer["ground truth"].fillna("").str.strip()

    # 如果question应该唯一，则去重
    df_a = df_a.drop_duplicates(subset=["question"])
    df_b = df_b.drop_duplicates(subset=["question"])
    df_answer = df_answer.drop_duplicates(subset=["question"])

    # =========================================================
    # 合并CSV A和CSV B，只保留共同question
    # =========================================================
    merged_df = pd.merge(
        df_a,
        df_b,
        on="question",
        how="inner",
        suffixes=("_a", "_b"),
    )

    # 判断initial_judge是否相同
    merged_df["judge_same"] = (
        merged_df["initial_judge_a"] == merged_df["initial_judge_b"]
    )

    # =========================================================
    # 根据question加入answer和ground_truth
    # =========================================================
    merged_df = pd.merge(
        merged_df,
        df_answer[
            [
                "question",
                "answer",
                "ground_truth",
            ]
        ],
        on="question",
        how="left",
    )

    # =========================================================
    # 调整主要列顺序
    # =========================================================
    first_columns = [
        "question",
        "answer",
        "ground_truth",
        "initial_judge_a",
        "initial_judge_b",
        "judge_same",
    ]

    other_columns = [col for col in merged_df.columns if col not in first_columns]

    merged_df = merged_df[first_columns + other_columns]

    # =========================================================
    # 统计
    # =========================================================
    total_common = len(merged_df)
    same_count = merged_df["judge_same"].sum()
    different_count = total_common - same_count

    same_ratio = same_count / total_common if total_common > 0 else 0

    missing_answer_count = (
        merged_df["answer"].isna() | merged_df["answer"].eq("")
    ).sum()

    missing_ground_truth_count = (
        merged_df["ground_truth"].isna() | merged_df["ground_truth"].eq("")
    ).sum()

    print(f"CSV A 行数: {len(df_a)}")
    print(f"CSV B 行数: {len(df_b)}")
    print(f"Answer CSV 行数: {len(df_answer)}")

    print(f"\n共同 question 数量: {total_common}")

    print(f"initial_judge 相同数量: {same_count}")
    print(f"initial_judge 不同数量: {different_count}")
    print(f"initial_judge 相同比例: {same_ratio:.2%}")

    print(f"\n未找到 answer 的数量: {missing_answer_count}")
    print(f"未找到 ground_truth 的数量: {missing_ground_truth_count}")

    # 保存
    merged_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n结果已保存到: {output_file}")

    return merged_df


LLMS = [
    # "gemini-2.5-flash",
    # "gemini-3-flash-preview",
    "qwen/qwen3.6-flash",
    # "deepseek/deepseek-v3.2",
    # "openai/gpt-5",
    # "meta-llama/llama-3.1-8b-instruct",
    # "x-ai/grok-4.20",
]

f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]
if __name__ == "__main__":
    four_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]
    for llm_id in range(len(LLMS)):
        save_dir = Path("{0}".format(LLMS[llm_id].replace(":", "_")))
        save_dir.mkdir(parents=True, exist_ok=True)
        # merge_and_compare_judge(
        #     save_dir / "NQ-open_test_mis_reasoning_modify_judge_explaination.csv",
        #     save_dir / "NQ-open_test_mis_reasoning_modify_judge_explaination_1.csv",
        #     save_dir / "NQ-open_initial_answer.csv",
        #     output_file="Initial_common.csv",
        # )
        text_judge("NQ-open", save_dir, LLMS[llm_id])
        # judge_reasoning_acc_explanation("math_train", save_dir, LLMS[llm_id])
        # judge_reasoning_acc_explanation("plain_text", save_dir, LLMS[llm_id])
    print("Done at {0}".format(datetime.now()))
