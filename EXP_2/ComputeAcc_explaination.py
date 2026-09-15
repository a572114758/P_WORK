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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image

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
    if (
        save_dir / "{0}_mis_reasoning_modify_explaination.csv".format(file)
    ).exists() == False:
        print(
            "Model: {0}\t{1}_mis_reasoning_modify_explaination.csv does not exist.".format(
                model, file
            )
        )
        return
    if file == "plain_text":
        original_df = pd.read_parquet("./{0}_train-00000-of-00001.parquet".format(file))
    df = pd.read_csv(
        save_dir / "{0}_mis_reasoning_modify_explaination.csv".format(file)
    )
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
        print(
            "Initial: {0} || Mis reasoning: {1} || Misleading: {2}".format(
                initial_answer, mis_reasoning_answer, mislead_answer
            )
        )
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


def count_arc_results_misleading(
    save_dir,
    file,
    export_grouped_rows=False,
):
    """
    统计多项选择题的六种回答变化情况。

    参数：
        save_dir:
            三个CSV所在目录。

        file:
            数据集名称，例如"ARC-challenge"。

        export_grouped_rows:
            是否分别导出count_1至count_6对应的数据。

    返回：
        counts:
            六类数据的数量。

        grouped_rows:
            六类数据对应的DataFrame。

        merged_df:
            合并后的完整DataFrame。
    """
    data_dir = Path(save_dir)

    initial_name = f"{file}_initial_answer_explaination.csv"
    misreasoning_name = f"{file}_misleading.csv"
    reasoning_name = f"{file}_background.csv"

    output_name = f"{file}_misleading_semantic_judgment_results.csv"

    initial_df = pd.read_csv(
        data_dir / initial_name,
        dtype=str,
    )

    misreasoning_df = pd.read_csv(
        data_dir / misreasoning_name,
        dtype=str,
    )

    # 修正：这里也需要添加data_dir
    reasoning_df = pd.read_csv(
        reasoning_name,
        dtype=str,
    )

    # 选择需要的列并重命名
    initial_df = initial_df[["question", "answer", "ground truth"]].rename(
        columns={
            "answer": "initial_answer",
            "ground truth": "ground_truth",
        }
    )

    misreasoning_df = misreasoning_df[
        [
            "question",
            "answer",
        ]
    ].rename(columns={"answer": "answer_under_misleading"})

    reasoning_df = reasoning_df[["question", "answer"]].rename(
        columns={
            "answer": "misleading_mis_answer",
        }
    )

    # 清理question，保证能够正确匹配
    for df in (
        initial_df,
        misreasoning_df,
        reasoning_df,
    ):
        df["question"] = df["question"].fillna("").str.strip()

    # 检查是否存在重复question
    dataframes = {
        initial_name: initial_df,
        misreasoning_name: misreasoning_df,
        reasoning_name: reasoning_df,
    }

    for filename, df in dataframes.items():
        duplicate_count = df["question"].duplicated().sum()

        if duplicate_count > 0:
            raise ValueError(f"{filename}中存在" f"{duplicate_count}个重复的question")

    # 根据question合并三个CSV
    merged_df = initial_df.merge(
        misreasoning_df,
        on="question",
        how="inner",
        validate="one_to_one",
    ).merge(
        reasoning_df,
        on="question",
        how="inner",
        validate="one_to_one",
    )

    # 统一答案格式
    answer_columns = [
        "initial_answer",
        "ground_truth",
        "answer_under_misleading",
        "misleading_mis_answer",
    ]

    for column in answer_columns:
        merged_df[column] = merged_df[column].fillna("").str.strip().str.upper()

    # 基础判断条件
    merged_df["initial_correct"] = (
        merged_df["initial_answer"] == merged_df["ground_truth"]
    )

    merged_df["misleading_correct"] = (
        merged_df["answer_under_misleading"] == merged_df["ground_truth"]
    )

    merged_df["misleading_equal_misleading_answer"] = (
        merged_df["answer_under_misleading"] == merged_df["misleading_mis_answer"]
    )

    merged_df["initial_equal_misleading_answer"] = (
        merged_df["initial_answer"] == merged_df["answer_under_misleading"]
    )

    initial_correct = merged_df["initial_correct"]

    misreasoning_correct = merged_df["misleading_correct"]

    misreasoning_equal_reasoning_answer = merged_df[
        "misleading_equal_misleading_answer"
    ]

    initial_equal_misreasoning_answer = merged_df["initial_equal_misleading_answer"]

    # 六种情况
    conditions = {
        # 1. initial == ground truth
        #    answer_under_mis_reasoning == ground truth
        "count_1": (initial_correct & misreasoning_correct),
        # 2. initial == ground truth
        #    answer_under_mis_reasoning != ground truth
        #    answer_under_mis_reasoning == reasoning_mis_answer
        "count_2": (
            initial_correct
            & ~misreasoning_correct
            & misreasoning_equal_reasoning_answer
        ),
        # 3. initial == ground truth
        #    answer_under_mis_reasoning != ground truth
        #    answer_under_mis_reasoning != reasoning_mis_answer
        "count_3": (
            initial_correct
            & ~misreasoning_correct
            & ~misreasoning_equal_reasoning_answer
        ),
        # 4. initial != ground truth
        #    answer_under_mis_reasoning == ground truth
        "count_4": (~initial_correct & misreasoning_correct),
        # 5. initial != ground truth
        #    answer_under_mis_reasoning != ground truth
        #    initial == answer_under_mis_reasoning
        "count_5": (
            ~initial_correct & ~misreasoning_correct & initial_equal_misreasoning_answer
        ),
        # 6. initial != ground truth
        #    answer_under_mis_reasoning != ground truth
        #    initial != answer_under_mis_reasoning
        "count_6": (
            ~initial_correct
            & ~misreasoning_correct
            & ~initial_equal_misreasoning_answer
        ),
    }
    # 统计每种情况的数量
    counts = {name: int(condition.sum()) for name, condition in conditions.items()}

    # 给每行添加所属类别
    merged_df["category"] = ""

    for name, condition in conditions.items():
        merged_df.loc[
            condition,
            "category",
        ] = name

    # 保存每种情况对应的具体行
    grouped_rows = {
        name: merged_df.loc[condition].copy() for name, condition in conditions.items()
    }

    # 保存全部分类结果
    output_path = data_dir / output_name

    merged_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 可选：分别保存六类数据
    if export_grouped_rows:
        for name, group_df in grouped_rows.items():
            group_output_path = data_dir / f"{file}_{name}.csv"

            group_df.to_csv(
                group_output_path,
                index=False,
                encoding="utf-8-sig",
            )

    print("*" * 80)
    print(f"File: {file}")
    print(f"initial文件问题数：{len(initial_df)}")
    print(f"misreasoning文件问题数：" f"{len(misreasoning_df)}")
    print(f"reasoning文件问题数：" f"{len(reasoning_df)}")
    print(f"三个CSV合并后的问题数：" f"{len(merged_df)}")

    print("1. initial正确，mis reasoning后正确：" f"{counts['count_1']}")

    print(
        "2. initial正确，mis reasoning后错误，"
        "且两个mis答案相同："
        f"{counts['count_2']}"
    )

    print(
        "3. initial正确，mis reasoning后错误，"
        "且两个mis答案不同："
        f"{counts['count_3']}"
    )

    print("4. initial错误，mis reasoning后正确：" f"{counts['count_4']}")

    print(
        "5. initial错误，mis reasoning后错误，"
        "且两次回答相同："
        f"{counts['count_5']}"
    )

    print(
        "6. initial错误，mis reasoning后错误，"
        "且两次回答不同："
        f"{counts['count_6']}"
    )

    print(f"六类数量之和：" f"{sum(counts.values())}")

    # 检查六类是否覆盖所有问题
    if sum(counts.values()) == len(merged_df):
        print("检查通过：所有问题均被归入且仅归入一个类别。")
    else:
        print("警告：六类数量之和与合并后的问题数不一致。")

    print(f"完整分类结果已保存至：{output_path}")
    print("*" * 80)

    return counts, grouped_rows, merged_df


def count_arc_results(
    save_dir,
    file,
    export_grouped_rows=False,
):
    """
    统计多项选择题的六种回答变化情况。

    参数：
        save_dir:
            三个CSV所在目录。

        file:
            数据集名称，例如"ARC-challenge"。

        export_grouped_rows:
            是否分别导出count_1至count_6对应的数据。

    返回：
        counts:
            六类数据的数量。

        grouped_rows:
            六类数据对应的DataFrame。

        merged_df:
            合并后的完整DataFrame。
    """
    data_dir = Path(save_dir)

    initial_name = f"{file}_initial_answer_explaination.csv"
    misreasoning_name = f"{file}_mis_reasoning_modify_explaination.csv"
    reasoning_name = f"{file}_reasoning.csv"

    output_name = f"{file}_semantic_judgment_results.csv"

    # 以字符串形式读取，避免答案1被读取为1.0
    initial_df = pd.read_csv(
        data_dir / initial_name,
        dtype=str,
    )

    misreasoning_df = pd.read_csv(
        data_dir / misreasoning_name,
        dtype=str,
    )

    # 修正：这里也需要添加data_dir
    reasoning_df = pd.read_csv(
        reasoning_name,
        dtype=str,
    )

    # 选择需要的列并重命名
    initial_df = initial_df[["question", "answer", "ground truth"]].rename(
        columns={
            "answer": "initial_answer",
            "ground truth": "ground_truth",
        }
    )

    misreasoning_df = misreasoning_df[
        [
            "question",
            "answer_under_mis_reasoning",
        ]
    ]

    reasoning_df = reasoning_df[["question", "answer"]].rename(
        columns={
            "answer": "reasoning_mis_answer",
        }
    )

    # 清理question，保证能够正确匹配
    for df in (
        initial_df,
        misreasoning_df,
        reasoning_df,
    ):
        df["question"] = df["question"].fillna("").str.strip()

    # 检查是否存在重复question
    dataframes = {
        initial_name: initial_df,
        misreasoning_name: misreasoning_df,
        reasoning_name: reasoning_df,
    }

    for filename, df in dataframes.items():
        duplicate_count = df["question"].duplicated().sum()

        if duplicate_count > 0:
            raise ValueError(f"{filename}中存在" f"{duplicate_count}个重复的question")

    # 根据question合并三个CSV
    merged_df = initial_df.merge(
        misreasoning_df,
        on="question",
        how="inner",
        validate="one_to_one",
    ).merge(
        reasoning_df,
        on="question",
        how="inner",
        validate="one_to_one",
    )

    # 统一答案格式
    answer_columns = [
        "initial_answer",
        "ground_truth",
        "answer_under_mis_reasoning",
        "reasoning_mis_answer",
    ]

    for column in answer_columns:
        merged_df[column] = merged_df[column].fillna("").str.strip().str.upper()

    # 基础判断条件
    merged_df["initial_correct"] = (
        merged_df["initial_answer"] == merged_df["ground_truth"]
    )

    merged_df["misreasoning_correct"] = (
        merged_df["answer_under_mis_reasoning"] == merged_df["ground_truth"]
    )

    merged_df["misreasoning_equal_reasoning_answer"] = (
        merged_df["answer_under_mis_reasoning"] == merged_df["reasoning_mis_answer"]
    )

    merged_df["initial_equal_misreasoning_answer"] = (
        merged_df["initial_answer"] == merged_df["answer_under_mis_reasoning"]
    )

    initial_correct = merged_df["initial_correct"]

    misreasoning_correct = merged_df["misreasoning_correct"]

    misreasoning_equal_reasoning_answer = merged_df[
        "misreasoning_equal_reasoning_answer"
    ]

    initial_equal_misreasoning_answer = merged_df["initial_equal_misreasoning_answer"]

    # 六种情况
    conditions = {
        # 1. initial == ground truth
        #    answer_under_mis_reasoning == ground truth
        "count_1": (initial_correct & misreasoning_correct),
        # 2. initial == ground truth
        #    answer_under_mis_reasoning != ground truth
        #    answer_under_mis_reasoning == reasoning_mis_answer
        "count_2": (
            initial_correct
            & ~misreasoning_correct
            & misreasoning_equal_reasoning_answer
        ),
        # 3. initial == ground truth
        #    answer_under_mis_reasoning != ground truth
        #    answer_under_mis_reasoning != reasoning_mis_answer
        "count_3": (
            initial_correct
            & ~misreasoning_correct
            & ~misreasoning_equal_reasoning_answer
        ),
        # 4. initial != ground truth
        #    answer_under_mis_reasoning == ground truth
        "count_4": (~initial_correct & misreasoning_correct),
        # 5. initial != ground truth
        #    answer_under_mis_reasoning != ground truth
        #    initial == answer_under_mis_reasoning
        "count_5": (
            ~initial_correct & ~misreasoning_correct & initial_equal_misreasoning_answer
        ),
        # 6. initial != ground truth
        #    answer_under_mis_reasoning != ground truth
        #    initial != answer_under_mis_reasoning
        "count_6": (
            ~initial_correct
            & ~misreasoning_correct
            & ~initial_equal_misreasoning_answer
        ),
    }

    # 统计每种情况的数量
    counts = {name: int(condition.sum()) for name, condition in conditions.items()}

    # 给每行添加所属类别
    merged_df["category"] = ""

    for name, condition in conditions.items():
        merged_df.loc[
            condition,
            "category",
        ] = name

    # 保存每种情况对应的具体行
    grouped_rows = {
        name: merged_df.loc[condition].copy() for name, condition in conditions.items()
    }

    # 保存全部分类结果
    output_path = data_dir / output_name

    merged_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 可选：分别保存六类数据
    if export_grouped_rows:
        for name, group_df in grouped_rows.items():
            group_output_path = data_dir / f"{file}_{name}.csv"

            group_df.to_csv(
                group_output_path,
                index=False,
                encoding="utf-8-sig",
            )

    print("*" * 80)
    print(f"File: {file}")
    print(f"initial文件问题数：{len(initial_df)}")
    print(f"misreasoning文件问题数：" f"{len(misreasoning_df)}")
    print(f"reasoning文件问题数：" f"{len(reasoning_df)}")
    print(f"三个CSV合并后的问题数：" f"{len(merged_df)}")

    print("1. initial正确，mis reasoning后正确：" f"{counts['count_1']}")

    print(
        "2. initial正确，mis reasoning后错误，"
        "且两个mis答案相同："
        f"{counts['count_2']}"
    )

    print(
        "3. initial正确，mis reasoning后错误，"
        "且两个mis答案不同："
        f"{counts['count_3']}"
    )

    print("4. initial错误，mis reasoning后正确：" f"{counts['count_4']}")

    print(
        "5. initial错误，mis reasoning后错误，"
        "且两次回答相同："
        f"{counts['count_5']}"
    )

    print(
        "6. initial错误，mis reasoning后错误，"
        "且两次回答不同："
        f"{counts['count_6']}"
    )

    print(f"六类数量之和：" f"{sum(counts.values())}")

    # 检查六类是否覆盖所有问题
    if sum(counts.values()) == len(merged_df):
        print("检查通过：所有问题均被归入且仅归入一个类别。")
    else:
        print("警告：六类数量之和与合并后的问题数不一致。")

    print(f"完整分类结果已保存至：{output_path}")
    print("*" * 80)

    return counts, grouped_rows, merged_df


def are_answers_equivalent(answer_a, answer_b, question):
    prompt = (
        "You are a strict answer-equivalence judge.\n"
        "\n"
        f"Question:\n{question}\n"
        "\n"
        f"Answer A:\n{answer_a}\n"
        "\n"
        f"Answer B:\n{answer_b}\n"
        "\n"
        "Determine whether Answer A and Answer B express the same final answer "
        "to the given question.\n"
        "\n"
        "Judgment rules:\n"
        "1. Judge semantic equivalence, not exact text matching.\n"
        "2. Different wording, abbreviations, aliases, units, or formatting are "
        "allowed if they represent the same answer.\n"
        "3. For mathematical answers, treat numerically or algebraically "
        "equivalent expressions as equivalent.\n"
        "4. A more detailed answer is equivalent if its additional information "
        "does not contradict or materially change the shared answer.\n"
        "5. If either answer contains extra claims that contradict the other "
        "answer or the question, return false.\n"
        "6. Refusals, irrelevant responses, uncertainty without an answer, and "
        "empty answers are not equivalent to a valid answer.\n"
        "7. Do not solve the question independently except when necessary to "
        "determine whether the two answers refer to the same result.\n"
        "\n"
        "Return only a valid JSON object in the following format:\n"
        '{"equivalent": true}\n'
        "or\n"
        '{"equivalent": false}'
    )
    for attempt in range(3):
        try:
            response = get_llm_response(JUDGE, prompt)

            # 如果返回值包含Markdown代码块，则将其移除
            response = response.strip()
            response = response.removeprefix("```json")
            response = response.removeprefix("```")
            response = response.removesuffix("```")
            response = response.strip()

            result = json.loads(response)
            equivalent = result.get("equivalent")
            # print("+" * 50)
            # print(
            #     "Question: {0}\nAnswer_A: {1}\nAnswer_B:{2}\nJUDGE:{3}".format(
            #         question, answer_a, answer_b, equivalent
            #     )
            # )
            # print("+" * 50)

            if isinstance(equivalent, bool):
                return equivalent

        except (json.JSONDecodeError, TypeError, AttributeError) as error:
            print(f"第{attempt + 1}次等价判断失败：{error}")


def count_semantic_results_misleading(save_dir, file, model):
    data_dir = Path(save_dir)
    initial_name = "{0}_initial_answer_explaination.csv".format(file)
    misleading_name = "{0}_misleading.csv".format(file)
    background_name = "{0}_background.csv".format(file)
    output_name = "{0}_misleading_semantic_judgment_results.csv".format(file)
    # 保存全部LLM判断结果
    output_path = data_dir / output_name
    initial_df = pd.read_csv(
        data_dir / initial_name,
        dtype=str,
    )

    misreasoning_df = pd.read_csv(
        data_dir / misleading_name,
        dtype=str,
    )

    reasoning_df = pd.read_csv(
        background_name,
        dtype=str,
    )

    min_length = min(
        len(initial_df),
        len(misreasoning_df),
        len(reasoning_df),
    )

    # 如果输出文件存在，则检查其数据行数
    if output_path.exists():
        output_df = pd.read_csv(output_path, dtype=str)

        if len(output_df) == min_length:
            print(f"模型：{model}的{file}数据集已经处理，跳过。")
            return

    # 选择需要的列并重命名
    initial_df = initial_df[["question", "answer", "ground truth"]].rename(
        columns={
            "answer": "initial_answer",
            "ground truth": "ground_truth",
        }
    )

    misreasoning_df = misreasoning_df[
        [
            "question",
            "answer",
        ]
    ].rename(columns={"answer": "answer_under_misleading"})

    # print(misreasoning_df)

    reasoning_df = reasoning_df[["question", "answer"]].rename(
        columns={
            "answer": "misleading_mis_answer",
        }
    )
    # print(reasoning_df)

    # 清理question
    for df in (
        initial_df,
        misreasoning_df,
        reasoning_df,
    ):
        df["question"] = df["question"].fillna("").str.strip()

    # print(reasoning_df)

    merged_df = initial_df.merge(
        misreasoning_df,
        on="question",
        how="inner",
        validate="one_to_one",
    ).merge(
        reasoning_df,
        on="question",
        how="inner",
        validate="one_to_one",
    )
    # print(merged_df)

    # 清理答案
    answer_columns = [
        "initial_answer",
        "ground_truth",
        "answer_under_misleading",
        "misleading_mis_answer",
    ]

    for column in answer_columns:
        merged_df[column] = merged_df[column].fillna("").str.strip()

    def equivalent(row, column_a, column_b):
        return are_answers_equivalent(
            answer_a=row[column_a],
            answer_b=row[column_b],
            question=row["question"],
        )

    print("*" * 30, "JUDGING FILE: {0} MODEL: {1}".format(file, model), "*" * 30)
    # initial与ground truth是否等价
    merged_df["initial_correct"] = merged_df.apply(
        lambda row: equivalent(
            row,
            "initial_answer",
            "ground_truth",
        ),
        axis=1,
    )

    # answer_under_misleadingg与ground truth是否等价
    merged_df["misleading_correct"] = merged_df.apply(
        lambda row: equivalent(
            row,
            "answer_under_misleading",
            "ground_truth",
        ),
        axis=1,
    )

    # answer_under_misleading与reasoning_mis_answer是否等价
    merged_df["misleading_equal_misleading_answer"] = merged_df.apply(
        lambda row: equivalent(
            row,
            "answer_under_misleading",
            "misleading_mis_answer",
        ),
        axis=1,
    )

    # initial与answer_under_misleading是否等价
    merged_df["initial_equal_misleading_answer"] = merged_df.apply(
        lambda row: equivalent(
            row,
            "initial_answer",
            "answer_under_misleading",
        ),
        axis=1,
    )
    print("*" * 30, "JUDGING DONE FILE: {0} MODEL: {1}".format(file, model), "*" * 30)

    # 这里懒得改名了，就用reasoning的变量名了，逻辑是不变的
    initial_correct = merged_df["initial_correct"]

    misreasoning_correct = merged_df["misleading_correct"]

    misreasoning_equal_reasoning_answer = merged_df[
        "misleading_equal_misleading_answer"
    ]

    initial_equal_misreasoning_answer = merged_df["initial_equal_misleading_answer"]

    # 六种情况
    conditions = {
        "count_1": (initial_correct & misreasoning_correct),
        "count_2": (
            initial_correct
            & ~misreasoning_correct
            & misreasoning_equal_reasoning_answer
        ),
        "count_3": (
            initial_correct
            & ~misreasoning_correct
            & ~misreasoning_equal_reasoning_answer
        ),
        "count_4": (~initial_correct & misreasoning_correct),
        "count_5": (
            ~initial_correct & ~misreasoning_correct & initial_equal_misreasoning_answer
        ),
        "count_6": (
            ~initial_correct
            & ~misreasoning_correct
            & ~initial_equal_misreasoning_answer
        ),
    }

    # 统计数量
    counts = {name: int(condition.sum()) for name, condition in conditions.items()}

    # 给每一行添加所属类别
    merged_df["category"] = ""

    for name, condition in conditions.items():
        merged_df.loc[
            condition,
            "category",
        ] = name

    # 获取每个类别对应的行
    grouped_rows = {
        name: merged_df.loc[condition].copy() for name, condition in conditions.items()
    }

    merged_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "*" * 30,
        "Model: {0}\tfile: {1}".format(
            model,
            file,
        ),
        "*" * 30,
    )

    print(f"合并后的问题数：{len(merged_df)}")
    print(f"count_1：{counts['count_1']}")
    print(f"count_2：{counts['count_2']}")
    print(f"count_3：{counts['count_3']}")
    print(f"count_4：{counts['count_4']}")
    print(f"count_5：{counts['count_5']}")
    print(f"count_6：{counts['count_6']}")
    print(f"六类数量之和：{sum(counts.values())}")
    print(f"判断结果保存至：{output_path}")

    if sum(counts.values()) == len(merged_df):
        print("检查通过：所有问题均归入一个类别。")
    else:
        print("警告：存在未分类或重复分类的数据。")

    print(
        "*" * 30,
        "Model: {0}\tfile: {1}".format(
            model,
            file,
        ),
        "*" * 30,
    )

    return counts, grouped_rows, merged_df


def count_semantic_results(save_dir, file, model):
    data_dir = Path(save_dir)

    initial_name = "{0}_initial_answer_explaination.csv".format(file)
    misreasoning_name = "{0}_mis_reasoning_modify_explaination.csv".format(file)
    reasoning_name = "{0}_reasoning.csv".format(file)

    output_name = "{0}_semantic_judgment_results.csv".format(file)
    # 保存全部LLM判断结果
    output_path = data_dir / output_name

    initial_df = pd.read_csv(
        data_dir / initial_name,
        dtype=str,
    )

    misreasoning_df = pd.read_csv(
        data_dir / misreasoning_name,
        dtype=str,
    )

    reasoning_df = pd.read_csv(
        reasoning_name,
        dtype=str,
    )
    # 三个输入文件中的最小数据行数
    min_length = min(
        len(initial_df),
        len(misreasoning_df),
        len(reasoning_df),
    )

    # 如果输出文件存在，则检查其数据行数
    if output_path.exists():
        output_df = pd.read_csv(output_path, dtype=str)

        if len(output_df) == min_length:
            print(f"{file} 此模型数据集已经处理，跳过。")
            return

    # 选择需要的列并重命名
    initial_df = initial_df[["question", "answer", "ground truth"]].rename(
        columns={
            "answer": "initial_answer",
            "ground truth": "ground_truth",
        }
    )

    misreasoning_df = misreasoning_df[
        [
            "question",
            "answer_under_mis_reasoning",
        ]
    ]

    reasoning_df = reasoning_df[["question", "answer"]].rename(
        columns={
            "answer": "reasoning_mis_answer",
        }
    )

    # 清理question
    for df in (
        initial_df,
        misreasoning_df,
        reasoning_df,
    ):
        df["question"] = df["question"].fillna("").str.strip()

    # 根据question合并三个CSV
    merged_df = initial_df.merge(
        misreasoning_df,
        on="question",
        how="inner",
        validate="one_to_one",
    ).merge(
        reasoning_df,
        on="question",
        how="inner",
        validate="one_to_one",
    )

    # 清理答案
    answer_columns = [
        "initial_answer",
        "ground_truth",
        "answer_under_mis_reasoning",
        "reasoning_mis_answer",
    ]

    for column in answer_columns:
        merged_df[column] = merged_df[column].fillna("").str.strip()

    def equivalent(row, column_a, column_b):
        return are_answers_equivalent(
            answer_a=row[column_a],
            answer_b=row[column_b],
            question=row["question"],
        )

    print("*" * 30, "JUDGING FILE: {0} MODEL: {1}".format(file, model), "*" * 30)
    # initial与ground truth是否等价
    merged_df["initial_correct"] = merged_df.apply(
        lambda row: equivalent(
            row,
            "initial_answer",
            "ground_truth",
        ),
        axis=1,
    )

    # answer_under_mis_reasoning与ground truth是否等价
    merged_df["misreasoning_correct"] = merged_df.apply(
        lambda row: equivalent(
            row,
            "answer_under_mis_reasoning",
            "ground_truth",
        ),
        axis=1,
    )

    # answer_under_mis_reasoning与reasoning_mis_answer是否等价
    merged_df["misreasoning_equal_reasoning_answer"] = merged_df.apply(
        lambda row: equivalent(
            row,
            "answer_under_mis_reasoning",
            "reasoning_mis_answer",
        ),
        axis=1,
    )

    # initial与answer_under_mis_reasoning是否等价
    merged_df["initial_equal_misreasoning_answer"] = merged_df.apply(
        lambda row: equivalent(
            row,
            "initial_answer",
            "answer_under_mis_reasoning",
        ),
        axis=1,
    )
    print("*" * 30, "JUDGING DONE FILE: {0} MODEL: {1}".format(file, model), "*" * 30)

    # 获取判断结果列
    initial_correct = merged_df["initial_correct"]

    misreasoning_correct = merged_df["misreasoning_correct"]

    misreasoning_equal_reasoning_answer = merged_df[
        "misreasoning_equal_reasoning_answer"
    ]

    initial_equal_misreasoning_answer = merged_df["initial_equal_misreasoning_answer"]

    # 六种情况
    conditions = {
        "count_1": (initial_correct & misreasoning_correct),
        "count_2": (
            initial_correct
            & ~misreasoning_correct
            & misreasoning_equal_reasoning_answer
        ),
        "count_3": (
            initial_correct
            & ~misreasoning_correct
            & ~misreasoning_equal_reasoning_answer
        ),
        "count_4": (~initial_correct & misreasoning_correct),
        "count_5": (
            ~initial_correct & ~misreasoning_correct & initial_equal_misreasoning_answer
        ),
        "count_6": (
            ~initial_correct
            & ~misreasoning_correct
            & ~initial_equal_misreasoning_answer
        ),
    }

    # 统计数量
    counts = {name: int(condition.sum()) for name, condition in conditions.items()}

    # 给每一行添加所属类别
    merged_df["category"] = ""

    for name, condition in conditions.items():
        merged_df.loc[
            condition,
            "category",
        ] = name

    # 获取每个类别对应的行
    grouped_rows = {
        name: merged_df.loc[condition].copy() for name, condition in conditions.items()
    }

    merged_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "*" * 30,
        "Model: {0}\tfile: {1}".format(
            model,
            file,
        ),
        "*" * 30,
    )

    print(f"合并后的问题数：{len(merged_df)}")
    print(f"count_1：{counts['count_1']}")
    print(f"count_2：{counts['count_2']}")
    print(f"count_3：{counts['count_3']}")
    print(f"count_4：{counts['count_4']}")
    print(f"count_5：{counts['count_5']}")
    print(f"count_6：{counts['count_6']}")
    print(f"六类数量之和：{sum(counts.values())}")
    print(f"判断结果保存至：{output_path}")

    if sum(counts.values()) == len(merged_df):
        print("检查通过：所有问题均归入一个类别。")
    else:
        print("警告：存在未分类或重复分类的数据。")

    print(
        "*" * 30,
        "Model: {0}\tfile: {1}".format(
            model,
            file,
        ),
        "*" * 30,
    )

    return counts, grouped_rows, merged_df


def plot_arc_category_pie(save_dir, file, model, misleading=False):
    """
    统计语义判断结果中每个类别的数量，并生成统计CSV和饼图。

    参数
    ----------
    save_dir : str或Path
        语义判断结果CSV所在目录。

    file : str
        数据集名称，例如"ARC-challenge"。

    model : str
        模型名称，用于饼图标题。

    misleading : bool, default=False
        True表示读取misleading结果；
        False表示读取misreasoning结果。

    返回
    ----------
    category_counts_df : pandas.DataFrame
        六类问题的数量及百分比。

    figure_path : pathlib.Path
        饼图保存路径。
    """
    data_dir = Path(save_dir)

    if misleading:
        input_path = data_dir / f"{file}_misleading_semantic_judgment_results.csv"
        counts_output_path = data_dir / f"{file}_misleading_category_counts.csv"
        figure_path = data_dir / f"{file}_misleading_category_pie.png"
    else:
        input_path = data_dir / f"{file}_semantic_judgment_results.csv"
        counts_output_path = data_dir / f"{file}_category_counts.csv"
        figure_path = data_dir / f"{file}_category_pie.png"

    # 检查输入文件
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在：{input_path}")

    # 读取分类结果
    df = pd.read_csv(
        input_path,
        dtype=str,
    )

    if "category" not in df.columns:
        raise ValueError(f"{input_path}中不存在category列")

    # 固定类别顺序
    category_order = [
        "count_1",
        "count_2",
        "count_3",
        "count_4",
        "count_5",
        "count_6",
    ]

    category_name_map = {
        "count_1": "Robust Correctness",
        "count_2": "Successful Misleading",
        "count_3": "Off-Target Error",
        "count_4": "Corrective Shift",
        "count_5": "Persistent Error",
        "count_6": "Error Shift",
    }

    # 统计每个类别，缺失类别补0
    category_counts = (
        df["category"]
        .fillna("")
        .str.strip()
        .value_counts()
        .reindex(
            category_order,
            fill_value=0,
        )
    )

    category_counts_df = pd.DataFrame(
        {
            "category": category_order,
            "category_name": [
                category_name_map[category] for category in category_order
            ],
            "count": [int(category_counts[category]) for category in category_order],
        }
    )

    total_count = int(category_counts_df["count"].sum())

    # 计算百分比
    if total_count > 0:
        category_counts_df["percentage"] = (
            category_counts_df["count"] / total_count * 100
        ).round(2)
    else:
        category_counts_df["percentage"] = 0.0

    # 保存统计结果
    category_counts_df.to_csv(
        counts_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    if total_count == 0:
        raise ValueError(f"{input_path}中没有有效的category分类结果，无法绘制饼图")

    # 绘图时过滤数量为0的类别
    plot_df = category_counts_df[category_counts_df["count"] > 0].copy()

    # 六个类别使用固定颜色
    category_color_map = {
        "count_1": "#4C78A8",
        "count_2": "#F58518",
        "count_3": "#E45756",
        "count_4": "#72B7B2",
        "count_5": "#54A24B",
        "count_6": "#B279A2",
    }

    colors = plot_df["category"].map(category_color_map)

    # 生成饼图
    fig, ax = plt.subplots(figsize=(10, 7))

    wedges, _, autotexts = ax.pie(
        plot_df["count"],
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda percentage: (f"{percentage:.2f}%" if percentage > 0 else ""),
        pctdistance=0.72,
        wedgeprops={
            "edgecolor": "white",
            "linewidth": 1.5,
        },
        textprops={
            "fontsize": 10,
        },
    )

    # 百分比文本样式
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    # 图例中显示类别名称、数量和百分比
    legend_labels = [
        (f"{row.category_name}: " f"{row['count']} ({row.percentage:.2f}%)")
        for _, row in plot_df.iterrows()
    ]

    ax.legend(
        wedges,
        legend_labels,
        title="Categories",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        title_fontsize=11,
    )

    ax.set_title(
        "Category Distribution of {0}, {1}, {2}".format(
            file,
            model,
            "misleading" if misleading else "misreasoning",
        ),
        fontsize=15,
        pad=15,
    )

    # 保证饼图为正圆
    ax.axis("equal")

    plt.tight_layout()

    # 保存饼图
    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("*" * 60)
    print(f"数据集：{file}")
    print(f"模型：{model}")
    print(category_counts_df.to_string(index=False))
    print(f"总问题数：{total_count}")
    print(f"统计结果保存至：{counts_output_path}")
    print(f"饼图保存至：{figure_path}")
    print("*" * 60)

    return category_counts_df, figure_path


def plot_arc_category_counts(save_dir, file, model, misleading=False):
    """
    统计classification_results.csv中每个类别的数量，
    并生成统计CSV和柱状图。

    参数：
        save_dir:
            分类结果CSV所在目录。

        file:
            数据集名称，例如"ARC-challenge"。

    返回：
        category_counts_df:
            六类数量统计结果。

        figure_path:
            柱状图保存路径。
    """
    data_dir = Path(save_dir)
    if misleading:
        input_path = data_dir / f"{file}_misleading_semantic_judgment_results.csv"

        counts_output_path = data_dir / f"{file}_misleading_category_counts.csv"

        figure_path = data_dir / f"{file}_misleading_category_counts.png"
    else:
        input_path = data_dir / f"{file}_semantic_judgment_results.csv"

        counts_output_path = data_dir / f"{file}_category_counts.csv"

        figure_path = data_dir / f"{file}_category_counts.png"

    # 读取分类结果
    df = pd.read_csv(
        input_path,
        dtype=str,
    )

    if "category" not in df.columns:
        raise ValueError(f"{input_path}中不存在category列")

    # 固定类别顺序
    category_order = [
        "count_1",
        "count_2",
        "count_3",
        "count_4",
        "count_5",
        "count_6",
    ]

    category_name_map = {
        "count_1": "Robust Correctness",
        "count_2": "Successful Misleading",
        "count_3": "Off-Target Error",
        "count_4": "Corrective Shift",
        "count_5": "Persistent Error",
        "count_6": "Error Shift",
    }

    # 统计每个类别，缺失类别补0
    category_counts = (
        df["category"]
        .fillna("")
        .str.strip()
        .value_counts()
        .reindex(
            category_order,
            fill_value=0,
        )
    )

    category_counts_df = pd.DataFrame(
        {
            "category": category_order,
            "count": [int(category_counts[category]) for category in category_order],
        }
    )

    total_count = int(category_counts_df["count"].sum())

    # 添加百分比
    if total_count > 0:
        category_counts_df["percentage"] = (
            category_counts_df["count"] / total_count * 100
        ).round(2)
    else:
        category_counts_df["percentage"] = 0.0

    # 保存统计结果
    category_counts_df.to_csv(
        counts_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 生成柱状图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 用于柱状图显示的类别名称
    display_categories = category_counts_df["category"].map(category_name_map)

    bars = ax.bar(
        display_categories,
        category_counts_df["count"],
        width=0.65,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_title(
        "Category Counts of {0}, {1}, {2}".format(
            file,
            model,
            "misleading" if misleading else "misreasoning",
        ),
        fontsize=15,
        pad=15,
    )

    ax.set_xlabel(
        "Category",
        fontsize=12,
    )

    ax.set_ylabel(
        "Number of Questions",
        fontsize=12,
    )

    # y轴只显示整数
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 添加横向辅助线
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    ax.set_axisbelow(True)

    # 在柱子顶部显示数量和百分比
    for bar, count, percentage in zip(
        bars,
        category_counts_df["count"],
        category_counts_df["percentage"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({percentage:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # 为顶部标签留出空间
    max_count = category_counts_df["count"].max()

    if max_count > 0:
        ax.set_ylim(
            0,
            max_count * 1.2,
        )
    else:
        ax.set_ylim(0, 1)

    plt.tight_layout()

    # 保存柱状图
    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("*" * 60)
    print(f"数据集：{file}")
    print(category_counts_df.to_string(index=False))
    print(f"总问题数：{total_count}")
    print(f"统计结果保存至：{counts_output_path}")
    print(f"柱状图保存至：{figure_path}")
    print("*" * 60)

    return category_counts_df, figure_path


def plot_count_radar(LLMS, file, misleading=False):
    """
    统计不同模型目录下CSV文件中count_1-count_6的占比，
    以对应CSV文件的总行数为基数，并绘制百分比雷达图。

    参数
    ----------
    LLMS : list
        模型名称列表。

    file : str
        数据集文件名，可带或不带.csv后缀。
    """
    categories = [f"count_{i}" for i in range(1, 7)]
    category_names = {
        "count_1": "Robust\nCorrectness",
        "count_2": "Successful\nMisleading",
        "count_3": "Off-Target\nError",
        "count_4": "Corrective\nShift",
        "count_5": "Persistent\nError",
        "count_6": "Error\nShift",
    }

    category_labels = [category_names[category] for category in categories]

    # 避免传入的file已经包含.csv后缀
    file_stem = Path(file).stem
    if misleading:
        file_name = f"{file_stem}_misleading_semantic_judgment_results.csv"
    else:
        file_name = f"{file_stem}_semantic_judgment_results.csv"

    statistics = {}
    row_counts = {}

    for llm in LLMS:
        save_dir = Path(llm.replace(":", "_"))
        csv_path = save_dir / file_name

        if not csv_path.exists():
            print(f"文件不存在，跳过：{csv_path}")
            continue

        df = pd.read_csv(csv_path, dtype=str)

        if "category" not in df.columns:
            raise ValueError(
                f"{csv_path}中不存在category列，" "无法统计count_1-count_6。"
            )

        total_rows = len(df)

        if total_rows == 0:
            print(f"CSV文件没有数据，跳过：{csv_path}")
            continue

        category_counts = df["category"].fillna("").str.strip().value_counts()

        # 以CSV总行数为基数计算百分比
        statistics[llm] = [
            category_counts.get(category, 0) / total_rows * 100
            for category in categories
        ]

        row_counts[llm] = total_rows

    if not statistics:
        print("没有找到可以统计的数据。")
        return pd.DataFrame()

    # 生成百分比统计结果表
    statistics_df = pd.DataFrame.from_dict(
        statistics,
        orient="index",
        columns=categories,
    )

    statistics_df.index.name = "LLM"

    # 保留两位小数
    statistics_df = statistics_df.round(2)

    # 添加CSV总行数，便于检查百分比基数
    statistics_df["total_rows"] = pd.Series(row_counts)

    print(statistics_df)

    # 雷达图的六个方向
    angles = np.linspace(
        0,
        2 * np.pi,
        len(categories),
        endpoint=False,
    ).tolist()

    closed_angles = angles + angles[:1]

    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw={"polar": True},
    )

    for llm, values in statistics.items():
        closed_values = values + values[:1]

        ax.plot(
            closed_angles,
            closed_values,
            linewidth=2,
            marker="o",
            label=llm,
        )

        ax.fill(
            closed_angles,
            closed_values,
            alpha=0.08,
        )

    ax.set_xticks(angles)
    ax.set_xticklabels(
        category_labels,
        fontsize=11,
    )

    # 将径向刻度显示成百分数
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, position: f"{value:.0f}%")
    )

    ax.set_title(
        f"{file_stem}: Response Behavior Distribution",
        fontsize=14,
        pad=25,
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1.05),
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.6,
    )

    if not misleading:
        output_path = Path(
            f"{file_stem}_semantic_judgment_results_percentage_radar.png"
        )
    else:
        output_path = Path(
            f"{file_stem}_misleading_semantic_judgment_results_percentage_radar.png"
        )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    print(f"百分比雷达图已保存至：{output_path}")

    return statistics_df


def judge_misleading_acc_explanation(file, save_dir, model):
    """
    只判断 misleading answer 是否正确。

    已有的 initial 判断结果直接从：
        {file}_mis_reasoning_modify_judge_explaination.csv

    读取。

    misleading answer 从：
        {file}_misleading.csv

    读取并进行判断。

    最终结果保存到：
        {file}_misleading_judge_explaination.csv
    """

    save_dir = Path(save_dir)

    judged_input_csv = save_dir / f"{file}_mis_reasoning_modify_judge_explaination.csv"

    misleading_input_csv = save_dir / f"{file}_misleading.csv"

    output_csv = save_dir / f"{file}_misleading_judge_explaination.csv"

    # 检查已经完成的 mis-reasoning 判断文件
    if not judged_input_csv.exists():
        print(f"Model: {model}\t" f"{judged_input_csv.name} does not exist.")
        return

    # 检查 misleading 结果文件
    if not misleading_input_csv.exists():
        print(f"Model: {model}\t" f"{misleading_input_csv.name} does not exist.")
        return

    # 读取已经完成的判断结果
    judged_df = pd.read_csv(judged_input_csv)

    # 读取 misleading 回答
    misleading_df = pd.read_csv(misleading_input_csv)

    # plain_text需要额外读取原始的background和situation
    if file == "plain_text":
        original_df = pd.read_parquet(f"./{file}_train-00000-of-00001.parquet")

    # 检查两个文件的行数是否一致
    if len(judged_df) != len(misleading_df):
        print(
            "Warning: The number of rows is inconsistent.\n"
            f"Judged file: {len(judged_df)} rows\n"
            f"Misleading file: {len(misleading_df)} rows"
        )

    # 防止行数不一致导致越界
    total_rows = min(len(judged_df), len(misleading_df))

    # 断点续传
    start = 0

    if output_csv.exists():
        result_df = pd.read_csv(output_csv)
        start = len(result_df)

    print(
        "----------------------------"
        f"Judge misleading answer {file} on {model}"
        "----------------------------"
    )

    for i in range(start, total_rows):
        # 直接读取已经完成的initial判断结果
        initial_judge = judged_df.iloc[i]["initial_judge"]
        initial_explaination = judged_df.iloc[i]["initial_explaination"]

        # misleading文件中的数据
        misleading_answer = misleading_df.iloc[i]["answer"]
        question = misleading_df.iloc[i]["question"]
        ground_truth = misleading_df.iloc[i]["ground truth"]

        print(
            "----------------------------"
            f"Q id: {i}\tModel: {model}"
            "----------------------------"
        )

        if file != "plain_text":
            misleading_judge, misleading_explaination = rest_judge_explaination(
                file,
                misleading_answer,
                ground_truth,
                question=question,
            )

        else:
            background = original_df.iloc[i]["background"]
            situation = original_df.iloc[i]["situation"]

            misleading_judge, misleading_explaination = rest_judge_explaination(
                file,
                misleading_answer,
                ground_truth,
                question=question,
                background=background,
                situation=situation,
            )

        row_df = pd.DataFrame(
            [
                {
                    "question": question,
                    "initial_judge": initial_judge,
                    "initial_explaination": initial_explaination,
                    "misleading_judge": misleading_judge,
                    "misleading_explaination": misleading_explaination,
                }
            ]
        )

        print(
            f"Misleading answer: {misleading_answer}\n" f"Ground truth: {ground_truth}"
        )

        print(
            f"Initial judge: {initial_judge}\t" f"Misleading judge: {misleading_judge}"
        )

        print(
            f"Initial explaination: {initial_explaination}\n"
            f"Misleading explaination: {misleading_explaination}"
        )

        row_df.to_csv(
            output_csv,
            mode="a",
            header=not output_csv.exists(),
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "----------------------------"
            f"Q id: {i}\tModel: {model} finished"
            "----------------------------"
        )

    print(
        "----------------------------"
        f"Judge misleading answer {file} on {model} finished"
        "----------------------------"
    )

    return


def add_original_question(csv_a, csv_b):
    """
    1. 将 CSV A 的 question 更名为 rewritten_problem
    2. 根据 rewritten_problem 匹配 CSV B 中的 question
    3. 将新增的 question 移动到第一列
    4. 直接覆盖 CSV A
    """
    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    # 将 A 中原来的 question 更名为 rewritten_problem
    df_a.rename(columns={"question": "rewritten_problem"}, inplace=True)

    # 创建 rewritten_problem -> question 映射
    question_mapping = df_b.drop_duplicates(subset=["rewritten_problem"]).set_index(
        "rewritten_problem"
    )["question"]

    # 添加匹配到的 question
    df_a["question"] = df_a["rewritten_problem"].map(question_mapping)

    # 将 question 移动到第一列
    question_column = df_a.pop("question")
    df_a.insert(0, "question", question_column)

    # 直接覆盖 CSV A
    df_a.to_csv(csv_a, index=False, encoding="utf-8-sig")

    print(f"已直接更新：{csv_a}")
    print(f"成功匹配：{df_a['question'].notna().sum()} 行")
    print(f"未匹配：{df_a['question'].isna().sum()} 行")

    return df_a


def merge_four_images(image_paths, output_path):
    """
    将四张图片按 2×2 排列合成一张图片。
    """
    if len(image_paths) != 4:
        raise ValueError("必须传入四张图片")

    images = [Image.open(path).convert("RGB") for path in image_paths]

    # 统一为第一张图片的尺寸
    width, height = images[0].size
    images = [
        image.resize((width, height), Image.Resampling.LANCZOS) for image in images
    ]

    # 创建画布
    merged_image = Image.new("RGB", (width * 2, height * 2), "white")

    positions = [
        (0, 0),
        (width, 0),
        (0, height),
        (width, height),
    ]

    for image, position in zip(images, positions):
        merged_image.paste(image, position)

    merged_image.save(output_path, quality=95)

    for image in images:
        image.close()

    print(f"合成图片已保存到：{output_path}")


LLMS = [
    "google/gemini-2.5-flash",
    "google/gemini-3-flash-preview",
    "qwen/qwen3.6-flash",
    "deepseek/deepseek-v3.2",
    "openai/gpt-5",
    "x-ai/grok-4.20",
    "meta-llama/llama-3.1-8b-instruct",
]


def draw_radar():
    plot_count_radar(LLMS, "ARC-challenge")
    plot_count_radar(LLMS, "NQ-open")
    plot_count_radar(LLMS, "math_train")
    plot_count_radar(LLMS, "plain_text")
    return


f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]

if __name__ == "__main__":
    four_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]
    # draw_radar()
    for llm_id in range(len(LLMS)):
        save_dir = Path("{0}".format(LLMS[llm_id].replace(":", "_")))
        save_dir.mkdir(parents=True, exist_ok=True)
        llm = LLMS[llm_id]

        # 下面的misreasoning代码在分类的时候已经将initial_answer和misreasoning_answer与ground truth进行对比了，所以包含了判断，无需额外判断
        # count_arc_results(save_dir, "ARC-challenge")
        # count_semantic_results(save_dir, "NQ-open", llm)
        # count_semantic_results(save_dir, "plain_text", llm)
        # count_semantic_results(save_dir, "math_train", llm)
        # plot_arc_category_pie(save_dir, "ARC-challenge", llm)
        # plot_arc_category_pie(save_dir, "NQ-open", llm)
        # plot_arc_category_pie(save_dir, "plain_text", llm)
        # plot_arc_category_pie(save_dir, "math_train", llm)
        fig_list = [
            save_dir / "{0}_category_pie.png".format(file) for file in four_list
        ]
        merge_four_images(fig_list, save_dir / "four.png")

        # add_original_question(
        #     save_dir / "{0}_misleading.csv".format("math_train"),
        #     "{0}_background.csv".format("math_train"),
        # )

        # 下面的misleading代码在分类的时候已经将initial_answer和misleading_answer与ground truth进行对比了，所以包含了判断，无需额外判断
        # count_arc_results_misleading(save_dir, "ARC-challenge")
        # count_semantic_results_misleading(save_dir, "NQ-open", llm)
        # count_semantic_results_misleading(save_dir, "plain_text", llm)
        count_semantic_results_misleading(save_dir, "math_train", llm)
        # plot_arc_category_pie(save_dir, "ARC-challenge", llm, misleading=True)
        # plot_arc_category_pie(save_dir, "NQ-open", llm, misleading=True)
        # plot_arc_category_pie(save_dir, "plain_text", llm, misleading=True)
        # plot_arc_category_pie(save_dir, "math_train", llm, misleading=True)
        fig_list = [
            save_dir / "{0}_misleading_category_pie.png".format(file)
            for file in four_list
        ]
        merge_four_images(fig_list, save_dir / "misleading four.png")
    print("Done at {0}".format(datetime.now()))
