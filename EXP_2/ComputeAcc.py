import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os
from pathlib import Path
import numpy as np
from datetime import datetime

from Correction import rest_judge
from PIL import Image, ImageDraw, ImageFont
import re

API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("未找到环境变量 OPENROUTER_API_KEY")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("未找到环境变量 HF_TOKEN")
file_list = ["math_train", "NQ-open", "plain_text"]

label_list = ["(b)", "(c)", "(d)"]


def judge_reasoning_acc(file, save_dir, model):
    if (save_dir / "{0}_mis_reasoning_modify.csv".format(file)).exists() == False:
        print(
            "Model: {0}\t{1}_mis_reasoning_modify.csv doex not exist.".format(
                model, file
            )
        )
        return
    df = pd.read_csv(save_dir / "{0}_mis_reasoning_modify.csv".format(file))
    output_csv = save_dir / "{0}_mis_reasoning_modify_judge.csv".format(file)
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
        ground_truth = df.iloc[i]["ground truth"]
        print(
            "----------------------------Q id: {0}\tModel: {1}----------------------------".format(
                i, model
            )
        )
        initial_judge = rest_judge(file, initial_answer, ground_truth)
        mis_reasoning_judge = rest_judge(file, mis_reasoning_answer, ground_truth)
        row_df = pd.DataFrame(
            [[df.iloc[i]["question"], initial_judge, mis_reasoning_judge]],
            columns=["question", "initial_judge", "mis_reasoning_judge"],
        )
        print(initial_answer, mis_reasoning_answer)
        print(ground_truth)
        print(
            "Initial judge: {0}\t Misleading judge: {1}".format(
                initial_judge, mis_reasoning_judge
            )
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
    print(
        "----------------------------Judge answer {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


def draw_reasoning_acc_one_model_four_files(model):
    save_dir = Path("{0}".format(model))
    save_dir.mkdir(parents=True, exist_ok=True)
    ARC_misreasoning_df = pd.read_csv(save_dir / "ARC-challenge_mis_reasoning.csv")
    initial_count = 0
    misleading_count = 0

    ACC_list = []
    for i in range(len(ARC_misreasoning_df)):
        initial_answer = ARC_misreasoning_df.iloc[i]["initial_answer"]
        misleading_answer = ARC_misreasoning_df.iloc[i]["answer_under_mis_reasoning"]
        ground_truth = ARC_misreasoning_df.iloc[i]["ground truth"]
        if initial_answer == ground_truth:
            initial_count += 1
        if misleading_answer == ground_truth:
            misleading_count += 1
    ACC_list.append(
        [
            initial_count / len(ARC_misreasoning_df),
            misleading_count / len(ARC_misreasoning_df),
        ]
    )

    for file in ["math_train", "NQ-open", "plain_text"]:
        judge_df = pd.read_csv(save_dir / "{0}_mis_reasoning_judge.csv".format(file))
        total = len(judge_df)
        initial_count = (judge_df["initial_judge"] == True).sum()
        misleading_count = (judge_df["mis_reasoning_judge"] == True).sum()
        ACC_list.append([initial_count / total, misleading_count / total])

    acc_initial = [x[0] for x in ACC_list]
    acc_misleading = [x[1] for x in ACC_list]
    f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]

    x = np.arange(len(f_list))
    width = 0.35

    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(x - width / 2, acc_initial, width, label="Acc_initial")
    bars2 = plt.bar(x + width / 2, acc_misleading, width, label="Acc_misleading")

    # 在柱子上显示数值
    plt.bar_label(bars1, fmt="%.2f", padding=3)
    plt.bar_label(bars2, fmt="%.2f", padding=3)

    plt.xlabel("File Name")
    plt.ylabel("Accuracy")
    plt.title("Misreasoning Comparison")
    plt.xticks(x, f_list, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "misreasoning accuracy.png", dpi=300, bbox_inches="tight")
    # plt.show()
    return


def judge_misleading(file, save_dir, model):
    if (save_dir / "{0}_misleading.csv".format(file)).exists() == False:
        print("Model: {0}\t{1}_misleading.csv does not exist.".format(save_dir, file))
        return
    df = pd.read_csv(save_dir / "{0}_misleading.csv".format(file))
    initial_df = pd.read_csv(save_dir / "{0}_initial_answer.csv".format(file))
    output_csv = save_dir / "{0}_misleading_judge.csv".format(file)
    output_path = Path(output_csv)
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    print(
        "----------------------------Judge misleading answer {1} on {0}----------------------------".format(
            model, file
        )
    )
    for i in range(start, len(df)):
        question = initial_df.iloc[i]["question"]
        initial_answer = initial_df.iloc[i]["answer"]
        misleading_answer = df.iloc[i]["answer"]
        ground_truth = df.iloc[i]["ground truth"]
        print(
            "----------------------------Q id: {0}\tModel: {1}----------------------------".format(
                i, model
            )
        )
        initial_judge = rest_judge(
            file, initial_answer, ground_truth, question=question
        )
        misleading_judge = rest_judge(
            file,
            misleading_answer,
            ground_truth,
            question=question,
        )
        row_df = pd.DataFrame(
            [[df.iloc[i]["question"], initial_judge, misleading_judge]],
            columns=["question", "initial_judge", "misleading_judge"],
        )
        print(initial_answer, misleading_answer)
        print(ground_truth)
        print(
            "Initial judge: {0}\t Misleading judge: {1}".format(
                initial_judge, misleading_judge
            )
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
    print(
        "----------------------------Judge misleading answer {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


def draw_ACC_fourfile_onemodel(model):
    save_dir = Path("{0}".format(model))
    save_dir.mkdir(parents=True, exist_ok=True)
    ARC_initial_df = pd.read_csv(save_dir / "ARC-challenge_initial_answer.csv")
    ARC_misleading_df = pd.read_csv(save_dir / "ARC-challenge_misleading.csv")
    length = min(len(ARC_initial_df), len(ARC_misleading_df))
    initial_count = 0
    misleading_count = 0

    ACC_list = []
    for i in range(length):
        initial_answer = ARC_initial_df.iloc[i]["answer"]
        misleading_answer = ARC_misleading_df.iloc[i]["answer"]
        assert (
            ARC_misleading_df.iloc[i]["ground truth"]
            == ARC_initial_df.iloc[i]["ground truth"]
        )
        ground_truth = ARC_misleading_df.iloc[i]["ground truth"]
        if initial_answer == ground_truth:
            initial_count += 1
        if misleading_answer == ground_truth:
            misleading_count += 1
    ACC_list.append([initial_count / length, misleading_count / length])

    for file in ["math_train", "NQ-open", "plain_text"]:
        judge_df = pd.read_csv(save_dir / "{0}_misleading_judge.csv".format(file))
        total = len(judge_df)
        initial_count = (judge_df["initial_judge"] == True).sum()
        misleading_count = (judge_df["misleading_judge"] == True).sum()
        ACC_list.append([initial_count / total, misleading_count / total])

    acc_initial = [x[0] for x in ACC_list]
    acc_misleading = [x[1] for x in ACC_list]
    f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]

    x = np.arange(len(f_list))
    width = 0.35

    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(x - width / 2, acc_initial, width, label="Acc_initial")
    bars2 = plt.bar(x + width / 2, acc_misleading, width, label="Acc_misleading")

    # 在柱子上显示数值
    plt.bar_label(bars1, fmt="%.2f", padding=3)
    plt.bar_label(bars2, fmt="%.2f", padding=3)

    plt.xlabel("File Name")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.xticks(x, f_list, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "misleading accuracy.png", dpi=300, bbox_inches="tight")
    # plt.show()


def draw_multi_turn_acc(model):
    save_dir = Path("{0}".format(model))
    save_dir.mkdir(parents=True, exist_ok=True)
    ARC_initial_df = pd.read_csv(save_dir / "ARC-challenge_initial_answer.csv")
    ARC_multi_turn_df = pd.read_csv(save_dir / "ARC-challenge_multi_turn.csv")
    length = min(len(ARC_initial_df), len(ARC_multi_turn_df))
    initial_count = 0

    ACC_list = []
    for i in range(length):
        initial_answer = ARC_initial_df.iloc[i]["answer"]
        ground_truth = ARC_initial_df.iloc[i]["ground truth"]
        if initial_answer == ground_truth:
            initial_count += 1
    multi_turn_count = (ARC_multi_turn_df["turn count"] == -1).sum()
    ACC_list.append([initial_count / length, multi_turn_count / length])

    for file in ["math_train", "NQ-open", "plain_text"]:
        judge_df = pd.read_csv(save_dir / "{0}_mis_reasoning_judge.csv".format(file))
        multi_turn_df = pd.read_csv(save_dir / "{0}_multi_turn.csv".format(file))
        total = min(len(judge_df), len(multi_turn_df))
        initial_count = 0
        multi_turn_count = 0
        print(total)
        for j in range(total):
            if judge_df.iloc[j]["initial_judge"] == True:
                initial_count += 1
            if multi_turn_df.iloc[j]["turn count"] == -1:
                multi_turn_count += 1
        print(f"{file}, {multi_turn_count}")
        ACC_list.append([initial_count / total, multi_turn_count / total])
    acc_initial = [x[0] for x in ACC_list]
    acc_misleading = [x[1] for x in ACC_list]
    f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]

    x = np.arange(len(f_list))
    width = 0.35

    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(x - width / 2, acc_initial, width, label="Acc_initial")
    bars2 = plt.bar(x + width / 2, acc_misleading, width, label="Acc_misleading")

    # 在柱子上显示数值
    plt.bar_label(bars1, fmt="%.2f", padding=3)
    plt.bar_label(bars2, fmt="%.2f", padding=3)

    plt.xlabel("File Name")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.xticks(x, f_list, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "multi_turn accuracy.png", dpi=300, bbox_inches="tight")
    # plt.show()
    return


def compute_multi_turn_distribution(model, correction=False):
    save_dir = Path("{0}".format(model))
    save_dir.mkdir(parents=True, exist_ok=True)
    f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]

    # 固定 x 轴取值
    x_values = [-1, 1, 2, 3, 4, 5, 6]

    for file in f_list:
        if correction:
            input_path = save_dir / "{0}_correction_multi_turn.csv".format(file)
            if not input_path.exists():
                print(f"File not found: {input_path}")
                continue
            df = pd.read_csv(input_path)
        else:
            input_path = save_dir / "{0}_multi_turn.csv".format(file)
            if not input_path.exists():
                print(f"File not found: {input_path}")
                continue
            df = pd.read_csv(input_path)

        column_name = "turn count"

        # 按固定 x_values 重新索引，缺失值填 0
        counts = df[column_name].value_counts().reindex(x_values, fill_value=0)

        ratios = counts / counts.sum() if counts.sum() != 0 else counts

        result = pd.DataFrame({"count": counts, "ratio": ratios})

        plt.figure(figsize=(10, 6))

        bars = plt.bar(result.index.astype(str), result["ratio"])

        plt.bar_label(bars, labels=[f"{v:.2%}" for v in result["ratio"]], padding=3)

        plt.xlabel(column_name)
        plt.ylabel("Proportion")

        if correction:
            plt.title(
                f"The proportion of reversals across different {column_name}, correction."
            )
        else:
            plt.title(f"The proportion of reversals across different {column_name}")

        max_ratio = result["ratio"].max()
        plt.ylim(0, max_ratio * 1.15 if max_ratio > 0 else 1)

        plt.tight_layout()

        if correction:
            plt.savefig(
                save_dir / "{0} correction turn distribution.png".format(file),
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                save_dir / "{0} turn distribution.png".format(file),
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()

    return


def merge_images_2x2(image_paths, output_path, label_font_size=40, label_margin=15):
    """
    将 4 张图片按照田字格顺序拼接，并给每个子图添加编号 (a)(b)(c)(d)。

    image_paths 顺序：
    [左上, 右上, 左下, 右下]
    """

    if len(image_paths) != 4:
        raise ValueError("image_paths 必须包含 4 张图片路径")

    labels = ["(a)", "(b)", "(c)", "(d)"]

    images = [Image.open(path).convert("RGBA") for path in image_paths]

    # 统一为第一张图片的大小
    width, height = images[0].size
    images = [img.resize((width, height)) for img in images]

    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", label_font_size)
    except:
        font = ImageFont.load_default()

    # 给每张子图加编号
    labeled_images = []
    for img, label in zip(images, labels):
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)

        # 文字位置：左上角
        x, y = label_margin, label_margin

        # 先画白底描边，增强可见性
        outline_offsets = [
            (-2, -2),
            (-2, 2),
            (2, -2),
            (2, 2),
            (-2, 0),
            (2, 0),
            (0, -2),
            (0, 2),
        ]
        for dx, dy in outline_offsets:
            draw.text((x + dx, y + dy), label, font=font, fill="white")

        # 再画黑字
        draw.text((x, y), label, font=font, fill="black")

        labeled_images.append(img_copy)

    # 创建透明背景画布
    new_img = Image.new("RGBA", (width * 2, height * 2), (255, 255, 255, 0))

    # 田字格位置
    positions = [
        (0, 0),  # 左上
        (width, 0),  # 右上
        (0, height),  # 左下
        (width, height),  # 右下
    ]

    for img, pos in zip(labeled_images, positions):
        new_img.paste(img, pos, img)

    new_img.save(output_path)
    print(f"拼接完成，保存到: {output_path}")


def compute_multi_turn_cumulative_distribution(model, correction=False):
    save_dir = Path("{0}".format(model))
    save_dir.mkdir(parents=True, exist_ok=True)

    f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]

    for file in f_list:
        if correction:
            input_path = save_dir / "{0}_correction_multi_turn.csv".format(file)
            if not input_path.exists():
                print(f"File not found: {input_path}")
                continue
            df = pd.read_csv(input_path)
        else:
            input_path = save_dir / "{0}_multi_turn.csv".format(file)
            if not input_path.exists():
                print(f"File not found: {input_path}")
                continue
            df = pd.read_csv(input_path)

        column_name = "turn count"

        # 原始统计横坐标仍然包含 -1
        fixed_index = [-1, 1, 2, 3, 4, 5, 6]

        counts = (
            df[column_name]
            .value_counts()
            .sort_index()
            .reindex(fixed_index, fill_value=0)
        )

        ratios = (
            df[column_name]
            .value_counts(normalize=True)
            .sort_index()
            .reindex(fixed_index, fill_value=0)
        )

        result = pd.DataFrame({"count": counts, "ratio": ratios})

        # 保持原来的累计逻辑
        y_values = result["ratio"].to_numpy(copy=True)

        for i in range(2, len(y_values)):
            y_values[i] = y_values[i] + y_values[i - 1]

        result["ratio"] = y_values

        # ============================
        # 修改点：
        # 1. 不显示 x = -1
        # 2. 剩余所有 x 坐标减 1
        # 3. y 值保持不变
        # ============================
        plot_result = result.drop(index=-1).copy()
        plot_x = plot_result.index - 1

        plt.figure(figsize=(10, 6))

        bars = plt.bar(plot_x.astype(str), plot_result["ratio"], color="#4C72B0")

        plt.bar_label(
            bars, labels=[f"{v:.2%}" for v in plot_result["ratio"]], padding=3
        )

        plt.xlabel("Turn count of converation")
        plt.ylabel("Percentage of incorrect answers")

        if correction:
            plt.title(f"The cumulative proportion of {file}, {model} correction.")
        else:
            plt.title(f"The cumulative proportion of {file}, {model}")

        max_ratio = plot_result["ratio"].max()

        if max_ratio > 0:
            plt.ylim(0, max_ratio * 1.15)
        else:
            plt.ylim(0, 1)

        plt.tight_layout()

        if correction:
            plt.savefig(
                save_dir
                / "{0} correction cumulative turn distribution.png".format(file),
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                save_dir / "{0} cumulative turn distribution.png".format(file),
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()

    if correction:
        images_list = [
            save_dir / "{0} correction cumulative turn distribution.png".format(file)
            for file in f_list
        ]
        merge_images_2x2(
            images_list, save_dir / "Cumulative distribution correction.png"
        )
    else:
        images_list = [
            save_dir / "{0} cumulative turn distribution.png".format(file)
            for file in f_list
        ]
        merge_images_2x2(images_list, save_dir / "Cumulative distribution.png")

    return


def draw_correction_easy(model):
    save_dir = Path("{0}".format(model))
    save_dir.mkdir(parents=True, exist_ok=True)
    f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]
    ACC_list = []
    for file in f_list:
        df = pd.read_csv(save_dir / "{0}_correction_easy.csv".format(file))
        multi_df = pd.read_csv(save_dir / "{0}_multi_turn.csv".format(file))
        correction_count = 0
        for i in range(len(df)):
            if df.iloc[i]["judge"] == True:
                correction_count += 1
        ACC_list.append([len(multi_df), len(df), correction_count])
    multi_initial = [x[0] for x in ACC_list]
    correction_total = [x[1] for x in ACC_list]
    correction_true = [x[2] for x in ACC_list]
    f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]

    x = np.arange(len(f_list))
    width = 0.25

    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(x - width, multi_initial, width, label="Multi_initial")
    bars2 = plt.bar(x, correction_total, width, label="Correction total")
    bars3 = plt.bar(x + width, correction_true, width, label="Correction true")

    # 在柱子上显示数值
    plt.bar_label(bars1, fmt="%.2f", padding=3)
    plt.bar_label(bars2, fmt="%.2f", padding=3)
    plt.bar_label(bars3, fmt="%.2f", padding=3)

    plt.xlabel("File Name")
    plt.ylabel("Number")
    plt.title("Correction count")
    plt.xticks(x, f_list, rotation=30, ha="right")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "Correction.png", dpi=300, bbox_inches="tight")
    # plt.show()
    return


def draw_correction_multi_turn(model):
    save_dir = Path("{0}".format(model))
    save_dir.mkdir(parents=True, exist_ok=True)
    f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]
    ACC_list = []
    for file in f_list:
        df = pd.read_csv(save_dir / "{0}_multi_turn.csv".format(file))
        multi_df = pd.read_csv(save_dir / "{0}_correction_multi_turn.csv".format(file))
        multi_total = len(df)  # 多轮的题目数量
        mistake_in_multi = len(multi_df)  # 多轮中被诱导然后答错题的数量
        correction_count = (multi_df["turn count"] != -1).sum()  # 成功改正的数量
        ACC_list.append(
            [multi_total, mistake_in_multi, correction_count]
        )  # 多轮的题目数量，被写入correction中的数量（也就是被质疑后答错的数量），以及correction改正的数量
    multi_initial = [x[0] for x in ACC_list]
    correction_total = [x[1] for x in ACC_list]
    correction_true = [x[2] for x in ACC_list]
    f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]

    x = np.arange(len(f_list))
    width = 0.25

    plt.figure(figsize=(10, 6))

    bars1 = plt.bar(x - width, multi_initial, width, label="Multi_initial")
    bars2 = plt.bar(x, correction_total, width, label="Correction total")
    bars3 = plt.bar(x + width, correction_true, width, label="Correction true")

    # 在柱子上显示数值
    plt.bar_label(bars1, fmt="%.2f", padding=3)
    plt.bar_label(bars2, fmt="%.2f", padding=3)
    plt.bar_label(bars3, fmt="%.2f", padding=3)

    plt.xlabel("File Name")
    plt.ylabel("Number")
    plt.title("Correction count")
    plt.xticks(x, f_list, rotation=30, ha="right")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_dir / "Correction_multi_turn.png", dpi=300, bbox_inches="tight")
    # plt.show()
    return


def compute_multi_turn_cumulative_distribution_modify(model, correction=False):
    save_dir = Path("{0}".format(model))
    save_dir.mkdir(parents=True, exist_ok=True)

    f_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]

    for file in f_list:
        if correction:
            input_path = save_dir / "{0}_correction_multi_turn_modify.csv".format(file)
            if not input_path.exists():
                print(f"File not found: {input_path}")
                continue
            df = pd.read_csv(input_path)
        else:
            input_path = save_dir / "{0}_multi_turn_modify.csv".format(file)
            if not input_path.exists():
                print(f"File not found: {input_path}")
                continue
            df = pd.read_csv(input_path)

        column_name = "turn count"

        # 原始统计横坐标仍然包含 -1
        fixed_index = [-1, 1, 2, 3, 4, 5, 6]

        counts = (
            df[column_name]
            .value_counts()
            .sort_index()
            .reindex(fixed_index, fill_value=0)
        )

        ratios = (
            df[column_name]
            .value_counts(normalize=True)
            .sort_index()
            .reindex(fixed_index, fill_value=0)
        )

        result = pd.DataFrame({"count": counts, "ratio": ratios})

        # 保持原来的累计逻辑
        y_values = result["ratio"].to_numpy(copy=True)

        for i in range(2, len(y_values)):
            y_values[i] = y_values[i] + y_values[i - 1]

        result["ratio"] = y_values

        # ============================
        # 修改点：
        # 1. 不显示 x = -1
        # 2. 剩余所有 x 坐标减 1
        # 3. y 值保持不变
        # ============================
        plot_result = result.drop(index=-1).copy()
        plot_x = plot_result.index - 1

        plt.figure(figsize=(10, 6))

        bars = plt.bar(plot_x.astype(str), plot_result["ratio"], color="#4C72B0")

        plt.bar_label(
            bars, labels=[f"{v:.2%}" for v in plot_result["ratio"]], padding=3
        )

        plt.xlabel("Turn count of converation")
        plt.ylabel("Percentage of incorrect answers")

        if correction:
            plt.title(f"The cumulative proportion of {file}, {model} correction.")
        else:
            plt.title(f"The cumulative proportion of {file}, {model}")

        max_ratio = plot_result["ratio"].max()

        if max_ratio > 0:
            plt.ylim(0, max_ratio * 1.15)
        else:
            plt.ylim(0, 1)

        plt.tight_layout()

        if correction:
            plt.savefig(
                save_dir
                / "{0} correction cumulative turn distribution modify.png".format(file),
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                save_dir / "{0} cumulative turn distribution modify.png".format(file),
                dpi=300,
                bbox_inches="tight",
            )

        plt.close()

    # if correction:
    #     images_list = [
    #         save_dir
    #         / "{0} correction cumulative turn distribution modify.png".format(file)
    #         for file in f_list
    #     ]
    #     merge_images_2x2(
    #         images_list, save_dir / "Cumulative distribution correction modify.png"
    #     )
    # else:
    #     images_list = [
    #         save_dir / "{0} cumulative turn distribution modify.png".format(file)
    #         for file in f_list
    #     ]
    #     merge_images_2x2(images_list, save_dir / "Cumulative distribution modify.png")

    return


def plot_turn_count_distribution(
    LLMS,
    file_list=["ARC-challenge", "math_train", "NQ-open", "plain_text"],
    base_dir=".",
    output_dir="turn_count_figures",
    turn_col="turn count",
    include_minus_one_in_denominator=True,
    correction=False,
):
    """
    统计不同 LLM 在不同 CSV 文件中的 turn count 累积分布，并绘制分组柱状图。

    横坐标:
    x = 0: 原数据 turn count = 1
    x = 1: 原数据 turn count = 1 + 2
    x = 2: 原数据 turn count = 1 + 2 + 3
    x = 3: 原数据 turn count = 1 + 2 + 3 + 4
    x = 4: 原数据 turn count = 1 + 2 + 3 + 4 + 5
    x = 5: 原数据 turn count = 1 + 2 + 3 + 4 + 5 + 6
    """

    base_dir = Path(base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_turns = [1, 2, 3, 4, 5, 6]
    x_labels = [0, 1, 2, 3, 4, 5]
    x = np.arange(len(x_labels))

    num_models = len(LLMS)
    bar_width = 0.8 / num_models

    for file in file_list:
        plt.figure(figsize=(10, 6))

        for i, llm in enumerate(LLMS):
            if not correction:
                csv_path = base_dir / llm / f"{file}_multi_turn_modify.csv"
            else:
                csv_path = base_dir / llm / f"{file}_correction_multi_turn_modify.csv"

            if not csv_path.exists():
                print(f"File not found: {csv_path}")
                percentages = [0] * len(original_turns)
            else:
                df = pd.read_csv(csv_path)

                if turn_col not in df.columns:
                    print(f"Column '{turn_col}' not found in {csv_path}")
                    percentages = [0] * len(original_turns)
                else:
                    turn_counts = df[turn_col]

                    if include_minus_one_in_denominator:
                        total = len(turn_counts)
                    else:
                        total = turn_counts.isin(original_turns).sum()

                    if total == 0:
                        percentages = [0] * len(original_turns)
                    else:
                        percentages = []

                        for idx in range(len(original_turns)):
                            current_turns = original_turns[: idx + 1]

                            count = turn_counts.isin(current_turns).sum()

                            percentage = count / total * 100
                            percentages.append(percentage)

            offset = (i - (num_models - 1) / 2) * bar_width

            bars = plt.bar(
                x + offset,
                percentages,
                width=bar_width,
                label=llm.split("/", 1)[1] if "/" in llm else llm,
            )

            plt.bar_label(
                bars, labels=[f"{p:.2f}%" for p in percentages], padding=3, fontsize=8
            )

        plt.xticks(x, x_labels)
        plt.xlabel("Turn of conversation")
        plt.ylabel("Percentage of incorrect answer")
        plt.title(f"Turn count cumulative distribution: {file}")
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()

        if not correction:
            save_path = output_dir / f"{file}_turn_count_cumulative_distribution.png"
        else:
            save_path = (
                output_dir / f"{file}_correction_turn_count_cumulative_distribution.png"
            )
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved figure: {save_path}")
    if not correction:
        images_list = [
            output_dir / "{0}_turn_count_cumulative_distribution.png".format(filename)
            for filename in file_list
        ]
    else:
        images_list = [
            output_dir
            / "{0}_correction_turn_count_cumulative_distribution.png".format(filename)
            for filename in file_list
        ]

    if not correction:
        merge_images_2x2(images_list, output_path=output_dir / "Four.png")
    else:
        merge_images_2x2(images_list, output_path=output_dir / "Four_correction.png")


def plot_judge_accuracy(LLMS, file_list, reasoning=False):
    """
    统计每个 LLM 目录下每个 CSV 中 initial_judge 和 misleading_judge 为 True 的比例，
    并为 file_list 中的每个 CSV 单独绘制一张柱状图。

    CSV 路径格式:
        ./LLM目录/{file}_misleading_judge.csv

    Parameters
    ----------
    LLMS : list[str]
        LLM 名称列表，每个名称对应一个目录。

    file_list : list[str]
        CSV 文件名前缀列表。
    """

    output_dir = Path("judge_accuracy_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    label_index = 0

    for file in file_list:
        llm_names = []
        initial_acc_list = []
        misleading_acc_list = []

        for llm in LLMS:
            llm_dir = llm.replace(":", "_")
            if reasoning == False:
                csv_path = Path(llm_dir) / f"{file}_misleading_judge_explaination.csv"
            else:
                csv_path = (
                    Path(llm_dir)
                    / f"{file}_mis_reasoning_modify_judge_explaination.csv"
                )

            if not csv_path.exists():
                print(f"[Warning] File not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            if "initial_judge" not in df.columns:
                raise ValueError(f"'initial_judge' not found in {csv_path}")

            if (
                "misleading_judge" not in df.columns
                and "mis_reasoning_judge" not in df.columns
            ):
                raise ValueError(
                    f"'misleading_judge' or 'mis_reasoning_judge' not found in {csv_path}"
                )

            initial_judge = (
                df["initial_judge"].astype(str).str.strip().str.lower().eq("true")
            )

            if reasoning == False:
                # misleading_judge = (
                #     df["misleading_judge"]
                #     .astype(str)
                #     .str.strip()
                #     .str.lower()
                #     .eq("true")
                # )
                misleading_judge = df["misleading_judge"].astype(
                    str
                ).str.strip().str.lower().eq("true") & df["initial_judge"].astype(
                    str
                ).str.strip().str.lower().eq(
                    "true"
                )
            else:
                # misleading_judge = (
                #     df["mis_reasoning_judge"]
                #     .astype(str)
                #     .str.strip()
                #     .str.lower()
                #     .eq("true")
                # )
                misleading_judge = df["mis_reasoning_judge"].astype(
                    str
                ).str.strip().str.lower().eq("true") & df["initial_judge"].astype(
                    str
                ).str.strip().str.lower().eq(
                    "true"
                )

            initial_acc = initial_judge.mean()
            misleading_acc = misleading_judge.mean()

            llm_names.append(llm)
            initial_acc_list.append(initial_acc)
            misleading_acc_list.append(misleading_acc)

        if len(llm_names) == 0:
            print(f"[Warning] No valid CSV found for {file}")
            continue

        x = np.arange(len(llm_names))
        width = 0.35

        plt.figure(figsize=(max(8, len(llm_names) * 1.2), 6))

        bars1 = plt.bar(x - width / 2, initial_acc_list, width, label="initial_acc")

        bars2 = plt.bar(
            x + width / 2, misleading_acc_list, width, label="misleading_acc"
        )

        plt.xlabel("LLM")
        plt.ylabel("Accuracy")

        if reasoning == False:
            plt.title(f"{label_list[label_index]} {file} background answer change")
        else:
            plt.title(f"{label_list[label_index]} {file} reasoning answer change")
        label_index += 1
        llm_names = [
            item.split("/", 1)[1] if "/" in item else item for item in llm_names
        ]
        plt.xticks(x, llm_names, ha="right", rotation=20)
        plt.ylim(0, 1.08)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.legend()

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.015,
                    f"{height * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()

        if reasoning == False:
            save_path = output_dir / f"{file}_judge_accuracy.png"
        else:
            save_path = output_dir / f"{file}_reasoning_judge_accuracy.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")

    entire_files = ["ARC-challenge"] + file_list

    suffix = "_reasoning_judge_accuracy.png" if reasoning else "_judge_accuracy.png"
    output_name = "Four_reasoning.png" if reasoning else "Four.png"

    images_list = [output_dir / f"{file}{suffix}" for file in entire_files]

    merge_images_2x2(images_list, output_dir / output_name)


def match_double_quote_answer(text, ground_truth):
    """
    捕获 text 中形如 ""A"" 或 ""1"" 的字符串。

    规则：
    1. 必须以两个双引号 "" 开头；
    2. 必须以两个双引号 "" 结尾；
    3. 中间只能有一个字母或数字；
    4. 如果 text 中有多个匹配项，返回 False；
    5. 如果只有一个匹配项，并且中间内容和 ground_truth 相同，返回 True；
    6. 否则返回 False。
    """
    # 专门为gemini-3-preview设计的，他的答案都是""A""这种格式
    text = str(text)
    ground_truth = str(ground_truth).strip()

    pattern = r'"([A-Za-z0-9])"'

    matches = re.findall(pattern, text)
    # print(matches, ground_truth, matches[0] == ground_truth)

    if len(matches) != 1:
        return False

    matched_answer = matches[0]

    return matched_answer == ground_truth


def evaluateAcc(file, model, reasoning=False):  # 专门计算多项选择数据集的acc
    save_dir = Path(model)

    df_initial = pd.read_csv(
        save_dir / f"{file}_initial_answer.csv",
        encoding="utf-8-sig",
    )

    if not reasoning:
        df_misleading = pd.read_csv(
            save_dir / f"{file}_misleading.csv",
            encoding="utf-8-sig",
        )
        misleading_answer_col = "answer"
    else:
        df_misleading = pd.read_csv(
            save_dir / f"{file}_mis_reasoning_modify.csv",
            encoding="utf-8-sig",
        )
        misleading_answer_col = "answer_under_mis_reasoning"

    length = min(len(df_initial), len(df_misleading))
    total_num = length

    def is_correct(ans, ground_truth):
        ans = str(ans).strip()
        ground_truth = str(ground_truth).strip()

        if ans == ground_truth:
            return True

        elif model == "gemini-3-flash-preview" and match_double_quote_answer(
            ans, ground_truth
        ):
            return True

        elif model == "x-ai/grok-4.20":
            ans_first = ans[0] if ans else ""
            if ans_first == ground_truth:
                return True

        return False

    correct_num_initial = 0
    initial_correct_list = []

    for i in range(length):
        ans = df_initial.iloc[i]["answer"]
        ground_truth = df_initial.iloc[i]["ground truth"]

        initial_correct = is_correct(ans, ground_truth)
        initial_correct_list.append(initial_correct)

        if initial_correct:
            correct_num_initial += 1

    accuracy_initial = correct_num_initial / total_num if total_num > 0 else 0

    correct_num_misleading = 0

    for i in range(length):
        if not initial_correct_list[i]:
            continue

        ans = df_misleading.iloc[i][misleading_answer_col]
        ground_truth = df_misleading.iloc[i]["ground truth"]

        if is_correct(ans, ground_truth):
            correct_num_misleading += 1

    accuracy_misleading = correct_num_misleading / total_num if total_num > 0 else 0

    print(
        "Model: {0}, initial_accuracy: {1}, misleading accuracy: {2}".format(
            model, accuracy_initial, accuracy_misleading
        )
    )

    return model, accuracy_initial, accuracy_misleading


LLMS = [
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "qwen/qwen3.6-flash",
    "deepseek/deepseek-v3.2",
    "openai/gpt-5",
    "meta-llama/llama-3.1-8b-instruct",
    "x-ai/grok-4.20",
]


def drawAccuracy(file, reasoning=False, reverse_cal=False):
    temp_list = LLMS
    final_list = []
    if not reverse_cal:
        for llm_id in range(len(temp_list)):
            temp = evaluateAcc(file, temp_list[llm_id], reasoning=reasoning)
            final_list.append(temp)
    else:
        for llm_id in range(len(temp_list)):
            save_dir = Path("{0}".format(temp_list[llm_id].replace(":", "_")))
            save_dir.mkdir(parents=True, exist_ok=True)
            c2f_acc = calculate_reverse_ratio(file, save_dir, temp_list[llm_id])
            f2c_acc = calculate_reverse_ratio(
                file, save_dir, temp_list[llm_id], reverse=True, reasoning=reasoning
            )
            temp = [temp_list[llm_id], c2f_acc, f2c_acc]
            final_list.append(temp)

    data = final_list
    names = [item[0].split("/", 1)[1] if "/" in item[0] else item[0] for item in data]
    acc_1 = [item[1] for item in data]
    acc_2 = [item[2] for item in data]

    x = np.arange(len(names))
    width = 0.3

    plt.figure(figsize=(10, 6))
    if not reverse_cal:
        bars1 = plt.bar(x - width / 2, acc_1, width=width, label="acc_initial")
        bars2 = plt.bar(x + width / 2, acc_2, width=width, label="acc_misleading")
    else:
        bars1 = plt.bar(x - width / 2, acc_1, width=width, label="acc_c2f")
        bars2 = plt.bar(x + width / 2, acc_2, width=width, label="acc_f2c")

    plt.xticks(x, names, ha="center", rotation=20)
    plt.ylim(0, 1)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    if reasoning == False:
        plt.title("(a) {0} {1} answer change".format(file, "background"))
    else:
        plt.title("(a) {0} {1} answer change".format(file, "reasoning"))
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
    if not reverse_cal:
        if reasoning == False:
            plt.savefig(
                "./judge_accuracy_figures/ARC-challenge_judge_accuracy.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                "./judge_accuracy_figures/ARC-challenge_reasoning_judge_accuracy.png",
                dpi=300,
                bbox_inches="tight",
            )
    else:
        if reasoning == False:
            plt.savefig(
                "./judge_accuracy_figures/ARC-challenge_judge_accuracy_reverse.png",
                dpi=300,
                bbox_inches="tight",
            )
        else:
            plt.savefig(
                "./judge_accuracy_figures/ARC-challenge_reasoning_judge_accuracy_reverse.png",
                dpi=300,
                bbox_inches="tight",
            )


def calculate_reverse_ratio(file, save_dir, model, reasoning=False, reverse=False):
    """
    统计 initial 和 misleading 之间答案正确性的变化比例。

    reverse=False:
        initial 正确，但 misleading 错误 的比例

    reverse=True:
        initial 错误，但 misleading 正确 的比例
    """

    save_dir = Path(save_dir)

    # 读取 initial 文件
    df_initial = pd.read_csv(
        save_dir / f"{file}_initial_answer.csv",
        encoding="utf-8-sig",
    )

    # 读取 misleading 文件
    if not reasoning:
        df_misleading = pd.read_csv(
            save_dir / f"{file}_misleading.csv",
            encoding="utf-8-sig",
        )
        misleading_answer_col = "answer"
    else:
        df_misleading = pd.read_csv(
            save_dir / f"{file}_mis_reasoning_modify.csv",
            encoding="utf-8-sig",
        )
        misleading_answer_col = "answer_under_mis_reasoning"

    def is_correct(ans, ground_truth):
        ans = str(ans).strip()
        ground_truth = str(ground_truth).strip()

        if ans == ground_truth:
            return True

        elif model == "gemini-3-flash-preview" and match_double_quote_answer(
            ans, ground_truth
        ):
            return True

        elif model == "x-ai/grok-4.20":
            ans_first = ans[0] if ans else ""
            if ans_first == ground_truth:
                return True

        return False

    # 检查必要列
    required_initial_cols = {"question", "answer", "ground truth"}
    required_misleading_cols = {"question", misleading_answer_col, "ground truth"}

    if not required_initial_cols.issubset(df_initial.columns):
        raise ValueError(
            f"df_initial 缺少必要列: {required_initial_cols - set(df_initial.columns)}"
        )

    if not required_misleading_cols.issubset(df_misleading.columns):
        raise ValueError(
            f"df_misleading 缺少必要列: {required_misleading_cols - set(df_misleading.columns)}"
        )

    # 用 question 对齐两个文件
    df_initial = df_initial.set_index("question")
    df_misleading = df_misleading.set_index("question")

    # 只保留两个文件共有的问题
    common_questions = df_initial.index.intersection(df_misleading.index)

    df_initial = df_initial.loc[common_questions]
    df_misleading = df_misleading.loc[common_questions]

    if len(common_questions) == 0:
        raise ValueError("两个文件中没有相同的 question，无法计算比例。")

    total = len(common_questions)

    count = 0

    for question in common_questions:
        initial_answer = df_initial.loc[question, "answer"]
        ground_truth = df_initial.loc[question, "ground truth"]

        misleading_answer = df_misleading.loc[question, misleading_answer_col]

        initial_correct = is_correct(initial_answer, ground_truth)
        misleading_correct = is_correct(misleading_answer, ground_truth)

        if not reverse:
            # initial 正确 -> misleading 错误
            if initial_correct and not misleading_correct:
                count += 1
        else:
            # initial 错误 -> misleading 正确
            if not initial_correct and misleading_correct:
                count += 1

    ratio = count / total

    return ratio


def plot_judge_acc_reverse(LLMS, file_list, reasoning=False):
    output_dir = Path("judge_accuracy_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    label_index = 0  # 给图标abcd

    for file in file_list:
        llm_names = []
        initial_acc_list = []
        misleading_acc_list = []

        for llm in LLMS:
            llm_dir = llm.replace(":", "_")
            if reasoning == False:
                csv_path = Path(llm_dir) / f"{file}_misleading_judge_explaination.csv"
            else:
                csv_path = (
                    Path(llm_dir)
                    / f"{file}_mis_reasoning_modify_judge_explaination.csv"
                )

            if not csv_path.exists():
                print(f"[Warning] File not found: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            if "initial_judge" not in df.columns:
                raise ValueError(f"'initial_judge' not found in {csv_path}")

            if (
                "misleading_judge" not in df.columns
                and "mis_reasoning_judge" not in df.columns
            ):
                raise ValueError(
                    f"'misleading_judge' or 'mis_reasoning_judge' not found in {csv_path}"
                )

            if reasoning == False:
                acc_c2f = df["initial_judge"].astype(str).str.strip().str.lower().eq(
                    "true"
                ) & df["misleading_judge"].astype(str).str.strip().str.lower().eq(
                    "false"
                )  # 先对后错
                acc_f2c = df["initial_judge"].astype(str).str.strip().str.lower().eq(
                    "false"
                ) & df["misleading_judge"].astype(str).str.strip().str.lower().eq(
                    "true"
                )
            else:
                acc_c2f = df["initial_judge"].astype(str).str.strip().str.lower().eq(
                    "true"
                ) & df["mis_reasoning_judge"].astype(str).str.strip().str.lower().eq(
                    "false"
                )
                acc_f2c = df["initial_judge"].astype(str).str.strip().str.lower().eq(
                    "false"
                ) & df["mis_reasoning_judge"].astype(str).str.strip().str.lower().eq(
                    "true"
                )

            initial_judge = acc_c2f
            misleading_judge = acc_f2c
            initial_acc = initial_judge.mean()
            misleading_acc = misleading_judge.mean()

            llm_names.append(llm)
            initial_acc_list.append(initial_acc)
            misleading_acc_list.append(misleading_acc)

        if len(llm_names) == 0:
            print(f"[Warning] No valid CSV found for {file}")
            continue

        x = np.arange(len(llm_names))
        width = 0.35

        plt.figure(figsize=(max(8, len(llm_names) * 1.2), 6))

        bars1 = plt.bar(x - width / 2, initial_acc_list, width, label="acc_c2f")

        bars2 = plt.bar(x + width / 2, misleading_acc_list, width, label="acc_f2c")

        plt.xlabel("LLM")
        plt.ylabel("Accuracy")
        if reasoning == False:
            plt.title(f"{label_list[label_index]} {file} background answer change")
        else:
            plt.title(f"{label_list[label_index]} {file} reasoning answer change")
        label_index += 1
        llm_names = [
            item.split("/", 1)[1] if "/" in item else item for item in llm_names
        ]
        plt.xticks(x, llm_names, ha="center", rotation=20)
        plt.ylim(0, 1.08)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.legend()

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.015,
                    f"{height * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()

        if reasoning == False:
            save_path = output_dir / f"{file}_judge_accuracy_reverse.png"
        else:
            save_path = output_dir / f"{file}_reasoning_judge_accuracy_reverse.png"
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")

    entire_files = ["ARC-challenge"] + file_list

    suffix = (
        "_reasoning_judge_accuracy_reverse.png"
        if reasoning
        else "_judge_accuracy_reverse.png"
    )
    output_name = "Four_reasoning_reverse.png" if reasoning else "Four_reverse.png"

    images_list = [output_dir / f"{file}{suffix}" for file in entire_files]

    merge_images_2x2(images_list, output_dir / output_name)

    return


def draw_acc_reverse(file_list):
    drawAccuracy("ARC-challenge", reverse_cal=True)  # 仅适用于多项选择
    drawAccuracy("ARC-challenge", reasoning=True, reverse_cal=True)
    plot_judge_acc_reverse(LLMS, four_list[1:4])
    plot_judge_acc_reverse(LLMS, four_list[1:4], reasoning=True)
    return


if __name__ == "__main__":
    four_list = ["ARC-challenge", "math_train", "NQ-open", "plain_text"]
    # draw_acc_reverse(four_list)
    # drawAccuracy("ARC-challenge")  # 仅适用于多项选择
    # drawAccuracy("ARC-challenge", reasoning=True)
    # plot_judge_accuracy(LLMS, four_list[1:4])
    plot_judge_accuracy(LLMS, four_list[1:4], reasoning=True)
    # plot_turn_count_distribution(LLMS)
    # plot_turn_count_distribution(LLMS, correction=True)
    for llm_id in range(len(LLMS)):
        save_dir = Path("{0}".format(LLMS[llm_id].replace(":", "_")))
        save_dir.mkdir(parents=True, exist_ok=True)
        llm = LLMS[llm_id]
        # judge_reasoning_acc("math_train", save_dir, LLMS[llm_id])
        # judge_reasoning_acc("NQ-open", save_dir, LLMS[llm_id])
        # judge_reasoning_acc("plain_text", save_dir, LLMS[llm_id])
        # judge_misleading("math_train", save_dir, LLMS[llm_id])
        # judge_misleading("plain_text", save_dir, LLMS[llm_id])
        # judge_misleading("NQ-open", save_dir, LLMS[llm_id])
        # compute_multi_turn_distribution(LLMS[llm_id])
        # compute_multi_turn_cumulative_distribution(LLMS[llm_id], correction=True)
        # compute_multi_turn_cumulative_distribution_modify(
        #     LLMS[llm_id], correction=True
        # )  # 测试修改prompt后的分布
    # draw_ACC_fourfile_onemodel(LLMS[0])
    # draw_reasoning_acc_one_model_four_files(LLMS[0])
    # draw_multi_turn_acc(LLMS[0])
    # draw_correction_multi_turn(LLMS[0])
    # compute_multi_turn_distribution(LLMS[0],correction=True)
    # draw_correction_easy(LLMS[0])
    print("Done at {0}".format(datetime.now()))
