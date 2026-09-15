import re
import json

# def fix_answer_if_mismatch(response_json):
# 这个只能处理数字
#     """
#     Match all 'final result = <number>' patterns in response_json["reasoning"],
#     store them in order, use the last matched number, and replace response_json["answer"]
#     if it does not match the last final result.

#     Supports integers, decimals, negative numbers, and fractions.
#     """
#     answer = str(response_json.get("answer", "")).strip()
#     reasoning = str(response_json.get("reasoning", "")).strip()

#     matches = list(
#         re.finditer(
#             r"final result\s*=\s*(-?(?:\d+(?:\.\d+)?|\d+/\d+))",
#             reasoning,
#             flags=re.IGNORECASE,
#         )
#     )

#     if not matches:
#         print(response_json)
#         raise ValueError("Cannot find any 'final result = <number>' in reasoning.")

#     final_results = [m.group(1).strip() for m in matches]
#     # print(final_results)
#     last_final_result = final_results[-1]

#     if answer != last_final_result:
#         response_json["answer"] = last_final_result

#     return response_json


# def fix_answer_if_mismatch(response_json):
#     """
#     Match all 'final result = <value>' patterns in response_json["reasoning"],
#     where <value> can be:这个函数可以处理final result=后面的字符串
#     - number: 42, -3.5, 1/2
#     - word: more, less
#     - phrase: insufficient data, cup Z, option A

#     Use the last matched value to replace response_json["answer"] if needed.
#     """
#     answer = str(response_json.get("answer", "")).strip()
#     reasoning = str(response_json.get("reasoning", "")).strip()

#     matches = list(
#         re.finditer(
#             r"final result\s*=\s*([^\n\r]+)",
#             reasoning,
#             flags=re.IGNORECASE,
#         )
#     )

#     if not matches:
#         print(response_json)
#         raise ValueError("Cannot find any 'final result = <value>' in reasoning.")

#     final_results = [m.group(1).strip() for m in matches]

#     # 使用最后一个 final result，并去掉末尾常见标点
#     last_final_result = final_results[-1].rstrip(".。;；").strip()

#     if answer != last_final_result:
#         response_json["answer"] = last_final_result

#     return response_json


def fix_answer_if_mismatch(response_json):
    """
    获取 reasoning 中最后一个 'final result =' 后面的内容，
    并用该内容替换 response_json["answer"]。
    """
    answer = str(response_json.get("answer", "")).strip()
    reasoning = str(response_json.get("reasoning", "")).strip()

    # 找到所有 final result =
    matches = list(
        re.finditer(
            r"final result\s*=",
            reasoning,
            flags=re.IGNORECASE,
        )
    )

    if not matches:
        print(response_json)
        raise ValueError("Cannot find any 'final result =' in reasoning.")

    # 取最后一个 final result = 后面的全部内容
    last_match = matches[-1]
    last_final_result = reasoning[last_match.end() :].strip()

    # 去掉末尾常见标点
    last_final_result = last_final_result.rstrip(".。;；").strip()

    if answer != last_final_result:
        response_json["answer"] = last_final_result

    return response_json


def extract_json(text):
    # gemini返回文本总是带json的标志
    # 这个函数只提取返回内容中{}中的内容
    text = text.strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    return match.group(0)


def convert_text_to_json_string(raw_text):
    """
    从原始文本中提取 answer 字段后的数字，
    并提取 Reasoning 字段后的内容作为 reasoning。
    返回值是可以被 json.loads() 解析的 JSON 字符串。
    """

    # 1. 提取 answer 后面的数字
    answer_pattern = r'"?answer"?\s*[:：]\s*"?([+-]?(?:\d+(?:\.\d+)?|\.\d+|\d+/\d+))"?'
    answer_match = re.search(answer_pattern, raw_text, flags=re.IGNORECASE)

    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        answer = "None"

    # 2. 提取 Reasoning 后面的全部内容
    reasoning_pattern = r'"?reasoning"?\s*[:：]\s*"?([\s\S]*)'
    reasoning_match = re.search(reasoning_pattern, raw_text, flags=re.IGNORECASE)

    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

        # 如果末尾有 JSON 字符串的结束引号或大括号，简单清理一下
        reasoning = re.sub(r'"\s*}\s*$', "", reasoning).strip()
    else:
        reasoning = raw_text

    json_obj = {"answer": answer, "reasoning": reasoning}

    return json.dumps(json_obj, ensure_ascii=False)
