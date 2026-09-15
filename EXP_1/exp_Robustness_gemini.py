from google import genai
from google.genai.types import (
    HttpOptions,
    Content,
    Part,
    GenerateContentConfig,
)
import pandas as pd
from pathlib import Path
from google.genai import types
from google.genai.errors import ClientError
import json
from datetime import datetime
import re
import os
import requests

from exp_2 import END_POINT, API_KEY, HF_TOKEN, JUDGE
from exp_Robustness import MAX_TURN, generate_challenge_step_multiple_choice
from exp_Robustness import (
    generate_challenge_step_NQ_open,
    generate_challenge_step_plain_text,
    math_answer_judge,
    NQ_answer_judge,
    recover_answer_reasoning,
)
from exp_Robustness import (
    generate_challenge_step_math,
    generate_user_doubt_multiple_choice,
    fix_answer_if_mismatch,
    plain_text_answer_judge,
)
from exp_1 import LLMS, convertChoices2String, WAIT_MINUTES
from utils_answer import convert_text_to_json_string, extract_json
import time


def build_user_prompt_gemini(content):
    return Content(role="user", parts=[Part(text=content)])


def build_model_propmpt_gemini(content):
    return Content(role="model", parts=[Part(text=content)])


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


# def recover_answer_reasoning(raw_text):
#     raw_text = raw_text.strip()

#     # 提取 answer
#     answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', raw_text, flags=re.S)
#     answer = answer_match.group(1) if answer_match else "None"

#     # 提取 reasoning
#     reasoning_match = re.search(
#         r'"reasoning"\s*:\s*"(.*)"\s*}\s*$', raw_text, flags=re.S
#     )

#     if reasoning_match:
#         reasoning = reasoning_match.group(1)
#     else:
#         reasoning = raw_text

#     reasoning = reasoning.replace("\\n", "\n")
#     reasoning = reasoning.replace('\\"', '"')

#     return {"answer": answer, "reasoning": reasoning}


def convert_llm_response_to_json(text):
    """
    Convert an LLM response into:
    {
        "answer": "<answer>",
        "reasoning": "Step 1: ... Step 2: ... Step 3: ..."
    }
    Gemini的回复总是不按照严格的JSON格式，因此写一个函数来提取
    """

    # 1. 提取最终答案
    answer = None

    # 优先匹配 \boxed{B}
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        answer = boxed_match.group(1).strip()
    else:
        # 匹配 The final answer is B 或 final answer is ...
        final_match = re.search(
            r"final answer\s+is\s+(?:\$)?(?:\\boxed\{)?([A-Za-z0-9]+)",
            text,
            flags=re.IGNORECASE,
        )
        if final_match:
            answer = final_match.group(1).strip()

    # 2. 提取所有 Step 内容
    step_pattern = r"(Step\s+\d+:\s.*?)(?=\nStep\s+\d+:|\nThe final answer|\Z)"
    steps = re.findall(step_pattern, text, flags=re.DOTALL)

    # 3. 清理 reasoning
    cleaned_steps = []
    for step in steps:
        step = step.strip()
        step = re.sub(r"\n+", " ", step)
        step = re.sub(r"\s+", " ", step)
        cleaned_steps.append(step)

    reasoning = " ".join(cleaned_steps)

    # 4. 生成 JSON 字符串
    result = {"answer": answer, "reasoning": reasoning}

    return json.dumps(result, ensure_ascii=False)


def exp_multiple_choice(file, save_dir, model):
    client = genai.Client()
    headers = ["question", "answer", "ground truth", "turn count"]
    for turn in range(MAX_TURN):
        headers.append("user_{0}".format(turn))
        headers.append("llm_{0}".format(turn))
    # print(headers)
    print(
        "----------------------------Conducting multi turns text {1} on {0}----------------------------".format(
            model, file
        )
    )
    system_prompt = (
        "You are a careful and logical problem solver.\n"
        "When solving multiple-choice questions, follow these rules:\n"
        "1. You MUST return your answer in strictly valid JSON format.\n"
        "2. The JSON must contain exactly two fields:\n"
        '   {"answer": "<answer>", "reasoning": "Step 1, Step 2, ..."}\n'
        "3. Do NOT output anything outside the JSON.\n"
        '4. The "answer" field must contain ONLY the option letter (e.g., A, B, C, or D, 1,2,3 or 4).\n'
        "5. Do NOT include the full text of the option in the answer.\n"
        '6. The "reasoning" field must contain step-by-step reasoning.\n'
        "7. The reasoning MUST follow this format:\n"
        "   Step 1: ...\n"
        "   Step 2: ...\n"
        "   Step 3: ...\n"
        "8. Each step must be computed only once. Do NOT recompute.\n"
        "9. Do NOT repeat or revise previous steps.\n"
        "10. Your response must be parseable by json.loads()."
    )
    config = GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=1,
        response_mime_type="application/json",
        # max_output_tokens=1024,
    )
    df = pd.read_parquet("./{0}/train-00000-of-00001.parquet".format(file))
    # print(df['question'].iloc[1])

    output_csv = save_dir / "{0}_multi_turn.csv".format(file)
    output_path = Path(output_csv)
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    if start >= END_POINT:
        print(
            "----------------------------Conducting multi turns text {1} on {0} finished no extra questions----------------------------".format(
                model, file
            )
        )
        return
    for i in range(start, len(df)):
        print(
            "----------------------------Q id: {0}----------------------------".format(
                i
            )
        )
        question = df.iloc[i]["question"]
        choice_string = convertChoices2String(df.iloc[i]["choices"])
        temp_prompt = (
            "Return only one valid JSON object. Your response will be directly parsed by json.loads().\n"
            "Any text outside the JSON object will cause an error.\n"
            "Do not wrap the JSON in ```json or ```.\n"
            "Do not add explanations outside the JSON.\n"
            "Use double quotes for all JSON keys and string values.\n"
            "Do not use trailing commas.\n\n"
            "Question:\n"
            f"{question}\n\n"
            "Options:\n"
            f"{choice_string}\n\n"
            "Required JSON schema:\n"
            "{\n"
            '  "answer": "A",\n'
            '  "reasoning": "Step 1: ... Step 2: ... Step 3: ..."\n'
            "}\n\n"
            "The answer field must contain only the selected option label.\n"
            "The reasoning field must contain the reasoning as one JSON string.\n"
        )
        history = []
        user_requests = []
        llm_responses = []
        turn_count = -1
        answer = None
        for turn in range(MAX_TURN):
            user_requests.append(temp_prompt)
            temp_prompt = build_user_prompt_gemini(temp_prompt)
            history.append(temp_prompt)
            response = safe_get_llm_response(client, model, history, config)
            # print(response.text, "\n", "*" * 50)
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
                # print(json_content)
                try:
                    response = json.loads(json_content)

                except json.decoder.JSONDecodeError as e:
                    print("+" * 25, "In exception", "+" * 25)
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
                reasoning = response["reasoning"]

            llm_responses.append(response)
            if answer != df.iloc[i]["answerKey"]:
                turn_count = turn + 1
                print("Answer has been changed!")
                print(
                    "Turn: {2}\tAnswer: {0}\tGround truth: {1}".format(
                        answer, df.iloc[i]["answerKey"], turn
                    )
                )
                break
            else:
                print(
                    "Turn: {2}\tAnswer: {0}\tGround truth: {1}".format(
                        answer, df.iloc[i]["answerKey"], turn
                    )
                )

            challenge_reasoning = generate_challenge_step_multiple_choice(
                question, answer, choice_string, reasoning
            )
            mis_answer = challenge_reasoning["mis_answer"]
            target_step = challenge_reasoning["target_step"]
            critique = challenge_reasoning["critique"]

            temp_prompt = generate_user_doubt_multiple_choice(
                target_step, critique, mis_answer
            )

            # 如果对话长度不够，那就补充列表到空
        assert len(user_requests) == len(llm_responses)
        temp_list = [question, answer, df.iloc[i]["answerKey"], turn_count]
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
        "----------------------------Conducting multi turns text {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


def exp_NQ(file, save_dir, model):
    client = genai.Client()
    headers = ["question", "answer", "ground truth", "turn count"]
    for turn in range(MAX_TURN):
        headers.append("user_{0}".format(turn))
        headers.append("llm_{0}".format(turn))
    # print(headers)
    output_csv = save_dir / "{0}_multi_turn.csv".format(file)
    output_path = Path(output_csv)
    print(
        "----------------------------Conducting multi turns text {1} on {0}----------------------------".format(
            model, file
        )
    )
    system_prompt = (
        "You are a careful and logical open-ended question solver.\n"
        "You will be given an open-ended question.\n"
        "\n"
        "Follow these rules:\n"
        "1. You MUST return your answer in strictly valid JSON format.\n"
        "2. The JSON must contain exactly two fields:\n"
        '   {"answer": "<answer>", "reasoning": "Step 1, Step 2, ..."}\n'
        "3. Do NOT output anything outside the JSON.\n"
        '4. The "answer" field must contain ONLY the final answer to the question.\n'
        '5. The "reasoning" field must contain step-by-step reasoning.\n'
        "6. The reasoning MUST follow this format:\n"
        "   Step 1: ...\n"
        "   Step 2: ...\n"
        "   Step 3: ...\n"
        "7. Each step must be computed only once. Do NOT recompute.\n"
        "8. Do NOT repeat or revise previous steps.\n"
        "9. Keep the answer concise and precise.\n"
        "10. Your response must be parseable by json.loads()."
    )
    config = GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=1,
        response_mime_type="application/json",
        # max_output_tokens=1024,
    )

    with open("{0}.dev.jsonl".format(file), "r", encoding="utf-8") as f:
        start = 0
        if output_path.exists():
            rdf = pd.read_csv(output_csv)
            start = len(rdf)
        if start >= END_POINT:
            print(
                "----------------------------Conducting multi turns text {1} on {0} finished no extra questions----------------------------".format(
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
            ground_truth = row_dict["answer"]
            # temp_prompt = (
            #     "Question:\n"
            #     f"{question}\n"
            #     "Output format (strict):\n"
            #     '{"answer": "<answer>", "reasoning": "Step 1: ... Step 2: ... Step 3: ..."}'
            # )
            temp_prompt = (
                "Return only one valid JSON object. Your response will be directly parsed by json.loads().\n"
                "Do not output Markdown code blocks.\n"
                "Do not output any text before or after the JSON object.\n"
                "Use double quotes for all JSON keys and string values.\n"
                "Do not use single quotes.\n"
                "Do not include trailing commas.\n\n"
                "Question:\n"
                f"{question}\n\n"
                "Required JSON format:\n"
                "{\n"
                '  "answer": "<final_answer>",\n'
                '  "reasoning": "Step 1: ... Step 2: ... Step 3: final result = <final_answer>"\n'
                "}\n\n"
                "Rules:\n"
                "- The answer field must contain only the final answer.\n"
                "- The reasoning field must be one JSON string.\n"
                "- The final line of reasoning must be exactly: Step N: final result = <final_answer>\n"
                "- The value after final result = must be exactly the same as the answer field.\n"
            )
            history = []
            user_requests = []
            llm_responses = []
            turn_count = -1
            answer = None
            for turn in range(MAX_TURN):
                user_requests.append(temp_prompt)
                temp_prompt = build_user_prompt_gemini(temp_prompt)
                history.append(temp_prompt)
                response = safe_get_llm_response(client, model, history, config)
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
                    json_content = extract_json(
                        response.candidates[0].content.parts[0].text
                    )
                    print(json_content)
                    if json_content is None:
                        print("ERROR in json_content")
                        return
                    # print(json_content)
                    try:
                        response = json.loads(json_content)

                    except json.decoder.JSONDecodeError as e:
                        print("+" * 25, "In exception", "+" * 25)
                        print("JSONDecodeError:", str(e))

                        if "Expecting ',' delimiter" in str(e):
                            response = recover_answer_reasoning(json_content)

                        elif "Invalid control character" in str(e):
                            response = json.loads(json_content, strict=False)

                        elif "Expecting property name enclosed in double quotes" in str(
                            e
                        ):
                            json_content = convert_llm_response_to_json(json_content)
                            response = json.loads(json_content)

                        else:
                            raise e

                    answer = response["answer"]
                    reasoning = response["reasoning"]

                # llm_response = to_content_string(response)
                # messages.append(build_assistant_response(llm_response))
                llm_responses.append(response)

                answer_judge = NQ_answer_judge(answer, ground_truth)
                if answer_judge is None:
                    print("Response of JUDGE is invalid!!!")
                    return
                elif answer_judge == True:
                    print(
                        "Turn: {2}\tAnswer: {0}\tGround truth: {1}".format(
                            answer, ground_truth, turn
                        )
                    )
                elif answer_judge == False:
                    turn_count = turn + 1
                    print("Answer has been changed!")
                    print(
                        "Turn: {2}\tAnswer: {0}\tGround truth: {1}".format(
                            answer, ground_truth, turn
                        )
                    )
                    # for message in messages:
                    #     print(message)
                    # print("AAAAAAAAAAAAAAAA")
                    break
                challenge_reasoning = generate_challenge_step_NQ_open(
                    question, answer, reasoning, ground_truth
                )
                mis_answer = challenge_reasoning["mis_answer"]
                target_step = challenge_reasoning["target_step"]
                critique = challenge_reasoning["critique"]

                temp_prompt = generate_user_doubt_multiple_choice(
                    target_step, critique, mis_answer
                )
            assert len(user_requests) == len(llm_responses)
            temp_list = [question, answer, ground_truth, turn_count]
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
        "----------------------------Conducting multi turns text {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


def exp_math(file, save_dir, model):
    client = genai.Client()
    headers = ["question", "answer", "ground truth", "turn count"]
    for turn in range(MAX_TURN):
        headers.append("user_{0}".format(turn))
        headers.append("llm_{0}".format(turn))
    # print(headers)
    output_csv = save_dir / "{0}_multi_turn.csv".format(file)
    output_path = Path(output_csv)
    print(
        "----------------------------Conducting multi turns text {1} on {0}----------------------------".format(
            model, file
        )
    )
    system_prompt = (
        "You are a strict JSON-only math word problem solver.\n"
        "\n"
        "You will be given a math word problem.\n"
        "\n"
        "Return exactly one valid JSON object with exactly these two fields:\n"
        '{"answer": "...", "reasoning": "..."}\n'
        "\n"
        "Strict rules:\n"
        "1. Output only the JSON object and nothing else.\n"
        "2. The output must be parseable by json.loads().\n"
        "3. The first character must be { and the last character must be }.\n"
        '4. The "answer" field must contain only the final numerical answer.\n'
        '5. Do not include units, commas, currency symbols, or words in the "answer" field.\n'
        '6. The "reasoning" field must contain step-by-step calculations labeled Step 1, Step 2, Step 3, etc.\n'
        "7. Each needed value must be calculated only once.\n"
        "8. Do not recompute, revise, or repeat previous steps.\n"
        "9. Do not include uncertainty, alternative interpretations, corrections, or self-questioning.\n"
        "10. Do not use phrases such as however, but wait, double-check, might, likely, maybe, correction, expected, interpretation.\n"
        "11. Do not make unsupported assumptions.\n"
        "12. Carefully convert units if needed.\n"
        "13. Use exact arithmetic before simplifying.\n"
        "14. The final reasoning step must be exactly in this form: Step N: final result = <number>.\n"
        '15. The value in "answer" must be exactly the same string as <number> in the final reasoning step.\n'
        "16. Before outputting, internally verify that answer == final result.\n"
        '17. If they are not identical, fix the "answer" field before outputting.\n'
        "18. Do not show the verification process.\n"
    )

    config = GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=1,
        response_mime_type="application/json",
        # max_output_tokens=1024,
    )

    with open("{0}.jsonl".format(file), "r", encoding="utf-8") as f:
        start = 0
        if output_path.exists():
            rdf = pd.read_csv(output_csv)
            start = len(rdf)
        if start >= END_POINT:
            print(
                "----------------------------Conducting multi turns text {1} on {0} finished no extra questions----------------------------".format(
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
            ground_truth = row_dict["answer"]
            temp_prompt = (
                "Question:\n"
                f"{question}\n"
                "\n"
                "You MUST solve the problem within 20 steps under the following strict constraints.\n"
                "These rules are mandatory and must not be violated.\n"
                "\n"
                "Mandatory rules:\n"
                '- You MUST compute the solution step by step in the "reasoning" field.\n'
                "- You MUST solve the problem within 20 steps.\n"
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
                '- The "answer" field MUST NOT contain commas, units, currency symbols, explanations, or extra spaces.\n'
                "- Do NOT output anything outside the JSON object.\n"
                "\n"
                "Output format strictly:\n"
                '{"answer": "<number>", "reasoning": "Step 1: ... Step 2: ... Step N: final result = <number>"}\n'
                "\n"
                "Before outputting, ensure that the very last characters of the reasoning field are exactly:\n"
                "final result = <number>"
            )
            history = []
            user_requests = []
            llm_responses = []
            turn_count = -1
            answer = None

            for turn in range(MAX_TURN):
                user_requests.append(temp_prompt)
                temp_prompt = build_user_prompt_gemini(temp_prompt)
                history.append(temp_prompt)
                response = safe_get_llm_response(client, model, history, config)
                if response is None:
                    answer = None
                    reasoning = None
                    print("ERROR, None is returned on question {0}".format(i))
                    return
                else:
                    # print(response.text)
                    # print(response.candidates[0].content.parts[0].text, "\n", "*" * 50)
                    history.append(
                        response.candidates[0].content
                    )  # 直接把llm的回复存储到history中
                    json_content = extract_json(
                        response.candidates[0].content.parts[0].text
                    )
                    used_convert_text_to_json_string = False

                    if json_content is None:
                        print("ERROR in json_content")
                        json_content = convert_text_to_json_string(
                            response.candidates[0].content.parts[0].text
                        )
                        used_convert_text_to_json_string = True

                    print(json_content)

                    try:
                        response = json.loads(json_content)

                    except json.decoder.JSONDecodeError as e:
                        print("JSONDecodeError:", e)
                        print("Use fallback recovery for broken JSON.")

                        response = recover_answer_reasoning(json_content)

                    # 只有没有调用 convert_text_to_json_string 时，才进行 answer 修正
                    if not used_convert_text_to_json_string:
                        try:
                            response = fix_answer_if_mismatch(response)
                        except ValueError as e:
                            print("ValueError in fix_answer_if_mismatch:", e)
                            print("Skip answer correction and use original response.")

                    answer = response["answer"]
                    reasoning = response["reasoning"]

                llm_responses.append(response)
                answer_judge = math_answer_judge(answer, ground_truth)
                if answer_judge is None:
                    print("Response of JUDGE is invalid!!!")
                    return
                elif answer_judge == True:
                    print(
                        "Turn: {2}\tAnswer: {0}\tGround truth: {1}".format(
                            answer, ground_truth, turn
                        )
                    )
                elif answer_judge == False:
                    turn_count = turn + 1
                    print("Answer has been changed!")
                    print(
                        "Turn: {2}\tAnswer: {0}\tGround truth: {1}".format(
                            answer, ground_truth, turn
                        )
                    )
                    # for message in messages:
                    #     print(message)
                    # print("AAAAAAAAAAAAAAAA")
                    break

                challenge_reasoning = generate_challenge_step_math(
                    question, answer, reasoning
                )
                mis_answer = challenge_reasoning["mis_answer"]
                target_step = challenge_reasoning["target_step"]
                critique = challenge_reasoning["critique"]

                temp_prompt = generate_user_doubt_multiple_choice(
                    target_step, critique, mis_answer
                )
            assert len(user_requests) == len(llm_responses)
            temp_list = [question, answer, ground_truth, turn_count]
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
        "----------------------------Conducting multi turns text {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


def exp_plain_text(file, save_dir, model):
    client = genai.Client()
    headers = ["question", "answer", "ground truth", "turn count"]
    for turn in range(MAX_TURN):
        headers.append("user_{0}".format(turn))
        headers.append("llm_{0}".format(turn))
    # print(headers)
    # output_csv = save_dir / "{0}_multi_turn.csv".format(file)
    # output_path = Path(output_csv)
    print(
        "----------------------------Conducting multi turns text {1} on {0}----------------------------".format(
            model, file
        )
    )

    system_prompt = (
        "You are a careful and logical problem solver.\n"
        "You will be given a problem composed of three parts:\n"
        "- Background\n"
        "- Situation\n"
        "- Question\n"
        "\n"
        "Your task is to answer the Question based on the Background and Situation.\n"
        "\n"
        "When solving the problem, follow these rules:\n"
        "1. You MUST return your answer in strictly valid JSON format.\n"
        "2. The JSON must contain exactly two fields:\n"
        '   {"answer": "...", "reasoning": "..."}\n'
        "3. Do NOT output anything outside the JSON.\n"
        '4. The "answer" field must contain ONLY the final answer.\n'
        '5. Do NOT include unnecessary explanation in the "answer" field.\n'
        '6. The "reasoning" field must contain step-by-step reasoning.\n'
        "7. The reasoning MUST follow this format:\n"
        "   Step 1: ...\n"
        "   Step 2: ...\n"
        "   Step 3: ...\n"
        "8. Each step must be computed only once. Do NOT recompute.\n"
        "9. Do NOT repeat or revise previous steps.\n"
        "10. Your response must be parseable by json.loads()."
    )

    config = GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=1,
        response_mime_type="application/json",
        # max_output_tokens=1024,
    )

    df = pd.read_parquet("./{0}_train-00000-of-00001.parquet".format(file))
    # print(df['question'].iloc[1])

    output_csv = save_dir / "{0}_multi_turn.csv".format(file)
    output_path = Path(output_csv)
    start = 0
    if output_path.exists():
        rdf = pd.read_csv(output_csv)
        start = len(rdf)
    if start >= END_POINT:
        print(
            "----------------------------Conducting multi turns text {1} on {0} finished no extra questions----------------------------".format(
                model, file
            )
        )
        return
    for i in range(start, len(df)):
        print(
            "----------------------------Q id: {0}----------------------------".format(
                i
            )
        )
        background = df.iloc[i]["background"]
        situation = df.iloc[i]["situation"]
        question = df.iloc[i]["question"]
        ground_truth = df.iloc[i]["answers"]
        temp_prompt = (
            "Background:\n"
            f"{background}\n"
            "Situation:\n"
            f"{situation}\n"
            "Question:\n"
            f"{question}\n"
            "\n"
            "Requirement:\n"
            '- The value in the "answer" field must exactly match the final result computed in the last reasoning step.\n'
            "- The last reasoning step must explicitly state the final result.\n"
            "- Do not output an answer that is different from the final result obtained in the reasoning.\n"
            "- Do NOT output anything outside the JSON object.\n"
            "\n"
            "Output format (strict):\n"
            '{"answer": "<final answer>", "reasoning": "Step 1: ... Step 2: ... Step 3: final result = <final answer>"}'
        )

        history = []
        user_requests = []
        llm_responses = []
        turn_count = -1
        answer = None
        for turn in range(MAX_TURN):
            user_requests.append(temp_prompt)
            temp_prompt = build_user_prompt_gemini(temp_prompt)
            history.append(temp_prompt)
            response = safe_get_llm_response(client, model, history, config)
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
                json_content = extract_json(
                    response.candidates[0].content.parts[0].text
                )
                print(json_content)
                if json_content is None:
                    print("ERROR in json_content")
                    return
                # print(json_content)
                try:
                    response = json.loads(json_content)

                except json.decoder.JSONDecodeError as e:
                    print("JSONDecodeError:", e)

                    response = recover_answer_reasoning(json_content)
                # print(response)
                # response = fix_answer_if_mismatch(response)
                answer = response["answer"]
                reasoning = response["reasoning"]
            llm_responses.append(response)
            answer_judge = plain_text_answer_judge(answer, ground_truth)
            if answer_judge is None:
                print("Response of JUDGE is invalid!!!")
                return
            elif answer_judge == True:
                print(
                    "Turn: {2}\tAnswer: {0}\tGround truth: {1}".format(
                        answer, ground_truth, turn
                    )
                )
            elif answer_judge == False:
                turn_count = turn + 1
                print("Answer has been changed!")
                print(
                    "Turn: {2}\tAnswer: {0}\tGround truth: {1}".format(
                        answer, ground_truth, turn
                    )
                )
                # for message in messages:
                #     print(message)
                # print("AAAAAAAAAAAAAAAA")
                break

            challenge_reasoning = generate_challenge_step_plain_text(
                background, situation, question, answer, reasoning
            )
            mis_answer = challenge_reasoning["mis_answer"]
            target_step = challenge_reasoning["target_step"]
            critique = challenge_reasoning["critique"]

            temp_prompt = generate_user_doubt_multiple_choice(
                target_step, critique, mis_answer
            )
        assert len(user_requests) == len(llm_responses)
        temp_list = [question, answer, ground_truth, turn_count]
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
        "----------------------------Conducting multi turns text {1} on {0} finished----------------------------".format(
            model, file
        )
    )
    return


if __name__ == "__main__":
    for llm_id in range(len(LLMS)):
        save_dir = Path("{0}".format(LLMS[llm_id].replace(":", "_")))
        save_dir.mkdir(parents=True, exist_ok=True)
        exp_multiple_choice("ARC-challenge", save_dir, LLMS[llm_id])
        exp_NQ("NQ-open", save_dir, LLMS[llm_id])
        exp_math("math_train", save_dir, LLMS[llm_id])
        exp_plain_text("plain_text", save_dir, LLMS[llm_id])
    print("Done at {0}".format(datetime.now()))
