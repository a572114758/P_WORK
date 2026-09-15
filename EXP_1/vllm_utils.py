import subprocess

def convertChoices2String(choices):
    ans=''
    for i in range(len(choices['text'])):
        # print(choices['label'][i])
        if i == len(choices['text'])-1:
            ans+=(choices['label'][i] +': ' + choices['text'][i]+'.')
        else:
            ans+=(choices['label'][i] +': ' + choices['text'][i]+', ')
    return ans

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
    
def extract_answer_math_json(text):
    """
    从任意返回文本中提取形如
    {"answer": "...", "reasoning": "..."}
    的 JSON 对象。

    返回：
        1. 成功时：返回 Python 字典
        2. 失败时：返回 None
    """
    decoder = json.JSONDecoder()

    for i, ch in enumerate(text):
        if ch == '{':
            try:
                obj, end = decoder.raw_decode(text[i:])
                if isinstance(obj, dict) and "answer" in obj and "reasoning" in obj:
                    return obj
            except json.JSONDecodeError:
                continue

    return None


def get_gpu_free_memory(gpu_list=None):
    """
    Get free memory for GPUs in gpu_list.

    gpu_list: list of physical GPU ids, such as [2, 3].
              If None, all GPUs will be considered.
    """
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        encoding="utf-8",
    )

    gpu_memory = {}

    for line in result.strip().split("\n"):
        gpu_id, free_mem = line.split(",")
        gpu_id = int(gpu_id.strip())
        free_mem = int(free_mem.strip())

        if gpu_list is None or gpu_id in gpu_list:
            gpu_memory[gpu_id] = free_mem

    if not gpu_memory:
        raise RuntimeError("No available GPU found in the given gpu_list.")

    return gpu_memory


