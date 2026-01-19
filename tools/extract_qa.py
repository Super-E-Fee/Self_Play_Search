import re
import torch


def extract_idx(data, tokenizer):
    question_list = []
    answer_list = []
    info_list = []
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem

        prompt_ids = data_item

        # prompt_length = prompt_ids.shape[-1]

        # decode
        sequences = prompt_ids
        sequences_str = tokenizer.decode(sequences)
        q_str, a_str, info_str = extract_qa(sequences_str)
        question_list.append(q_str)
        answer_list.append(a_str)
        info_list.append(info_str)
    
    return question_list, answer_list, info_list


def extract_qa(solution_str):
    """Extract the equation from the solution string."""

    qa_pattern = r'<proposed_qa>(.*?)</proposed_qa>'
    match = re.finditer(qa_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # 这时候其实是我的prompt里的例子
    if len(matches) <= 4:
        return ('', '', '')
    # 最后一个是模型形成的最终答案
    final_str = matches[-1].group(1).strip()
    # If there are 2 or more matches, return the last one
    """根据#*#标记切分字符串"""
    
    parts = final_str.split('#*#', 1)
    if len(parts) > 1:
        q_str = parts[0]
        a_str = parts[1]
        info_pattern = r'<information>(.*?)</information>'
        info_match = re.finditer(info_pattern, solution_str, re.DOTALL)
        info_matches = list(info_match)[1:]
        info_str =''
        for info in info_matches:
            info_str = info.group(1).strip() + info_str # 越靠后的的搜索结果越接近出题时的重点
        return q_str, a_str, info_str
    else:
        return ('', '', '')
