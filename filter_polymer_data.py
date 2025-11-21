#!/usr/bin/env python3
"""
从JSON文件中筛选包含碳碳双键(C=C)或碳碳三键(C#C)的SMILES条目
这些化合物可能是高分子相关的（含有可聚合的双键或三键）
"""

import json
import re
import glob
from pathlib import Path


def extract_smiles(question_text):
    """从question文本中提取SMILES字符串，正确处理嵌套括号"""
    # 首先处理带外层括号的情况 (SMILES: ...)
    # 查找 (SMILES: 或 (SMILES 
    paren_patterns = [
        r'\(SMILES:\s*',      # (SMILES: 
        r'\(SMILES\s+',       # (SMILES 
    ]
    
    for pattern in paren_patterns:
        match = re.search(pattern, question_text, re.IGNORECASE)
        if match:
            start_pos = match.end()  # SMILES: 或 SMILES 之后的位置
            # 从start_pos开始，使用括号计数找到匹配的右括号
            paren_count = 1  # 已经有一个左括号（外层括号）
            i = start_pos
            
            while i < len(question_text) and paren_count > 0:
                if question_text[i] == '(':
                    paren_count += 1
                elif question_text[i] == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        # 找到匹配的右括号
                        smiles = question_text[start_pos:i].strip()
                        smiles = smiles.rstrip('.,;')
                        return smiles
                i += 1
    
    # 处理不带外层括号的情况 SMILES: ... 或 SMILES ...
    # 需要找到SMILES字符串，它可能包含括号，但会在遇到空格+单词、句号、逗号等时结束
    no_paren_patterns = [
        r'SMILES:\s*',      # SMILES: 
        r'SMILES\s+',       # SMILES 
    ]
    
    for pattern in no_paren_patterns:
        match = re.search(pattern, question_text, re.IGNORECASE)
        if match:
            start_pos = match.end()
            i = start_pos
            # 提取SMILES，直到遇到空格后跟字母（可能是下一个单词）或句子结束
            while i < len(question_text):
                char = question_text[i]
                # 如果遇到空格，检查后面是否跟字母（可能是下一个单词）
                if char == ' ':
                    # 检查空格后是否有字母（可能是新单词）
                    if i + 1 < len(question_text):
                        next_char = question_text[i + 1]
                        # 如果空格后跟大写字母，可能是新句子开始
                        if next_char.isupper() and i > start_pos + 5:  # 至少提取了几个字符
                            break
                        # 如果空格后跟小写字母，可能是新单词，但也可能是SMILES的一部分
                        # 检查是否是常见的SMILES字符
                        if next_char.islower() and next_char not in 'cnos()=[]':
                            # 可能是新单词，停止
                            break
                # 如果遇到句号、逗号、分号等，且不在括号内，可能是句子结束
                elif char in '.,;' and i > start_pos + 5:
                    # 检查是否在括号内
                    paren_count = 0
                    for j in range(start_pos, i):
                        if question_text[j] == '(':
                            paren_count += 1
                        elif question_text[j] == ')':
                            paren_count -= 1
                    if paren_count == 0:
                        break
                i += 1
            
            smiles = question_text[start_pos:i].strip()
            smiles = smiles.rstrip('.,;')
            # 移除末尾可能的标点符号（如果SMILES本身不包含这些）
            # 但不能移除括号内的内容
            return smiles if smiles else None
    
    return None


def has_cc_double_or_triple_bond(smiles):
    """
    检查SMILES是否包含碳碳双键(C=C)或碳碳三键(C#C)
    """
    if not smiles:
        return False
    
    # 1. 检查碳碳三键 C#C
    if re.search(r'[Cc]#[Cc]', smiles):
        return True
    
    # 2. 检查碳碳双键 C=C
    # 查找所有 = 的位置
    i = 0
    while i < len(smiles):
        if smiles[i] == '=':
            # 检查前后字符
            if i > 0 and i < len(smiles) - 1:
                before = smiles[i-1]
                after = smiles[i+1]
                
                # 检查是否是 C=C
                if before.upper() == 'C' and after.upper() == 'C':
                    # 排除 C(=O) 等情况
                    if i > 1 and smiles[i-2] == '(':
                        # 可能是 C(=O)，检查括号后的原子
                        if after.upper() in ['O', 'N', 'S', 'P', 'F']:
                            i += 1
                            continue
                    # 检查是否是 /C=C/ 或 \C=C\ (立体化学标记)
                    if i > 1 and i < len(smiles) - 2:
                        if (smiles[i-2] in ['/', '\\'] and smiles[i+2] in ['/', '\\']):
                            return True
                    # 其他 C=C 情况
                    return True
                
                # 检查 [...]C=C 或 C=C[...] (方括号内的原子)
                if before == ']' and after.upper() == 'C':
                    return True
                if before.upper() == 'C' and after == '[':
                    return True
        
        i += 1
    
    return False


def filter_json_files(input_pattern, output_suffix='_polymer'):
    """筛选JSON文件中包含碳碳双键或三键SMILES的条目"""
    json_files = glob.glob(input_pattern)
    
    if not json_files:
        print(f"未找到匹配 {input_pattern} 的文件")
        return
    
    for json_file in json_files:
        print(f"\n处理文件: {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  错误: 无法读取文件 {json_file}: {e}")
            continue
        
        if not isinstance(data, list):
            print(f"  警告: {json_file} 不是列表格式，跳过")
            continue
        
        filtered_data = []
        stats = {
            'total': len(data),
            'with_smiles': 0,
            'with_cc_bond': 0,
            'double_bond': 0,
            'triple_bond': 0,
            'no_smiles': 0
        }
        
        for item in data:
            if not isinstance(item, dict) or 'question' not in item:
                continue
            
            question = item.get('question', '')
            smiles = extract_smiles(question)
            
            if smiles:
                stats['with_smiles'] += 1
                has_bond = has_cc_double_or_triple_bond(smiles)
                
                if has_bond:
                    stats['with_cc_bond'] += 1
                    
                    # 检查是双键还是三键
                    if '#' in smiles:
                        stats['triple_bond'] += 1
                    if '=' in smiles:
                        stats['double_bond'] += 1
                    
                    item['extracted_smiles'] = smiles
                    filtered_data.append(item)
            else:
                stats['no_smiles'] += 1
        
        # 生成输出文件名
        input_path = Path(json_file)
        output_file = input_path.parent / f"{input_path.stem}{output_suffix}{input_path.suffix}"
        
        # 保存筛选后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f"  总条目数: {stats['total']}")
        print(f"  包含SMILES: {stats['with_smiles']}")
        print(f"  包含碳碳双/三键: {stats['with_cc_bond']}")
        print(f"    - 包含双键: {stats['double_bond']}")
        print(f"    - 包含三键: {stats['triple_bond']}")
        print(f"  无SMILES: {stats['no_smiles']}")
        print(f"  筛选结果保存到: {output_file}")


if __name__ == '__main__':
    filter_json_files('multimodal_*.json', output_suffix='_polymer')
    print("\n筛选完成！")
