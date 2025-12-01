#!/usr/bin/env python3
"""
从multimodal_cot_multimodal_full_polymer.json中筛选出包含碳碳双键(C=C)或三键(C#C)的条目
"""

import json
import re
from pathlib import Path


def has_cc_double_bond(smiles):
    """
    检查SMILES是否包含碳碳双键(C=C)
    排除 C=O, C=N 等非碳碳双键
    """
    if not smiles:
        return False
    
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


def has_cc_triple_bond(smiles):
    """检查SMILES是否包含碳碳三键(C#C)"""
    if not smiles:
        return False
    return re.search(r'[Cc]#[Cc]', smiles) is not None


def filter_double_or_triple_bond(input_file, output_file=None):
    """筛选出包含碳碳双键或三键的条目"""
    if output_file is None:
        output_file = input_file
    
    print(f"读取文件: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取文件 {input_file}: {e}")
        return
    
    if not isinstance(data, list):
        print(f"警告: {input_file} 不是列表格式")
        return
    
    filtered_data = []
    stats = {
        'total': len(data),
        'with_double_bond': 0,
        'with_triple_bond': 0,
        'with_both': 0,
        'no_bond': 0
    }
    
    for item in data:
        if not isinstance(item, dict):
            continue
        
        smiles = item.get('extracted_smiles', '')
        if not smiles:
            stats['no_bond'] += 1
            continue
        
        # 检查是否包含双键或三键
        has_double = has_cc_double_bond(smiles)
        has_triple = has_cc_triple_bond(smiles)
        
        if has_double or has_triple:
            if has_double and has_triple:
                stats['with_both'] += 1
            elif has_double:
                stats['with_double_bond'] += 1
            else:
                stats['with_triple_bond'] += 1
            filtered_data.append(item)
        else:
            stats['no_bond'] += 1
    
    # 保存筛选后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"  总条目数: {stats['total']}")
    print(f"  包含碳碳双键: {stats['with_double_bond']}")
    print(f"  包含碳碳三键: {stats['with_triple_bond']}")
    print(f"  同时包含双键和三键: {stats['with_both']}")
    print(f"  无键: {stats['no_bond']}")
    print(f"  筛选后条目数: {len(filtered_data)}")
    print(f"  筛选结果保存到: {output_file}")


if __name__ == '__main__':
    input_file = 'multimodal_cot_multimodal_full_polymer.json'
    filter_double_or_triple_bond(input_file)
    print("\n筛选完成！")

