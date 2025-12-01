#!/usr/bin/env python3
"""
将multimodal_cot_multimodal_full_polymer.json重构为新的格式
"""

import json
import re


def extract_keywords_from_solution(solution):
    """
    从solution中提取关键词/要点
    提取Step 1和Step 2之间的内容，Step 2和Step 3之间的内容，等等
    实际上就是提取每个Step的完整内容（从Step N: 到下一个Step之前）
    """
    keywords = []
    
    if not solution:
        return keywords
    
    # 移除开头的"Step-by-step solution"等前缀
    solution = re.sub(r'^Step-by-step\s+solution[:\s]*', '', solution, flags=re.IGNORECASE)
    solution = solution.strip()
    
    # 找到所有Step的位置
    # 匹配 "Step N:" 或数字编号 "N." (如 "1.", "2.") 的模式
    step_pattern = r'(?:Step\s+\d+:|^\d+\.\s+|\n\d+\.\s+)'
    matches = list(re.finditer(step_pattern, solution, re.IGNORECASE | re.MULTILINE))
    
    if not matches:
        # 如果没有找到Step格式，尝试提取段落
        paragraphs = solution.split('\n\n')
        for para in paragraphs:
            para = para.strip()
            # 跳过包含"Conclusion"、"Summary"等的段落
            if para and len(para) > 20 and not re.search(r'^(Conclusion|Summary|Final|In\s+summary)', para, re.IGNORECASE):
                keywords.append(para)
                if len(keywords) >= 10:
                    break
        return keywords
    
    # 提取每个Step的内容
    for i, match in enumerate(matches):
        start_pos = match.end()  # Step N: 或 N. 之后的位置
        
        # 确定结束位置
        if i < len(matches) - 1:
            # 还有下一个Step，提取到下一个Step之前
            end_pos = matches[i + 1].start()
        else:
            # 最后一个Step，检查是否有Conclusion等
            # 查找Conclusion、Summary等关键词的位置
            conclusion_match = re.search(r'\n(Conclusion|Summary|Final|In\s+summary)[:\s]', solution[start_pos:], re.IGNORECASE)
            if conclusion_match:
                end_pos = start_pos + conclusion_match.start()
            else:
                end_pos = len(solution)
        
        # 提取Step的内容
        step_content = solution[start_pos:end_pos].strip()
        
        # 清理内容：移除多余的空白和换行
        step_content = re.sub(r'\s+', ' ', step_content)  # 将多个空白字符替换为单个空格
        step_content = step_content.strip()
        
        # 移除开头的"**"等markdown标记
        step_content = re.sub(r'^\*\*[^*]+\*\*:\s*', '', step_content)
        step_content = step_content.strip()
        
        # 移除开头的数字和点（如果还有残留）
        step_content = re.sub(r'^\d+\.\s*', '', step_content)
        step_content = step_content.strip()
        
        if step_content and len(step_content) > 10:
            keywords.append(step_content)
    
    return keywords


def transform_data(input_file, output_file):
    """将数据转换为新格式"""
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
    
    transformed_data = []
    
    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        
        # 提取关键词
        solution = item.get('solution', '')
        keywords = extract_keywords_from_solution(solution)
        
        # Path直接使用image字段的值
        image_path = item.get('image', '')
        
        # 构建新格式的数据
        new_item = {
            "id": idx,
            "Type": "多模态分析",
            "Topic": "功能高分子基础知识",
            "mm": True,
            "Path": image_path,  # 直接使用image字段
            "Question": item.get('question', ''),
            "Answer": solution,
            "Keywords": keywords
        }
        
        transformed_data.append(new_item)
    
    # 保存转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=2)
    
    print(f"  转换完成: {len(transformed_data)} 条数据")
    print(f"  保存到: {output_file}")


if __name__ == '__main__':
    input_file = 'multimodal_cot_multimodal_full_polymer.json'
    output_file = 'multimodal_cot_multimodal_full_polymer_transformed.json'
    transform_data(input_file, output_file)
    print("\n转换完成！")

