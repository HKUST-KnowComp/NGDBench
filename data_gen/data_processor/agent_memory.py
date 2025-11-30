import json
import re
from datetime import datetime, timedelta
import random

data_path = "../gnd_dataset/agent_memory/conflict1.json"
output_path = "../gnd_dataset/agent_memory/conflict1_modified.json"

def process_context(context_text):
    """
    将 context 中的 '\n数字.' 替换为 '\nRecord_time+时间戳.'
    确保时间戳按照原数字顺序递增
    """
    # 找到所有 \n数字. 的模式
    pattern = r'\n(\d+)\.'
    matches = list(re.finditer(pattern, context_text))
    
    if not matches:
        return context_text
    
    # 提取所有数字并排序
    numbers = sorted([int(match.group(1)) for match in matches])
    
    # 生成时间戳映射：数字越小，时间越早
    # 起始时间为 2024-01-01 00:00:00
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    time_mapping = {}
    
    for i, num in enumerate(numbers):
        # 每条记录间隔 1-48 小时（随机），让时间跨度可以有几天
        random_hours = random.randint(1, 48)
        random_minutes = random.randint(0, 59)
        current_time = base_time + timedelta(hours=i * 12 + random_hours, minutes=random_minutes)
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
        time_mapping[num] = time_str
    
    # 替换所有匹配项
    def replace_func(match):
        num = int(match.group(1))
        time_str = time_mapping[num]
        return f'\nRecord_time: {time_str}.'
    
    result = re.sub(pattern, replace_func, context_text)
    return result

def main():
    print("开始处理 JSON 文件...")
    
    # 读取 JSON 文件
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"读取完成，开始修改 context 字段...")
    
    # 修改 context 字段
    if 'context' in data:
        original_context = data['context']
        data['context'] = process_context(original_context)
        print("Context 字段已修改")
    
    # 保存修改后的文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！文件已保存到: {output_path}")
    
    # 显示一个示例
    print("\n修改后的 context 前500个字符:")
    print(data['context'][:500])

if __name__ == "__main__":
    main()
