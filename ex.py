# 模拟一段日志字符串
dummy_log = """
2024-01-01 12:00:00,000 - mmseg - INFO - Iter [2000/2000] eta: 0:00:00
+-------+-------+-------+
|  aAcc |  mIoU |  mAcc |
+-------+-------+-------+
| 92.07 | 70.37 | 83.66 |
+-------+-------+-------+
"""

def parse_mmseg_table(log_output):
    lines = log_output.strip().split('\n')
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if '|' in line and 'mIoU' in line and 'aAcc' in line:
            value_row_index = i + 2
            if value_row_index < len(lines):
                val_line = lines[value_row_index]
                parts = val_line.split('|')
                if len(parts) >= 3:
                    try:
                        return float(parts[2].strip())
                    except ValueError:
                        continue
    return 0.0

# 运行测试
result = parse_mmseg_table(dummy_log)
print(f"提取结果: {result}") 
# 应该输出: 提取结果: 70.37