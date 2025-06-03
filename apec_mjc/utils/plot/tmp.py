import pandas as pd
import numpy as np
import ast  # 用于将字符串列表转换为实际的列表

# 读取数据
df = pd.read_csv('/home/ubuntu/duxinghao/imitation_pref/eval/nonoise_baselines3/coverage/total_coverage_data.csv', index_col=0)

# 计算每列的均值和标准差
def avg_std_from_list(val_list):
    avg = np.mean(val_list)
    std = np.std(val_list)
    return avg, std

# 计算每个方法在每个任务上的均值和标准差
def calculate_performance_stats(col):
    task_mean_std = [avg_std_from_list([ast.literal_eval(x) for x in col])]
    return task_mean_std

# 计算任务内的均值和标准差
task_results = df.apply(lambda col: [f"{np.mean(ast.literal_eval(x)):.2f} ± {np.std(ast.literal_eval(x)):.2f}" for x in col])

# 计算每个方法在所有任务上的总体均值和标准差
def overall_avg_std_across_tasks(col):
    all_values = np.concatenate([ast.literal_eval(x) for x in col])
    avg = np.mean(all_values)
    std = np.std(all_values)
    return f"{avg:.2f}"

overall_results = df.apply(overall_avg_std_across_tasks)

# 创建结果表格
final_results = task_results.copy()

# 添加总体均值和标准差行
final_results.loc['Average'] = overall_results

# 打印结果
print(final_results)

# 将结果保存到 CSV 文件
final_results.to_csv('/home/ubuntu/duxinghao/imitation_pref/eval/nonoise_baselines3/coverage/total_coverage.csv')
