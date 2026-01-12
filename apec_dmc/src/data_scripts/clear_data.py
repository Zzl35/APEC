import os
import time


def delete_files_in_folder(folder_path, condition_func):
    """
    删除符合条件的所有文件
    :param folder_path: 目标文件夹路径
    :param condition_func: 判断文件是否符合条件的函数
    """
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            
            # 如果文件符合条件，删除文件
            if condition_func(file_path):
                try:
                    os.remove(file_path)
                    print(f"删除文件: {file_path}")
                except Exception as e:
                    print(f"删除文件 {file_path} 时出错: {e}")

def example_condition(file_path):
    iteration = int(file_path.split('_')[-2]) 
    if iteration >= 1000:
        return True
    else:
        return False


# 示例使用：删除文件夹 "/path/to/folder" 下所有符合条件的文件
folder_path = '/home/ubuntu/duxinghao/APEC/apec_dmc/src/buffer/dmc/cheetah_run/2/buffer/irl'
delete_files_in_folder(folder_path, example_condition)
