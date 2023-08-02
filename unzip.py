import os
import zipfile

# 压缩包所在目录
dir_path = './datasets'

# 遍历目录下的所有文件
for file_name in os.listdir(dir_path):
    # 如果是zip文件，则解压
    #if file_name.endswith('.zip'):
    if file_name == 'makeup_data.zip':
        file_path = os.path.join(dir_path, file_name)
        # 创建解压目录
        extract_path = os.path.join(dir_path, file_name[:-4])
        os.makedirs(extract_path, exist_ok=True)
        # 解压
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f'{file_name} 解压完成')