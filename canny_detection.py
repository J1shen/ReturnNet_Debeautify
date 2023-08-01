import os
import cv2

# 原始图像所在目录
input_dir = 'datasets/makeup_data/original_400_400'
# 存储canny图像的目录
output_dir = 'datasets/makeup_data/original_400_canny'

# 如果存储canny图像的目录不存在，则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历原始图像所在目录
for file_name in os.listdir(input_dir):
    # 如果不是图像文件，则跳过
    if not file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        continue

    # 读取原始图像
    img = cv2.imread(os.path.join(input_dir, file_name))

    # 提取canny图像
    edges = cv2.Canny(img, 100, 200)

    # 存储canny图像
    cv2.imwrite(os.path.join(output_dir, file_name[:-4] + '_canny' + file_name[-4:]), edges)