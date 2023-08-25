import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载图像
img = cv2.imread('test_result/test3.jpg', cv2.IMREAD_GRAYSCALE)

# 进行傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# 构造3D坐标轴
x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
z = magnitude_spectrum

# 颜色映射
colors = plt.cm.viridis(z/z.max())

# 绘制3D频谱图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z, facecolors=colors, shade=False)
surf.set_facecolor((0,0,0,0))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Magnitude')

# 保存图像
plt.savefig('test_result/spectrogram3.png', dpi=300, bbox_inches='tight')