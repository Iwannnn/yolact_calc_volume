import numpy as np
import cv2

# 读取深度图像
depth_image = cv2.imread("test/3_d.pgm", cv2.IMREAD_UNCHANGED)

# fx, fy, cx, cy = 365.4947, 365.1759, 256.7344, 202.3221
# # 内参矩阵
# K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# # 选择两个点
# x1, y1 = 135, 129
# x2, y2 = 135, 297

# # 计算两点深度值
# z1 = depth_image[x1, y1]
# z2 = depth_image[x2, y2]

# x1 = (x1 - cx) * z1 / fx
# y1 = (y1 - cy) * z1 / fy
# x2 = (x2 - cx) * z2 / fx
# y2 = (y2 - cy) * z2 / fy


# 将像素坐标转换为相机坐标系下的三维坐标

# print("x1:", x1, "y1:", y1, "z1:", z1)
# print("x2:", x2, "y2:", y2, "z2:", z2)

# res = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1))

# print("res:", res)

print("d", depth_image[198, 241])
