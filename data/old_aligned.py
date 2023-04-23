import cv2
import numpy as np

# 读取深度图和灰度图
depth_img = cv2.imread("test/81_d.pgm", cv2.IMREAD_ANYDEPTH)
gray_img = cv2.imread("test/81_r.png", cv2.IMREAD_GRAYSCALE)


# 深度相机内参矩阵参数
depth_fx, depth_fy, depth_cx, depth_cy = 444.8667, 444.6116, 331.6182, 231.8496

# 深度相机外参矩阵
depth_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
depth_T = np.array([0, 0, 0])

depth_k1, depth_k2, depth_p1, depth_p2 = -0.2007, 0.0703, 0, 0

# 灰度相机内参矩阵参数
gray_fx, gray_fy, gray_cx, gray_cy = 796.1702, 796.4280, 650.7882, 433.7383

# 灰度相机外参矩阵
gray_R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
gray_T = np.array([0, 0, 0])


gray_k1, gray_k2, gray_p1, gray_p2 = 0.0290, -0.0216, 0, 0

depth_img_ = np.zeros(depth_img.shape, dtype=np.float32)
gray_img_resized = cv2.resize(gray_img, (depth_img.shape[1], depth_img.shape[0]))


height, width = depth_img.shape

camera_factor = 1000

for v in range(height):
    for u in range(width):
        depth = depth_img[v, u]
        if depth == 0:
            continue

        z_c = depth
        x_c = (u - depth_cx) * z_c / depth_fx
        y_c = (v - depth_cy) * z_c / depth_fy

        # 相机坐标转世界坐标
        point_c = np.array([x_c, y_c, z_c])
        point_w = np.dot(depth_R, point_c) + depth_T

        # 世界坐标转灰度相机坐标
        point_c_gray = np.dot(gray_R, point_w) + gray_T

        x_gray_distorted = x_c / z_c
        y_gray_distorted = y_c / z_c

        r_squared = (
            x_gray_distorted * x_gray_distorted + y_gray_distorted * y_gray_distorted
        )
        x_gray_distorted = (
            x_gray_distorted
            * (1 + gray_k1 * r_squared + gray_k2 * r_squared * r_squared)
            + 2 * gray_p1 * x_gray_distorted * y_gray_distorted
            + gray_p2 * (r_squared + 2 * x_gray_distorted * x_gray_distorted)
        )

        gray_x = int((x_gray_distorted * gray_fx + gray_cx) / 2)
        gray_y = int((y_gray_distorted * gray_fy + gray_cy) / 2)

        depth_img_[gray_y, gray_x] = depth

depth_color = cv2.applyColorMap(
    cv2.convertScaleAbs(depth_img_, alpha=0.1), cv2.COLORMAP_JET
)
depth_color = cv2.medianBlur(depth_color, 5)

cv2.imshow("GolorImage", depth_color)


cv2.imwrite("1.png", depth_color)
cv2.imwrite("2.png", gray_img_resized)

# aligned_img = cv2.merge((gray_img_resized, depth_color))


# # 显示带深度通道的图像
# cv2.imshow("Gray Depth Image", aligned_img)
