import cv2
import numpy as np


# 读取深度图和灰度图
depth_img = cv2.imread("test/112_d.pgm", cv2.IMREAD_ANYDEPTH)
gray_img = cv2.imread("test/112_r.png", cv2.IMREAD_GRAYSCALE)

rgb_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

# rgb_img = rgb_img
rgb_img = cv2.resize(rgb_img, (depth_img.shape[1], depth_img.shape[0]))

width, height = depth_img.shape[1], depth_img.shape[0]

camera_factor = 1000


depth = np.asarray(depth_img, dtype=np.uint16).T

Z = depth / camera_factor
fx, fy, cx, cy = 400, 450, 325, 230

X = np.zeros((width, height))
Y = np.zeros((width, height))

for i in range(width):
    X[i, :] = np.full(X.shape[1], i)

X = ((X - cx / 2) * Z) / fx
for i in range(height):
    Y[:, i] = np.full(Y.shape[0], i)
Y = ((Y - cy / 2) * Z) / fy

data_ply = np.zeros((6, width * height))
data_ply[0] = X.T.reshape(-1)
data_ply[1] = -Y.T.reshape(-1)
data_ply[2] = -Z.T.reshape(-1)
img = np.array(rgb_img, dtype=np.uint8)
data_ply[3] = img[:, :, 0:1].reshape(-1)
data_ply[4] = img[:, :, 1:2].reshape(-1)
data_ply[5] = img[:, :, 2:3].reshape(-1)


float_formatter = lambda x: "%.4f" % x
points = []
for i in data_ply.T:
    points.append(
        "{} {} {} {} {} {} 0\n".format(
            float_formatter(i[0]),
            float_formatter(i[1]),
            float_formatter(i[2]),
            int(i[3]),
            int(i[4]),
            int(i[5]),
        )
    )
file = open("save_ply.ply", "w")
file.write(
    """ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
"""
    % (len(points), "".join(points))
)
file.close()
