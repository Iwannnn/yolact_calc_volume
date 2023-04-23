import cv2
import numpy as np

depth_img = cv2.imread("test/1_d.pgm", cv2.IMREAD_ANYDEPTH)
bgr_img = cv2.imread("test/1_r.png")


bgr_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGRA2BGR)
bgr_fx, bgr_fy, bgr_cx, bgr_cy = 1086.0402, 1086.3857, 985.9641, 515.5833
ir_fx, ir_fy, ir_cx, ir_cy = 365.4947, 365.1759, 256.7344, 202.3221

# bgr_R = np.array(
#     [[0.9993, -0.0080, -0.0026], [0.0080, 0.9993, -0.0018], [0.0027, 0.0018, 1.0000]]
# )
# bgr_T = np.array([[-27.4436], [-109.7310], [23.3539]])


# ir_R = np.array(
#     [[0.9974, 0.0341, 0.0246], [-0.0336, 0.9980, -0.0115], [-0.0241, 0.0110, 0.9985]]
# )
# ir_T = np.array([[-57.0745], [-67.9168], [20.5080]])

# ir2bgr_R = bgr_R @ np.linalg.inv(ir_R)
# ir2bgr_T = bgr_T - ir2bgr_R @ ir_T

ir2bgr_R = np.array(
    [
        [0.99983816, 0.00969128, -0.01424244],
        [-0.00973783, 0.99995766, 0.0062798],
        [0.01411798, 0.0064265, 0.99988306],
    ]
)

ir2bgr_T = np.array([[46.14159475], [3.18711363], [7.84516403]])
# ir2bgr_T = np.array([[0], [0], [0]])


bgr_K = np.array([[bgr_fx, 0, bgr_cx], [0, bgr_fy, bgr_cy], [0, 0, 1]])
ir_K = np.array([[ir_fx, 0, ir_cx], [0, ir_fy, ir_cy], [0, 0, 1]])

R = bgr_K @ ir2bgr_R @ np.linalg.inv(ir_K)
T = bgr_K @ ir2bgr_T

# print("R:", R)
# print("T:", T)

aligned_bgr = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)

i = 0
for row in range(424):
    for col in range(512):
        depth_value = depth_img[row, col]
        if depth_value != 0 and depth_value != 65535:
            uv_depth = np.array([[col], [row], [1.0]])
            uv_bgr = (depth_value / 1000.0 * R) @ uv_depth + T / 1000

            x = int(uv_bgr[0] / uv_bgr[2])
            y = int(uv_bgr[1] / uv_bgr[2])
            if x >= 0 and x < 1920 and y >= 0 and y < 1080:
                aligned_bgr[row, col, 0] = bgr_img[y, x, 0]
                aligned_bgr[row, col, 1] = bgr_img[y, x, 1]
                aligned_bgr[row, col, 2] = bgr_img[y, x, 2]
            else:
                aligned_bgr[row, col, 0] = 0
                aligned_bgr[row, col, 1] = 0
                aligned_bgr[row, col, 2] = 0
        else:
            aligned_bgr[row, col, 0] = 0
            aligned_bgr[row, col, 1] = 0
            aligned_bgr[row, col, 2] = 0

depth_color = cv2.applyColorMap(
    cv2.convertScaleAbs(depth_img, alpha=0.2), cv2.COLORMAP_JET
)
cv2.imshow("depth_color", depth_color)
cv2.imshow("aligned_bgr", aligned_bgr)

res = cv2.add(aligned_bgr, depth_color)

cv2.imwrite("test/algined_r.png", res)


cv2.waitKey(0)
cv2.destroyAllWindows
