import cv2
import numpy as np

t = cv2.imread("test/4_d.pgm", cv2.IMREAD_ANYDEPTH)

print(np.max(t))

depth_color = cv2.applyColorMap(cv2.convertScaleAbs(t, alpha=0.1), cv2.COLORMAP_JET)


cv2.imshow("ttt", depth_color)

cv2.waitKey(0)
cv2.destroyAllWindows
