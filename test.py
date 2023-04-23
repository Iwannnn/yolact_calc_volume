import cv2

test = cv2.imread("t.png", cv2.IMREAD_UNCHANGED)

# Convert the depth map to float values
# test = test.astype("float32")

cv2.imshow("test", test)
print(test)
cv2.waitKey(0)

cv2.destroyAllWindows
