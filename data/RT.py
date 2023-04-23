import numpy as np

bgr_R = np.array(
    [[0.9998, -0.0187, -0.0100], [0.0194, 0.9976, 0.0664], [0.0087, -0.0666, 0.9977]]
)
bgr_T = np.array([[-120.8259], [-77.8176], [472.8877]])


ir_R = np.array(
    [[0.9996, -0.0294, 0.0035], [0.0291, 0.9969, 0.0727], [-0.0056, -0.0726, 0.9973]]
)
ir_T = np.array([[-179.4418], [-75.7635], [528.5057]])

ir2bgr_R = np.array(
    [[0.9659, 0.0155, 0.2586], [-0.0224, 0.9995, 0.0237], [-0.2581, -0.0287, 0.9657]]
)
# ir2bgr_R = bgr_R @ np.linalg.inv(ir_R)
ir2bgr_T = bgr_T - ir2bgr_R @ ir_T

print(ir2bgr_R)
print(ir2bgr_T)
