import cv2
import numpy as np


def rotate_part(part, center, theta):

    left = part.get('left')
    top = part.get('top')
    width = part.get('width')
    height = part.get('height')
    data = part.get('data')

    # 计算四个顶点坐标
    R1 = [left, top]
    R2 = [left + width, top]
    R3 = [left, top + height]
    R4 = [left + width, top + height]
    R = [R1, R2, R3, R4]

    # 计算旋转后顶点坐标
    M = cv2.getRotationMatrix2D(center, theta, 1.0)
    R_rotate = []
    for x, y in R:
        r_x = M[0][0] * x + M[0][1] * y + M[0][2]
        r_y = M[1][0] * x + M[1][1] * y + M[1][2]
        r_x = int(round(r_x))
        r_y = int(round(r_y))
        R_rotate.append([r_x, r_y])
    R_rotate = np.array(R_rotate)
    R_rotate = np.transpose(R_rotate)

    X_max = max(R_rotate[0])
    X_min = min(R_rotate[0])
    Y_max = max(R_rotate[1])
    Y_min = min(R_rotate[1])
    print(X_min, Y_min, X_max, Y_max)

    # 计算data部分位置转换
    dict = {}
    for x in range(left, left+width):
        for y in range(top, top+height):
            r_x = M[0][0]*x+M[0][1]*y+M[0][2]
            r_y = M[1][0]*x+M[1][1]*y+M[1][2]
            r_x = int(round(r_x))
            r_y = int(round(r_y))
            dict["%d_%d" % (r_x, r_y)] = data[x-left, y-top]

    # 新区域赋值
    r_data = np.ones([X_max-X_min, Y_max-Y_min, 3])
    for x in range(X_min, X_max):
        for y in range(Y_min, Y_max):
            if "%d_%d" % (x, y) in dict:
                r_data[x - X_min][y - Y_min] = dict["%d_%d" % (x, y)]
            else:
                r_data[x - X_min][y - Y_min] = [255, 255, 255]
    print(r_data.shape)
    cv2.imwrite("./r.png", r_data)


if __name__ == '__main__':
    data = cv2.imread('./test.png')
    # part region
    part = {"left": 10, "top": 10, "width": 48, "height": 48, "data": data}
    # rotate center location
    center = (40, 40)
    # rotate angle
    theta = 15
    rotate_part(part=part, center=center, theta=theta)
