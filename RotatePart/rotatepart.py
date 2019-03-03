import cv2


def rotate_part(data, center, theta):

    left = data.get('left')
    top = data.get('top')
    width = data.get('width')
    height = data.get('height')
    img = data.get('img')
    i_w, i_h, _ = img.shape

    # extract part
    part = img[top:top+height, left:left+width]
    cv2.imwrite("./part.png", part)

    # calculate rotate location(r_x, r_y)
    M = cv2.getRotationMatrix2D(center, theta, 1.0)
    for x in range(top, top+height):
        for y in range(left, left+width):
            r_x = M[0][0]*x+M[0][1]*y+M[0][2]
            r_y = M[1][0]*x+M[1][1]*y+M[1][2]
            r_x = int(round(r_x))
            r_y = int(round(r_y))
            # deal out of bounds
            if r_x >= i_w or r_y >= i_h:
                continue
            img[r_x][r_y] = img[x][y]
    # set rotated part background
    img[top:top + height, left:left + width] = (255, 255, 255)
    cv2.imwrite("./rotated.png", img)


if __name__ == '__main__':
    img = cv2.imread('./test.png')
    # roi region
    data = {"left": 10, "top": 5, "width": 20, "height": 20, "img": img}
    # rotate center location
    center = (40, 40)
    # rotate angle
    theta = 30
    rotate_part(data=data, center=center, theta=theta)
