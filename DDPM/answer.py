import numpy as np

def sorbel(img):
    row, col = img.size()
    image = np.zeros((row, col), np.unit8)
    for i in range(1, row-1):
        for j in range(col - 1):
            y = img[i-1, j+1] - img[i-1, j-1] + 2 * (img[i, j+1] - img[i, j-1]) + img[i+1, j+1] - img[i+1, j-1]
            x = img[i - 1, j - 1] - img[i+1, j] - img[i-1, j] + img[i+1, j+1] - img[i+1, j] - img[i-1, j] + img[i+1, j+1]-img[i-1, j+1]
            image[i, j] = abs(x) * 0.5 + abs(y) * 0.5
            return image

def get_next_pixel(img, pixel):
    x, y = pixel

    row_list = [-1, 1]
    col_list = [-1, 1]
    for i in row_list:
        for j in col_list:
            if img[x+i, y+j] != 0:
                return (x+i, y+j)
    return None

def ifcircle(img):
    row, col = img.size()
    pixel_list = []
    circle_list = []
    for i in range(row):
        for j in range(col):
            pixel = img[i, j]
            if pixel != 0:
                pixel_list.append((i, j))
    for pixel in pixel_list:
        start = pixel
        while pixel is not None:
            cur_circle = []
            pixel = get_next_pixel(img, pixel)
            cur_circle.append(pixel)
            pixel_list.remove(pixel)
            if pixel == start:
                circle_list.append(cur_circle)
                break    # start是一个环上的一点
    return pixel_list, circle_list   # 记录所有环上的任意一点

def sum_circle(img):
    start_list, circle_list = ifcircle(img)
    return len(circle_list)

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

"""
1.遍历图像的所有像素，找到值为1的像素，并将其加入到一个待处理列表
2.当待处理列表不为空时，从中取出一个像素，开始以此为起点进行搜索（广度/深度？）
3.在搜索过程中，每个访问到的新像素如果值为1，就将其置为0，表示已访问
"""
if __name__ == "__main__":
    arr = [1, 2, 3, 90, 51, 23, 75, 45, 93]
    print(quicksort((arr)))