import os
import cv2
import argparse
import numpy as np
from imutils import contours


def parse():
    """设置自己的参数"""
    parser = argparse.ArgumentParser(description="set your identity parameters")
    parser.add_argument("-i", "--image", default="./images", type=str, help="path to input image")
    parser.add_argument("-t", "--template", default="./ocr_a_reference.png", type=str,
                        help="path to template OCR-A image")
    opt = parser.parse_args()
    # opt = vars(opt)   # 可用于返回参数的‘字典对’对象
    return opt


def cv_show(name, img):
    """绘图，避免重复造轮子"""
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sort_contours(cnts, method="left-to-right"):
    """对轮廓进行排序"""
    # 设置标识
    reverse = False     # 表示排序是否需要反转：正序（左->右，上->下）为False  逆序（右->左，下->上）为True
    i = 0               # 表示按x轴排序还是按y轴排序：上下为1   左右为0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 对轮廓进行排序。用一个最小的矩形，把找到的形状包起来x,y,h,w，按x或y进行排序
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i],
                                        reverse=reverse))
    return cnts, boundingBoxes


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """根据自定的宽/高进行等比例缩放图像"""
    dim = None              # 缩放后的图像尺寸
    h, w = image.shape[:2]  # 原始图像尺寸
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def deal_template(opt):
    """预处理模板图像，将每个类别的模板图像保存在字典中"""
    img = cv2.imread(opt.template)  # 读取模板图像
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将模板图像转换为灰度图
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]  # 二值化模板的灰度图像
    # -------------------------------------------------------
    # cv2.findContours(): 寻找二值图中的轮廓
    # 参数：
    #   cv2.RETR_EXTERNAL：      只检测外轮廓
    #   cv2.CHAIN_APPROX_SIMPLE：只保留终点坐标
    # 返回的list中每个元素都是图像中的一个轮廓
    # 注：函数接受的参数为二值图，即黑白的（不是灰度图）
    # -------------------------------------------------------
    refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找模板中的每个外轮廓
    cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)  # 将轮廓绘制在原始图像上
    print(np.array(refCnts, dtype=object).shape)  # 打印轮廓的个数
    refCnts = sort_contours(refCnts, method="left-to-right")[0]  # 对轮廓进行排序。从左到右，从上到下

    # 遍历每一个轮廓，并存储每个类别模板
    digits = {}
    for i, c in enumerate(refCnts):
        (x, y, w, h) = cv2.boundingRect(c)  # 计算轮廓的最小外接矩形
        roi = ref[y:y + h, x:x + w]  # 从二值图中提取当前类别模板
        roi = cv2.resize(roi, (57, 88))  # 将模板resize到合适尺寸
        digits[i] = roi  # 存储当前类别模板
    return digits


def template_math(img_path):
    image = cv2.imread(img_path)                    # 读取待识别图像
    image = resize(image, width=300)                # 将图像根据指定宽进行缩放
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    cv_show("image", image)

    # 初始化卷积核
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)   # 礼帽操作，突出更明亮的区域

    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)  # 检测x方向的边缘。ksize=-1相当于用3*3的核
    gradX = np.absolute(gradX)                                          # 取绝对值
    minVal, maxVal = np.min(gradX), np.max(gradX)
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))              # 归一化
    gradX = gradX.astype("uint8")

    print(np.array(gradX).shape)

    # 通过闭操作（先膨胀，再腐蚀）将数字连在一起
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    # THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # 再来一个闭操作
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    # 计算轮廓
    threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = threshCnts
    cur_img = image.copy()
    cv2.drawContours(cur_img, cnts, -1, (0, 0, 255), 3)

    # 遍历轮廓，保留符合先验信息的轮廓
    locs = []
    for (i, c) in enumerate(cnts):
        # 计算矩形
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
        if 2.5 < ar < 4.0:
            if 40 < w < 55 and 10 < h < 20:
                # 符合的留下来
                locs.append((x, y, w, h))

    # 将符合的轮廓从左到右排序
    locs = sorted(locs, key=lambda x: x[0])

    # 遍历每一个轮廓中的数字
    output = []
    for i, (gX, gY, gW, gH) in enumerate(locs):
        # initialize the list of group digits
        groupOutput = []

        # 根据坐标提取每一个组
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        # 二值化
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # 计算每一组的轮廓
        digitCnts, hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

        # 计算每一组中的每一个数值
        for c in digitCnts:
            # 找到当前数值的轮廓，resize成合适的的大小
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))

            # 计算匹配得分
            scores = []

            # 在模板中计算每一个得分
            for digit, digitROI in digits.items():
                # 模板匹配
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                _, score, _, _ = cv2.minMaxLoc(result)
                scores.append(score)

            # 得到最合适的数字
            groupOutput.append(str(np.argmax(scores)))

        # 画出来
        cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # 得到结果
        output.extend(groupOutput)
    # 绘制与打印结果
    cv_show("Image", image)
    print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
    print("Credit Card #: {}".format("".join(output)))
    return output


if __name__ == '__main__':
    # =================== 参数预处理 ===================
    FIRST_NUMBER = {
        "3": "American Express",
        "4": "Visa",
        "5": "MasterCard",
        "6": "Discover Card"
    }
    opt = parse()
    # ================== 模板图像预处理 ==================
    digits = deal_template(opt)

    # =================== 模板匹配 ==================
    for basename in os.listdir(opt.image):
        image_path = opt.image + "/" + basename
        output = template_math(image_path)



