import numpy as np
import cv2
from numba import njit, prange


def otsu_canny(image, lowrate=0.1):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding
    ret, _ = cv2.threshold(image, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU))
    edged = cv2.Canny(image, threshold1=(ret * lowrate), threshold2=ret)

    # return the edged image
    return edged


def single_filt(img):
    median = np.median(img)
    print('med ', median)
    sigma = 0.3
    low = int(max(0, (1 - sigma) * median))
    up = int(min(255, (1 + sigma) * median))
    print(low, ' ', up)
    lower = np.array(low)
    upper = np.array(up)
    img = cv2.inRange(img, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    img = cv2.dilate(img, kernel)
    return img


def three_filt(img):
    # coef = 0.0038
    median = np.median(img)
    print('med ', median)
    sigma = 0.3
    low = int(max(0, (1 - sigma) * median))
    up = int(min(255, (1 + sigma) * median))
    print(low, ' ', up)
    lower = np.array([low, low, low])
    upper = np.array([up, up, up])
    img = cv2.inRange(img, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    img = cv2.dilate(img, kernel)
    return img


def Hough(edge_image, max_line_gap=7, min_line_length=50):
    rho_res = .1  # [pixels]
    theta_res = np.pi / 180.  # [radians]
    threshold = 0  # [# votes]
    # min_line_length = 50  # [pixels]
    # max_line_gap = 7  # [pixels]
    lines = cv2.HoughLinesP(edge_image, rho_res, theta_res, threshold, np.array([]),
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines


def get_large_rec(contours):
    maxS = 0
    rec = None
    if len(contours) != 0:
        for (i, c) in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 300:
                r = cv2.boundingRect(c)
                S = r[2] * r[3]
                if S > maxS:
                    maxS = S
                    rec = r
    return rec


@njit(fastmath=True, cache=True, parallel=True)
def light_compare(edges, found):
    for i in prange(edges.shape[0]):
        for j in prange(edges.shape[1]):
            if edges[i][j] != 0:
                found[i][j][0] = 0
                found[i][j][1] = 255
                found[i][j][2] = 0


def light_canny(found):
    median = np.median(found)
    sigma = 0.3
    low = int(max(0, (1 - sigma) * median))
    up = int(min(255, (1 + sigma) * median))
    print(low, ' ', up)
    edges = cv2.Canny(found, low, up)  ##slkjdghsjodhgkjsbgklsdbg gkjasdbgkjadhbgkjladhgkljad
    light_compare(edges, found)


@njit(fastmath=True, cache=True, parallel=True)
def lower(img, delta=0.1):
    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            img[i][j] = img[i][j] * delta


@njit(fastmath=True, cache=True, parallel=True)
def compare(edges, copy, ed, f):
    for i in prange(edges.shape[0]):
        for j in prange(edges.shape[1]):
            if edges[i][j] != 0 and copy[i][j][0] == 0 and copy[i][j][1] == 0 and copy[i][j][2] == 255:
                f[i][j][0] = 0
                f[i][j][1] = 255
                f[i][j][2] = 0
                ed[i][j] = 255
                if i + 1 < edges.shape[0]:
                    ed[i + 1][j] = 255
                if j + 1 < edges.shape[1]:
                    ed[i][j + 1] = 255
                    if i + 1 < edges.shape[0]:
                        ed[i + 1][j + 1] = 255

            else:
                ed[i][j] = 0


def detection(found, edges, gap, leng):
    lines = Hough(edges, max_line_gap=gap, min_line_length=leng)
    copy = found.copy()
    f = found.copy()
    if lines is not None:
        for i in range(0, len(lines)):
            coords = lines[i][0]
            cv2.line(copy, (coords[0], coords[1]), (coords[2], coords[3]), (0, 0, 255), 8, cv2.LINE_AA)
    ed = edges.copy()
    if edges is not None:
        compare(edges, copy, ed, f)
    return f, ed


def cont(img, result):  # dark pics
    contours, hierarchy = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im_copy = img.copy()
    if len(contours) != 0:
        for (i, c) in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 100000:
                r = cv2.boundingRect(c)
                found = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
                # cv2.imshow('found', found)
                cv2.drawContours(im_copy, c, -1, (255, 255, 0), 12)
                x, y, w, h = r[0:]
                # print(x, y, w, h)

                cv2.rectangle(im_copy, (x, y), (x + w, y + h), (255, 0, 255), 12)
                edges = TH(found)
                gaps = [20, 4, 40]
                leng = [25, 40, 130]
                save = found.copy()
                found = (255 - found)
                for i in range(3):
                    found, edges = detection(save, edges, gaps[i], leng[i])
                    cp = found.copy()
                    max_dimension = float(max(cp.shape))
                    scale = 900 / max_dimension
                    cp = cv2.resize(cp, None, fx=scale, fy=scale)
                    #cv2.imshow(str(i) + ' found', cp)

                # cv2.waitKey(1000)

                img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = found
                # max_dimension = float(max(im_copy.shape))
                # scale = 900 / max_dimension
                # im_copy = cv2.resize(im_copy, None, fx=scale, fy=scale)

    # if len(contours) != 0:
    #     for (i, c) in enumerate(contours):
    #         area = cv2.contourArea(c)
    #         if area > 100000:
    #             cv2.drawContours(img, c, -1, (255, 255, 0), 12)

    # r = get_large_rec(contours)
    # found = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
    # # cv2.imshow('found', found)
    # # cv2.drawContours(img, c, -1, (255,255,0), 12)
    # x, y, w, h = r[0:]
    # # print(x, y, w, h)
    #
    # cv2.rectangle(im_copy, (x, y), (x + w, y + h), (255, 0, 255), 12)
    # edges = TH(found)
    # gaps = [20, 4, 40]
    # leng = [25, 40, 130]
    # save = found.copy()
    # for i in range(3):
    #     found, edges = detection(save, edges, gaps[i], leng[i])
    #     cp = found.copy()
    #     max_dimension = float(max(cp.shape))
    #     scale = 900 / max_dimension
    #     cp = cv2.resize(cp, None, fx=scale, fy=scale)
    #     cv2.imshow(str(i) + ' found', cp)
    # cv2.waitKey(1000)
    #
    # img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = found
    max_dimension = float(max(im_copy.shape))
    scale = 900 / max_dimension
    im_copy = cv2.resize(im_copy, None, fx=scale, fy=scale)
    # print(r)

    cv2.imshow('recs', im_copy)
    return img


def light_cont(img, result):  # light pics
    contours, hierarchy = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im_copy = img.copy()
    if len(contours) != 0:
        for (i, c) in enumerate(contours):
            area = cv2.contourArea(c)
            if area > 100000:
                r = cv2.boundingRect(c)
                found = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]
                cv2.drawContours(im_copy, c, -1, (255, 255, 0), 12)
                x, y, w, h = r[0:]

                cv2.rectangle(im_copy, (x, y), (x + w, y + h), (255, 0, 255), 12)
                light_canny(found)
                img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = found

    max_dimension = float(max(im_copy.shape))
    scale = 900 / max_dimension
    im_copy = cv2.resize(im_copy, None, fx=scale, fy=scale)

    cv2.imshow('recs', im_copy)
    return img


def edge_demo(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # Находим градиент по оси X
    grad_x = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0)
    # Находим градиент в направлении y
    grad_y = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1)
    # Преобразуем значение градиента в 8 бит
    x_grad = cv2.convertScaleAbs(grad_x)
    y_grad = cv2.convertScaleAbs(grad_y)
    # Объединить два градиента
    src1 = cv2.addWeighted(x_grad, 0.5, y_grad, 0.5, 0)

    # Объедините градиенты, используя хитрый алгоритм, где 50 и 100 - пороги
    edge = cv2.Canny(src1, 50, 100)
    # cv2.imshow("Canny_edge_1", edge)
    edge1 = cv2.Canny(grad_x, grad_y, 10, 100)
    # cv2.imshow("Canny_edge_2", edge1)
    # Используйте край как маску для выполнения побитовых и побитовых операций
    edge2 = cv2.bitwise_and(image, image, mask=edge1)
    # cv2.imshow("bitwise_and", edge2)
    return edge


def morph(im):
    morph = im.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # take morphological gradient
    gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

    # split the gradient image into channels
    image_channels = np.split(np.asarray(gradient_image), 3, axis=2)

    channel_height, channel_width, _ = image_channels[0].shape

    # apply Otsu threshold to each channel
    for i in range(0, 3):
        _, image_channels[i] = cv2.threshold(~image_channels[i], 254, 255,
                                             cv2.THRESH_BINARY_INV | cv2.THRESH_TOZERO_INV)
        image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))

    # merge the channels
    image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
    return image_channels


def TH(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (71, 71))
    diff = cv2.subtract(blur, gray)
    ret, th = cv2.threshold(diff, 13, 255, cv2.THRESH_BINARY)
    return th
    # cv2.imshow("threshold", th)


if __name__ == "__main__":
    img = cv2.imread('4.bmp')
    median = np.median(img)
    print('median: ', median)
    if median < 60:
        ret, thresh5 = cv2.threshold(img, 255, 13, cv2.THRESH_TOZERO_INV)
        result = thresh5
        result = cv2.GaussianBlur(result, (75, 75), 0)
        result = three_filt(result)
        img = cont(img, result)
    else:
        lower(img, delta=1.0)
        result = edge_demo(img)
        result = cv2.GaussianBlur(result, (75, 75), 0)
        result = single_filt(result)
        img = light_cont(img, result)

    # ret, thresh1 = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
    # ret, thresh2 = cv2.threshold(img, 100, 180, cv2.THRESH_BINARY_INV)
    # ret, thresh4 = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)
    # ret, thresh5 = cv2.threshold(img, 255, 13, cv2.THRESH_TOZERO_INV)

    # cv2.imshow('Binary Threshold', thresh1)
    # cv2.imshow('Binary Threshold Inverted', thresh2)
    # cv2.imshow('Set to 0', thresh4)
    # cv2.imshow('Set to 0 Inverted', thresh5)

    # Filtering
    # result = thresh5
    # result = cv2.GaussianBlur(result, (17,17), 0)
    # cv2.imshow('blurred', result)
    # result = filtering(result)
    # img = cont(img, result)

    # result = thresh4
    # result = cv2.GaussianBlur(result, (17,17), 0)
    # result = filtering(result)
    # img = cont(img, result)
    #
    # result = thresh2
    # result = cv2.GaussianBlur(result, (17,17), 0)
    # result = filtering(result)
    # img = cont(img, result)

    re = result.copy()
    max_dimension = float(max(re.shape))
    scale = 900 / max_dimension
    re = cv2.resize(re, None, fx=scale, fy=scale)
    cv2.imshow('resf', re)

    # img = cont(img, result)

    # Resize
    max_dimension = float(max(result.shape))
    scale = 900 / max_dimension
    img = cv2.resize(img, None, fx=scale, fy=scale)

    # Show results
    cv2.imwrite('result.jpg', img)
    cv2.imshow('res', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
