# Ramiro Isa-Jara, ramiro.isaj@gmail.com
# Vision Interface to use for viewing and saving images from Video Camera Input
# with Analysis of area from experiments with well using HSV space ----- version 0.2.1
import cv2
import os
import glob
import numpy as np
import pandas as pd
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu


def dist(xp, yp):
    return np.sqrt(np.sum((xp-yp)**2))


def bytes_(img, m, n):
    ima = cv2.resize(img, (m, n))
    return cv2.imencode('.png', ima)[1].tobytes()


def camera_idx():
    # checks the first 10 indexes.
    index = 10
    videos = []
    for idx in range(index):
        cap = cv2.VideoCapture(idx)
        if cap.read()[0]:
            videos.append(idx)
            cap.release()
    return videos


def update_dir(path):
    path_s = path.split('/')
    cad, path_f = len(path_s), path_s[0]
    for p in range(1, cad):
        path_f += '\\' + path_s[p]
    return path_f


def save_csv_file(data_, des, header):
    # Save data in csv file
    _root_result = os.path.join(des, header+'.csv')
    data_.to_csv(_root_result, index=False)
    print('----------------------------------------------')
    print('..... Save data in CSV file successfully .....')
    print('----------------------------------------------')


def graph_data(des, header):
    _root_data = os.path.join(des, header+'.csv')
    data_ = pd.read_csv(_root_data)
    y = np.array(data_['Percentage'])
    x = np.arange(1, len(y) + 1, 1)
    fig = plt.figure()
    plt.plot(x, y, 'o')
    plt.grid()
    plt.xlabel('N. of image')
    plt.ylabel('Percentage')
    _root_fig = os.path.join(des, 'Percentage_'+header+'.jpg')
    fig.tight_layout()
    plt.savefig(_root_fig)
    plt.close()


def preprocessing(img):
    image_gray_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clh = cv2.createCLAHE(clipLimit=5.0)
    clh_img = clh.apply(image_gray_)

    blurred = cv2.GaussianBlur(clh_img, (5, 5), 0)

    return clh_img, blurred


def f_sorted(files_, id_sys):
    symbol = '\\' if id_sys == 0 else '/'
    ids = []
    for f in files_:
        parts = f.split(symbol)
        name_i = parts[len(parts) - 1]
        ids.append(name_i.split('.')[0].split('_')[-1])
    ids = list(map(int, ids))
    ids.sort(key=int)
    file_r = []
    for i in range(len(files_)):
        parts = files_[i].split(symbol)
        name = parts[len(parts) - 1].split('.')
        exp = name[0].split('_')
        if len(exp) >= 2:
            n_exp = exp[0]
            for j in range(1, len(exp)-1):
                n_exp += '_' + exp[j]
            n_name = n_exp + '_' + str(ids[i]) + '.' + name[1]
        else:
            n_name = str(ids[i]) + '.' + name[1]

        if id_sys == 0:
            n_file = parts[0] + symbol
        else:
            n_file = (symbol + parts[0])
        for j in range(1, len(parts)-1):
            n_file += (parts[j] + symbol)
        n_file += n_name
        file_r.append(n_file)

    return file_r


def load_image_i(orig, i, type_, filenames, id_sys):
    symbol = '\\' if id_sys == 0 else '/'
    if len(filenames) == 0:
        filenames = [img for img in glob.glob(orig+'*'+type_)]
        filenames = f_sorted(filenames, id_sys)
        # filenames.sort()
    if i < len(filenames):
        name = filenames[i]
        parts = name.split(symbol)
        name_i = parts[len(parts)-1]
        image_ = cv2.imread(name)
    else:
        image_, name_i = [], []
    return filenames, image_, name_i, len(filenames)


def calculate_contour(binary_):
    contours, hierarchy = cv2.findContours(binary_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    f_contours, area = [], []
    for c in contours:
        area.append(cv2.contourArea(c))
    area_min = max(area) / 3
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_min:
            f_contours.append(c)
    return f_contours


def binary_contours(img, binary_):
    contours, hierarchy = cv2.findContours(binary_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
    return img


def build_filters():
    filters_, k_size, sigma = [], 21, [4.0]
    for s in sigma:
        for theta in np.arange(0, np.pi, np.pi / 4):
            kern = cv2.getGaborKernel((k_size, k_size), s, theta, 10.0, 0.9, 0, ktype=cv2.CV_32F)
            kern /= 1.5*kern.sum()
            filters_.append(kern)
    return filters_


def apply_gabor(img, filters):
    gabor_img_ = np.zeros_like(img)
    for kern in filters:
        np.maximum(gabor_img_, cv2.filter2D(img, cv2.CV_8UC3, kern), gabor_img_)
    return gabor_img_


def p_circle(binary_):
    cords = []
    contours = calculate_contour(binary_)
    contour = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour1 in contour:
        (x_, y_), radius_ = cv2.minEnclosingCircle(contour1)
        cords.append([x_, y_, radius_])
    cords = np.array(cords)
    (x_, y_), radius_ = cv2.minEnclosingCircle(contour[0])
    if len(contour) > 1:
        idx = np.where(cords == np.max(cords[:, 2]))[0]
        x1, y1 = cords[idx, 0], cords[idx, 1]
        if dist(np.array([x_, y_]), np.array([x1, y1])) < 200:
            x_, y_, radius_ = np.round((x_+x1)/2), np.round((y1 + y_)/2), cords[idx, 2]+5
    radius_ -= 15
    (x_, y_), radius_ = (int(np.round(x_)), int(np.round(y_))), int(np.round(radius_))
    return x_, y_, radius_


def well_region(img, filters):
    ima_gray, final_ima = preprocessing(img)
    gabor_img = apply_gabor(final_ima, filters)
    thresh = threshold_otsu(gabor_img)
    thresh = cv2.threshold(gabor_img, thresh, 255, cv2.THRESH_TOZERO_INV)[1]
    total = thresh.shape[0] * thresh.shape[1]
    total_n = np.sum(thresh == 0)
    per = np.round(total_n / total, 2)
    if 0.38 > per >= 0.35:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
        markers = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        markers = cv2.morphologyEx(markers, cv2.MORPH_CLOSE, kernel, iterations=1)
    elif per < 0.35:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        markers = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations=2)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        markers = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=1)

    arr = markers > 0
    markers = morphology.remove_small_objects(arr, min_size=2000, connectivity=1).astype(np.uint8)
    markers = morphology.remove_small_holes(markers.astype(np.bool), area_threshold=4000, connectivity=1)
    markers = markers.astype(np.uint8)
    # compute well area
    x_, y_, radius_ = p_circle(markers)
    image_r = np.copy(img)
    cv2.circle(image_r, (x_, y_), radius_, (35, 255, 12), 3)
    return image_r, x_, y_, radius_


def eval_cords(cords_, x_, y_, radius_):
    cords_ = np.array(cords_)
    xp, yp, rdp = np.round(np.average(cords_[:, 0])), np.round(np.average(cords_[:, 1])), np.round(np.average(cords_[:, 2]))
    distance, rel = dist(np.array([x_, y_]), np.array([xp, yp])), np.round(min(rdp, radius_) / max(rdp, radius_), 2)
    if distance < 50 and rel > 0.92:
        return True
    else:
        return False


def seq_circular(img, cords_):
    image_r, cords_ = np.copy(img), np.array(cords_)
    x_ = int(np.round(np.average(cords_[:, 0])))
    y_ = int(np.round(np.average(cords_[:, 1])))
    radius_ = int(np.round(np.average(cords_[:, 2])))
    cv2.circle(image_r, (x_, y_), radius_, (35, 255, 12), 3)
    return image_r, x_, y_, radius_


def gray_circle(binary_g, x_, y_):
    contour = cv2.findContours(binary_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[0] if len(contour) == 2 else contour[1]
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    (xr, yr), rad_r = cv2.minEnclosingCircle(contour[0])
    distance = dist(np.array([x_, y_]), np.array([xr, yr]))
    if distance < 30 and rad_r > 50:
        binary_g = np.zeros_like(binary_g, dtype=np.uint8)
        cv2.circle(binary_g, (x_, y_), int(rad_r), 255, -1)
    return binary_g


def sobel_filter(img):
    ima_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ima_norm = ima_gray / np.max(ima_gray)
    thresh_norm_ = threshold_otsu(ima_norm)
    enhanced_ima = 1 - np.exp(-ima_norm ** 2 / 0.5)
    ima = np.array(enhanced_ima * 255).astype(np.uint8)
    dx = cv2.Sobel(ima, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(ima, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.convertScaleAbs(dx)
    gy = cv2.convertScaleAbs(dy)
    combined = cv2.addWeighted(gx, 2.5, gy, 2.5, 0)
    thresh_val_ = threshold_otsu(combined)
    thresh_sobel_ = np.array((255 * (combined > thresh_val_))).astype(np.uint8)
    return thresh_sobel_, thresh_val_, thresh_norm_


def hsv_space(img):
    hsv = rgb2hsv(img)
    image = (255 * hsv[:, :, 1]).astype(np.uint8)
    th_hsv_ = threshold_otsu(image)
    thresh_hsv = (255 * np.array(image > 30)).astype(np.uint8)
    return thresh_hsv, th_hsv_


def roi_region(bin_img, x_, y_, radius_, val_):
    roi_img = np.zeros_like(bin_img, dtype=np.uint8)
    rad_n = radius_ - val_
    cv2.circle(roi_img, (x_, y_), rad_n, 255, -1)
    idx = np.where(roi_img == 255)
    roi_img[idx] = bin_img[idx]
    return roi_img


def opera_sobel(bin_sobel_, x_, y_, radius_, ctr_):
    morph_img = roi_region(bin_sobel_, x_, y_, radius_, 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    binary = cv2.morphologyEx(morph_img, cv2.MORPH_DILATE, kernel, iterations=1)
    if ctr_:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=1)
        arr = binary > 0
        binary = morphology.remove_small_objects(arr, min_size=100, connectivity=1)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=2)
        arr = binary > 0
        binary = morphology.remove_small_objects(arr, min_size=5, connectivity=1)

    binary = morphology.remove_small_holes(binary.astype(np.bool), area_threshold=100000, connectivity=1)
    binary = (255 * binary).astype(np.uint8)
    return binary


def opera_sobel_hsv(bin_sob_hsv, x_, y_, radius_):
    morph_img = roi_region(bin_sob_hsv, x_, y_, radius_, 40)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(morph_img, cv2.MORPH_DILATE, kernel, iterations=1)
    binary = morphology.remove_small_holes(binary.astype(np.bool), area_threshold=1000, connectivity=1)
    binary = binary.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=2)
    arr = binary > 0
    binary = morphology.remove_small_objects(arr, min_size=50, connectivity=1)
    binary = morphology.remove_small_holes(binary.astype(np.bool), area_threshold=50000, connectivity=1)
    binary = (255 * binary).astype(np.uint8)
    return binary


def opera_gray(bin_gray_, x_, y_, radius_):
    morph_gray = roi_region(bin_gray_, x_, y_, radius_, 50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(morph_gray, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = morphology.remove_small_holes(binary.astype(np.bool), area_threshold=1000, connectivity=1)
    binary = binary.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=1)
    arr = binary > 0
    binary = morphology.remove_small_objects(arr, min_size=100, connectivity=1)
    binary = morphology.remove_small_holes(binary.astype(np.bool), area_threshold=50000, connectivity=1)
    binary = (255 * binary).astype(np.uint8)
    binary = gray_circle(binary, x_, y_)
    return binary


def binary_regions(img, x_, y_, radius_):
    _, original_gray = preprocessing(img)
    thresh_gray = threshold_otsu(original_gray)
    norm_img = cv2.normalize(img, None, alpha=-0.1, beta=1.1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = (255 * norm_img).astype(np.uint8)
    # ---> obtain binary regions of images
    bin_sobel, th_sobel, th_norm = sobel_filter(img)
    bin_hsv, th_hsv = hsv_space(img)
    bin_gray = cv2.threshold(original_gray, thresh_gray, 255, cv2.THRESH_TOZERO_INV)[1]
    # ---> choose better binary image
    param = np.round(thresh_gray * th_norm)
    print(param, th_sobel, th_hsv)
    if (param >= 70 and th_sobel <= 145) or (param >= 79 and th_sobel >= 150 and th_hsv > 133) or \
       (param >= 80 and th_sobel >= 150):
        rel_ = np.round(min(th_hsv, th_sobel) / max(th_hsv, th_sobel), 2)
        if 0.80 < rel_ < 0.95:
            print('SOBEL')
            ctr = True if param > 85 or th_sobel > 140 else False
            ima_binary = opera_sobel(bin_sobel, x_, y_, radius_, ctr)
            return ima_binary
        else:
            print('or')
            bin_hsv_sobel = cv2.bitwise_or(bin_sobel, bin_hsv)
            ima_binary = opera_sobel_hsv(bin_hsv_sobel, x_, y_, radius_)
            return ima_binary
    else:
        print('gray')
        ima_binary = opera_gray(bin_gray, x_, y_, radius_)

    return ima_binary


def well_analysis(img, img_s, x_, y_, radius_):
    binary = binary_regions(img, x_, y_, radius_)
    idx = np.where(binary == 255)
    seg_image = binary_contours(img_s, binary)
    # image binary result
    binary2 = np.zeros_like(binary, dtype=np.uint8)
    cv2.circle(binary2, (x_, y_), radius_+1, 255, 1)
    binary2[idx] = binary[idx]
    # compute segmented area
    area_detected_ = np.sum(binary == 255)
    # compute well area
    well = np.zeros_like(binary, dtype=np.uint8)
    cv2.circle(well, (x_, y_), radius_, 255, -1)
    area_well_ = np.sum(well == 255)
    percent_well_ = np.round((area_detected_ * 100) / area_well_, 2)
    return binary2, seg_image, area_well_, area_detected_, percent_well_


def well_main(img, img_r, k, x_, y_, radius_):
    bin_final, img_final, area_well, area_detected, percent_well = well_analysis(img, img_r, x_, y_, radius_)
    print('Processing image  ----->  ' + str(k))
    table = [['Total area      : ', str(area_well)],
             ['Detected area   : ', str(area_detected)],
             ['Percentage      : ', str(percent_well)]]
    for line in table:
        print('{:>10} {:>10}'.format(*line))
    print('-------------------------------------------------------------------------')

    return bin_final, img_final, area_well, area_detected, percent_well

