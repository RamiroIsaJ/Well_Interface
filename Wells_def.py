# Ramiro Isa-Jara, ramiro.isaj@gmail.com
# Vision Interface to use for viewing and saving images from Video Camera Input
# with Analysis of area from experiments with well ----- version 0.1.1

import cv2
import os
import glob
import numpy as np
import pandas as pd
from skimage import morphology
import matplotlib.pyplot as plt


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
    clh = cv2.createCLAHE(clipLimit=5)
    clh_img = clh.apply(image_gray_)

    blurred = cv2.GaussianBlur(clh_img, (5, 5), 0)

    return clh_img, blurred


def f_sorted(files_, id_sys):
    symbol = '\\' if id_sys == 0 else '/'
    ids = []
    for f in files_:
        parts = f.split(symbol)
        name_i = parts[len(parts) - 1]
        ide = name_i.split('.')[0].split('_')[1]
        ids.append(ide)
    ids = list(map(int, ids))
    ids.sort(key=int)
    file_r = []
    for i in range(len(files_)):
        parts = files_[i].split(symbol)
        name_i = parts[len(parts) - 1]
        name = name_i.split('.')
        exp = name[0].split('_')[0]
        n_name = exp + '_' + str(ids[i]) + '.' + name[1]
        if id_sys == 0:
            n_file = parts[0]
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

    if i < len(filenames):
        name = filenames[i]
        parts = name.split(symbol)
        name_i = parts[len(parts)-1]
        # read image
        image_ = cv2.imread(name)
    else:
        image_, name_i = [], []

    return filenames, image_, name_i, len(filenames)


def p_circle(binary_):
    contour = cv2.findContours(binary_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = contour[0] if len(contour) == 2 else contour[1]
    contour = sorted(contour, key=cv2.contourArea, reverse=True)

    (x_, y_), radius_ = cv2.minEnclosingCircle(contour[0])
    (x_, y_), radius_ = (int(np.round(x_)), int(np.round(y_))), int(np.round(radius_))
    return x_, y_, radius_


def circular_reg(img):
    ima_gray, final_ima = preprocessing(img)
    '''
    gradient = rank.gradient(final_ima, disk(15))
    thresh = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 125, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    '''
    thresh = cv2.threshold(final_ima, 125, 255, cv2.THRESH_TOZERO_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel, iterations=3)
    binary = morphology.remove_small_holes(binary.astype(np.bool), area_threshold=500, connectivity=2)
    binary = binary.astype(np.uint8)

    x_, y_, radius_ = p_circle(binary)

    image_r = np.copy(img)
    cv2.circle(image_r, (x_, y_), radius_, (35, 255, 12), 3)
    return image_r, x_, y_, radius_


def seq_circular(img, x_, y_, radius_):
    image_r = np.copy(img)
    cv2.circle(image_r, (x_, y_), radius_, (35, 255, 12), 3)
    return image_r


def c_grow(img_bin, x_, y_, radius_ini, error_):
    grow = np.zeros_like(img_bin, dtype=np.uint8)
    cv2.circle(grow, (x_, y_), radius_ini, 255, -1)
    grow_ref = np.copy(grow)
    # compute regions
    idx = np.where(grow_ref == 255)
    grow[idx] = 1
    area_tot = np.sum(grow_ref == 255)
    # replace well detection
    grow_ref[idx] = img_bin[idx]
    # compute area
    area_well = np.sum(grow_ref == 1)
    relation_ = np.round(area_well / area_tot, 2)

    ctr_ = False
    if relation_ < error_:
        ctr_ = True

    return ctr_, grow


def control_grow(img, x_, y_):
    thresh = cv2.threshold(img, 95, 255, cv2.THRESH_TOZERO_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary = morphology.remove_small_holes(binary.astype(np.bool), area_threshold=500, connectivity=2)
    binary = binary.astype(np.uint8)

    x1, y1, radius1 = p_circle(binary)
    xp_, yp_ = np.array([x_, y_]), np.array([x1, y1])

    distance = dist(xp_, yp_)
    ctr_ = False
    if distance <= 80 and radius1 >= 30:
        ctr_ = True

    return ctr_


def wells_analysis(img, k, x_, y_, radius_, ctr_, rad_ini_):
    ima_gray, final_ima = preprocessing(img)

    if ctr_ is False:
        ctr = control_grow(final_ima, x_, y_)
    else:
        ctr = True

    # well interest region
    thresh = cv2.adaptiveThreshold(final_ima, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 75, 35)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    markers = morphology.remove_small_holes(binary.astype(np.bool), area_threshold=800, connectivity=2)
    markers = markers.astype(np.uint8)

    ctr2, ima_grow = c_grow(markers, x_, y_, rad_ini_, 0.05)

    # complementary grow region
    thresh1 = cv2.adaptiveThreshold(final_ima, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 6)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    binary1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary1 = cv2.morphologyEx(binary1, cv2.MORPH_CLOSE, kernel, iterations=3)
    markers1 = morphology.remove_small_holes(binary1.astype(np.bool), area_threshold=800, connectivity=2)
    markers1 = markers1.astype(np.uint8)
    # remove region outside interest area
    arr = markers1 > 0
    markers1 = morphology.remove_small_objects(arr, min_size=9000, connectivity=1).astype(np.uint8)

    markers = cv2.bitwise_or(markers, ima_grow)
    if ctr2:
        rad_ini_ += 2
    if ctr:
        markers = cv2.bitwise_or(markers, markers1)

    mask = np.zeros_like(markers, dtype=np.uint8)
    cv2.circle(mask, (x_, y_), radius_, 255, -1)
    idx = np.where(mask == 255)
    # compute total area
    area_total = np.sum(mask == 255)

    # replace well detection
    mask[idx] = markers[idx]

    # compute area
    area_well = np.sum(mask == 1)
    percent_well = np.round((area_well * 100) / area_total, 2)

    print('Processing image  ----->  ' + str(k))
    table = [['Total area  : ', str(area_total)],
             ['Well area   : ', str(area_well)],
             ['Percentage  : ', str(percent_well)]]
    for line in table:
        print('{:>10} {:>10}'.format(*line))
    print('-------------------------------------------------------------------------')

    return mask, ctr, area_total, area_well, percent_well, rad_ini_
