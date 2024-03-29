# Ramiro Isa-Jara, ramiro.isaj@gmail.com
# Vision Interface to use for viewing and saving images from Video Camera Input
# with Analysis of area from experiments with well ----- version 0.2.1

import cv2
import os
import glob
import numpy as np
import pandas as pd
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu


class SegmentYeast:
    def __init__(self):
        self.img, self.img_ = None, None
        self.buffer_size = 0
        self.filters, self.f_contours = [], []

    def dist(self, xp, yp):
        return np.sqrt(np.sum((xp - yp) ** 2))

    def build_filters(self):
        k_size, sigma = 21, [4.0]
        for s in sigma:
            for theta in np.arange(0, np.pi, np.pi / 4):
                kern = cv2.getGaborKernel((k_size, k_size), s, theta, 10.0, 0.9, 0, ktype=cv2.CV_32F)
                kern /= 1.5 * kern.sum()
                self.filters.append(kern)

    def apply_gabor(self, img_g):
        gabor_img_ = np.zeros_like(img_g)
        for kern in self.filters:
            np.maximum(gabor_img_, cv2.filter2D(img_g, cv2.CV_8UC3, kern), gabor_img_)
        return gabor_img_

    def preprocessing(self):
        image_gray_ = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        clh = cv2.createCLAHE(clipLimit=5.0)
        clh_img = clh.apply(image_gray_)
        blurred = cv2.GaussianBlur(clh_img, (5, 5), 0)
        return clh_img, blurred

    def calculate_contour(self, binary_):
        contours, hierarchy = cv2.findContours(binary_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area = []
        for c in contours:
            area.append(cv2.contourArea(c))
        area_min = max(area) / 3
        for c in contours:
            area = cv2.contourArea(c)
            if area > area_min:
                self.f_contours.append(c)

    def binary_contours(self, img_s, binary_, x_, y_, radius_):
        img_c = np.copy(img_s)
        contours, hierarchy = cv2.findContours(binary_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contours_f = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                dist_rel = self.dist(np.array([cx, cy]), np.array([x_, y_])) / radius_
                if dist_rel < 0.50:
                    contours_f.append(c)
            else:
                contours_f.append(c)
        cv2.drawContours(img_c, contours_f, -1, (0, 0, 255), 2)
        return img_c, len(contours_f)

    def p_circle(self, binary_):
        cords = []
        self.calculate_contour(binary_)
        contour = sorted(self.f_contours, key=cv2.contourArea, reverse=True)
        for contour1 in contour:
            (x_, y_), radius_ = cv2.minEnclosingCircle(contour1)
            cords.append([x_, y_, radius_])
        cords = np.array(cords)
        (x_, y_), radius_ = cv2.minEnclosingCircle(contour[0])
        if len(contour) > 1:
            idx = np.where(cords == np.max(cords[:, 2]))[0]
            x1, y1 = cords[idx, 0], cords[idx, 1]
            if self.dist(np.array([x_, y_]), np.array([x1, y1])) < 200:
                x_, y_, radius_ = np.round((x_ + x1) / 2), np.round((y1 + y_) / 2), cords[idx, 2] + 5
        radius_ -= 15
        self.f_contours = []
        (x_, y_), radius_ = (int(np.round(x_)), int(np.round(y_))), int(np.round(radius_))
        return x_, y_, radius_

    def well_region(self):
        ima_gray, final_ima = self.preprocessing()
        gabor_img = self.apply_gabor(final_ima)
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
        x_, y_, radius_ = self.p_circle(markers)
        image_r = np.copy(self.img)
        cv2.circle(image_r, (x_, y_), radius_, (35, 255, 12), 3)
        return image_r, x_, y_, radius_

    def eval_cords(self, cords_, x_, y_, radius_):
        cords_ = np.array(cords_)
        xp, yp, rdp = np.round(np.average(cords_[:, 0])), np.round(np.average(cords_[:, 1])), np.round(
            np.average(cords_[:, 2]))
        distance, rel = self.dist(np.array([x_, y_]), np.array([xp, yp])), np.round(min(rdp, radius_) / max(rdp, radius_), 2)
        if distance < 50 and rel > 0.92:
            return True
        else:
            return False

    def seq_circular(self, cords_):
        image_r, cords_ = np.copy(self.img), np.array(cords_)
        x_ = int(np.round(np.average(cords_[:, 0])))
        y_ = int(np.round(np.average(cords_[:, 1])))
        radius_ = int(np.round(np.average(cords_[:, 2])))
        cv2.circle(image_r, (x_, y_), radius_, (35, 255, 12), 3)
        return image_r, x_, y_, radius_

    def gray_circle(self, binary_g, x_, y_):
        contour = cv2.findContours(binary_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contour[0] if len(contour) == 2 else contour[1]
        contour = sorted(contour, key=cv2.contourArea, reverse=True)
        (xr, yr), rad_r = cv2.minEnclosingCircle(contour[0])
        distance = self.dist(np.array([x_, y_]), np.array([xr, yr]))
        if distance < 30 and rad_r > 50:
            binary_g = np.zeros_like(binary_g, dtype=np.uint8)
            cv2.circle(binary_g, (x_, y_), int(rad_r), 255, -1)
        return binary_g

    def sobel_filter(self):
        ima_gray = cv2.cvtColor(self.img_, cv2.COLOR_BGR2GRAY)
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

    def hsv_space(self, param_):
        hsv = rgb2hsv(self.img_)
        image = (255 * hsv[:, :, 1]).astype(np.uint8)
        th_hsv_ = threshold_otsu(image)
        if param_ < 75:
            th_value = 90 if th_hsv_ < 100 else 100
        else:
            th_value = 50 if th_hsv_ < 100 else 60
        thresh_hsv = (255 * np.array(image > th_value)).astype(np.uint8)
        return thresh_hsv, th_value

    def roi_region(self, bin_img, x_, y_, radius_, val_):
        roi_img = np.zeros_like(bin_img, dtype=np.uint8)
        rad_n = radius_ - val_
        cv2.circle(roi_img, (x_, y_), rad_n, 255, -1)
        idx = np.where(roi_img == 255)
        roi_img[idx] = bin_img[idx]
        return roi_img

    def opera_sobel(self, bin_sobel_, x_, y_, radius_, ctr_):
        morph_img = self.roi_region(bin_sobel_, x_, y_, radius_, 40)
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

    def opera_sobel_hsv(self, bin_sob_hsv, x_, y_, radius_):
        morph_img = self.roi_region(bin_sob_hsv, x_, y_, radius_, 40)
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

    def opera_gray(self, bin_gray_, x_, y_, radius_):
        morph_gray = self.roi_region(bin_gray_, x_, y_, radius_, 50)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(morph_gray, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = morphology.remove_small_holes(binary.astype(np.bool), area_threshold=1000, connectivity=1)
        binary = binary.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel, iterations=1)
        arr = binary > 0
        binary = morphology.remove_small_objects(arr, min_size=1000, connectivity=1)
        binary = morphology.remove_small_holes(binary.astype(np.bool_), area_threshold=50000, connectivity=1)
        binary = (255 * binary).astype(np.uint8)
        binary = self.gray_circle(binary, x_, y_)
        return binary

    def binary_regions(self, x_, y_, radius_):
        _, original_gray = self.preprocessing()
        thresh_gray = threshold_otsu(original_gray)
        norm_img = cv2.normalize(self.img, None, alpha=-0.1, beta=1.1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.img_ = (255 * norm_img).astype(np.uint8)
        # ---> obtain binary regions of images
        bin_sobel, th_sobel, th_norm = self.sobel_filter()
        bin_gray = cv2.threshold(original_gray, thresh_gray, 255, cv2.THRESH_TOZERO_INV)[1]
        # ---> choose better binary image
        param = np.round(thresh_gray * th_norm)
        bin_hsv, th_hsv = self.hsv_space(param)
        if (param >= 70 and th_sobel <= 145) or (param >= 79 and th_sobel >= 150 and th_hsv > 133) or \
                (param >= 80 and th_sobel >= 150):
            rel_ = np.round(min(th_hsv, th_sobel) / max(th_hsv, th_sobel), 2)
            if 0.80 < rel_ < 0.95:
                ctr = True if param > 85 or th_sobel > 140 else False
                ima_binary = self.opera_sobel(bin_sobel, x_, y_, radius_, ctr)
                return ima_binary
            else:
                bin_hsv_sobel = cv2.bitwise_or(bin_sobel, bin_hsv)
                ima_binary = self.opera_sobel_hsv(bin_hsv_sobel, x_, y_, radius_)
                return ima_binary
        else:
            ima_binary = self.opera_gray(bin_gray, x_, y_, radius_)
        return ima_binary

    def well_analysis(self, img_s, x_, y_, radius_, ctr_b, bin_alt):
        if ctr_b == 0:
            binary = self.binary_regions(x_, y_, radius_)
        else:
            binary = np.copy(bin_alt)
        # analysis values
        idx = np.where(binary == 255)
        seg_image, total_cont = self.binary_contours(img_s, binary, x_, y_, radius_)
        # compute segmented area
        area_detected_ = np.sum(binary == 255)
        # compute well area
        well = np.zeros_like(binary, dtype=np.uint8)
        cv2.circle(well, (x_, y_), radius_, 255, -1)
        area_well_ = np.sum(well == 255)
        # control zero area
        if total_cont == 0:
            percent_well_ = 0.0
        else:
            percent_well_ = np.round((area_detected_ * 100) / area_well_, 2)
        return binary, seg_image, percent_well_

    def binary_seq(self, binary_):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        bina_ = cv2.morphologyEx(binary_, cv2.MORPH_CLOSE, kernel, iterations=1)
        arr = bina_ > 0
        bina_ = morphology.remove_small_objects(arr, min_size=100, connectivity=1)
        bina_ = morphology.remove_small_holes(bina_.astype(np.bool_), area_threshold=50000, connectivity=1)
        return (255 * bina_).astype(np.uint8)

    def well_main(self, des, img_r, ima_name, i, ctr_exp, bin_ref, prt_ref, cont_zero, x_, y_, radius_):
        binary_, img_final, prt_well = self.well_analysis(img_r, x_, y_, radius_, 0, 0)
        print('----->' + str(prt_well))
        if i == 0 and cont_zero == 0:
            bin_ref, prt_ref = np.copy(binary_), np.copy(prt_well)
        else:
            if cont_zero > 2 and prt_well > 0.0:
                bin_ref, prt_ref = np.copy(binary_), np.copy(prt_well)
                cont_zero = 0

        if ctr_exp == 0:
            if prt_well > 0:
                if prt_ref > prt_well or np.round(prt_ref / prt_well, 2) > 0.97:
                    bina_ = cv2.bitwise_or(bin_ref, binary_)
                    bina_ = self.binary_seq(bina_)
                    binary_, img_final, prt_well = self.well_analysis(img_r, x_, y_, radius_, 1, bina_)
                    bin_ref, prt_ref = np.copy(binary_), np.copy(prt_well)
                    cont_zero = 0
            else:
                cont_zero += 1
        else:
            if prt_well > 0:
                if prt_well < prt_ref and np.round(prt_well / prt_ref, 2) > 0.95:
                    bina_ = cv2.bitwise_and(bin_ref, binary_)
                    bina_ = self.binary_seq(bina_)
                    binary_, img_final, prt_well = self.well_analysis(img_r, x_, y_, radius_, 1, bina_)
                    bin_ref, prt_ref = np.copy(binary_), np.copy(prt_well)
                elif prt_well < prt_ref and np.round(prt_well / prt_ref, 2) < 0.95:
                    bina_ = cv2.bitwise_or(bin_ref, binary_)
                    bina_ = self.binary_seq(bina_)
                    binary_, img_final, prt_well = self.well_analysis(img_r, x_, y_, radius_, 1, bina_)
                    bin_ref, prt_ref = np.copy(binary_), np.copy(prt_well)
                cont_zero = 0
            else:
                cont_zero += 1

        # Output image
        print(des, ima_name)
        root_des = des + ima_name
        cv2.imwrite(root_des, img_final)
        return prt_well, img_final, bin_ref, prt_ref, cont_zero

    def ini_well(self, img, cont_ini, cords_well):
        self.img = img
        if cont_ini < 6:
            ima_res, x, y, radius = self.well_region()
            cords_well.append([x, y, radius])
        else:
            _, x, y, radius = self.well_region()
            ctr_range = self.eval_cords(cords_well, x, y, radius)
            if ctr_range:
                cords_well.append([x, y, radius])
            ima_res, x, y, radius = self.seq_circular(cords_well)
        cont_ini += 1
        return cont_ini, cords_well, ima_res, x, y, radius


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
    y = np.array(data_['Area'])
    x = np.arange(1, len(y) + 1, 1)
    fig = plt.figure()
    plt.plot(x, y, 'o-')
    plt.grid()
    plt.xlabel('N. of image')
    plt.ylabel('Area Yeast (um.)')
    _root_fig = os.path.join(des, 'Percentage_'+header+'.jpg')
    fig.tight_layout()
    plt.savefig(_root_fig)
    plt.close()


def bytes_(img, m, n):
    ima = cv2.resize(img, (m, n))
    return cv2.imencode('.png', ima)[1].tobytes()


def camera_idx():
    # checks the first 10 indexes.
    index = 3
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
    if i < len(filenames):
        name = filenames[i]
        parts = name.split(symbol)
        name_i = parts[len(parts)-1]
        image_ = cv2.imread(name)
    else:
        image_, name_i = [], []
    return filenames, image_, name_i, len(filenames)
