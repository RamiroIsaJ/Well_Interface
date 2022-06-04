# Ramiro Isa-Jara, ramiro.isaj@gmail.com
# Vision Interface to use for viewing and saving images from Video Camera Input
# with Analysis of area from experiments with well ----- version 0.2.1

import cv2
import math
import time as tm
import numpy as np
import pandas as pd
import Wells_def as wd
import PySimpleGUI as sg
from datetime import datetime


# -------------------------------
# Adjust size screen
# -------------------------------
Screen_size = 10
# -------------------------------
sg.theme('LightGrey1')
m1, n1 = 450, 400
img = np.ones((m1, n1, 1), np.uint8)*255

layout1 = [[sg.Radio('Windows', "RADIO1", default=True, key='_SYS_')],
           [sg.Radio('Linux', "RADIO1")], [sg.Text('')]]

layout2 = [[sg.Checkbox('*.jpg', default=True, key="_IN1_")], [sg.Checkbox('*.png', default=False, key="_IN2_")],
           [sg.Checkbox('*.tiff', default=False, key="_IN3_")]]

idx = ['0']
layout3 = [[sg.Radio('Minutes', "RADIO2", default=True, key='_TMI_'), sg.Radio('Seconds', "RADIO2")],
           [sg.Text('Time wait:', size=(10, 1)), sg.InputText('5', key='_INF_', size=(6, 1))],
           [sg.Text('Id_ini image:', size=(10, 1)), sg.InputText('1', key='_IDI_', size=(6, 1))],
           [sg.Text('Video input: ', size=(10, 1)),
            sg.Combo(values=idx, size=(5, 1), enable_events=True, key='_VIN_')]]

layout3a = [[sg.Radio('On-line', "RADIO3", default=True, key='_MET_')],
            [sg.Radio('Off-line', "RADIO3")], [sg.Text('')]]

layout4 = [[sg.Text('Radius Well: ', size=(10, 1)), sg.InputText('700', size=(6, 1), key='_RAW_'),
            sg.Text('um.', size=(8, 1)),
            sg.Text('Name images: ', size=(12, 1)), sg.InputText('Experiment1_', size=(15, 1), key='_NGR_')],
           [sg.Text('Save/Read Images: ', size=(15, 1)), sg.InputText(size=(32, 1), key='_SIM_'), sg.FolderBrowse()],
           [sg.Text('Save Results: ', size=(15, 1)), sg.InputText(size=(32, 1), key='_DBN_'), sg.FolderBrowse()],
           ]

layout6 = [[sg.T("", size=(25, 1)), sg.Text('NO PROCESS', size=(33, 1), key='_MES_', text_color='DarkRed')]]

layout7 = [[sg.T("", size=(20, 1)), sg.Text('Current time: ', size=(10, 1)), sg.Text('', size=(10, 1), key='_TAC_')],
           [sg.T("", size=(5, 1)),
            sg.Text('Start time: ', size=(8, 1)), sg.Text('-- : -- : --', size=(10, 1), key='_TIN_', text_color='blue'),
            sg.T("", size=(7, 1)),
            sg.Text('Finish time: ', size=(8, 1)), sg.Text('-- : -- : --', size=(10, 1), key='_TFI_', text_color='red')],
           [sg.Text('Time remaining : ', size=(17, 1)), sg.InputText('', key='_RES_', size=(7, 1)),
            sg.Text('...', size=(4, 1), key='_ITM_'),
            sg.Text('Current image: ', size=(14, 1)), sg.InputText('', key='_ICR_', size=(8, 1)), sg.T("", size=(1, 1))],
           [sg.Text('Total save images: ', size=(17, 1)), sg.InputText('', key='_CIM_', size=(7, 1)),
            sg.T("", size=(4, 1)),
            sg.Text('Total image set: ', size=(14, 1)), sg.InputText('', key='_CTI_', size=(8, 1)), sg.T("", size=(1, 1))],
           [sg.Text('Area well: ', size=(14, 1)), sg.InputText('', key='_AWL_', size=(10, 1)), sg.Text('um.', size=(4, 1)),
            sg.T("", size=(4, 1))],
           [sg.Text('Area yeast: ', size=(14, 1)), sg.InputText('', key='_AYE_', size=(10, 1)), sg.Text('um.', size=(4, 1)),
            sg.Text('% Area yeast: ', size=(14, 1)), sg.InputText('', key='_PYE_', size=(8, 1)), sg.T("", size=(1, 1))],
           ]


v_image = [sg.Image(filename='', key="_IMA_")]
# columns
col_1 = [[sg.Frame('', [v_image])]]
col_2 = [[sg.Frame('Operative S.: ', layout1, title_color='Blue'),
          sg.Frame('Type image: ', layout2, title_color='Blue'), sg.Frame('Method: ', layout3a, title_color='Blue'),
          sg.Frame('Settings: ', layout3, title_color='Blue')],
         [sg.Frame('Settings and directories: ', layout4, title_color='Blue')],
         [sg.T(" ", size=(8, 1)), sg.Button('View', size=(8, 1)), sg.Button('Save', size=(8, 1)),
          sg.Button('Analysis', size=(8, 1)), sg.Button('Finish', size=(8, 1))],
         [sg.Frame('', layout6)], [sg.Frame('', layout7)]]

layout = [[sg.Column(col_1), sg.Column(col_2)]]

# Create the Window
window = sg.Window('WELLS Analysis Interface', layout, font="Helvetica "+str(Screen_size), finalize=True)
window['_IMA_'].update(data=wd.bytes_(img, m1, n1))
# ----------------------------------------------------------------------------------
time, id_image, id_sys, method, i = 0, 1, 0, 0, -1
x, y, radius, cont_ini, area_total = 0, 0, 0, 0, 0
view_, save_, analysis_, finish_, control, analysis_b, save_only = False, False, False, False, False, False, False
video, name, image, ini_time, type_i, filename = None, None, None, None, None, None
path_des1, path_des2, filenames, cords_well = [], [], [], []
segYes = wd.SegmentYeast()
segYes.build_filters()
# Variable to save results
results = pd.DataFrame(columns=['Image', 'Percentage', 'Area'])
# -----------------------------------------------------------------------------------

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read(timeout=10)
    window.Refresh()
    now = datetime.now()
    now_time = now.strftime("%H : %M : %S")
    window['_TAC_'].update(now_time)

    if event == sg.WIN_CLOSED:
        break

    if event == '_VIN_':
        index = wd.camera_idx()
        window.Element('_VIN_').update(values=index)

    if event == 'Finish' or finish_:
        print('FINISH')
        window['_MES_'].update('Process finished')
        if view_:
            now_time = now.strftime("%H : %M : %S")
            window['_TFI_'].update(now_time)
            view_, save_ = False, False
            video.release()

        if finish_ or analysis_ or analysis_b:
            now_time = now.strftime("%H : %M : %S")
            window['_TFI_'].update(now_time)
            finish_, analysis_, analysis_b, save_only = False, False, False, False
            i, filenames, data, x, y, radius, cont, ini_rad = -1, [], [], 0, 0, 0, 0, 15
            # save results and events
            header = values['_NGR_'].split('_')[0]
            wd.save_csv_file(results, path_des2, header)
            wd.graph_data(path_des2, header)
            for i in range(1, 100, 25):
                sg.OneLineProgressMeter('Saving RESULTS in CSV files', i + 25, 100, 'single')
                tm.sleep(1)

        if save_only:
            now_time = now.strftime("%H : %M : %S")
            window['_TFI_'].update(now_time)
            finish_, analysis_, analysis_b, save_only = False, False, False, False

    if event == 'View':
        window['_MES_'].update('TEST VIDEO INPUT')
        idx = values['_VIN_']
        if view_ is False and idx != '':
            video = cv2.VideoCapture(int(idx))
            view_ = True
        elif idx == '':
            sg.Popup('Error', ['Not selected Input Video'])
        else:
            sg.Popup('Warning', ['Process is running'])

    if view_:
        ret, image = video.read()
        if ret:
            window['_IMA_'].update(data=wd.bytes_(image, m1, n1))
            window['_MES_'].update('View video frame')

    if event == 'Save':
        name = values['_NGR_']
        id_image = int(values['_IDI_'])
        now_time1 = now.strftime("%H : %M : %S")
        window['_TIN_'].update(now_time1)
        window['_TFI_'].update('-- : -- : --')
        # -------------------------------------------------------------------
        method = 0 if values['_MET_'] else 1
        id_sys = 0 if values['_SYS_'] else 1
        # -------------------------------------------------------------------
        if values['_TMI_']:
            window['_ITM_'].update('min')
        else:
            window['_ITM_'].update('sec')
        # -------------------------------------------------------------------
        if values['_IN2_']:
            type_i = ".png"
        elif values['_IN3_']:
            type_i = ".tiff"
        else:
            type_i = ".jpg"
        # -------------------------------------------------------------------
        window['_MES_'].update('ONLY SAVE IMAGES')
        print('SAVE IMAGES')
        if id_sys == 0:
            path_des1 = wd.update_dir(values['_SIM_']) + "\\"
            path_des1 = r'{}'.format(path_des1)
        else:
            path_des1 = values['_SIM_'] + '/'
        # ------------------------------------------------------------------
        if len(path_des1) > 1 and len(name) > 1 and save_ is False:
            ini_time = datetime.now()
            time = float(values['_INF_'])
            save_, save_only = True, True
        elif len(path_des1) > 1 and len(name) > 1 and save_:
            sg.Popup('Warning', ['Process is running'])
        else:
            sg.Popup('Error', ['Information not valid'])

    if event == 'Analysis':
        now_time1 = now.strftime("%H : %M : %S")
        window['_TIN_'].update(now_time1)
        window['_TFI_'].update('-- : -- : --')
        id_image = int(values['_IDI_'])
        radius = float(values['_RAW_'])
        area_total = np.round(math.pi * radius**2, 2)
        # -------------------------------------------------------------------
        method = 0 if values['_MET_'] else 1
        id_sys = 0 if values['_SYS_'] else 1
        # -------------------------------------------------------------------
        if values['_IN2_']:
            type_i = ".png"
        elif values['_IN3_']:
            type_i = ".tiff"
        else:
            type_i = ".jpg"
        # -------------------------------------------------------------------
        if method == 0:
            window['_MES_'].update('ONLINE PROCESSING')
            print('SAVE and Image PROCESSING')
            # -------------------------------------------------------------------
            if values['_TMI_']:
                window['_ITM_'].update('min')
            else:
                window['_ITM_'].update('sec')
            # -------------------------------------------------------------------
            if id_sys == 0:
                path_des1 = wd.update_dir(values['_SIM_']) + "\\"
                path_des1 = r'{}'.format(path_des1)
                path_des2 = wd.update_dir(values['_DBN_']) + "\\"
                path_des2 = r'{}'.format(path_des2)
            else:
                path_des1 = values['_DGR_'] + '/'
                path_des2 = values['_DBN_'] + '/'
            # ------------------------------------------------------------------
            if len(path_des1) > 1 and len(path_des2) > 1 and save_ is False:
                ini_time = datetime.now()
                name = values['_NGR_']
                time = float(values['_INF_'])
                save_, analysis_b = True, True
            elif len(path_des1) > 1 and len(path_des2) > 1 and save_:
                sg.Popup('Warning', ['Process is running'])
            else:
                sg.Popup('Error', ['Information not valid'])
        else:
            window['_MES_'].update('OFFLINE PROCESSING')
            print('ONLY Image PROCESSING')
            if id_sys == 0:
                path_des1 = wd.update_dir(values['_SIM_']) + "\\"
                path_des1 = r'{}'.format(path_des1)
                path_des2 = wd.update_dir(values['_DBN_']) + "\\"
                path_des2 = r'{}'.format(path_des2)
            else:
                path_des1 = values['_SIM_'] + '/'
                path_des2 = values['_DBN_'] + '/'
            # ------------------------------------------------------------------
            if len(path_des2) > 1 and len(path_des1) > 1 and analysis_ is False:
                ini_time = datetime.now()
                time = float(values['_INF_'])
                analysis_ = True
            elif len(path_des2) > 1 and len(path_des1) > 1 and analysis_:
                sg.Popup('Warning', ['Process is running'])
            else:
                sg.Popup('Error', ['Information not valid'])

    if save_:
        filename = path_des1 + name + str(id_image) + type_i
        # -----------------------------------------------------------------
        now_time_ = datetime.now()
        delta = now_time_ - ini_time
        time_sleep = delta.seconds
        if values['_TMI_']:
            time_sleep /= 60
        rest_time = np.round(time - time_sleep, 4)
        window['_RES_'].update(rest_time)
        # -----------------------------------------------------------------
        if (view_ and time == time_sleep) or id_image == 1:
            print(filename)
            print('SAVE IMAGE SUCCESSFULLY')
            cv2.imwrite(filename, image)
            window['_CIM_'].update(id_image)
            ini_time = datetime.now()
            if save_only:
                id_image += 1
            if analysis_b:
                analysis_ = True

    if analysis_:
        i += 1
        window['_ICR_'].update(i)
        if method == 0:
            image_ = cv2.imread(filename)
            id_image += 1
            analysis_ = False
        else:
            filenames, image_, filename, total_i = wd.load_image_i(path_des1, i, type_i, filenames, id_sys)
            window['_CTI_'].update(total_i)
            if len(image_) == 0 and i > 0:
                finish_ = True
                continue
            elif len(image_) == 0 and i == 0:
                finish_ = True
                continue

        cont_ini, cords_well, ima_res, x, y, radius = segYes.ini_well(image_, cont_ini, cords_well)
        # main process to detect area
        percent, img_f = segYes.well_main(path_des2, ima_res, filename, type_i, i, x, y, radius)
        area_yeast = np.round((area_total * percent) / 100, 2)
        # save results
        results = results.append({'Image': filename, 'Percentage': percent, 'Area': area_yeast}, ignore_index=True)
        print('Processing image  ----->  ' + str(i))
        table = [['Total area      : ', str(area_total)],
                 ['Detected area   : ', str(area_yeast)],
                 ['Percentage      : ', str(percent)]]
        for line in table:
            print('{:>10} {:>10}'.format(*line))
        print('-------------------------------------------------------------------------')
        window['_IMA_'].update(data=wd.bytes_(img_f, m1, n1))
        window['_AWL_'].update(area_total)
        window['_AYE_'].update(area_yeast)
        window['_PYE_'].update(percent)


print('CLOSE WINDOW')
window.close()
