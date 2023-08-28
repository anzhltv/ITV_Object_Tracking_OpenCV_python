import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyautogui

from tracker import *


def clean_array(arr_id):
    for i in range(len(arr_id)):
        arr_id[i] = 0


def if_same_object(arr_id, array_hist, count_same, id_save):
    # если был найден существующий объект, то очистка собранных гистограмм и массива id + увеличение числа совпадающих объектов
    if max(arr_id) >= count_frame // 6:
        array_hist[id_save - count_same] = 0
        count_same += 1
    clean_array(arr_id)
    return count_same


def find_max_same_id(arr_id, count_same, id):
    # если максимальное совпадение больше чем 1/6 от общего числа кадров проверки
    if max(arr_id) >= count_frame // 6:
        print(count_frame // 6)
        # то берем найденный id
        correct_id = arr_id.index(max(arr_id))
    else:
        # иначе следующий по порядку
        correct_id = id - count_same
    return correct_id


def search_compare(global_id, opt_param, arr_id):
    obj_id, obj_k, max = (0, 0, 0)
    for i in range(global_id):
        compareHist = cv2.compareHist(
            array_hist[global_id].astype(np.float32),
            array_hist[i].astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
        # print('COMPARE: ', compareHist, 'i = ', i)
        if compareHist > opt_param and compareHist != 1:
            if max < compareHist:
                max = compareHist
                obj_id = i
                obj_k += 1
    if obj_k > 0:
        # увеличиваем значение элемента с индексом id у которого максимальное совпадение с гистограммой
        arr_id[obj_id] += 1
    return


def detect_countour(size, contours, detections):
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > size:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])


def get_screen_resolution():
    screen_width, screen_height = pyautogui.size()
    return screen_width, screen_height


def if_border(y, h):
    if y < 50:
        y1 = y + h + 30
    else:
        y1 = y - 15
    return y1


def image_hist(frame_hist, arr_hist, count_id, count_frame):
    alpha = 0.1
    if count_frame > 0:
        b, g, r = cv2.split(frame_hist)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        frame_hist = cv2.merge((b_eq, g_eq, r_eq))
        frame_hist = cv2.calcHist(
            [frame_hist], [0, 1, 2], None, [160, 160, 160], [0, 256, 0, 256, 0, 256]
        )
        cv2.normalize(frame_hist, None, 0, 1.0, cv2.NORM_MINMAX)
        # если первая запись гистограммы - записываем всю, иначе как 0.1 нового значения и 0.9 старого
        if np.all(np.abs(arr_hist[count_id]) < EPSILON):
            arr_hist[count_id] = frame_hist
        else:
            arr_hist[count_id] = alpha * (frame_hist) + (1 - alpha) * arr_hist[count_id]

    return


parser = argparse.ArgumentParser(description="Input video file")
parser.add_argument("video_path1", type=str, help="Path to the video file number 1")
parser.add_argument("video_path2", type=str, help="Path to the video file number 2")
args = parser.parse_args()

path1 = args.video_path1
path2 = args.video_path2

cap1 = cv2.VideoCapture(path1)
cap2 = cv2.VideoCapture(path2)
tracker1 = EuclideanDistTracker()
tracker2 = EuclideanDistTracker()

EPSILON = 1e-6
width, height = get_screen_resolution()
print(width, height)
N = width * height

size1, size2 = (
    int(round(0.02 * width * height)),
    int(round(0.0034 * width * height)),
)  # минимальные размеры боксов

id2_save, id1_save = (-1, -1)
count_frame = 40

ret1, frame1_1 = cap1.read()
ret2, frame2_2 = cap2.read()
height_frame, width_frame, _ = frame1_1.shape

arr_id1, arr_id2 = ([0] * 10, [0] * 10)
array_hist = np.zeros((200, 160, 160, 160))  # массив для накопления гистограмм объектов

opt_param1, opt_param2 = (0.5, 0.21)  # границы сравнения гистограмм для каждого кадра
count_same = 0  # переменная для подсчета одинаковых объектов

report2 = True
output_frames = []

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    roi1 = frame1[0:height_frame, width_frame // 8 : width_frame // 2]
    roi2 = frame2[0:height_frame, int(width_frame // 3.2) : int(width_frame / 1.6)]
    # 1. Object Detection
    diff1 = cv2.absdiff(frame1_1, frame1)
    diff1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)
    _, mask1 = cv2.threshold(diff1, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    mask1 = cv2.dilate(mask1, kernel, iterations=3)

    diff2 = cv2.absdiff(frame2_2, frame2)
    diff2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)
    diff2 = diff2[0:height_frame, int(width_frame // 3.2):int(width_frame / 1.6)]
    _, mask2 = cv2.threshold(diff2, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    mask2 = cv2.dilate(mask2, kernel, iterations=7)

    contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections1 = []
    detections2 = []
    detect_countour(size1, contours1, detections1)
    detect_countour(size2, contours2, detections2)

    # трекинг кадра с первой камеры
    boxes_ids1 = tracker1.update(detections1)
    report1 = True
    for box_id in boxes_ids1:
        x, y, w, h, id1 = box_id
        # если надпись окажется за пределами
        y1 = if_border(y, h)
        report1 = False
        frame_plt1 = frame1[y : y + h, x : x + w]

        # если новый объект
        if id1 != id1_save:
            # проверяем содержимое массива id и увеличиваем кол-во одинаковых объектов
            count_same = if_same_object(arr_id1, array_hist, count_same, id1_save)
            # если нет объекта во второй камере
            if report2 == True:
                # проверяем содержимое массива id и увеличиваем кол-во одинаковых объектов
                count_same = if_same_object(arr_id2, array_hist, count_same, id2_save)
            # обновляем счетчик кадров для гистограмм и для детектора
            count_frame1 = count_frame

        global_id1 = id1 - count_same
        id1_save = id1

        # копим новые гистограммы
        image_hist(frame_plt1, array_hist, global_id1, count_frame1)

        # пока счетчик кадров больше нуля, считаем сходства новой гистограммы с уже имеющимися
        if count_frame1 > 0:
            search_compare(global_id1, opt_param1, arr_id1)
            count_frame1 -= 1
            correct_id1 = global_id1
        else:
            # иначе ищем id с макс совпадением
            correct_id1 = find_max_same_id(arr_id1, count_same, id1)
            id1 = correct_id1
            cv2.putText(
                frame1,
                "Object " + str(id1),
                (x, y1),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 3)
        tracker2.id_count = tracker1.id_count

    # трекинг кадра со второй камеры
    report2 = True
    boxes_ids2 = tracker2.update(detections2)
    for box_id in boxes_ids2:
        x, y, w, h, id2 = box_id
        # если надпись окажется за пределами
        y1 = if_border(y, h)
        frame_plt2 = roi2[y : y + h, x : x + w]
        report2 = False

        # если новый объект
        if id2 != id2_save:
            # проверяем содержимое массива id и увеличиваем кол-во одинаковых объектов
            count_same = if_same_object(arr_id2, array_hist, count_same, id2_save)
            # если нет объекта в первой камере
            if report1 == True:
                # проверяем содержимое массива id и увеличиваем кол-во одинаковых объектов
                count_same = if_same_object(arr_id1, array_hist, count_same, id1_save)
            # обновляем счетчик кадров для гистограмм и для детектора
            count_frame2 = count_frame

        id2_save = id2
        global_id2 = id2 - count_same

        # копим новые гистограммы
        image_hist(frame_plt2, array_hist, global_id2, count_frame2)

        # пока счетчик кадров больше нуля, считаем сходства новой гистограммы с уже имеющимися
        if count_frame2 > 0:
            search_compare(global_id2, opt_param2, arr_id2)
            count_frame2 -= 1
            correct_id2 = global_id2
        else:
            # иначе ищем id с макс совпадением
            correct_id2 = find_max_same_id(arr_id2, count_same, id2)
            id2 = correct_id2
            cv2.putText(
                roi2,
                "Object " + str(id2),
                (x, y1),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )

        cv2.rectangle(roi2, (x, y), (x + w, y + h), (255, 0, 0), 3)
        tracker1.id_count = tracker2.id_count
    resized_frame_1 = cv2.resize(frame1, (width // 2, height // 2))
    resized_frame_2 = cv2.resize(frame2, (width // 2, height // 2))
    combined_frame = cv2.hconcat([resized_frame_1, resized_frame_2])
    output_frames.append(combined_frame)
    cv2.imshow("combined", combined_frame)
    key = cv2.waitKey(30)
    if key == 27:
        break
cap1.release()
cap2.release()
cv2.destroyAllWindows()
