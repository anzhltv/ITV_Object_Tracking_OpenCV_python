"""
    Это документационная строка для main.py.

    Описание алгоритма.
Получаем кадры с обеих камер
1) Изначально, если айди объекта сменился на новый, увеличиваем счетчик кадров, проверяем наличие массива array_id,
	если под прошлым айди был найден уже существующий объект,
		то увеличиваем количество одинаковых объектов
		очищаем элемент массива векторов относящийся к этому объекту
Если объектов во втором кадре нет, то так же проверяем array_id со второй камеры, чтобы увеличить счетчик одинаковых элементов

Далее проверка нового объекта
Если счетчик кадров больше 40, считаем вектор текущего элемента как 0.9 старого значения и 0.1 нового и заносим в массив векторов под текущим id
	Параллельно считаем сходство данного вектора с уже имеющимися векторами, перебирая массив векторов.
		Если значение параметра схожести векторов больше нужной величины, то в массиве array_id инкрементируем элемент под найденным id
Иначе ищем максимальный элемент массива array_id,
	если он больше хотя бы 1/6 количества кадров определения айди, то записываем в correct_id номер элемента
	иначе отдаем ему id идущий по порядку

2) Аналогично для второй камеры

***Трекер работает по принципу подсчета расстояния между центрами поступающих ему боксов
***Перед входом в циклы, есть замер времени и поиск центра бокса, это нужно для трекера,
если сеть на несколько кадров теряет человека, то если время и расстояние между центрами последнего бокса и нового меньше определенной величины, то отдаем ему тот же айди

    Пример использования:
    python main.py 3.Camera 4.Camera GeneralNMHuman_v1.0_IR10_FP16 original_reid
    """


import argparse
import cmath
from datetime import datetime

import cv2
import numpy as np

# from network import NeuralNetworkDetector
from small_utils import (
    calculate_area,
    center_point_save,
    clean_array,
    get_screen_resolution,
    if_border,
    resize_frame,
)
from tracker import EuclideanDistTracker

parser = argparse.ArgumentParser(description="Input video file")
parser.add_argument("video_path1", type=str, help="Path to the video file number 1")
parser.add_argument("video_path2", type=str, help="Path to the video file number 2")
# parser.add_argument("openvino_path", type=str, help="Path to the openvino model")
# parser.add_argument("onnx_path", type=str, help="Path to the onnx model")
args = parser.parse_args()

EPSILON = 1e-6


def substractor(frame1_1, frame1, kern, iter):
    diff1 = cv2.absdiff(frame1_1, frame1)
    diff1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)
    _, mask1 = cv2.threshold(diff1, 20, 255, cv2.THRESH_BINARY)
    kernel = np.ones((kern, kern), np.uint8)
    mask1 = cv2.dilate(mask1, kernel, iterations=iter)
    return mask1

def detect_countour(size, contours, detections):
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > size:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])


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
            arr_hist[count_id] = alpha * frame_hist + (1 - alpha) * arr_hist[count_id]

    return


# метод на случай, если найден тот же объект
# если максимальное число совпадений больше, чем хотя бы 1/6 от количества кадров проверки,
# то очищаем текущий элемент с собранным вектором для этого объекта, так как он уже есть
# + увеличиваем количество совпадающих объектов
# Input:
# arr_id - массив с совпадающими айди, count_same - подсчет одинаковых элементов, id_save - айди предыдущего объекта
# Output:
# количество одинаковых элементов
def if_same_object(arr_id, count_same, id_save):
    # если был найден существующий объект, то очистка собранного вектора и массива id + увеличение числа совпадающих объектов
    if max(arr_id) >= count_frame // 6:
        vector[id_save - count_same] = 0
        count_same += 1
    clean_array(arr_id)
    return count_same


# метод для определения корректного айди для объекта
# если максимальное число совпадений больше, чем хотя бы 1/6 от количества кадров проверки,
# то отдаем объекту найденный айди,
# иначе айди по порядку
# Input:
# arr_id - массив с совпадающими айди, count_same - подсчет одинаковых элементов, global_id - айди текущего объекта
# Output:
# корректный айди объекта
def find_max_same_id(arr_id, count_same, global_id):
    arr_id_list = arr_id.tolist()
    # если максимальное совпадение больше чем 1/6 от общего числа кадров проверки
    if max(arr_id_list) >= count_frame // 6:
        # то берем найденный id
        corr_id = arr_id_list.index(max(arr_id_list))
    else:
        # иначе следующий по порядку
        corr_id = global_id - count_same
    return corr_id


# метод подсчета совпадений
# Input:
# global_id - id текущего объекта по порядку, opt_param - наименьший параметр схожести,
# arr_id - массив, содержащий количество совпадений с каждым существующим id
# Output:
# заполненный массив arr_id
def search_compare(global_id, opt_param, arr_id):
    obj_id, obj_k, max = (0, 0, 0)
    for i in range(global_id):
        compareHist = cv2.compareHist(
            vector[global_id].astype(np.float32),
            vector[i].astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
        print('COMPARE: ', compareHist, 'i = ', i)
        if compareHist > opt_param and compareHist != 1:
            if max < compareHist:
                max = compareHist
                obj_id = i
                obj_k += 1
    if obj_k > 0:
        # увеличиваем значение элемента с индексом id у которого максимальное совпадение с гистограммой
        arr_id[obj_id] += 1
    print('arr_id ', arr_id)
    return


# метод для увеличения count_same - если новый объект, а старый был уже существующим
# Input:
# id текущего объекта по порядку, num_cam, num_cam2 - наименьший параметр схожести,
# Output:
# заполненный массив arr_id
def if_new_object(id, num_cam, num_cam2):
    global count_same
    # если новый объект
    if id != id_save[num_cam]:
        # проверяем содержимое массива id для первой камеры и увеличиваем кол-во одинаковых объектов
        count_same = if_same_object(array_id[num_cam], count_same, id_save[num_cam])
        # если нет объекта во второй камере
        if report[num_cam2]:
            # проверяем содержимое массива id для второй камеры и увеличиваем кол-во одинаковых объектов
            count_same = if_same_object(
                array_id[num_cam2], count_same, id_save[num_cam2]
            )
        # обновляем счетчик кадров для гистограмм и для детектора
        count_frame_cam[num_cam] = count_frame


# метод для определения нового объекта на полученном кадре
# Input:
# boxes_ids - координаты бокса и id нового объекта, num_cam - номер камеры, frame - сам кадр
# Output:
# верно определенный id объекта и бокс на кадре
def camera_tracking(boxes_ids, num_cam, frame):
    global count_same, correct_id
    num_cam2 = (num_cam + 1) % 2
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        # если надпись окажется за пределами
        y1 = if_border(y, h)
        report[num_cam] = False
        frame_plt1 = frame[y : y + h, x : x + w]
        # если новый объект, проверяем наличие массивов id (второй массив, только если отсутствует объект во 2 кадре)
        if_new_object(id, num_cam, num_cam2)
        global_id1 = id - count_same
        id_save[num_cam] = id
        # пока счетчик кадров больше нуля, считаем сходства новой гистограммы с уже имеющимися
        if count_frame_cam[num_cam] > 0:
            # ищем вектор схожести
            image_hist(frame_plt1, vector, global_id1, count_frame_cam[num_cam])
            # net_search_vector(frame_plt1, global_id1)
            # считаем сходства с существующими id
            search_compare(global_id1, opt_param[num_cam], array_id[num_cam])
            # net_count_compare(global_id1, opt_param[num_cam], array_id[num_cam])
            count_frame_cam[num_cam] -= 1
            correct_id[num_cam] = global_id1
        else:
            # иначе ищем id с макс совпадением
            correct_id[num_cam] = find_max_same_id(array_id[num_cam], count_same, id)
            id = correct_id[num_cam]
            cv2.putText(
                frame,
                "Object " + str(id),
                (x, y1),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 255),
                2,
            )
        center[num_cam] = center_point_save(x, w, y, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        time[num_cam][0] = datetime.now().timestamp()
        tracker[num_cam2].id_count = tracker[num_cam].id_count
        print('num_cam', num_cam, 'count_same', count_same, 'global_id', global_id1)
    return


# метод для обновления трекера и использование метода camera_tracking
# Input:
# frame - сам кадр, num_cam - номер камеры, size - минимальный и максимальный размер бокса
# Output:
# output метода camera_tracking
def update_camera_tracking(frame, num_cam, detections):
    # detections = []
    # detect_contour(size, contours[num_cam], detections) #///////////////////////////////////////
    time[num_cam][1] = datetime.now().timestamp()
    # разница времени с последнего бокса в первом кадре с текущим моментом
    delta_time = int(time[num_cam][1]) - int(time[num_cam][0])
    report[num_cam] = True
    # обновление трекера
    boxes_ids1 = tracker[num_cam].update(detections, center[num_cam], delta_time)
    # трекинг, назначение id, отрисовка боксов
    camera_tracking(boxes_ids1, num_cam, frame)
    return


tracker = [EuclideanDistTracker(), EuclideanDistTracker()]

video_path1 = args.video_path1
video_path2 = args.video_path2
cap1 = cv2.VideoCapture(video_path1)
cap2 = cv2.VideoCapture(video_path2)
width, height = get_screen_resolution()

id_save = [-1, -1]
count_frame = 40  # количество кадров для идентификации человека
vector_size = 256  # размер выходного вектора из сети onnx

count_frame_cam = [0, 0]
ret1, frame1_1 = cap1.read()
ret2, frame2_2 = cap2.read()
height_frame, width_frame, _ = frame2_2.shape
frame2_2=frame2_2[0:height_frame, int(width_frame // 3.2):int(width_frame / 1.6)]
size1, size2 = (
    int(round(0.02 * width * height)),
    int(round(0.0034 * width * height)),
)  # минимальные размеры боксов

# минимальные и максимальные размеры боксов
size_box1 = (
    int(round(0.045 * width_frame * height_frame)),
    int(round(width_frame * height_frame / 3)),
)
size_box2 = (
    int(round(0.0076 * width_frame * height_frame)),
    int(round(width_frame * height_frame / 3)),
)

array_id = np.zeros((2, 10))  # массив для накопления совпадений с конкретным объектом

# vector = np.zeros((200, vector_size))
vector = np.zeros((200, 160, 160, 160))  # массив для накопления гистограмм объектов
opt_param = [0.5, 0.21]  # границы сравнения векторов для каждого кадра
count_same = 0  # переменная для подсчета одинаковых объектов

report = [True, True]
center = np.zeros((2, 2))
correct_id = [0, 0]
time = np.zeros((2, 2))

def main():
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        roi2 = frame2[0:height_frame, int(width_frame // 3.2): int(width_frame / 1.6)]
        # 1. Object Detection

        mask1 = substractor(frame1_1, frame1, 2, 3)
        mask2 = substractor(frame2_2, roi2, 3, 7)

        contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detections1 = []
        detections2 = []
        detect_countour(size1, contours1, detections1)
        detect_countour(size2, contours2, detections2)

        # трекинг кадра с первой камеры
        update_camera_tracking(frame1, 0, detections1)

        # трекинг кадра со второй камеры
        update_camera_tracking(roi2, 1, detections2)

        combined_frame = resize_frame(frame1, frame2, width, height)
        cv2.imshow("combined", combined_frame)
        key = cv2.waitKey(30)
        if key == 27:
            break
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


<<<<<<< HEAD
if __name__ == "__main__":
    main()
=======
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
    print("count_same = ", count_same)
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
>>>>>>> 1569b6f1b2f3ff29647207c327483796ce9c4ff1
