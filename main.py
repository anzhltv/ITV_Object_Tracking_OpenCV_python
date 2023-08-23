import time
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
from scipy.optimize import minimize

from tracker import *


def get_screen_resolution():
    screen_width, screen_height = pyautogui.size()
    return screen_width, screen_height

def if_edge(y):
    if y < 50:
        y1 = y + 50
    else:
        y1 = y - 15
    return y1

# яркость/контраст в доработке
def objective(I_new, I_et, N):
    I2 = I_et
    I1 = I_new
    # a=(I1_sum-I2_sum*I_sum)/(N*I2_sum_q-I2_sum**2)
    # b=(N*I_sum-I1_sum*I2_sum)/(N*I2_sum_q-I2_sum**2)
    # a=(I_sum-I1_sum*I2_sum/N)/I2_sum_q-I2_sum_q/N
    # b=I1_sum-a*I2_sum/N
    n = I1.shape[0] * I1.shape[1]
    sum_I1_squared = np.sum(I1 * I1)
    sum_I2 = np.sum(I2)
    sum_I1_I2 = np.sum(I1 * I2)
    sum_I1 = np.sum(I1)
    # a = (sum_I1 * sum_I2_squared - sum_I1_I2 * sum_I2) / (n * sum_I2_squared - sum_I2 ** 2)
    # b = (n * sum_I1_I2 - sum_I1 * sum_I2) / (n * sum_I2_squared - sum_I2 ** 2)
    # a=-1*(sum_I2*sum_I1+n*sum_I1_I2)/(sum_I1_squared*(n-1))
    # a=sum_I1_I2/sum_I1_squared
    # b=(a*sum_I1-sum_I2)/n
    # # print('a= ',a,'b= ', b)
    # b=(I1_sum-I_sum)/N-I2_sum
    # a=(I1_sum-b*N)/I2_sum
    a = np.array([[2 * sum_I1_squared, 2 * sum_I1], [2 * sum_I1, 2 * N]])
    b = np.array([2 * sum_I1_I2, 2 * sum_I2])
    solution = np.linalg.solve(a, b)
    a = solution[0]
    b = solution[1]
    # a=20
    # b=sum_I1-a*sum_I2
    min = a * sum_I1 + b - sum_I2
    print("min ", min)
    print("a= ", a, "b= ", b)
    # new_image = cv2.convertScaleAbs(I_et, alpha=40, beta=20)
    gray_image = cv2.cvtColor(I_et, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray_image.mean()
    new_image = cv2.convertScaleAbs(I_et, alpha=1.5, beta=0)
    # new_image = np.clip(I_et * (1.0 + 20), 0, 255).astype(np.uint8)
    # matrix = np.array([[1.0, 0.0, 0.0],
    #                    [0.0, 1.0, 0.0],
    #                    [0.0, 0.0, 1.0 + 20]])
    # new_image = cv2.transform(I_et, matrix)
    # new_image = np.clip(new_image, 0, 255)
    cv2.imshow("New Image", new_image)
    cv2.imshow("Image", I_new)
    return


def image_hist(frame_hist, arr_hist, count_id, frame, k, stop):
    alpha = 0.1
    # color = ('b', 'g', 'r')
    # for channel, col in enumerate(color):
    #     hist_color = cv2.calcHist([frame_hist], [channel], None, [256], [0, 256])
    #     plt.plot(hist_color, color=col)
    # plt.show()
    # # time.sleep(3)
    # plt.close('all')
    # frame_hist = cv2.cvtColor(frame_hist, cv2.COLOR_BGR2RGB)
    if stop==False:
        b, g, r = cv2.split(frame_hist)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        frame_hist = cv2.merge((b_eq, g_eq, r_eq))
        frame_hist = cv2.calcHist(
            [frame_hist], [0, 1, 2], None, [160, 160, 160], [0, 256, 0, 256, 0, 256] )
        cv2.normalize(frame_hist, None, 0, 1.0, cv2.NORM_MINMAX)
        EPSILON = 1e-6
        compare = cv2.compareHist(arr_hist[count_id][frame].astype(np.float32), frame_hist.astype(np.float32),
                              cv2.HISTCMP_CORREL)
        if k[0]>0:
            if np.all(np.abs(arr_hist[count_id]) < EPSILON):
                arr_hist[count_id][frame] = frame_hist
            else:
                arr_hist[count_id][frame] = (
                        alpha * (frame_hist) + (1 - alpha) * arr_hist[count_id][frame]
                )
            k[0]-=1
        else:
            if (frame==0 and compare>0.1) or (frame==1 and compare>0.1):
                if np.all(np.abs(arr_hist[count_id]) < EPSILON):
                    arr_hist[count_id][frame] = frame_hist
                else:
                    arr_hist[count_id][frame] = (
                    alpha * (frame_hist) + (1 - alpha) * arr_hist[count_id][frame]
                    )
    return count_id



path1 = "3.Camera 2017-05-29 16-23-04_137 [3m3s].avi"
path2 = "4.Camera 2017-05-29 16-23-04_137 [3m3s].avi"
cap1 = cv2.VideoCapture(path1)
cap2 = cv2.VideoCapture(path2)
tracker1 = EuclideanDistTracker()
tracker2 = EuclideanDistTracker()
I_new = cv2.VideoCapture("ex1_2.png")
I_et = cv2.VideoCapture("ex1_1.png")
_, I_new = I_new.read()
_, I_et = I_et.read()

# object_detector1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=60000, varThreshold=10)
# object_detector2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=40000, varThreshold=10)
EPSILON = 1e-6
width, height = get_screen_resolution()
N = width * height
size1 = int(round(0.02 * width * height))
size2 = int(round(0.0034 * width * height))
boxes_ids2_save=[]
id_save, id2_save, id1_save = (0, 0, 0)
id2, id1 = (0, 0)
id2_new, id1_new = (0, 0)
flag1, flag2 = (1, 1)
correct_id2, correct_id1 = (0, 0)
ret1, frame1_1 = cap1.read()
ret2, frame2_2 = cap2.read()
arr_id1=[0]*10
arr_id2=[0]*10
array_hist = np.zeros((200, 1, 160, 160, 160))
k, k1, k2 = (0,0,0)
global_id, global_id1, global_id2 = (0,0,0)
opt_param1, opt_param2=(0.5, 0.21)
count_same=0
k=[30]
k1=30
id_number=0
save_global_id1, save_global_id2=(0,0)
stop1, stop2=(False, False)
some=5
output_frames=[]
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    roi1 = frame1[0:720, 150:600]
    roi2 = frame2[0:720, 400:800]

    # 1. Object Detection
    diff1 = cv2.absdiff(frame1_1, frame1)
    diff1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)

    diff2 = cv2.absdiff(frame2_2, frame2)
    diff2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)

    diff2 = diff2[0:720, 400:800]
    _, mask1 = cv2.threshold(diff1, 20, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(diff2, 20, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    mask1 = cv2.dilate(mask1, kernel, iterations=3)

    kernel = np.ones((3, 3), np.uint8)
    mask2 = cv2.dilate(mask2, kernel, iterations=7)

    contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # similarity = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I1, 0.0)
    # print('similarity=', similarity)
    # print('contours:', contours1)
    detections1 = []
    detections2 = []
    for cnt in contours1:
        area = cv2.contourArea(cnt)
        if area > size1:
            # cv2.drawContours(frame1, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections1.append([x, y, w, h])

    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area > size2:
            # cv2.drawContours(roi2, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections2.append([x, y, w, h])

    # Треккинг ----------------------------------------------------------------------------------

    boxes_ids1 = tracker1.update(detections1)
    report = False
    for box_id in boxes_ids1:
        x, y, w, h, id1 = box_id
        y1 = if_edge(y)
        report = True
        frame_plt = frame1[y : y + h, x : x + w]
        if id1 != id1_save:
            if flag1 == 2:
                if np.all(array_hist[correct_id1][0] < EPSILON):
                    array_hist[correct_id1][0] = array_hist[global_id1][0]
                array_hist[global_id1][0] = 0
                count_same+=1
                flag1 = 1
                for i in range(len(arr_id1)):
                    arr_id1[i]=0
                k[0] = 20

        # /////////////////////////////////
        if flag2 == 2:
            if np.all(array_hist[correct_id2][0] < EPSILON):
                array_hist[correct_id2][0] = array_hist[global_id2][0]
            array_hist[global_id2][0] = 0
            flag2 = 1
            if max(arr_id2) > 15:
               count_same += 1
            for i in range(len(arr_id2)):
                arr_id2[i] = 0
            k[0] = 20

        global_id1 = id1 - count_same
        if save_global_id1!=global_id1:
            k[0]=20
            if k1>0:
                if max(arr_id1) > 15:
                    correct_id1 = arr_id1.index(max(arr_id1))
                else:
                    correct_id1 = id1 - count_same
                    flag1 = 1
        if flag1 != 2 and id1_save != id1:
            k1 = 30
            stop1=False
        id1_save = id1
        save_global_id1 = image_hist(frame_plt, array_hist, global_id1, 0, k, stop1)
        max1 = 0
        if k1 > 0:
            for i in range(global_id1):
                for j in range(1):
                    compareHist1 = cv2.compareHist(
                        array_hist[global_id1][0].astype(np.float32),
                        array_hist[i][j].astype(np.float32),
                        cv2.HISTCMP_CORREL,
                    )
                    if compareHist1 > opt_param1 and compareHist1!=1 :
                        arr_id1[i] += 1
                        if max1 < compareHist1:
                            boxes_ids1[:][-1] = i
                            id1_new = id1
                            flag1 = 2
                            max1 = compareHist1
            k1 -= 1
        if k1>0:
            correct_id1=id1-count_same
        else:
            if max(arr_id1) >= 6:
                correct_id1 = arr_id1.index(max(arr_id1))
            else:
                correct_id1 = id1 - count_same
                flag1 = 1
            stop1 = True
        if id1 == id1_new and flag1 == 2:
            id1 = correct_id1
        else:
            if flag1 == 2:
                flag1 = 3
            else:
                id1 = id1 - count_same
        cv2.putText(frame1, "Object " + str(id1), (x, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2,)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 3)
        tracker2.id_count = tracker1.id_count

    # ---------------------------------------------------------------------------------------------------------------
    boxes_ids2 = tracker2.update(detections2)
    count_id = 0
    for box_id in boxes_ids2:
        count_id += 1
        x, y, w, h, id2 = box_id
        y1 = if_edge(y)
        frame_plt2 = roi2[y : y + h, x : x + w]
        if flag1 == 2 and report==False:
            if np.all(array_hist[correct_id1][0] < EPSILON):
                array_hist[correct_id1][0] = array_hist[global_id - 1][0]
            array_hist[global_id - 1][0] = 0
            flag1 = 1
            if report==False:
                if max(arr_id1) > 15:
                    count_same += 1
                    for i in range(len(arr_id1)):
                        arr_id1[i] = 0
            k[0] = 20
        # ////////////////////////////////////////////
        global_id2=id2-count_same
        if save_global_id2!=global_id2:
            k[0]=20
            if k2>0:
                if max(arr_id2) > 15:
                    correct_id2 = arr_id2.index(max(arr_id2))
                else:
                    correct_id2 = id2 - count_same
                    flag2 = 1
        max2 = 0
        if flag2 != 2:
            k2 = 30
            stop2=False
        save_global_id2 = image_hist(frame_plt2, array_hist, global_id2, 0, k, stop2)
        if k2 > 0:
            arr_id=0
            obj_k=0
            for i in range(global_id2):
                for j in range(1):
                    compareHist2 = cv2.compareHist(
                        array_hist[global_id2][0].astype(np.float32),
                        array_hist[i][j].astype(np.float32),
                        cv2.HISTCMP_CORREL,
                    )
                    if compareHist2 > opt_param2 and compareHist2!=1:
                        # arr_id2[i] += 1
                        if max2 < compareHist2:
                            # boxes_ids2[:][-1] = i
                            id2_new = id2
                            # correct_id2 = i
                            flag2 = 2
                            max2 = compareHist2
                            arr_id=i
                            obj_k+=1
            if obj_k>0:
                arr_id2[arr_id] += 1
            k2 -= 1
        if k2>0:
            correct_id2=id2-count_same
        else:
            if max(arr_id2) > 15:
               correct_id2=arr_id2.index(max(arr_id2))
            else:
                correct_id2=id2-count_same
                flag2=1
            stop2 = True
        if id2 == id2_new and flag2 == 2:
            id2 = correct_id2
        else:
            if flag2 == 2:
                flag2 = 3
            else:
                id2 = id2 - count_same
        if flag2 == 3:
            if np.all(array_hist[correct_id2][0] < EPSILON):
                array_hist[correct_id2][0] = array_hist[global_id2][0]
            count_same += 1
            array_hist[global_id2][0] = 0
            flag2 = 1
            k[0] = 20
            for i in range(len(arr_id2)):
                arr_id2[i] = 0
        cv2.putText(roi2, "Object " + str(id2), (x, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.rectangle(roi2, (x, y), (x + w, y + h), (255, 0, 0), 3)
        tracker1.id_count = tracker2.id_count
    resized_frame_1 = cv2.resize(frame1, (width // 2 , height // 2 ))
    resized_frame_2 = cv2.resize(frame2, (width // 2 , height // 2))
    combined_frame = cv2.hconcat([resized_frame_1, resized_frame_2])
    output_frames.append(combined_frame)
    cv2.imshow('combined', combined_frame)
    key = cv2.waitKey(30)
    if key == 27:
        break
cap1.release()
cap2.release()
cv2.destroyAllWindows()
