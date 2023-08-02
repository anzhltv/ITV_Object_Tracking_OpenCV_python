from tracker import *
import pyautogui
import cv2
from datetime import datetime
import time

def get_screen_resolution():
    screen_width, screen_height = pyautogui.size()
    return screen_width, screen_height
def center_point_save(x, y):
    return(x + x + y) // 2

def if_edge(y):
    if y < 50:
        y1 = y + 50
    else:
        y1 = y - 15
    return (y1)


path1 = '3.Camera 2017-05-29 16-23-04_137 [3m3s].avi'
path2 = '4.Camera 2017-05-29 16-23-04_137 [3m3s].avi'
cap1 = cv2.VideoCapture(path1)
cap2 = cv2.VideoCapture(path2)
tracker1 = EuclideanDistTracker()
tracker2 = EuclideanDistTracker()

object_detector1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=60000, varThreshold=10)
object_detector2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True, history=40000, varThreshold=10)

width, height = get_screen_resolution()
id1_same_object, id2_same_object = (0, 0)
time_1, time_1_2 = (0, 0)
time_c1_1, time_c2_1, time_c2_2,time_c1_2 = (0, 0, 0, 0)
cx_1, cy_1, cx_2, cy_2= (0, 0, 0, 0)
delta_time = 20
id_save = 0
id2, id1=(0, 0)
same_obj_1, same_obj_2 =(False, False)
flag = False

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    #height, width, _ = frame1.shape
    #print(height, width)

    roi1 = frame1[0: 720, 150: 600]
    roi2 = frame2[0: 720, 300: 900]

    # 1. Object Detection
    mask1 = object_detector1.apply(roi1)
    mask2 = object_detector2.apply(roi2)

    _, mask1 = cv2.threshold(mask1, 254, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(mask2, 254, 255, cv2.THRESH_BINARY)

    contours1, _ = cv2.findContours(mask1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections1 = []
    detections2 = []
    for cnt in contours1:
        area = cv2.contourArea(cnt)
        if area > 40000:
           #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections1.append([x, y, w, h])

    for cnt in contours2:
        area = cv2.contourArea(cnt)
        if area > 7000:
            #cv2.drawContours(roi2, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections2.append([x, y, w, h])

    # Треккинг
    time_c1_2 = datetime.now().timestamp()
    r=int(time_c1_2)-int(time_c1_1)
    boxes_ids1 = tracker1.update(detections1, cx_1, cy_1, r, same_obj_1, id_save)
    same_obj_1 = False
    for box_id in boxes_ids1:
        x, y, w, h, id1 = box_id
        y1 = if_edge(y)
        time_2_2 = datetime.now().timestamp()
        if int(time_2_2)-int(time_1_2)<10 and int(time_2_2)-int(time_1_2)>3:
            boxes_ids1[:][-1] = id2_same_object
            id1=id2_same_object
            same_obj_1 = True
        cv2.putText(roi1, 'Object ' + str(id1), (x, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.rectangle(roi1, (x, y), (x + w, y + h), (255, 0, 0), 3)
        if x>50 and y>300:
            time_1 = datetime.now().timestamp()
            id1_same_object = id1
        time_c1_1 = datetime.now().timestamp()
        cx_1 = center_point_save(x, w)
        cy_1 = center_point_save(y, h)
        id_save = id1
    if boxes_ids1==[]:
        flag=False

    time_c2_2 = datetime.now().timestamp()
    r=int(time_c2_2)-int(time_c2_1)
    boxes_ids2 = tracker2.update(detections2, cx_2, cy_2, r, same_obj_2, id_save)
    same_obj_2 = False
    for box_id in boxes_ids2:
        x, y, w, h, id2 = box_id
        y1 = if_edge(y)
        time_2 = datetime.now().timestamp()
        if int(time_2)-int(time_1)<delta_time and int(time_2)-int(time_1)>3:
            boxes_ids2[:][-1]=id1_same_object
            id2=id1_same_object
            same_obj_2=True
        cv2.putText(roi2, 'Object ' + str(id2), (x, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.rectangle(roi2, (x, y), (x + w, y + h), (255, 0, 0), 3)
        time_c2_1 = datetime.now().timestamp()
        cx_2 = center_point_save(x, w)
        cy_2 = center_point_save(y, h)

        if y>350:
            time_1_2 = datetime.now().timestamp()
            id2_same_object = id2
        id_save = id2

    resized_frame_1 = cv2.resize(frame1, (width//2-70, height//2-70))
    cv2.imshow("Frame_1", resized_frame_1)
   # pyautogui.moveTo(10, 10, duration=0)
    resized_frame_2 = cv2.resize(frame2, (width // 2-70, height // 2-70))
    cv2.imshow("Frame_2", resized_frame_2)

    key = cv2.waitKey(30)
    if key == 27:
        break
cap1.release()
cap2.release()
cv2.destroyAllWindows()