import math


class EuclideanDistTracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
        self.save_id = 0

    def update(self, objects_rect,cx_last, cy_last, r, f, id_new):
        objects_bbs_ids = []
        # Центр нового объекта
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            # Поиск уже существующего объекта
            same_object_detected = False
            dist = 0
            dist2 = 0
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 100:
                    self.center_points[id] = (cx, cy)
                    print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    self.save_id = id
                    break
            if dist==0:
                dist2=math.hypot(cx - cx_last, cy - cy_last)
                if dist2<150 and r<7:
                    self.center_points[self.save_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.save_id])
                    same_object_detected = True

            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
            if f==True:
                self.id_count=id_new+1

        # Очистка от ненужных id
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Обновление
        self.center_points = new_center_points.copy()
        return objects_bbs_ids



