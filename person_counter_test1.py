import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker
import csv
# import datetime


# print("Date and time is:", dt)
# print("Timestamp is:", ts)

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


def main():
    cap = cv2.VideoCapture(0)

    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    lpc_count = 0
    opc_count = 0

    #  Timestamp

    # Getting the current date and time
    dt = datetime.datetime.now()

    # getting the timestamp
    # ts = datetime.timestamp(dt)


    object_id_list = []



    f = open('Data_test.csv', 'a+')
    writer = csv.writer(f)
    # writer.writerow(header)

    Data = []
    isDataWriteLocked = False

    while True:
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        dt = datetime.datetime.now()

        date_time_str = dt.strftime("%Y-%m-%d %H:%M")


        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)

        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            if objectId not in object_id_list:
                object_id_list.append(objectId)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        lpc_count = len(objects)
        opc_count = len(object_id_list)

        # Here we are locking the data at 59th second and unlocking the data at 57th second excludng the 0th second as per the algorithm.
        # Algortihm we used is by taking the modulus of the second 
            # 1) second % 59 =0   [ locking the data at this instance]
            # 2) second % 57 =0   [ unlocking the data at this instance]
            # but 0%0 = 0 then here the data will automatically gets unlocked and also at upper loop 0 % 0 =0 here the data will again gets unlocked and this continues untill second =1

        if(datetime.datetime.now().second % 59 == 0 and not isDataWriteLocked):
            Data.append([opc_count, lpc_count, date_time_str])
            print(f'isDataLocked ? : {isDataWriteLocked} and time: {dt}')
            isDataWriteLocked = True

        if(datetime.datetime.now().second % 57 == 0 and datetime.datetime.now().second != 0):
            print(f'isDataLocked ? : {isDataWriteLocked} and time: {dt}')
            isDataWriteLocked = False
        lpc_txt = "LPC: {}".format(lpc_count)
        opc_txt = "OPC: {}".format(opc_count)
        ts_txt = "TIME: {}".format(date_time_str)

        cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.putText(frame, opc_txt, (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.putText(frame, ts_txt, (5, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        # print(Data)
        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        # f.close()
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    for i in Data:
        print(i)
        writer.writerow(i)
    f.close()

main()




# I have to run it every 5 minutes and then it will automatically get locked and unlocked and then it will write the data in csv file.
# how to run it every 2 minutes ??
# how to lock and unlock the data ??
# how to write the data in csv file ??
# how to run it in background ??
# how to run it in background even if the system is off ??
# how to run it in background even if the system is off and then it will automatically get on and run the code and then again get off ??
# how to run it in background even if the system is off and then it will automatically get on and run the code and then again get off and this process will continue for 24 hours ??

'''cronjob = CronTab(user='root')
job = cronjob.new(command='python3 /home/akshay/Desktop/Project/Project.py')
job.minute.every(2)
cronjob.write()'''
