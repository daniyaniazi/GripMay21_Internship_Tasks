"""
TASK NO : 3 SOCIAL DISTANCING DETECTION 

SUBMITTED BY DANIYA

"""
#IMPORTING DEPENECIES
import cv2
import datetime
import imutils
from os.path import dirname, join
import numpy as np
import random
from centroidtracker import CentroidTracker
from itertools import combinations  # 0 1 2 => 01,02,12 ,012
import math

# CLASSES OF OUR PRE TRAINED MODEL
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# no of frame our tracker keeps waiting for object to be here and assign the same id
tracker = CentroidTracker(maxDisappeared=10, maxDistance=100)

COLORS = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for i in CLASSES]

#PATH OF PRETRAINED CAFFEE MODEL
protoPath = join(dirname(__file__), "MobileNetSSD_deploy.prototxt")
modelPath = join(dirname(__file__), "MobileNetSSD_deploy.caffemodel")

detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# FOR A APPROX BOX
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
    cap = cv2.VideoCapture('dataset/covid_distance.mp4')
    #FPS CALCULATION
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    while True: 
        ret, frame = cap.read(1)

        centriod_dict = {}
        distance_line = []
        # if frame == None:
        #     break
        if ret:        
            frame = imutils.resize(frame, width=600)
            total_frames = total_frames + 1

            (H, W) = frame.shape[:2]
            birdView = np.zeros([H, W, 3], dtype=np.uint8)
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

            detector.setInput(blob)
            person_detections = detector.forward()
            rects = []
            
            #DETECTING PERSONS ONLY
            for i in np.arange(0, person_detections.shape[2]):
                confidence = person_detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(person_detections[0, 0, i, 1])
                    if CLASSES[idx] != "person":
                        continue
                    person_box = person_detections[0, 0,
                                                   i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = person_box.astype("int")
                    rects.append(person_box)

            # PERSONS DETECTIONS
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
                #CALCUTATING CENTER POINT
                centriodX = int((x1+x2)/2.0)
                centriodY = int((y1+y2)/2.0)

                centriod_dict[objectId] = (
                    centriodX, centriodY, x1, y1, x2, y2)

                text = "Person No: {}".format(objectId)
                cv2.putText(frame, text, (x1, y1-5),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 0), 1)

            
            red_zone_list = []
            # iterarte the centriod
            for (id1, p1), (id2, p2) in combinations(centriod_dict.items(), 2):
                # p1 = centriod of one persion
                # p2 - centriod of second person
                dx, dy = p1[0] - p2[0], p1[1] - p2[1]
                distance_line.append((id1, id2,
                                      (p1[0], p1[1]), (p2[0], p2[1])))
                # distance
                distance = math.sqrt(dx*dx+dy*dy)
                cv2.circle(birdView, (p1[0], p1[1]) , 5, (255,255,0), -1)
                cv2.circle(birdView, (p2[0], p2[1]) , 5, (255,255,0), -1)
                # take a threshold
                if distance < 75.0:
                    # if the distance between 2 is less than red mark
                    cv2.line(frame, (p1[0], p1[1]),
                             (p2[0], p2[1]), (0, 0, 255), 1)
                    cv2.line(birdView, (p1[0], p1[1]),
                             (p2[0], p2[1]), (0, 0, 255), 1)
                    
                    # adding person in red zones
                    if id1 not in red_zone_list:
                        red_zone_list.append(id1)
                    if id2 not in red_zone_list:
                        red_zone_list.append(id2)
                else:
                    cv2.line(frame, (p1[0], p1[1]),
                             (p2[0], p2[1]), (0, 255, 0), 1)
                    cv2.line(birdView, (p1[0], p1[1]),
                             (p2[0], p2[1]), (0, 255, 0), 1)
                    
                    
            # DRAW RED, GREEN BOXES
            for id, box in centriod_dict.items():
                if id in red_zone_list:
                    cv2.rectangle( frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
                else:
                    cv2.rectangle(
                        frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)

            fps_end_time = datetime.datetime.now()
            time_diff = fps_end_time - fps_start_time
            if time_diff.seconds == 0:
                fps = 0.0
            else:
                fps = (total_frames / time_diff.seconds)
            fps_text = "FPS: {:.2f}".format(fps)
            cv2.putText(frame, fps_text, (5, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            
            #PERSPECTIVE AREA
            cv2.circle(frame, (574,8), 5, (0,0,255), -1) #tr
            cv2.circle(frame, (338,11),5, (0,0,255), -1) #tl
            cv2.circle(frame, (44,187) , 5, (0,0,255), -1) #blQ
            cv2.circle(frame, (520,303),5, (0,0,255), -1) #br
        
            #tl , tr ,bl,br
            pts1 = np.float32([[338,11],[574,8],[44,187],[520,303]])
            pts2=np.float32([[0,0],[337,0],[0,337],[337,337]])
            matrix=cv2.getPerspectiveTransform(pts1,pts2)
            results = cv2.warpPerspective(birdView,matrix,(337,337))

            #CONCAT FRAMES
            vis1 = np.concatenate((frame, birdView), axis=1)
            vis2= np.concatenate((results, vis1), axis=1)
            cv2.setMouseCallback('Application',draw_circle)
            cv2.imshow("Bird View | Live | Demo ", vis2)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


# I USED THIS FUNCTION TO GET CO-ORDINATES OF RESPECTIVE REGION
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(x,y)
        mouseX,mouseY = x,y

main()
