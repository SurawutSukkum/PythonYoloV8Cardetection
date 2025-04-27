import cv2
import os
from ultralytics import YOLO

#v4l2-ctl --list-devices  คำสั่งสำหรับหา กล้อง
video_capture = cv2.VideoCapture(0) #ใส่ตัวเลขให้ตรง
model_path = '/home/smart/test/yolo11n.pt' # Folder ที่ใช้เก็บโมเดล หลังจากที่ เทรนแล้ว
# Load a model
model = YOLO(model_path)  # load a custom model
threshold = 0.5
class_name_dict = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

while True:
    ret, video_frame = video_capture.read()  
    if ret is False:
        break 
      
    results = model(video_frame)[0]
    print(results)
    for result in results.boxes.data.tolist():
        print(result)
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
         cv2.rectangle(video_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
         cv2.putText(video_frame, class_name_dict[int(class_id)].upper(), (int(x1), int(y1 - 10)),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

    cv2.namedWindow('detectLabel', cv2.WINDOW_NORMAL)
    cv2.imshow("detectLabel", video_frame )
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
