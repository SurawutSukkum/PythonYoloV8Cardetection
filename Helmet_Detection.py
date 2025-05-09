import cv2
import os
from ultralytics import YOLO

#v4l2-ctl --list-devices  คำสั่งสำหรับหา กล้อง
video_capture = cv2.VideoCapture(0) #ใส่ตัวเลขให้ตรง
model_path = '/home/smart/test/best.pt' # Folder ที่ใช้เก็บโมเดล หลังจากที่ เทรนแล้ว
# Load a model
model = YOLO(model_path)  # load a custom model
threshold = 0.5
class_name_dict = {0: 'Head', 1: 'Helmet'}

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
