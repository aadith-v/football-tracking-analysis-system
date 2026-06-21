from ultralytics import YOLO
import cv2

# LOAD MODEL
model = YOLO("models/player_detection_best.pt")
model.to("cuda")
print(model.device)

# VIDEO
cap = cv2.VideoCapture("videos/ronaldoshot.mp4")

while True:

    success, frame = cap.read()

    if not success:
        break

    # DETECTION
    results = model.track(
    frame,
    persist=True,
    imgsz=416,
    verbose=False
)

    # DRAW DETECTIONS
    annotated_frame = results[0].plot()

    # SHOW VIDEO
    cv2.imshow("Football Model Test", annotated_frame)

    # ESC TO EXIT
    if cv2.waitKey(1) == 27:
        break

# CLEANUP
cap.release()
cv2.destroyAllWindows()