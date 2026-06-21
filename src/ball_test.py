from ultralytics import YOLO
import cv2

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture("videos/ronaldoshot.mp4")

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if model.names[cls] == "sports ball" and conf > 0.6:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            cv2.putText(
                frame,
                "BALL",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

    cv2.imshow("Ball Test", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()