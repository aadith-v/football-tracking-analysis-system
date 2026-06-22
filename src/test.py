from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("models/football-pitch-detection.pt")

print(model.names)

cap = cv2.VideoCapture("videos/ronaldoshot.mp4")

last_points = {}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        frame,
        imgsz=960,
        verbose=False
    )

    annotated = frame.copy()

    valid_count = 0

    if (
        results[0].keypoints is not None
        and len(results[0].keypoints.xy) > 0
    ):

        pts = results[0].keypoints.xy[0].cpu().numpy()
        confs = results[0].keypoints.conf[0].cpu().numpy()

        current_points = {}

        for i, ((x, y), conf) in enumerate(zip(pts, confs)):

            if conf > 0.3 and 5 < x < 1915 and 5 < y < 1075:

                valid_count += 1

                current_points[i] = (int(x), int(y))

                last_points[i] = (int(x), int(y))

                cv2.circle(
                    annotated,
                    (int(x), int(y)),
                    6,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    annotated,
                    str(i),
                    (int(x) + 10, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Draw old points if missing this frame
        for kp_id, point in last_points.items():

            if kp_id not in current_points:

                cv2.circle(
                    annotated,
                    point,
                    5,
                    (0, 255, 255),
                    -1
                )

                cv2.putText(
                    annotated,
                    str(kp_id),
                    (point[0] + 10, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

    else:
        print("NO PITCH DETECTED")

    cv2.putText(
        annotated,
        f"VALID KEYPOINTS: {valid_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    color = (0, 255, 0) if valid_count >= 4 else (0, 0, 255)

    cv2.putText(
        annotated,
        "HOMOGRAPHY OK" if valid_count >= 4 else "NOT ENOUGH POINTS",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )

    cv2.imshow("Pitch Detection", annotated)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()