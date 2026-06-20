from ultralytics import YOLO
import cv2
import math

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("videos/NeymarFoul.mp4")

players = {}
next_id = 1
MAX_DISTANCE = 50

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model(frame)

    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if model.names[cls] == "person" and conf > 0.5:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            new_center = (center_x, center_y)

            nearest_id = None
            smallest_distance = 999999

            # Find nearest existing player
            for player_id, position in players.items():

                distance = math.dist(position, new_center)

                if distance < smallest_distance:

                    smallest_distance = distance
                    nearest_id = player_id

            # Update existing player or create new player
            if nearest_id is not None and smallest_distance < MAX_DISTANCE:

                players[nearest_id] = new_center
                current_id = nearest_id

            else:

                players[next_id] = new_center
                current_id = next_id
                next_id += 1

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Draw ID
            cv2.putText(
                frame,
                f"ID {current_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

    cv2.imshow("Nearest Neighbor Tracker", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()