from ultralytics import YOLO
from sklearn.cluster import KMeans
import cv2
import math


# CONFIG

TARGET_ID = 25


# PLAYER DATA

player_positions = {}
total_distance = {}
player_speed = {}

player_paths = {}
heatmap_data = {}

player_colors = {}
player_teams = {}



# MODEL + VIDEO

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture("videos/ronaldoshot.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

cv2.namedWindow("Football Analytics", cv2.WINDOW_NORMAL)
# MAIN LOOP

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model.track(frame, persist=True, classes=[0])

    
    # PLAYER PROCESSING
    
    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if model.names[cls] != "person":
            continue

        if conf < 0.5:
            continue

        if box.id is None:
            continue

        track_id = int(box.id[0])

        
        # BOUNDING BOX
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        new_center = (center_x, center_y)

        
        # JERSEY COLOR EXTRACTION 

        player_crop = frame[y1:y2, x1:x2]

        if player_crop.size != 0:

            h, w, _ = player_crop.shape

            jersey_crop = player_crop[
                int(h * 0.2) : int(h * 0.5), int(w * 0.2) : int(w * 0.8)
            ]

            average_color = cv2.mean(jersey_crop)

            player_colors[track_id] = [
                average_color[0],
                average_color[1],
                average_color[2],
            ]

        
        # FIRST TIME PLAYER
        
        if track_id not in player_positions:

            player_positions[track_id] = new_center

            total_distance[track_id] = 0

            player_speed[track_id] = 0

            player_paths[track_id] = [new_center]

            heatmap_data[track_id] = {}

        
        # UPDATE PLAYER
        
        else:

            old_position = player_positions[track_id]

            distance = math.dist(old_position, new_center)

            speed = distance * fps

            player_speed[track_id] = speed

            total_distance[track_id] += distance

            player_positions[track_id] = new_center

            player_paths[track_id].append(new_center)

            if new_center in heatmap_data[track_id]:

                heatmap_data[track_id][new_center] += 1

            else:

                heatmap_data[track_id][new_center] = 1

            if len(player_paths[track_id]) > 20:

                player_paths[track_id].pop(0)

    
    # TEAM DETECTION
    
    if len(player_colors) >= 4:

        color_data = list(player_colors.values())

        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)

        kmeans.fit(color_data)

        labels = kmeans.labels_

        player_teams = {}

        for player_id, team in zip(player_colors.keys(), labels):

            player_teams[player_id] = team

    
    # DRAWING

    for box in results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if model.names[cls] != "person":
            continue

        if conf < 0.5:
            continue

        if box.id is None:
            continue

        track_id = int(box.id[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        
        # BOX COLOR BY TEAM
        
        box_color = (0, 255, 0)

        if track_id in player_teams:

            if player_teams[track_id] == 0:

                box_color = (255, 0, 0)

            else:

                box_color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)

        if track_id == TARGET_ID:

            path = player_paths[track_id]

            for i in range(1, len(path)):

                cv2.line(frame, path[i - 1], path[i], (255, 0, 255), 2)

        team_text = "?"

        if track_id in player_teams:

            if player_teams[track_id] == 0:

                team_text = "A"

            else:

                team_text = "B"

        cv2.putText(
            frame,
            f"T:{team_text} ID:{track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
        )

    
    # DISPLAY
    
    cv2.imshow("Football Analytics", frame)

    if cv2.waitKey(1) == 27:
        break


# CLEANUP

cap.release()
cv2.destroyAllWindows()
