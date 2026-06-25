from time import time
from tracemalloc import start
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans
import cv2
import math
from pitch_config import PITCH_POINTS
from pitch_map import create_pitch
# CONFIG

TARGET_ID = 25


# PLAYER DATA
DEBUG = False
player_positions = {}
total_distance = {}
player_speed = {}

player_paths = {}
heatmap_data = {}

player_colors = {}
player_teams = {}
ball_position = None
ball_path = []
ball_positions = []
interpolated_ball_positions = []
total_frames = 0
ball_detected_frames = 0


# MODEL + VIDEO

player_model = YOLO("models/player_detection_best.pt")
ball_model = YOLO("models/football-ball-detection.pt")
pitch_model = YOLO("models/football-pitch-detection.pt")

print("PLAYER MODEL:", player_model.names)
print("BALL MODEL:", ball_model.names)
print("PITCH MODEL:", pitch_model.names)
player_model.to("cuda")
ball_model.to("cuda")
pitch_model.to("cuda")


cap = cv2.VideoCapture("videos/messi.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

cv2.namedWindow("Football Analytics", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Football Analytics", 1200, 700)
pitch_template = cv2.imread("src/pitch.png")

# MAIN LOOP
frame_count = 0
while True:
    success, frame = cap.read()
    if not success:
        break
    
    pitch = create_pitch()
    
    
    ball_found = False
    frame_count += 1
    total_frames += 1
   
    
    player_results = player_model.track(
        frame,
        persist=True,
        imgsz=960,
        verbose=False
    )

    ball_results = ball_model.predict(
        frame,
        imgsz=960,
        conf=0.2,
        verbose=False
    )
    
    pitch_results = pitch_model.predict(
    frame,
    imgsz=960,
    verbose=False
    )
    
    
    
    if (
        pitch_results[0].keypoints is not None
        and len(pitch_results[0].keypoints.xy) > 0
    ):

        pts = pitch_results[0].keypoints.xy[0].cpu().numpy()
        confs = pitch_results[0].keypoints.conf[0].cpu().numpy()

        valid_count = 0
        image_points = []
        pitch_points = []

        for i, ((x, y), conf) in enumerate(zip(pts, confs)):

            if conf > 0.3:
                if i in PITCH_POINTS:

                    image_points.append(
                    (int(x), int(y))
                    )

                    pitch_points.append(
                        PITCH_POINTS[i]
                    )
                
                valid_count += 1

                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    5,
                    (0, 255, 0),
                    -1
                )

                cv2.putText(
                    frame,
                    f"KP {i}",
                    (int(x) + 8, int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
                
                
        cv2.putText(
        frame,
        f"KEYPOINTS: {valid_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
        
    
    if len(image_points) >= 4:

        image_points = np.array(
            image_points,
            dtype=np.float32
        )

        pitch_points = np.array(
            pitch_points,
            dtype=np.float32
        )

        homography_matrix, status = cv2.findHomography(
            image_points,
            pitch_points
        )
        
        
        if homography_matrix is not None:

            determinant = np.linalg.det(homography_matrix)

            print(f"H DET = {determinant:.6f}")
            
        if homography_matrix is not None:
            print("H MATRIX CREATED")
    
        if homography_matrix is not None:

            cv2.putText(
                frame,
                "H MATRIX READY",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )
    
    
    
    # PLAYER PROCESSING
    for box in ball_results[0].boxes:

        conf = float(box.conf[0])

        if conf < 0.2:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(
        frame,
        f"BALL {conf:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
         2
        )
        
        
        ball_position = (center_x, center_y)
        if len(ball_positions) > 0 and ball_positions[-1] is not None:

            old_x, old_y = ball_positions[-1]

            jump = math.dist((old_x, old_y), ball_position)

            if jump > 40:
                ball_found = False
                continue
        
        ball_found = True
            

        ball_path.append(ball_position)

        if len(ball_path) > 50:
            ball_path.pop(0)
                 
        ball_detected_frames += 1     
        break
        
        
    for box in player_results[0].boxes:
           
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        label = player_model.names[cls]
        
        
        # if label == "ball":
           
               
        #    x1, y1, x2, y2 = map(int, box.xyxy[0])
        #    center_x = (x1 + x2) // 2
        #    center_y = (y1 + y2) // 2
           
        #    ball_position = (center_x,center_y)
        #    ball_found = True
        #    ball_detected_frames += 1
        #    ball_path.append(ball_position)
        #    if len(ball_path) > 50:
        #        ball_path.pop(0)
        #    print(f"BALL ID={ball_id} CONF={conf:.2f} POS=({center_x},{center_y})")
    
    
        if player_model.names[cls] != "player":
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
        
        if homography_matrix is not None:

            image_point = np.array(
                [[[center_x, center_y]]],
                dtype=np.float32
        )

        pitch_point = cv2.perspectiveTransform(
            image_point,
            homography_matrix
        )

        pitch_x = pitch_point[0][0][0]
        pitch_y = pitch_point[0][0][1]

        print(
            f"Player {track_id} -> "
            f"Pitch ({pitch_x:.2f}, {pitch_y:.2f})"
        )
        
        if not (0 <= pitch_x <= 105 and 0 <= pitch_y <= 68):
            continue
        draw_x = int(pitch_x * 10)
        draw_y = int(pitch_y * 10)

        cv2.circle(
            pitch,
            (draw_x, draw_y),
            5,
            (0, 0, 255),
            -1
        )
        
        cv2.putText(
            pitch,
            str(track_id),
            (draw_x + 8, draw_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

        
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
                
    if ball_found:
        ball_positions.append(ball_position)
    else:
        ball_positions.append(None)
       
    if len(ball_positions) >= 3:

       p1 = ball_positions[-3]
       p2 = ball_positions[-2]
       p3 = ball_positions[-1]

       if p1 is not None and p2 is None and p3 is not None:

           x = (p1[0] + p3[0]) // 2
           y = (p1[1] + p3[1]) // 2

           ball_positions[-2] = (x, y)
           
           
    if DEBUG and len(ball_positions) >= 5:
        print(ball_positions[-5:])
    
    
    # TEAM DETECTION
    
    if len(player_colors) >= 4 and frame_count % 30 == 0:

        color_data = list(player_colors.values())

        kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)

        kmeans.fit(color_data)

        labels = kmeans.labels_

        player_teams = {}

        for player_id, team in zip(player_colors.keys(), labels):

            player_teams[player_id] = team

    
    # DRAWING

    for box in player_results[0].boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])

        label = player_model.names[cls]

        if label != "player":
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
        
        for i in range(1,len(ball_path)):
            cv2.line(frame, ball_path[i-1], ball_path[i], (0, 255, 255), 2)

        if track_id == TARGET_ID and track_id in player_paths:

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
            f"{label} ID:{track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            box_color,
            2,
        )

    
    # DISPLAY
    
    cv2.imshow("Football Analytics", frame)
    cv2.imshow("Mini Pitch", pitch)

    if cv2.waitKey(1) == 27:
        break


# CLEANUP
if DEBUG:
    print()
    print("TOTAL FRAMES:", total_frames)
    print("BALL DETECTED:", ball_detected_frames)

if total_frames > 0:
    print(
        "DETECTION RATE:",
        round(ball_detected_frames / total_frames * 100, 2),
        "%"
    )
cap.release()
cv2.destroyAllWindows()
