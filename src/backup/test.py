# IMPORTS

from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
from pitch_config import PITCH_POINTS


# MODEL & VIDEO

model = YOLO("models/football-pitch-detection.pt")

print(model.names)

cap = cv2.VideoCapture("videos/MbappeGoal.mp4")


# VARIABLES

last_points = {}

kp_positions = {}
kp_counter = defaultdict(int)

total_frames = 0
homography_frames = 0

homography_matrix = None


# MAIN LOOP

while True:

    ret, frame = cap.read()

    if not ret:
        break

    total_frames += 1

    annotated = frame.copy()

    results = model.predict(
        frame,
        imgsz=960,
        verbose=False
    )

    valid_count = 0


    # KEYPOINT DETECTION

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
                kp_counter[i] += 1

                current_points[i] = (int(x), int(y))

                if i not in kp_positions:
                    kp_positions[i] = []

                kp_positions[i].append((int(x), int(y)))

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


        # HOMOGRAPHY POINT COLLECTION

        image_points = []
        pitch_points = []

        for kp_id, point in current_points.items():

            if kp_id in PITCH_POINTS:

                image_points.append(point)
                pitch_points.append(
                    PITCH_POINTS[kp_id]
                )


        # HOMOGRAPHY MATRIX

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


        # HOMOGRAPHY STATUS

        if homography_matrix is not None:

            cv2.putText(
                annotated,
                "H MATRIX READY",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                3
            )


        # LAST KNOWN KEYPOINTS

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


    # FRAME STATUS

    if valid_count >= 4:
        homography_frames += 1

    color = (
        (0, 255, 0)
        if valid_count >= 4
        else (0, 0, 255)
    )

    cv2.putText(
        annotated,
        f"VALID KEYPOINTS: {valid_count}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        3
    )

    cv2.putText(
        annotated,
        "HOMOGRAPHY OK"
        if valid_count >= 4
        else "NOT ENOUGH POINTS",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        3
    )


    # DISPLAY

    cv2.imshow(
        "Pitch Detection",
        annotated
    )

    if cv2.waitKey(1) == 27:
        break


# HOMOGRAPHY REPORT

print()
print("TOTAL FRAMES:", total_frames)
print("HOMOGRAPHY FRAMES:", homography_frames)

if total_frames > 0:

    print(
        "HOMOGRAPHY SUCCESS RATE:",
        round(
            homography_frames / total_frames * 100,
            2
        ),
        "%"
    )


# KEYPOINT FREQUENCIES

print("\nKEYPOINT FREQUENCIES")

for kp_id, count in sorted(
    kp_counter.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(f"KP {kp_id}: {count}")


# KEYPOINT STABILITY

print("\nKEYPOINT STABILITY")

for kp_id, positions in kp_positions.items():

    if len(positions) < 20:
        continue

    pts = np.array(positions)

    std_x = np.std(pts[:, 0])
    std_y = np.std(pts[:, 1])

    print(
        f"KP {kp_id}: "
        f"STD_X={std_x:.1f} "
        f"STD_Y={std_y:.1f}"
    )


# CLEANUP

cap.release()
cv2.destroyAllWindows()