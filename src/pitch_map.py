import cv2
import numpy as np

# -----------------------
# PITCH SIZE
# -----------------------

WIDTH = 1050
HEIGHT = 680

pitch = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
pitch[:] = (0, 110, 0)


# -----------------------
# OUTER BOUNDARY
# -----------------------

cv2.rectangle(
    pitch,
    (20, 20),
    (WIDTH - 20, HEIGHT - 20),
    (255, 255, 255),
    3
)


# -----------------------
# HALF WAY LINE
# -----------------------

cv2.line(
    pitch,
    (WIDTH // 2, 20),
    (WIDTH // 2, HEIGHT - 20),
    (255, 255, 255),
    3
)


# -----------------------
# CENTRE CIRCLE
# -----------------------

cv2.circle(
    pitch,
    (WIDTH // 2, HEIGHT // 2),
    75,
    (255, 255, 255),
    3
)

cv2.circle(
    pitch,
    (WIDTH // 2, HEIGHT // 2),
    4,
    (255, 255, 255),
    -1
)


# -----------------------
# LEFT PENALTY AREA
# -----------------------

cv2.rectangle(
    pitch,
    (20, 170),
    (180, 510),
    (255, 255, 255),
    3
)


# -----------------------
# LEFT GOAL AREA
# -----------------------

cv2.rectangle(
    pitch,
    (20, 250),
    (80, 430),
    (255, 255, 255),
    3
)


# -----------------------
# LEFT PENALTY SPOT
# -----------------------

cv2.circle(
    pitch,
    (140, HEIGHT // 2),
    4,
    (255,255,255),
    -1
)


# -----------------------
# LEFT PENALTY ARC
# -----------------------




# -----------------------
# RIGHT PENALTY AREA
# -----------------------

cv2.rectangle(
    pitch,
    (WIDTH-180,170),
    (WIDTH-20,510),
    (255,255,255),
    3
)


# -----------------------
# RIGHT GOAL AREA
# -----------------------

cv2.rectangle(
    pitch,
    (WIDTH-80,250),
    (WIDTH-20,430),
    (255,255,255),
    3
)


# -----------------------
# RIGHT PENALTY SPOT
# -----------------------

cv2.circle(
    pitch,
    (WIDTH - 140,HEIGHT//2),
    4,
    (255,255,255),
    -1
)


# -----------------------
# RIGHT PENALTY ARC
# -----------------------


# -----------------------
# LEFT GOAL
# -----------------------

cv2.rectangle(
    pitch,
    (5,290),
    (20,390),
    (255,255,255),
    2
)


# -----------------------
# RIGHT GOAL
# -----------------------

cv2.rectangle(
    pitch,
    (WIDTH-20,290),
    (WIDTH-5,390),
    (255,255,255),
    2
)


# -----------------------
# CORNER ARCS
# -----------------------

cv2.ellipse(pitch,(20,20),(20,20),0,0,90,(255,255,255),2)
cv2.ellipse(pitch,(WIDTH-20,20),(20,20),0,90,180,(255,255,255),2)
cv2.ellipse(pitch,(20,HEIGHT-20),(20,20),0,270,360,(255,255,255),2)
cv2.ellipse(pitch,(WIDTH-20,HEIGHT-20),(20,20),0,180,270,(255,255,255),2)


# -----------------------
# SHOW
# -----------------------

def create_pitch():
    return pitch.copy()