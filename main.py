import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)

# Previous finger position
prev_x = 0
prev_y = 0

# White canvas
canvas = None

# Drawing mode
drawing_mode = True

# Default color - Red
draw_color = (0, 0, 255)

# -----------------------------
# Main Loop
# -----------------------------
while True:

    success, frame = cap.read()

    if not success:
        print("Camera not found!")
        break

    # Flip camera
    frame = cv2.flip(frame, 1)

    # Frame size
    h, w, _ = frame.shape

    # Create white canvas
    if canvas is None:
        canvas = np.ones(
            (h, w, 3),
            dtype=np.uint8
        ) * 255

    # -----------------------------
    # Hand Detection
    # -----------------------------
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip
            index_finger = hand_landmarks.landmark[8]

            x = int(index_finger.x * w)
            y = int(index_finger.y * h)

            # Finger position
            cv2.circle(
                frame,
                (x, y),
                10,
                draw_color,
                -1
            )

            # -----------------------------
            # Draw or Erase
            # -----------------------------
            if prev_x != 0 and prev_y != 0:

                if drawing_mode:

                    # Draw using selected color
                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        draw_color,
                        5
                    )

                else:

                    # Eraser
                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        (255, 255, 255),
                        30
                    )

            # Save current position
            prev_x = x
            prev_y = y

    else:

        # Reset position
        prev_x = 0
        prev_y = 0

    # -----------------------------
    # Instructions
    # -----------------------------
    cv2.putText(
        frame,
        "R=Red B=Blue G=Green Y=Yellow",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "D=Draw E=Eraser C=Clear Q=Quit",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # Current Mode
    # -----------------------------
    if drawing_mode:

        cv2.putText(
            frame,
            "MODE: DRAW",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            draw_color,
            2
        )

    else:

        cv2.putText(
            frame,
            "MODE: ERASER",
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # -----------------------------
    # Show selected color
    # -----------------------------
    cv2.rectangle(
        frame,
        (10, 110),
        (60, 160),
        draw_color,
        -1
    )

    cv2.putText(
        frame,
        "COLOR",
        (70, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # Combine Camera + Canvas
    # -----------------------------
    combined = np.hstack(
        (frame, canvas)
    )

    cv2.imshow(
        "Virtual Drawing Board",
        combined
    )

    # -----------------------------
    # Keyboard Controls
    # -----------------------------
    key = cv2.waitKey(1) & 0xFF

    # Quit
    if key == ord('q'):
        break

    # Draw mode
    elif key == ord('d'):
        drawing_mode = True
        prev_x = 0
        prev_y = 0

    # Eraser
    elif key == ord('e'):
        drawing_mode = False
        prev_x = 0
        prev_y = 0

    # Clear
    elif key == ord('c'):
        canvas = np.ones(
            (h, w, 3),
            dtype=np.uint8
        ) * 255

        prev_x = 0
        prev_y = 0

    # Red
    elif key == ord('r'):
        draw_color = (0, 0, 255)
        drawing_mode = True
        prev_x = 0
        prev_y = 0

    # Blue
    elif key == ord('b'):
        draw_color = (255, 0, 0)
        drawing_mode = True
        prev_x = 0
        prev_y = 0

    # Green
    elif key == ord('g'):
        draw_color = (0, 255, 0)
        drawing_mode = True
        prev_x = 0
        prev_y = 0

    # Yellow
    elif key == ord('y'):
        draw_color = (0, 255, 255)
        drawing_mode = True
        prev_x = 0
        prev_y = 0


# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()