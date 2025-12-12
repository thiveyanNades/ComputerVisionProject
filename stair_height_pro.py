import cv2
import numpy as np
import time

# --- Configuration Constant ---
ASSUMED_RISER_HEIGHT_CM = 19.0

# --- Helper Functions for Geometry (The 3D Challenge) ---

def calculate_step_height_real_world(step_lines, camera_matrix, focal_length_mm, sensor_height_mm):
    """
    Counts the distinct steps by clustering horizontal lines that are close together.
    """
    if not step_lines:
        return 0, 0.0

    horizontal_y_coords = []
    ANGLE_TOLERANCE = 30

    for line in step_lines:
        x1, y1, x2, y2 = line
        angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi

        is_horizontal = abs(angle) < ANGLE_TOLERANCE or abs(angle) > (180 - ANGLE_TOLERANCE)
        if is_horizontal:
            horizontal_y_coords.append((y1 + y2) / 2)

    if not horizontal_y_coords:
        return 0, 0.0

    CLUSTER_THRESHOLD = 15
    horizontal_y_coords.sort()

    distinct_steps = 0
    last_y = -CLUSTER_THRESHOLD

    for current_y in horizontal_y_coords:
        if current_y > last_y + CLUSTER_THRESHOLD:
            distinct_steps += 1
            last_y = current_y

    average_pixel_height = 0.0
    return distinct_steps, average_pixel_height

# --- Real-Time Detection Logic ---

def detect_stairs_and_height_realtime():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    placeholder_matrix = np.eye(3)

    print("--- Real-Time Stair Detector Running ---")
    print(f"Assumed Step Riser Height: {ASSUMED_RISER_HEIGHT_CM:.1f} cm")
    print("Point the camera at stairs. Press 'q' to quit.")

    frame_count = 0
    start_time = time.time()
    fps = 0.0

    num_steps = 0
    total_height_cm = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # --- Preprocessing ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # --- Edge Detection ---
        edges = cv2.Canny(blur, 30, 90)

        # --- Hough Line Transform ---
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=15,
            minLineLength=20,
            maxLineGap=10
        )

        detected_step_lines = []
        frame_display = frame.copy()

        ANGLE_TOLERANCE = 30

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi

                is_horizontal = abs(angle) < ANGLE_TOLERANCE or abs(angle) > (180 - ANGLE_TOLERANCE)
                is_vertical = 75 < abs(angle) < 105

                if is_horizontal or is_vertical:
                    cv2.line(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    detected_step_lines.append(line[0])

            num_steps, _ = calculate_step_height_real_world(
                detected_step_lines,
                placeholder_matrix,
                focal_length_mm=4.0,
                sensor_height_mm=3.6
            )

            total_height_cm = num_steps * ASSUMED_RISER_HEIGHT_CM

            # --- Display Text ---
            step_count_text = f"Steps Counted: {num_steps}"
            height_text = f"Total H: {total_height_cm:.1f} cm (RISER={ASSUMED_RISER_HEIGHT_CM:.1f}cm)"

            CYAN = (255, 255, 0)
            BLACK = (0, 0, 0)
            RED = (0, 0, 255)

            cv2.putText(frame_display, step_count_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)

            # Outline (black)
            cv2.putText(frame_display, height_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 4)

            # Main text (red)
            cv2.putText(frame_display, height_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

        else:
            cv2.putText(frame_display, "Stairs Not Detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- FPS Calculation ---
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            start_time = time.time()
            frame_count = 0

        cv2.putText(frame_display, f"FPS: {fps:.1f}",
                    (frame_display.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)

        cv2.imshow("Real-Time Stair Detection", frame_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Execution ---
if __name__ == "__main__":
    detect_stairs_and_height_realtime()
