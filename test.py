import cv2
import pyvirtualcam
import time

def main():
    width = 1280
    height = 720
    fps = 30

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print("Cannot open webcam")
        return

    with pyvirtualcam.Camera(width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        print(f'Virtual camera started: {cam.device}')

        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            cam.send(frame)
            cam.sleep_until_next_frame()

            frame_count += 1
            elapsed = time.time() - start_time

            if elapsed >= 1.0:
                print(f"FPS: {frame_count / elapsed:.2f}")
                frame_count = 0
                start_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
