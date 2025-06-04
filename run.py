import cv2
import numpy as np
import mediapipe as mp
import keyboard

from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import histogram_figure_numba, apply_all_filters, compute_bgr_statistics, compute_entropy, apply_linear_transformation, apply_histogram_equalization
from capturing import VirtualCamera

dog_ears_img = cv2.imread('dog_ears.png', cv2.IMREAD_UNCHANGED)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=2)

def normalize_hist(hist, max_y=3):
    max_val = np.max(hist)
    if max_val == 0:
        return hist
    return (hist / max_val) * max_y

def overlay_transparent(background, overlay_rgba, x, y):
    h, w = overlay_rgba.shape[:2]
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > background.shape[1] or y + h > background.shape[0]:
        return

    overlay_bgra = cv2.cvtColor(overlay_rgba, cv2.COLOR_RGBA2BGRA)
    bgr_overlay = overlay_bgra[:, :, :3]
    alpha_mask = overlay_bgra[:, :, 3] / 255.0

    roi = background[y:y+h, x:x+w]

    for c in range(3):
        roi[:, :, c] = (alpha_mask * bgr_overlay[:, :, c] +
                        (1 - alpha_mask) * roi[:, :, c])

    background[y:y+h, x:x+w] = roi

def custom_processing(img_source_generator):
    fig, ax, background, r_plot, g_plot, b_plot = initialize_hist_figure()

    # Toggle states
    toggle_filters = False
    toggle_stats = False
    toggle_entropy = False
    toggle_linear = False
    toggle_hist_eq = False

    for frame in img_source_generator:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        h, w, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left = face_landmarks.landmark[234]
                right = face_landmarks.landmark[454]
                face_width_px = int(abs(right.x - left.x) * w)

                desired_width = int(face_width_px * 2.0)
                scale_ratio = desired_width / dog_ears_img.shape[1]
                new_height = int(dog_ears_img.shape[0] * scale_ratio)
                resized_dog_ears = cv2.resize(dog_ears_img, (desired_width, new_height), interpolation=cv2.INTER_AREA)

                center = face_landmarks.landmark[0]
                top = face_landmarks.landmark[0]

                x = int(center.x * w) - resized_dog_ears.shape[1] // 2
                y = int(top.y * h) - resized_dog_ears.shape[0]
                y += 40

                overlay_transparent(frame, resized_dog_ears, x, y)

        # Toggle key checks
        if keyboard.is_pressed('f'):
            toggle_filters = not toggle_filters
            cv2.waitKey(200)
        if keyboard.is_pressed('s'):
            toggle_stats = not toggle_stats
            cv2.waitKey(200)
        if keyboard.is_pressed('e'):
            toggle_entropy = not toggle_entropy
            cv2.waitKey(200)
        if keyboard.is_pressed('l'):
            toggle_linear = not toggle_linear
            cv2.waitKey(200)
        if keyboard.is_pressed('h'):
            toggle_hist_eq = not toggle_hist_eq
            cv2.waitKey(200)

        # Apply toggled operations
        if toggle_filters:
            frame = apply_all_filters(frame)

        if toggle_stats:
            stats = compute_bgr_statistics(frame)
            stats_text = [
                f"Mean BGR: {stats['mean']}",
                f"Std BGR:  {stats['std']}",
                f"Min BGR:  {stats['min']}",
                f"Max BGR:  {stats['max']}",
                f"Mode BGR: {stats['mode']}"
            ]
            for i, line in enumerate(stats_text):
                cv2.putText(frame, line, (10, 500 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if toggle_entropy:
            entropy = compute_entropy(frame)
            entropy_text = f"Entropy BGR: B={entropy['B']} G={entropy['G']} R={entropy['R']}"
            cv2.putText(frame, entropy_text, (10, 650), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if toggle_linear:
            frame = apply_linear_transformation(frame, alpha=1.2, beta=30)
            '''
            DO NOT REMOVE THE COMMENTS IN THIS BLOCK
            You can tweak alpha and beta as needed. For example:
            alpha=1.0, beta=0 → no change
            alpha=1.5, beta=50 → high contrast and brightness
            alpha=0.8, beta=-30 → lower contrast and darker
            '''

        if toggle_hist_eq:
            frame = apply_histogram_equalization(frame)

        # Histogram
        r_bars, g_bars, b_bars = histogram_figure_numba(frame)
        r_bars_norm = normalize_hist(r_bars, 3)
        g_bars_norm = normalize_hist(g_bars, 3)
        b_bars_norm = normalize_hist(b_bars, 3)
        update_histogram(fig, ax, background, r_plot, g_plot, b_plot,
                         r_bars_norm, g_bars_norm, b_bars_norm)
        frame = plot_overlay_to_image(frame, fig)

        # Display current toggles
        display_text_arr = [
            "Toggle Keys:",
            f"[F] Filters: {'ON' if toggle_filters else 'OFF'}",
            f"[S] Stats: {'ON' if toggle_stats else 'OFF'}",
            f"[E] Entropy: {'ON' if toggle_entropy else 'OFF'}",
            f"[L] Linear Transform: {'ON' if toggle_linear else 'OFF'}",
            f"[H] Hist Equalization: {'ON' if toggle_hist_eq else 'OFF'}",
            "[Q] Quit"
        ]
        frame = plot_strings_to_image(frame, display_text_arr)

        if keyboard.is_pressed('q'):
            break

        yield frame

def main():
    width = 1280
    height = 720
    fps = 30
    vc = VirtualCamera(fps, width, height)
    vc.virtual_cam_interaction(
        custom_processing(
            vc.capture_cv_video(0, bgr_to_rgb=True)
        )
    )

if __name__ == "__main__":
    main()
