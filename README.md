# 📷 VirtualCamera

This project is a simple virtual camera application with computer vision features like dog ear overlays and basic image processing filters using OpenCV. It captures video, applies real-time effects, and displays the output using your webcam feed.

---

## ⚙️ Requirements

**Python 3.10** is required for compatibility with compiled Python files.

Install all necessary dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Navigate to the project directory:

```bash
cd VirtualCamera
```

Run the application:

```bash
python run.py
```

This will open your webcam and apply a **sharpen filter** by default along with any overlay logic configured. 

The Toggles are 'OFF' by default to reduce the workload on your computer. To turn them 'ON', please refer the info chart on the overlay.

You will also need OBS Studio with a virtual camera source to run the Virtual Camera <- which you can use in Discord or other applications.

---

## 🎨 Filters

Inside `basics.py`, several filters are defined such as:

- Grayscale
- Sepia
- Cartoon
- Negative
- Sharpen (enabled by default)

By default, **only the sharpen filter is active**. If you want to enable other filters:

1. Open `basics.py`
2. Locate the filter section.
3. Uncomment the desired filters and comment out the ones you don't need.

```python
# Example: To use grayscale instead of sharpen
# frame = apply_sharpen(frame)  # Comment this
frame = apply_grayscale(frame)  # Uncomment this
```

---

## 🐶 Dog Ears Overlay

The project includes a `dog_ears.png` file used for adding a dog ear overlay using facial landmarks. This logic is implemented in `overlays.py` and used in `capturing.py`.

---

## 👨🏻‍💻 Credits

Harmish Tanna : harmishhtanna@gmail.com <br>
Adil Ahmed : ada3971@thi.de
