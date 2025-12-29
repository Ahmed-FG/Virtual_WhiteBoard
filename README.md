# Virtual WhiteBoard

A lightweight, webcam-driven virtual whiteboard that uses MediaPipe gesture recognition to draw, erase, and display animated GIFs on a dotted whiteboard surface in real time.

This README documents the exact requirements and behavior implemented in `main.py`.

## Requirements

- Python: 3.10.11
- mediapipe: 0.10.9
- opencv-python (OpenCV): 4.10.0
- numpy: 1.26.4
- Pillow: 11.0.0

Install the Python packages with:
```bash
python -m pip install mediapipe opencv-python numpy pillow
```

## Repository assets required (place in repository root)
- `gesture_recognizer.task` — MediaPipe gesture recognizer model used by the main.
- `waving-hand.gif` — GIF shown for Open_Palm gesture.
- `victory-hand.gif` — GIF shown for Victory gesture.

## Quick start

1. Create & activate a virtual environment (recommended):
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
python -m pip install mediapipe opencv-python numpy pillow
```

3. Make sure the model file and GIFs listed above are in the same folder as `main.py`.

4. Run the app:
```bash
python main.py
```

5. The app opens a resizable window titled `Whiteboard Window`. Press `q` to quit.

## How it works (overview)

- The app captures frames from your webcam (camera index 0), flips them horizontally for a mirror effect, and sends them to MediaPipe GestureRecognizer.
- A dotted whiteboard is rendered and a separate `mask` image stores drawing strokes and erasures.
- The app combines the whiteboard and mask each frame and overlays a small, resized camera preview in the bottom-right corner.
- GIF frames are loaded using Pillow and overlaid with alpha blending when certain gestures are detected.

## Gestures & behavior

- Closed_Fist
  - Enables drawing mode (sets a flag). After this gesture, drawing gestures will be accepted.
- Pointing_Up
  - When drawing mode is enabled, draws a line following the index fingertip (landmark 8). Uses `thickness = 4` and `color = (0, 0, 139)` (BGR).
- Thumb_Up
  - Acts as an eraser: draws an eraser rectangle centered at the thumb tip (landmark 4) and clears that area from the mask.
- Open_Palm
  - Displays `waving-hand.gif` centered on landmark 9 (palm) and advances GIF frames each time it is shown.
- Victory
  - Displays `victory-hand.gif` centered on landmark 9 and advances GIF frames.
- Thumb_Down
  - Clears the entire mask (erases all drawings).
- Any other or missing gesture resets the previous point for drawing.

Notes:
- Drawing coordinates are mapped from normalized landmark coordinates to the current window size.
- The whiteboard is recreated each loop iteration to support window resizing correctly; the mask is preserved and resized with nearest-neighbor interpolation if the window size changes.
- The small camera preview is resized to 200×150 and placed in the bottom-right corner of the whiteboard output.

## Display & window behavior

- Window title: `Whiteboard Window`
- Initial resize: 800×600
- Camera capture requested frame size: 1280×720 (via OpenCV camera properties), but the app adapts drawing coordinates to the actual displayed window size.
- To quit: press `q` while the window is focused.

## Troubleshooting

- Webcam not detected: ensure your camera is available and not used by another app. Try different camera index or check OS privacy permissions.
- MediaPipe model error: ensure `gesture_recognizer.task` is present and compatible with the installed MediaPipe version.
- GIFs not visible: verify `waving-hand.gif` and `victory-hand.gif` exist; GIF frames are converted to BGRA and blended onto the whiteboard—transparent areas should show the whiteboard.
- OpenCV GUI not opening or crashing: some headless environments (servers, Docker without display) will not support GUI windows. Run locally with a display.
- Performance: MediaPipe gesture recognition and GIF overlay are processed per frame — slower machines may observe dropped frames. Use a smaller camera resolution or skip GIF overlays to improve performance.

## Customization

- Change draw color and thickness by editing:
  - `thickness = 4`
  - `color = (0, 0, 139)` (BGR)
- Change eraser rectangle size in `draw_eraser_rectangle(...)` call (50×50 in the code).
- Change camera index or capture resolution via OpenCV `VideoCapture` settings.

## Development notes

- The app uses MediaPipe Tasks API:
  - `mp.tasks.BaseOptions`
  - `mp.tasks.vision.GestureRecognizer`
  - `mp.tasks.vision.GestureRecognizerOptions`
  - `mp.tasks.vision.RunningMode.VIDEO`
- GIF frames are loaded using Pillow `ImageSequence` and converted to BGRA for alpha compositing with OpenCV arrays.

## Contributing

Feel free to open issues or pull requests to:
- Add persistence for saved boards (files or DB).
- Add multi-user collaboration (networked sync).
- Improve gesture-to-action mapping or make gestures configurable.
- Add unit tests and CI.

## License(MIT)

## Author / Contact

Ahmed-FG
