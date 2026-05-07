# Handwritten Digit Recognition

A CNN-based handwritten digit classifier trained on the MNIST dataset. Supports three input modes — drawing on a canvas, loading a static image, or streaming from a live phone camera — all wrapped in a simple Tkinter launcher.

## Features

- **CNN model** trained on MNIST (28×28 grayscale, digits 0–9)
- **Three prediction modes:**
  - Draw a digit with your mouse on an OpenCV canvas
  - Predict from a saved image file
  - Predict from a live phone camera feed (via IP Webcam)
- **Tkinter launcher** (`main.py`) — one window to launch any mode
- **Evaluation script** — plots training/validation loss curves and a confusion matrix

## Project Structure

```
Handwritten-Digit-Recognition/
├── main.py           # Tkinter launcher (TrackPad / Image / Live Image)
├── app.py            # Mouse-drawing canvas → predict on keypress
├── image.py          # Static image file → predict
├── image_live.py     # Live phone camera → predict on keypress
├── history_acc.py    # Plot training history + confusion matrix
├── CNN_nodel.h5      # Trained CNN model weights
├── CNN_model.h5      # Alternate trained model
├── history.pkl       # Saved training history (loss/val_loss)
└── 3.jpg             # Sample test image
```

## Requirements

- Python 3.7+
- TensorFlow / Keras
- OpenCV (`opencv-python`)
- NumPy
- Pillow
- SciPy
- scikit-learn
- Matplotlib
Install all dependencies:

```bash
pip install -r requirements.txt
```

> On Apple Silicon Macs (M1/M2/M3/M4) this automatically installs `tensorflow-macos` and `tensorflow-metal` instead of the standard `tensorflow` package.

## Usage

### Launch the GUI

```bash
python main.py
```

A window opens with three buttons:

| Button | Script | What it does |
|--------|--------|--------------|
| **TrackPad** | `app.py` | Draw a digit with your mouse on a black canvas. Press `p` to predict, `c` to clear, `q` to quit. |
| **Image** | `image.py` | Loads `files3.jpg`, preprocesses it (threshold → center-of-mass shift → resize to 28×28), and prints the prediction. |
| **Live Image** | `image_live.py` | Streams frames from a phone camera over HTTP. Press `p` to capture and predict, `q` to quit. |

### Run modes directly

```bash
python app.py          # Mouse drawing canvas
python image.py        # Static image prediction
python image_live.py   # Live phone camera prediction
```

### Live camera setup

`image_live.py` uses your laptop's built-in webcam (device index `0`). No setup required — just run it and point the camera at a handwritten digit.

### Evaluate the model

```bash
python history_acc.py
```

Plots training vs. validation loss curves and a confusion matrix evaluated on the MNIST test set.

## Model

The CNN is trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) (60,000 training / 10,000 test images of handwritten digits). Input images are preprocessed to 28×28 grayscale before inference.

Preprocessing pipeline (for external images):
1. Invert and threshold (Otsu)
2. Crop whitespace
3. Fit to 20×20 maintaining aspect ratio
4. Pad to 28×28
5. Center-of-mass shift for alignment

## License

MIT License © 2026 Dhruvil Vasoya
