# Virtual Air Canvas ✏️

A real-time air drawing application using your index finger and webcam.
Built with OpenCV and MediaPipe.

## Demo
> Draw in the air using just your index finger — no mouse, no touch needed!

## Features
- ☝️ Index finger to draw
- ✌️ Two fingers to pause
- 🎨 Color palette — Yellow, Green, Red, Blue, White, Pink
- ↩️ Undo with Z key
- ↪️ Redo with Y key
- 🗑️ Clear canvas with C key
- 📷 Live webcam feed with drawing overlay

## Requirements
- Python 3.12
- OpenCV
- MediaPipe
- NumPy

## Installation
pip install opencv-python mediapipe numpy

## How to Run
python air_drawing2.py

## Controls
| Action | Control |
|--------|---------|
| Draw | Index finger up |
| Pause | Two fingers up |
| Change color | Point at color button |
| Undo | Z key |
| Redo | Y key |
| Clear | C key |
| Quit | Q key |

## What I Learned
- Real time video processing with OpenCV
- Hand landmark detection with MediaPipe
- Gesture recognition using finger coordinates
- Canvas compositing with bitwise operations
- Debugging Windows camera driver issues
