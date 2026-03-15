import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading model... wait")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
canvas  = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

prev_x, prev_y = None, None
draw_color = (0, 255, 255)
brush_size = 10

smooth_x, smooth_y = None, None
SMOOTH = 0.6

stroke_history = []
redo_history   = []
current_stroke = []

def redraw_canvas():
    c = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    for stroke in stroke_history:
        for i in range(1, len(stroke)):
            cv2.line(c, stroke[i-1], stroke[i], draw_color, brush_size)
    return c

with HandLandmarker.create_from_options(options) as detector:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            cx = int(lm[8].x * frame_w)
            cy = int(lm[8].y * frame_h)

            if smooth_x is None:
                smooth_x, smooth_y = cx, cy
            else:
                smooth_x = int(smooth_x * SMOOTH + cx * (1 - SMOOTH))
                smooth_y = int(smooth_y * SMOOTH + cy * (1 - SMOOTH))

            index_up  = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[9].y

            if index_up and not middle_up:
                cv2.circle(frame, (smooth_x, smooth_y), brush_size, draw_color, -1)
                if prev_x is not None:
                    cv2.line(canvas, (prev_x, prev_y), (smooth_x, smooth_y), draw_color, brush_size)
                    current_stroke.append((prev_x, prev_y))  # save actual drawn point
                prev_x, prev_y = smooth_x, smooth_y
            else:
                if current_stroke:
                    stroke_history.append(current_stroke.copy())
                    redo_history.clear()
                    current_stroke = []
                prev_x = prev_y = None

        else:
            if current_stroke:
                stroke_history.append(current_stroke.copy())
                redo_history.clear()
                current_stroke = []
            smooth_x = smooth_y = None
            prev_x = prev_y = None

        gray     = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask  = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg       = cv2.bitwise_and(frame,  frame,  mask=mask_inv)
        fg       = cv2.bitwise_and(canvas, canvas, mask=mask)
        output   = cv2.add(bg, fg)

        cv2.rectangle(output, (0, frame_h-28), (frame_w, frame_h), (20,20,20), -1)
        cv2.putText(output, "Index=draw | 2 fingers=pause | Z=undo | Y=redo | C=clear",
                    (8, frame_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)

        cv2.imshow("Air Drawing", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('z'):
            if stroke_history:
                redo_history.append(stroke_history.pop())
                canvas = redraw_canvas()
        elif key == ord('y'):
            if redo_history:
                stroke_history.append(redo_history.pop())
                canvas = redraw_canvas()
        elif key == ord('c'):
          stroke_history.clear()
          redo_history.clear()
          current_stroke = []
          canvas = np.zeros((frame_h, frame_w, 3), dtype=np.unit8)
        elif key == ord('z'):
          if stroke_history:
             redo_history.append(stroke_history.pop())
             canvas = redraw_canvas()
        elif key == ord('y'):
          if redo_history:
            stroke_history.append(redo_history.pop())
            canvas = redraw_canvas()      

cap.release()
cv2.destroyAllWindows()