import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

prev_x, prev_y = None, None
draw_color = (0, 255, 255)
brush_size = 8

lower_skin = np.array([0,  20, 70],  dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.erode(mask,  None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        hand = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand) > 8000:
            topmost = tuple(hand[hand[:, :, 1].argmin()][0])
            cx, cy  = topmost
            cv2.circle(frame, (cx, cy), 10, draw_color, -1)
            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), draw_color, brush_size)
            prev_x, prev_y = cx, cy
        else:
            prev_x = prev_y = None
    else:
        prev_x = prev_y = None

    gray     = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask2 = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask2)
    bg       = cv2.bitwise_and(frame,  frame,  mask=mask_inv)
    fg       = cv2.bitwise_and(canvas, canvas, mask=mask2)
    output   = cv2.add(bg, fg)

    cv2.rectangle(output, (0, frame_h-28), (frame_w, frame_h), (20,20,20), -1)
    cv2.putText(output, "Raise your hand to draw | Q=quit | C=clear",
                (8, frame_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)

    cv2.imshow("Air Drawing", output)

    key = cv2.waitKey(1) & 0xFF
    if   key == ord('q'): break
    elif key == ord('c'): canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()
