import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

prev_x, prev_y = None, None
draw_color = (0, 255, 255)
brush_size = 8

lower_blue = np.array([100, 120, 70])
upper_blue = np.array([130, 255, 255])

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera read failed")
        break

    frame = cv2.flip(frame, 1)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask,  None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) > 500:
            M  = cv2.moments(biggest)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), brush_size + 4, draw_color, 2)
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
    cv2.putText(output, "Hold BLUE object to draw | Q=quit | C=clear",
                (8, frame_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160,160,160), 1)

    cv2.imshow("Air Drawing", output)

    key = cv2.waitKey(1) & 0xFF
    if   key == ord('q'): break
    elif key == ord('c'): canvas = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

cap.release()
cv2.destroyAllWindows()