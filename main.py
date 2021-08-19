import sys
import numpy as np
import cv2
from utils import calibrate_model
from process import Sudoku

if __name__ == '__main__':
    # Ensure that the model works
    calibrate_model()
    sudoku = Sudoku()
    cap = cv2.VideoCapture(0)
    window = "Sudoku Solver by @victor-hugo-dc"
    
    if not cap.isOpened():
        sys.exit()
    
    ret, frame = cap.read()
    while ret:
        frame = sudoku.process_frame(frame)

        if cv2.waitKey(1) == ord('q'):
            break

        cv2.imshow(window, frame)
        ret, frame = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()