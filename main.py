import sys
import numpy as np
import cv2
from utils import calibrate_model, process_image, extract_largest_contour, extract_board, extract_squares, predict_squares, overlay
from sudoku import solve

if __name__ == '__main__':
    # Ensure that the model works
    calibrate_model()

    window = "Sudoku Solver by @victor-hugo-dc"
    height = width = 450 # dimensions of the frames
    cap = cv2.VideoCapture(0)
    previous_squares = "1" * 81 # the previous sequence of predicted squares
    sudoku = None

    if not cap.isOpened():
        sys.exit()
    
    ret, frame = cap.read()
    while ret:
        frame = cv2.resize(frame, (width, height)) 
        contours, _ = cv2.findContours(process_image(frame), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = extract_largest_contour(contours) # assume the largest square contour is the board
        if largest_contour is not None:
            (corners, board) = extract_board(frame, largest_contour)
            extracted_squares = extract_squares(board) # list of each square (image)
            predicted_squares = predict_squares(extracted_squares) # list of predicted values
            pos_array = np.where(predicted_squares > 0, 0, 1) # array that allows us to show solved numbers
            predicted_squares = np.array2string(predicted_squares, max_line_width = 85, separator = '').strip('[]')

            if previous_squares != predicted_squares:
                # The predicted numbers differ from the previously predicted numbers,
                # recalculate the Sudoku board.
                solved = solve(predicted_squares)
                if solved != False:
                    sudoku = [*map(int, solved.values())]
            
            if sudoku is not None:
                solved = sudoku * pos_array
                frame = overlay(frame, solved, corners)
            
            previous_squares = predicted_squares

        if cv2.waitKey(1) == ord('q'):
            break

        cv2.imshow(window, frame)
        ret, frame = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()