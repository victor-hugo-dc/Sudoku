import sys
import numpy as np
import cv2
from utils import *
from sudoku import solve

if __name__ == '__main__':

    # ensure that the model works
    calibrate_model()

    window = "Sudoku Solver by @victor-hugo-dc"
    height = width = 450 # dimensions of the frames
    dimensions = np.float32([[0, 0],[width, 0], [0, height],[width, height]])
    cap = cv2.VideoCapture(0)
    previous_squares = "1" * 81 # the previous sequence of predicted squares
    sudoku = None

    if not cap.isOpened():
        sys.exit()
    
    ret, frame = cap.read()
    while ret:
        frame = cv2.resize(frame, (width, height)) 
        image = process_image(frame)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        board = extract_largest_contour(contours) # assume the largest square contour is the board
        if board is not None:
            corners = extract_corners(board)
            board = warp(frame.copy(), corners, dimensions, (width, height))
            board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
            board = crop_image(board, 0.98) # shave off the borders

            list_of_squares = extract_squares(board) # list of each square (image)
            predicted_squares = predict_squares(np.vstack(list_of_squares)) # list of predicted values
            pos_array = np.where(np.asarray(predicted_squares) > 0, 0, 1) # array that allows us to show solved numbers
            predicted_squares = ''.join(map(str, predicted_squares))

            if previous_squares != predicted_squares:
                # the predicted numbers differ from the previously predicted numbers,
                # recalculate the Sudoku board.
                solved = solve(predicted_squares)
                if solved != False:
                    sudoku = [*solved.values()]
                    sudoku = [*map(int, sudoku)]
            
            if sudoku is not None:
                solved = sudoku * pos_array
                frame = overlay(frame, solved, corners, height, width)
            
            previous_squares = predicted_squares

        if cv2.waitKey(1) == ord('q'):
            break

        cv2.imshow(window, frame)
        ret, frame = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()

