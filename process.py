import cv2
import numpy as np
from sudoku import solve
from utils import process_image, extract_largest_contour, extract_board, extract_squares, predict_squares, overlay

class Sudoku(object):
    def __init__(self) -> None:
        self.previous_squares = "1" * 81
        self.sudoku = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = cv2.flip(frame, 1)
        contours, _ = cv2.findContours(process_image(frame), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = extract_largest_contour(contours) # assume the largest square contour is the board
        if largest_contour is not None:
            (corners, board) = extract_board(frame, largest_contour)
            extracted_squares = extract_squares(board) # list of each square (image)
            predicted_squares = predict_squares(extracted_squares) # list of predicted values
            pos_array = np.where(predicted_squares > 0, 0, 1) # array that allows us to show solved numbers
            predicted_squares = np.array2string(predicted_squares, max_line_width = 85, separator = '').strip('[]')

            if self.previous_squares != predicted_squares:
                # The predicted numbers differ from the previously predicted numbers,
                # recalculate the Sudoku board.
                solved = solve(predicted_squares)
                if solved != False:
                    self.sudoku = [*map(int, solved.values())]
                
            if self.sudoku is not None:
                solved = self.sudoku * pos_array
                frame = overlay(frame, solved, corners)
                
            previous_squares = predicted_squares
        return frame