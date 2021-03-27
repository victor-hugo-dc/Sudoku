import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow

def init_model() -> tensorflow.keras.Sequential:
    """
    Loads and returns pre-trained Keras model.
    :return: Sequential model with pre-trained weights.
    :rtype: tensorflow.keras.Sequential
    """
    model = load_model('./resources/model.h5', compile = False)
    return model

# Model used for digit recognition.
model = init_model()

def calibrate_model():
    print("Calibrating model.")
    six = cv2.imread('./resources/test.jpg')
    six = process_image(six)
    six = process_square(six)
    prediction = model.predict(six)
    six = np.argmax(prediction) + 1
    assert (six == 6), "Model prediction error."

def process_image(frame: np.ndarray) -> np.ndarray:
    """
    Converts the input image to grayscale, applies a Gaussian blur and an adaptive threshold,
    and returns the processed image.
    :param frame: Image to be processed.
    :type frame: NumPy array (np.ndarray).
    :return: A processed image.
    :rtype: NumPy array (np.ndarray).
    """
    result = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
    result = cv2.GaussianBlur(result, (9, 9), 1)
    result = cv2.adaptiveThreshold(result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 1, 11, 2)
    return result

def extract_largest_contour(contours: list, min_contour_area: int = 20000) -> np.ndarray:
    """
    Finds and returns the largest four-side contour in the input list.
    :param contours: Contours from an image.
    :type contours: List.
    :param min_contour_area: The minimum area for the largest contour, meant to reduce noise.
    :type min_contour_area: Int.
    :return: The largest contour that satisfies the aforementioned conditions.
    :rtype: NumPy array (np.ndarray).
    """
    for c in sorted(contours, key = cv2.contourArea, reverse = True)[:10]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4 and cv2.contourArea(c) >= min_contour_area:
            return approx
    
    return None

def extract_corners(pts: np.ndarray) -> np.ndarray:
    """
    Gets the coordinates of the corners from the input points.
    :param pts: A contour.
    :type pts: NumPy array (np.ndarray).
    :return: An ordered list of the corners of the input contour.
    :rtype: NumPy array (np.ndarray).
    """
    pts = pts.reshape((4, 2))
    rect = np.zeros((4, 1, 2), dtype = np.int32)
 
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
    return np.float32(rect)

def crop_image(frame: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Crop the image by a certain scale.
    :param frame: Input image to be cropped.
    :type frame: NumPy array (np.ndarray).
    :param scale: The scale for which to crop the image by.
    :type scale: Float.
    :return: The input image cropped by the given scale.
    :rtype: NumPy array (np.ndarray).
    """
    shape = np.array(frame.shape[::-1])
    center = shape / 2
    offset = scale * shape / 2
    l_x, t_y = np.subtract(center, offset).astype(int)
    r_x, b_y = np.add(center, offset).astype(int)
    crop = frame[t_y: b_y, l_x: r_x]
    return crop

def process_square(square: np.ndarray) -> np.ndarray:
    """
    Process the input image to prepare for model prediction.
    :param square: Image of a single square in the Sudoku puzzle.
    :type square: NumPy array (np.ndarray).
    :return: Processed square image of dimension 28 pixels by 28 pixels.
    :rtype: NumPy array (np.ndarray).
    """
    _, square = cv2.threshold(square, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    square = crop_image(square, 0.9)
    square = np.asarray(square)
    square = cv2.resize(square, (28, 28))
    square = square / 255
    square = square.reshape(-1, 28, 28, 1)
    return square

def extract_squares(board: np.ndarray) -> np.ndarray:
    """
    Divides the square board into 9 by 9 squares and processes each square image.
    :param board: Source image.
    :type board: NumPy array (np.ndarray).
    :return: NumPy array of the processed squares in the Sudoku puzzle.
    :rtype: Vertically stacked NumPy arrays (np.ndarray).
    """
    squares = [process_square(square) for row in np.vsplit(board, 9) for square in np.hsplit(row, 9)]
    return np.vstack(squares)

def predict_squares(squares: np.ndarray, min_confidence: float = 0.8) -> np.ndarray:
    """
    Predicts every square in the Sudoku puzzle.
    :param squares: List of the processed square images of the Sudoku puzzle.
    :type squares: NumPy array (np.ndarray).
    :param min_confidence: Minimum confidence threshold.
    :type min_confidence: Float.
    :return: List of predicted squares.
    :rtype: NumPy array (np.ndarray) of ints.
    """
    predictions = [np.argmax(prediction) + 1 if np.amax(prediction) > min_confidence else 0 for prediction in model.predict(squares)]
    return np.asarray(predictions)

def warp(image: np.ndarray, src: np.ndarray, dst: np.ndarray, dsize: tuple) -> np.ndarray:
    """
    Applies a geometric transformation on the input image.
    :param image: Source image.
    :type image: NumPy array (np.ndarray).
    :param src: Coordinates of quadrangle vertices in the source image.
    :type src: NumPy array (np.ndarray).
    :param dst: Coordinates of the corresponding quadrangle vertices in the destination image.
    :type dst: NumPy array (np.ndarray).
    :param dsize: Size of output image.
    :type dsize: Tuple.
    :return: Geometrically transformed image.
    :rtype: NumPy array (np.ndarray).
    """
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, dsize)

def extract_board(frame: np.ndarray, largest_contour: np.ndarray) -> tuple:
    """
    Given the largest contour in the frame, extract the corners of the board and the cropped board.
    :param frame: Image of the entire frame.
    :type frame: NumPy array (np.ndarray).
    :param largest_contour: Largest contour found in the frame.
    :type largest_contour: NumPy array (np.ndarray).
    :return: A tuple of the corners and the cropped board.
    :rtype: Tuple.
    """
    (height, width, _) = frame.shape
    dimensions = np.float32([[0, 0],[width, 0], [0, height],[width, height]])
    corners = extract_corners(largest_contour)
    board = warp(frame.copy(), corners, dimensions, (width, height))
    board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    board = crop_image(board, 0.98)
    return (corners, board)

def overlay_numbers(frame: np.ndarray, numbers: np.ndarray, color: tuple = (0,255,0)) -> np.ndarray:
    """
    Overlays the solved numbers over the correct part of the Sudoku puzzle.
    :param frame: An empty image.
    :type frame: NumPy array (np.ndarray).
    :param numbers: The solved numbers from the Sudoku puzzle.
    :type numbers: List of NumPy array (np.ndarray) of ints.
    :param color: The color for the overlay numbers.
    :type color: Tuple of dimension 3 (RGB).
    :return: An image with the numbers overlayed in the correct position with respect to the Sudoku puzzle.
    :rtype: NumPy array (np.ndarray).
    """
    sec_h, sec_w = (np.array(frame.shape[:2]) / 9).astype(int)
    for x in range(9):
        for y in range(9):
            number = numbers[(y * 9) + x]
            if number != 0:
                cv2.putText(frame, str(number), (x * sec_w + int(sec_w / 2) - 10, int((y + 0.8) * sec_h)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, color, 2, cv2.LINE_AA)
    return frame

def overlay(frame: np.ndarray, solved: list, corners: np.ndarray) -> np.ndarray:
    """
    Overlay the solution on the Sudoku puzzle.
    :param frame: The frame captured by the camera.
    :type frame: NumPy array (np.ndarray).
    :param solved: The solution of the Sudoku board.
    :type solved: List of ints.
    :param corners: Corners of the Sudoku board.
    :type corners: NumPy array (np.ndarray).
    :return: Full frame with solution overlayed.
    :rtype: NumPy array (np.ndarray).
    """
    (height, width, _) = frame.shape
    image = np.zeros((height, width, 3), np.uint8)
    image = overlay_numbers(image, solved)

    pts1 = np.float32([[0, 0],[width, 0], [0, height],[width, height]]) 
    pts2 = np.float32(corners)

    inv_warp_c = warp(image, pts1, pts2, (width, height))
    inv_perspective = cv2.addWeighted(inv_warp_c, 1, frame, 0.5, 1)
    return inv_perspective