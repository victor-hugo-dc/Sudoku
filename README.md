# Sudoku AI
Finding a solution, in real-time, to a Sudoku puzzle using **Computer Vision**, **Neural Networks**, and the **Backtracking Algorithm** with constraint propogation.

## Table of Contents
1. [Introduction](#Introduction)
2. [Prerequisites](#Prerequisites)
3. [Usage](#Usage)
4. [Procedure](Procedure)

## Introduction
Sudoku is a combinatorial number-placement puzzle in which the goal is to fill a 9×9 grid with digits such that every square in each column, row, and box 
(3×3 subgrids) that compose the grid are filled with a permutation of the digits 1 to 9.

## Prerequisites

### Python
This project was written in Python version 3.7.7.

### OpenCV
OpenCV is open source computer vision library that includes functions primarily aimed at real-time computer vision.

### TensorFlow
TensorFlow is an open-source software library for machine learning, primarily focused on training and inference of deep neural networks.

To install all dependencies, run ```pip install -r requirements.txt```

## Usage
To run this project, clone the repository 
```
git clone https://github.com/victor-hugo-dc/Sudoku.git
```
and run
```
python3 main.py
```

## Procedure
1. **Image Processing**\
1.1 Gaussian Blur: Apply Gaussian smoothing to reduce image noise.\
1.2 Adaptive Threshold: Segment the regions in the image to identify the puzzle.
2. **Sudoku Puzzle Extraction**\
2.1 Find Contours, assume the largest four-side contour is the Sudoku puzzle.\
2.2 Find the coordinates of the corners of the contour found in the previous step.\
2.3 Warp the image: isolate the puzzle and warp it into a square image which is easier to process.\
2.4 Divide the board 9 ways horizontally and 9 ways vertically in order to extract each individual square.
3. **Predict the Digits**\
3.1 Process every square image.\
3.2 Use the pre-trained model to predict every number in each square.
4. **Solving the Puzzle**\
4.1 Use Peter Norvig's backtracking algorithm with constraint propogation to find the solution.\
4.2 Overlay the solution over the input frame.

## Resources
[OpenCV Sudoku Solver and OCR by Adrian Rosebrock](https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/)
[OpenCV Sudoku Solver by 
Murtaza's Workshop](https://www.youtube.com/watch?v=qOXDoYUgNlU)\
[Solving Every Sudoku Puzzle by Peter Norvig](https://norvig.com/sudoku.html)