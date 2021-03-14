# The following code was written following Peter Norvig's essay on Solving Every
# Sudoku Puzzle. It has been adapted to Python3 and is being primarily used for
# solving Sudoku Puzzles in real time using OpenCV and TensorFlow.

# The following function returns the cross product of the two input strings, A and B.
# This function is primarily used for creating neccessary lists containing information
# relating to the grid.
def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]

# In Sudoku, columns are numbered 1-9 and rows are labeled A-I.
digits   = '123456789'
rows     = 'ABCDEFGHI'
cols     = digits
squares  = cross(rows, cols)
unitlist = ([cross(rows, c) for c in cols] + [cross(r, cols) for r in rows] + [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')])
# The following descriptions for units and peers are from Norvig's essay.
# units is a dictionary where each square maps to the list of units that contain the square
units = dict((s, [u for u in unitlist if s in u]) for s in squares)
# peers is a dictionary where each square s maps to the set of squares formed by the union 
# of the squares in the units of s, but not s itself.
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in squares)

def parse_grid(grid: str):
    """Convert grid to a dict of possible values, {square: digits}, or
    return False if a contradiction is detected."""
    ## To start, every square can be any digit; then assign values from the grid.
    values = dict((s, digits) for s in squares)
    for square, value in grid_values(grid).items():
        # assign(values, square, value) will return the updated values, but if the assignment cannot be made then assign returns False.
        # assign(values, square, value) is similar to values[square] = value but with constraint propagation
        if value in digits and not assign(values, square, value):
            return False # assignment could not be made
    return values

def grid_values(grid):
    "Convert grid into a dict of {square: char} with '0' or '.' for empties."
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))

def assign(values: dict, square: str, value: str):
    """Eliminate all the other values (except d) from values[s] and propagate.
    Return values, except return False if a contradiction is detected."""
    # This function serves as an alternative to values[square] = value.
    # We implement this method not by assigning a value to the square, but rather by eliminating
    # one of the possible values for a square.
    other_values = values[square].replace(value, '')
    # The below code eliminates the values `val` from the other values in values[square]
    if all(eliminate(values, square, val) for val in other_values):
        return values
    else:
        return False

def eliminate(values, s, d):
    """Eliminate d from values[s]; propagate when values or places <= 2.
    Return values, except return False if a contradiction is detected."""
    if d not in values[s]:
        return values ## Already eliminated
    values[s] = values[s].replace(d,'')
    ## (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
    if len(values[s]) == 0:
	    return False ## Contradiction: removed last value
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    ## (2) If a unit u is reduced to only one place for a value d, then put it there.
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False ## Contradiction: no place for this value
        elif len(dplaces) == 1:
            # d can only be in one place in unit; assign it there
            if not assign(values, dplaces[0], d):
                return False
    return values

def solve(grid: str): 
    return search(parse_grid(grid))

def search(values):
    "Using depth-first search and propagation, try all possible values."
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in squares): 
        return values ## Solved!
    ## Chose the unfilled square s with the fewest possibilities
    n,s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    return some(search(assign(values.copy(), s, d)) for d in values[s])

def some(seq):
    "Return some element of seq that is true."
    for e in seq:
        if e: return e
    return False