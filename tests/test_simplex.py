import pytest
from services.simplex_methods import SimplexSimple

def test_simplex_basic():
    c = [3, 5]
    A = [[2, 3],
         [2, 1]]
    b = [8, 4]

    simplex = SimplexSimple()
    solution, z = simplex.solve(c, A, b)

    assert round(z, 2) == 12.0
    assert solution[0] == 0
    assert solution[1] == 4
