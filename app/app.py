from services.simplex_methods import SimplexSimple
from utils.helpers import print_tableau
import numpy as np

def main():
    # Ejemplo básico: 
    # Max Z = 3x1 + 5x2
    # Sujeto a:
    # 2x1 + 3x2 <= 8
    # 2x1 + x2  <= 4
    # x1, x2 >= 0
    import numpy as np

    c = np.array([3, 5], dtype=float)
    A = np.array([[2, 3],
              [2, 1]], dtype=float)
    b = np.array([8, 4], dtype=float)

    simplex = SimplexSimple()
    solution, z = simplex.solve(c, A, b)

    print("Solución óptima:", solution)
    print("Valor óptimo Z:", z)

if __name__ == "__main__":
    main()
