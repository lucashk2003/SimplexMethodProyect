import numpy as np
from services.simplex_methods import SimplexSimple, SimplexIntermediate, SimplexAdvanced

def get_user_input():
    """Obtiene los datos del problema desde la consola."""
    try:
        n = int(input("Ingrese el número de variables de decisión: "))
        m = int(input("Ingrese el número de restricciones: "))
        is_max = input("¿Es un problema de maximización? (s/n): ").lower() == 's'

        print("Ingrese los coeficientes de la función objetivo (separados por espacios):")
        c = np.array([float(x) for x in input().split()], dtype=float)
        if len(c) != n:
            raise ValueError("El número de coeficientes no coincide con el número de variables.")

        A = []
        b = []
        constraints_types = []
        print("Ingrese las restricciones (coeficientes de A, tipo (<=, >=, =), y b):")
        for i in range(m):
            print(f"Restricción {i+1}:")
            row = [float(x) for x in input("Coeficientes de A (separados por espacios): ").split()]
            if len(row) != n:
                raise ValueError(f"Restricción {i+1} tiene un número incorrecto de coeficientes.")
            A.append(row)
            constraint_type = input("Tipo de restricción (<=, >=, =): ")
            if constraint_type not in ['<=', '>=', '=']:
                raise ValueError("Tipo de restricción inválido.")
            constraints_types.append(constraint_type)
            b.append(float(input("Valor de b: ")))

        return (np.array(c), np.array(A), np.array(b), constraints_types, is_max)
    except ValueError as e:
        print(f"Error en la entrada: {e}")
        return None

def main():
    print("Programa para resolver problemas de programación lineal con el método Simplex")
    data = get_user_input()
    if data is None:
        return

    c, A, b, constraints_types, is_max = data

    # Seleccionar el método según la complejidad
    if all(t == '<=' for t in constraints_types) and is_max:
        simplex = SimplexSimple()
    elif any(t in ['>=', '='] for t in constraints_types):
        simplex = SimplexIntermediate()
    else:
        simplex = SimplexAdvanced()

    try:
        solution, z = simplex.solve(c, A, b, constraints_types, is_max)
        print("Solución óptima:", solution)
        print("Valor óptimo Z:", z)
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()