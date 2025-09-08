def print_tableau(tableau):
    """
    Imprime la tabla simplex de manera legible.
    """
    for row in tableau:
        print(" | ".join([f"{x:8.2f}" for x in row]))
    print("-" * 50)
