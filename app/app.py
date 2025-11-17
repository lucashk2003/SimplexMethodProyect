from services.simplex_methods import SimplexSimple, SimplexAdvanced
from utils.helpers import print_tableau # Importar print_tableau para usarlo en caso de que lo necesite en el futuro.
import numpy as np

# Constante de tolerancia para comparaciones de punto flotante
TOLERANCE = 1e-9

def solicitar_entero(mensaje):
    """
    Solicita un número entero al usuario con validación.

    Args:
        mensaje (str): Mensaje a mostrar al usuario

    Returns:
        int: Número entero ingresado correctamente
    """
    while True:
        try:
            valor = int(input(mensaje))
            if valor <= 0:
                print("❌ Error: Debe ser un número mayor a 0")
                continue
            return valor
        except ValueError:
            print("❌ Error: Debe ingresar un número entero válido")

def solicitar_si_no(mensaje):
    """
    Solicita respuesta sí/no al usuario con validación.

    Args:
        mensaje (str): Pregunta a mostrar

    Returns:
        bool: True si responde 's', False si responde 'n'
    """
    while True:
        respuesta = input(mensaje).strip().lower()
        if respuesta in ['s', 'si', 'sí']:
            return True
        elif respuesta in ['n', 'no']:\
            return False
        else:
            print("❌ Error: Responda 's' para sí o 'n' para no")

def solicitar_float(mensaje):
    """
    Solicita un número de punto flotante al usuario con validación.

    Args:
        mensaje (str): Mensaje a mostrar al usuario

    Returns:
        float: Número flotante ingresado
    """
    while True:
        try:
            return float(input(mensaje))
        except ValueError:
            print("❌ Error: Debe ingresar un número válido")

def solicitar_array(mensaje, longitud_esperada, nombre_variable):
    """
    Solicita un array de números al usuario con validación.

    Args:
        mensaje (str): Mensaje a mostrar al usuario
        longitud_esperada (int): Número de elementos esperados
        nombre_variable (str): Nombre de la variable (ej: 'c', 'A') para mensajes

    Returns:
        np.array: Array de números flotantes
    """
    while True:
        entrada = input(mensaje).replace(',', ' ').split()
        if len(entrada) != longitud_esperada:
            print(f"❌ Error: Debe ingresar {longitud_esperada} valores para {nombre_variable}.")
            continue
        try:
            return np.array([float(x) for x in entrada], dtype=float)
        except ValueError:
            print("❌ Error: Todos los valores deben ser números válidos.")

def main():
    """
    Función principal de la aplicación.
    Recoge datos del usuario y resuelve el problema de programación lineal.
    """
    print("\n" + "="*60)
    print("      PROYECTO MÉTODO SIMPLEX - INGRESO DE DATOS")
    print("="*60)

    # 1. Definir la Función Objetivo
    es_maximizacion = solicitar_si_no("¿El problema es de Maximización? (s/n): ")
    if es_maximizacion:
        print("Función Objetivo: MAXIMIZAR Z")
    else:
        print("Función Objetivo: MINIMIZAR Z")

    # 2. Número de Variables
    num_vars = solicitar_entero("Ingrese el número de variables (x1, x2, ...): ")
    print(f"✅ {num_vars} variables definidas.")

    # 3. Coeficientes de la Función Objetivo (c)
    print("\n  COEFICIENTES DE LA FUNCIÓN OBJETIVO (c)")
    print("-" * 40)
    c = solicitar_array(
        f"  Ingrese los {num_vars} coeficientes de Z (separados por espacio): ",
        num_vars, "Z"
    )
    terminos_z = [f"{c[i]}x{i+1}" for i in range(num_vars)]
    z_str = " + ".join(terminos_z).replace("+ -", "- ")
    print(f"   → Z = {z_str}")

    # 4. Número de Restricciones
    num_restr = solicitar_entero("\nIngrese el número de restricciones: ")
    print(f"✅ {num_restr} restricciones definidas.")

    # Inicializar estructuras para restricciones
    A = np.zeros((num_restr, num_vars), dtype=float)
    b = np.zeros(num_restr, dtype=float)
    constraints_types = []

    # 5. Ingreso de Restricciones (Matriz A, Tipos, Vector b)
    print("\n  DEFINICIÓN DE RESTRICCIONES (A*x tipo b)")
    print("-" * 40)
    for i in range(num_restr):
        print(f"\n  RESTRICCIÓN {i+1} de {num_restr}:")

        # Coeficientes de la restricción (fila i de A)
        print("  Ingrese los coeficientes (A) de la restricción, separados por espacio:")
        A[i] = solicitar_array(f"    Coeficientes de x1 a x{num_vars}: ", num_vars, f"A[{i+1}]")

        # Tipo de restricción
        while True:
            tipo = input("  Tipo de restricción ('<=', '>=' o '='): ").strip()
            if tipo in ['<=', '>=', '=']:
                constraints_types.append(tipo)
                break
            else:
                print("❌ Error: Tipo de restricción inválido. Use '<=', '>=' o '='.")

        # # Código original comentado: Solo permitía '<='
        # # if tipo != '<=':
        # #    print("   ⚠️  Advertencia: Esta versión solo maneja restricciones <=\")
        # #    print("      La restricción se tratará como <=\")

        # Lado derecho (b)
        b[i] = solicitar_float(f"  Valor del lado derecho (b{i+1}): ")

        # Mostrar restricción ingresada
        terminos_restr = [f"{A[i][j]}x{j+1}" for j in range(num_vars)]
        restriccion = " + ".join(terminos_restr).replace("+ -", "- ")
        print(f"   → {restriccion} {tipo} {b[i]}")  # ✅ Mostrar el tipo correcto

    # =========================================================================
    # 6. LÓGICA DE SELECCIÓN AUTOMÁTICA DEL MÉTODO (Opción 2)
    # =========================================================================
    # Criterio para SimplexSimple:
    # 1. Debe ser Maximización.
    # 2. Todas las restricciones deben ser de tipo '<='.
    # 3. El vector b (lado derecho) debe ser no negativo (b >= 0).

    # np.all(b >= -TOLERANCE) verifica si todos los b[i] son >= 0, 
    # usando tolerancia para evitar errores de punto flotante.
    is_simple_form = (
        es_maximizacion and
        all(t == '<=' for t in constraints_types) and
        np.all(b >= -TOLERANCE)
    )

    if is_simple_form:
        simplex = SimplexSimple()
        print("\n⚙️ MÉTODO ELEGIDO: Simplex Simple (El problema está en forma estándar).")
    else:
        simplex = SimplexAdvanced()
        print("\n⚙️ MÉTODO ELEGIDO: Simplex Avanzado (Minimización o uso de variables artificiales).")
    
    # Resolver el problema
    print("\n" + "="*60)
    print("  RESOLVIENDO...")
    print("="*60 + "\n")

    try:
        # Código original: (comentado)
        # simplex = SimplexSimple() 
        # simplex = SimplexAdvanced()
        
        # El objeto 'simplex' ya está instanciado por la lógica de selección.
        solution, z = simplex.solve(c, A, b, constraints_types, es_maximizacion)

        # Mostrar resultados
        print("✅ SOLUCIÓN ÓPTIMA ENCONTRADA")
        print("-" * 40)
        for i in range(num_vars):
            print(f"   x{i+1} = {solution[i]:.4f}")

        print(f"\n🎯 Valor óptimo de Z = {z:.4f}")

    except ValueError as e:
        print("\n❌ ERROR DE SOLUCIÓN")
        print("-" * 40)
        print(f"   {e}")
        print("   No se pudo encontrar una solución o el problema es infactible/no acotado.")
    except Exception as e:
        print("\n❌ ERROR INESPERADO")
        print("-" * 40)
        print(f"   Ocurrió un error durante la ejecución: {e}")


if __name__ == "__main__":
    # La solución de Simplex Advanced requiere un número mayor de iteraciones
    # y la lógica interna es más robusta.
    # Por ejemplo, SimplexAdvanced usa una lógica más compleja para Minimizacion.

    # Establecer la configuración de impresión de numpy para una mejor visualización de números grandes
    np.set_printoptions(precision=4, suppress=True) 

    # Llamada a la función principal
    main()