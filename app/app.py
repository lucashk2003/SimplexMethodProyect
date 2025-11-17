from services.simplex_methods import SimplexSimple, SimplexAdvanced
from utils.helpers import print_tableau
import numpy as np

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
        elif respuesta in ['n', 'no']:
            return False
        else:
            print("❌ Error: Responda 's' para sí o 'n' para no")

def solicitar_array(mensaje, longitud_esperada, nombre_variable):
    """
    Solicita un array de números al usuario con validación.
    
    Args:
        mensaje (str): Mensaje a mostrar
        longitud_esperada (int): Cantidad de números esperados
        nombre_variable (str): Nombre descriptivo para mensajes de error
        
    Returns:
        np.array: Array de floats con los valores ingresados
    """
    while True:
        try:
            # Obtener entrada del usuario
            entrada = input(mensaje).strip()
            
            # Validar que no esté vacío
            if not entrada:
                print(f"❌ Error: Debe ingresar {longitud_esperada} números")
                print(f"   Formato correcto: número1 número2 ... (separados por espacios)")
                continue
            
            # Validar que no use comas
            if ',' in entrada:
                print(f"❌ Error: No use comas para separar los números")
                print(f"   Use ESPACIOS. Ejemplo: 3 5 2")
                continue
            
            # Intentar convertir a array de floats
            valores = list(map(float, entrada.split()))
            
            # Validar longitud
            if len(valores) != longitud_esperada:
                print(f"❌ Error: Se esperaban {longitud_esperada} números, pero ingresó {len(valores)}")
                print(f"   Variable: {nombre_variable}")
                continue
            
            # Todo OK, retornar array
            return np.array(valores, dtype=float)
            
        except ValueError:
            print(f"❌ Error: Todos los valores deben ser números válidos")
            print(f"   Asegúrese de ingresar solo números separados por espacios")
            print(f"   Ejemplo correcto: 2.5 3 1.8")

def solicitar_tipo_restriccion():
    """
    Solicita el tipo de restricción con validación.
    
    Returns:
        str: Tipo de restricción ('<=', '>=', o '=')
    """
    while True:
        tipo = input("Tipo de restricción (<=, >=, =): ").strip()
        if tipo in ['<=', '>=', '=']:
            return tipo
        else:
            print("❌ Error: Tipo de restricción inválido")
            print("   Opciones válidas: <=  >=  =")

def solicitar_float(mensaje):
    """
    Solicita un número decimal al usuario con validación.
    
    Args:
        mensaje (str): Mensaje a mostrar
        
    Returns:
        float: Número ingresado correctamente
    """
    while True:
        try:
            valor = float(input(mensaje))
            return valor
        except ValueError:
            print("❌ Error: Debe ingresar un número válido (entero o decimal)")

def main():
    """
    Función principal que solicita datos del problema de programación lineal
    y resuelve usando el método Simplex.
    """
    print("\n" + "="*60)
    print("  MÉTODO SIMPLEX - Resolución de Problemas de Optimización")
    print("="*60 + "\n")
    
    # Solicitar número de variables de decisión
    print("📊 CONFIGURACIÓN DEL PROBLEMA")
    print("-" * 40)
    
    num_vars = solicitar_entero("Ingrese el número de variables de decisión (x1, x2, ...): ")
    
    # Solicitar número de restricciones
    num_restr = solicitar_entero("Ingrese el número de restricciones: ")

    # Preguntar si es maximización o minimización
    es_maximizacion = solicitar_si_no("\n¿Es un problema de MAXIMIZACIÓN? (s/n): ")
    tipo_problema = "Maximizar" if es_maximizacion else "Minimizar"
    
    print(f"\n✓ Configuración: {tipo_problema} con {num_vars} variables y {num_restr} restricciones\n")
    
    # Solicitar coeficientes de la función objetivo
    print("🎯 FUNCIÓN OBJETIVO")
    print("-" * 40)
    print(f"Ingrese los coeficientes de la funcion objetivo Z = c1*x1 + c2*x2 + ... + c{num_vars}*x{num_vars}")
    c = solicitar_array(
        f"Coeficientes (separados por espacios): ",
        num_vars,
        "función objetivo"
    )
    
    # Mostrar función objetivo ingresada
    terminos = [f"{c[i]}x{i+1}" for i in range(num_vars)]
    funcion_obj = " + ".join(terminos).replace("+ -", "- ")
    print(f"   → {tipo_problema} Z = {funcion_obj}\n")
    
    # Inicializar matrices para restricciones
    A = np.zeros((num_restr, num_vars))
    b = np.zeros(num_restr)
    constraints_types = []  # ✅ Inicializar acá
    
    # Solicitar restricciones
    print("📋 RESTRICCIONES")
    print("-" * 40)
    for i in range(num_restr):
        print(f"\nRestricción {i+1}:")
        
        # Coeficientes de A
        A[i] = solicitar_array(
            f"  Coeficientes (a{i+1}1 a{i+1}2 ... a{i+1}{num_vars}): ",
            num_vars,
            f"restricción {i+1}"
        )
        
        # Tipo de restricción
        tipo = solicitar_tipo_restriccion()
        constraints_types.append(tipo)  # ✅ Guardar cada tipo
        
        # Eliminar si las linea comentadas a continuacion si no generan inconvenientes
        #if tipo != '<=':
        #    print("   ⚠️  Advertencia: Esta versión solo maneja restricciones <=")
        #    print("      La restricción se tratará como <=")
        
        # Lado derecho (b)
        b[i] = solicitar_float(f"  Valor del lado derecho (b{i+1}): ")
        
        # Mostrar restricción ingresada
        terminos_restr = [f"{A[i][j]}x{j+1}" for j in range(num_vars)]
        restriccion = " + ".join(terminos_restr).replace("+ -", "- ")
        print(f"   → {restriccion} {tipo} {b[i]}")  # ✅ Mostrar el tipo correcto
    
    # Resolver el problema
    print("\n" + "="*60)
    print("  RESOLVIENDO...")
    print("="*60 + "\n")
    
    try:
        # simplex = SimplexSimple()
        simplex = SimplexAdvanced()
        solution, z = simplex.solve(c, A, b, constraints_types, es_maximizacion)
        
        # Mostrar resultados
        print("✅ SOLUCIÓN ÓPTIMA ENCONTRADA")
        print("-" * 40)
        for i in range(num_vars):
            print(f"   x{i+1} = {solution[i]:.4f}")
        
        print(f"\n🎯 Valor óptimo de Z = {z:.4f}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR al resolver el problema:")
        print(f"   {str(e)}")
        print("\nVerifique que:")
        print("  • Las restricciones sean válidas")
        print("  • Los valores ingresados sean correctos")
        print("  • El problema tenga solución factible\n")

if __name__ == "__main__":
    main()