# SimplexMethodProyect
Es software proporciona las herramientas esenciales para resolver problemas de optimización mediante el método simplex

## Autor

- **Lucas Candia** – [@lucashk2003](https://github.com/lucashk2003)

## Desarrollador principal
- Lucas Candia – ([@lucashk2003](https://github.com/lucashk2003))

## Tecnologías

Este proyecto actualmente utiliza:

- **Python 3.13.4** – Lenguaje principal

## Como ejecutar:
desde la raiz del proyecto: python -m app.app

# Programa para resolver problemas de Programación Lineal (Método Simplex)

## Uso

Ejecutar el programa e ingresar los datos solicitados por consola.

### Ejemplo

Maximizar:
> Z = 3x1 + 5x2

Sujeto a:
> 2x1 + 3x2 ≤ 8  
> 2x1 + x2 ≤ 4  
> x1, x2 ≥ 0  

Entradas en el programa:

Ingrese el número de variables de decisión: 2

Ingrese el número de restricciones: 2

¿Es un problema de maximización? (s/n): s

Ingrese los coeficientes de la función objetivo: 3 5

Restricción 1: 2 3 <= 8

Restricción 2: 2 1 <= 4


Resultado esperado:
> x1 = 0, x2 = 8/3  
> Z = 5 * 8/3 = 13.33
