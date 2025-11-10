from abc import ABC, abstractmethod

class SimplexBase(ABC):
    """
    Clase abstracta que define la estructura básica para los métodos de optimización con Simplex.
    """

    @abstractmethod
    def solve(self, c, A, b, constraints_types, is_max):
        """
        Método principal para resolver un problema de programación lineal.
        :param c: coeficientes de la función objetivo
        :param A: matriz de restricciones
        :param b: lado derecho de las restricciones
        :param constraints_types: lista de tipos de restricciones ('<=', '>=', '=')
        :param is_max: True si es maximización, False si es minimización
        :return: solución óptima y valor de la función objetivo
        """
        pass