from abc import ABC, abstractmethod

class SimplexBase(ABC):
    """
    Clase abstracta que define la estructura básica
    para los métodos de optimización con Simplex.
    """

    @abstractmethod
    def solve(self, c, A, b):
        """
        Método principal para resolver un problema de programación lineal.
        :param c: coeficientes de la función objetivo
        :param A: matriz de restricciones
        :param b: lado derecho de las restricciones
        :return: solución óptima
        """
        pass
