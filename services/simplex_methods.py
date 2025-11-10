import numpy as np
from services.simplex_base import SimplexBase

class SimplexSimple(SimplexBase):
    """
    Implementación básica del método simplex para problemas en forma estándar (<=, maximización).
    """
    def solve(self, c, A, b, constraints_types, is_max):
        if not all(t == '<=' for t in constraints_types):
            raise ValueError("SimplexSimple solo soporta restricciones <=.")
        if not is_max:
            raise ValueError("SimplexSimple solo soporta maximización.")

        num_vars = len(c)
        num_restr = len(b)

        # Validaciones
        if A.shape != (num_restr, num_vars):
            raise ValueError("Dimensiones de A no coinciden con el número de variables/restricciones.")
        if len(b) != num_restr:
            raise ValueError("Dimensiones de b no coinciden con el número de restricciones.")
        if np.any(b < 0):
            raise ValueError("Los valores de b deben ser no negativos.")

        # Construcción del tableau
        tableau = np.zeros((num_restr + 1, num_vars + num_restr + 1))
        tableau[:-1, :num_vars] = A
        tableau[:-1, num_vars:num_vars + num_restr] = np.identity(num_restr)
        tableau[:-1, -1] = b
        tableau[-1, :num_vars] = -c

        # Iteraciones
        while np.any(tableau[-1, :-1] < 0):
            pivot_col = np.argmin(tableau[-1, :-1])
            ratios = []
            for i in range(num_restr):
                if tableau[i, pivot_col] > 0:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(np.inf)
            pivot_row = np.argmin(ratios)
            if ratios[pivot_row] == np.inf:
                raise ValueError("El problema es no acotado.")

            # Pivotaje
            tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
            for i in range(num_restr + 1):
                if i != pivot_row:
                    tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        # Extraer solución
        solution = np.zeros(num_vars)
        for i in range(num_restr):
            col = np.where(tableau[i, :num_vars] == 1)[0]
            if len(col) == 1:
                solution[col[0]] = tableau[i, -1]
        
        return solution, tableau[-1, -1]

class SimplexIntermediate(SimplexBase):
    """
    Implementación para problemas con restricciones <=, >=, = (método de dos fases).
    """
    def solve(self, c, A, b, constraints_types, is_max):
        num_vars = len(c)
        num_restr = len(b)

        # Validaciones
        if A.shape != (num_restr, num_vars):
            raise ValueError("Dimensiones de A no coinciden con el número de variables/restricciones.")
        if len(b) != num_restr or len(constraints_types) != num_restr:
            raise ValueError("Dimensiones de b o constraints_types no coinciden con el número de restricciones.")

        # Normalizar restricciones a forma estándar (<=)
        A_norm = A.copy()
        b_norm = b.copy()
        slack_vars = 0
        surplus_vars = 0
        artificial_vars = 0
        var_types = []

        for i, constraint_type in enumerate(constraints_types):
            if constraint_type == '<=':
                slack_vars += 1
                var_types.append('slack')
            elif constraint_type == '>=':
                A_norm[i, :] = -A_norm[i, :]  # Convertir a <=
                b_norm[i] = -b_norm[i]
                surplus_vars += 1
                var_types.append('surplus')
                artificial_vars += 1
                var_types.append('artificial')
            elif constraint_type == '=':
                artificial_vars += 1
                var_types.append('artificial')

        # Construir tableau inicial con variables de holgura, exceso y artificiales
        total_vars = num_vars + slack_vars + surplus_vars + artificial_vars
        tableau = np.zeros((num_restr + 1, total_vars + 1))
        col = num_vars
        artificial_cols = []

        for i, constraint_type in enumerate(constraints_types):
            tableau[i, :num_vars] = A_norm[i, :]
            if constraint_type == '<=':
                tableau[i, col] = 1  # Variable de holgura
                col += 1
            elif constraint_type == '>=':
                tableau[i, col] = -1  # Variable de exceso
                col += 1
                tableau[i, col] = 1  # Variable artificial
                artificial_cols.append(col)
                col += 1
            elif constraint_type == '=':
                tableau[i, col] = 1  # Variable artificial
                artificial_cols.append(col)
                col += 1
            tableau[i, -1] = b_norm[i]

        # Fase 1: Minimizar la suma de variables artificiales
        if artificial_vars > 0:
            tableau[-1, artificial_cols] = -1  # Función objetivo auxiliar
            while np.any(tableau[-1, :-1] < 0):
                pivot_col = np.argmin(tableau[-1, :-1])
                ratios = []
                for i in range(num_restr):
                    if tableau[i, pivot_col] > 0:
                        ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                    else:
                        ratios.append(np.inf)
                pivot_row = np.argmin(ratios)
                if ratios[pivot_row] == np.inf:
                    raise ValueError("El problema no tiene solución factible.")

                tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
                for i in range(num_restr + 1):
                    if i != pivot_row:
                        tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

            if abs(tableau[-1, -1]) > 1e-10:
                raise ValueError("El problema no tiene solución factible.")

        # Fase 2: Optimizar la función objetivo original
        tableau[-1, :] = 0
        tableau[-1, :num_vars] = -c if is_max else c
        tableau[-1, -1] = 0

        while np.any(tableau[-1, :-1] < 0):
            pivot_col = np.argmin(tableau[-1, :-1])
            ratios = []
            for i in range(num_restr):
                if tableau[i, pivot_col] > 0:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(np.inf)
            pivot_row = np.argmin(ratios)
            if ratios[pivot_row] == np.inf:
                raise ValueError("El problema es no acotado.")

            tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
            for i in range(num_restr + 1):
                if i != pivot_row:
                    tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        # Extraer solución
        solution = np.zeros(num_vars)
        for i in range(num_restr):
            col = np.where(tableau[i, :num_vars] == 1)[0]
            if len(col) == 1:
                solution[col[0]] = tableau[i, -1]

        z = tableau[-1, -1]
        return solution, z if is_max else -z

class SimplexAdvanced(SimplexBase):
    """
    Implementación avanzada con soporte para minimización y dual simplex.
    """
    def solve(self, c, A, b, constraints_types, is_max):
        # Usar SimplexIntermediate como base, pero agregar dual simplex para casos específicos
        num_vars = len(c)
        num_restr = len(b)

        # Validaciones
        if A.shape != (num_restr, num_vars):
            raise ValueError("Dimensiones de A no coinciden con el número de variables/restricciones.")
        if len(b) != num_restr or len(constraints_types) != num_restr:
            raise ValueError("Dimensiones de b o constraints_types no coinciden con el número de restricciones.")

        # Normalizar restricciones como en SimplexIntermediate
        A_norm = A.copy()
        b_norm = b.copy()
        slack_vars = 0
        surplus_vars = 0
        artificial_vars = 0
        var_types = []

        for i, constraint_type in enumerate(constraints_types):
            if constraint_type == '<=':
                slack_vars += 1
                var_types.append('slack')
            elif constraint_type == '>=':
                A_norm[i, :] = -A_norm[i, :]
                b_norm[i] = -b_norm[i]
                surplus_vars += 1
                var_types.append('surplus')
                artificial_vars += 1
                var_types.append('artificial')
            elif constraint_type == '=':
                artificial_vars += 1
                var_types.append('artificial')

        # Construir tableau inicial
        total_vars = num_vars + slack_vars + surplus_vars + artificial_vars
        tableau = np.zeros((num_restr + 1, total_vars + 1))
        col = num_vars
        artificial_cols = []

        for i, constraint_type in enumerate(constraints_types):
            tableau[i, :num_vars] = A_norm[i, :]
            if constraint_type == '<=':
                tableau[i, col] = 1
                col += 1
            elif constraint_type == '>=':
                tableau[i, col] = -1
                col += 1
                tableau[i, col] = 1
                artificial_cols.append(col)
                col += 1
            elif constraint_type == '=':
                tableau[i, col] = 1
                artificial_cols.append(col)
                col += 1
            tableau[i, -1] = b_norm[i]

        # Fase 1 (si hay variables artificiales)
        if artificial_vars > 0:
            tableau[-1, artificial_cols] = -1
            while np.any(tableau[-1, :-1] < 0):
                pivot_col = np.argmin(tableau[-1, :-1])
                ratios = []
                for i in range(num_restr):
                    if tableau[i, pivot_col] > 0:
                        ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                    else:
                        ratios.append(np.inf)
                pivot_row = np.argmin(ratios)
                if ratios[pivot_row] == np.inf:
                    raise ValueError("El problema no tiene solución factible.")

                tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
                for i in range(num_restr + 1):
                    if i != pivot_row:
                        tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

            if abs(tableau[-1, -1]) > 1e-10:
                raise ValueError("El problema no tiene solución factible.")

        # Fase 2 o dual simplex
        tableau[-1, :] = 0
        tableau[-1, :num_vars] = -c if is_max else c
        tableau[-1, -1] = 0

        # Usar dual simplex si b tiene valores negativos
        if np.any(tableau[:-1, -1] < 0):
            while np.any(tableau[:-1, -1] < 0):
                pivot_row = np.argmin(tableau[:-1, -1])
                ratios = []
                for j in range(total_vars):
                    if tableau[pivot_row, j] < 0:
                        ratios.append(tableau[-1, j] / tableau[pivot_row, j])
                    else:
                        ratios.append(np.inf)
                pivot_col = np.argmin(ratios)
                if ratios[pivot_col] == np.inf:
                    raise ValueError("El problema no tiene solución factible.")

                tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
                for i in range(num_restr + 1):
                    if i != pivot_row:
                        tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
        else:
            # Simplex primal si la solución es factible
            while np.any(tableau[-1, :-1] < 0):
                pivot_col = np.argmin(tableau[-1, :-1])
                ratios = []
                for i in range(num_restr):
                    if tableau[i, pivot_col] > 0:
                        ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                    else:
                        ratios.append(np.inf)
                pivot_row = np.argmin(ratios)
                if ratios[pivot_row] == np.inf:
                    raise ValueError("El problema es no acotado.")

                tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
                for i in range(num_restr + 1):
                    if i != pivot_row:
                        tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        # Extraer solución
        solution = np.zeros(num_vars)
        for i in range(num_restr):
            col = np.where(tableau[i, :num_vars] == 1)[0]
            if len(col) == 1:
                solution[col[0]] = tableau[i, -1]

        z = tableau[-1, -1]
        return solution, z if is_max else -z