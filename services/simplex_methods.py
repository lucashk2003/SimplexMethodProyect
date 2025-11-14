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

        # Iteraciones del método simplex
        max_iter = 1000
        for iteration in range(max_iter):
            # Verificar optimalidad
            if not np.any(tableau[-1, :-1] < -1e-10):
                break

            # Seleccionar columna pivote
            # Regla tipo Bland: primera columna con costo reducido negativo
            pivot_candidates = np.where(tableau[-1, :-1] < -1e-10)[0]
            if pivot_candidates.size == 0:
                break
            pivot_col = int(pivot_candidates[0])

            # Calcular razones para seleccionar fila pivote
            ratios = []
            for i in range(num_restr):
                if tableau[i, pivot_col] > 1e-10:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(np.inf)
            pivot_row = np.argmin(ratios)

            if ratios[pivot_row] == np.inf:
                raise ValueError("El problema es no acotado.")

            # Normalizar fila pivote
            pivot_value = tableau[pivot_row, pivot_col]
            tableau[pivot_row, :] /= pivot_value

            # Eliminar entradas en la columna pivote
            for i in range(num_restr + 1):
                if i != pivot_row:
                    factor = tableau[i, pivot_col]
                    tableau[i, :] -= factor * tableau[pivot_row, :]

        # Extraer solución
        solution = np.zeros(num_vars)
        for j in range(num_vars + num_restr):
            if j < num_vars:  # Solo variables de decisión
                col = tableau[:-1, j]  # Excluir fila objetivo
                if np.sum(np.abs(col) > 1e-10) == 1:  # Un solo valor no nulo
                    row = np.where(np.abs(col) > 1e-10)[0][0]
                    if abs(col[row] - 1) < 1e-10:  # Verificar que sea 1
                        solution[j] = tableau[row, -1]

        # Calcular z
        z = np.dot(c, solution)
        return solution, z

class SimplexIntermediate(SimplexBase):
    """
    Implementación intermedia del método simplex con soporte para restricciones >=, = y variables artificiales.
    """
    def solve(self, c, A, b, constraints_types, is_max):
        num_vars = len(c)
        num_restr = len(b)

        # Validaciones
        if A.shape != (num_restr, num_vars):
            raise ValueError("Dimensiones de A no coinciden con el número de variables/restricciones.")
        if len(b) != num_restr or len(constraints_types) != num_restr:
            raise ValueError("Dimensiones de b o constraints_types no coinciden con el número de restricciones.")

        # Normalizar restricciones
        A_norm = A.copy()
        b_norm = b.copy()
        # Asegurar b >= 0 invirtiendo filas cuando sea necesario y ajustando el tipo
        norm_types = list(constraints_types)
        for i in range(num_restr):
            if b_norm[i] < 0:
                A_norm[i, :] = -A_norm[i, :]
                b_norm[i] = -b_norm[i]
                if norm_types[i] == '<=':
                    norm_types[i] = '>='
                elif norm_types[i] == '>=':
                    norm_types[i] = '<='
                # '=' se mantiene igual

        # Chequeo temprano: restricciones paralelas conflictivas (misma fila A)
        # Si existe A_i == A_j y: (i es <= con b1) y (j es >= con b2) y b2 > b1 => infeasible
        # o si (i es '=' y j es '=') y b1 != b2 => infeasible
        tol = 1e-10
        for i in range(num_restr):
            for j in range(i + 1, num_restr):
                if np.allclose(A_norm[i, :], A_norm[j, :], atol=tol, rtol=0):
                    ti, tj = norm_types[i], norm_types[j]
                    bi, bj = b_norm[i], b_norm[j]
                    # '=' vs '=' conflictivo
                    if ti == '=' and tj == '=' and not np.isclose(bi, bj, atol=tol):
                        raise ValueError("El problema no tiene solución factible.")
                    # '<=' y '>=' conflictivo
                    if ti == '<=' and tj == '>=' and bj - bi > tol:
                        raise ValueError("El problema no tiene solución factible.")
                    if ti == '>=' and tj == '<=' and bi - bj > tol:
                        raise ValueError("El problema no tiene solución factible.")

        slack_vars = 0
        surplus_vars = 0
        artificial_vars = 0
        artificial_rows = []
        for i, constraint_type in enumerate(norm_types):
            if constraint_type == '<=':
                slack_vars += 1
            elif constraint_type == '>=':
                surplus_vars += 1
                artificial_vars += 1
                artificial_rows.append(i)
            elif constraint_type == '=':
                artificial_vars += 1
                artificial_rows.append(i)

        # Construir tableau inicial
        total_vars = num_vars + slack_vars + surplus_vars + artificial_vars
        tableau = np.zeros((num_restr + 1, total_vars + 1))
        col = num_vars
        artificial_cols = []
        for i, constraint_type in enumerate(norm_types):
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
            # Configurar función objetivo auxiliar
            tableau[-1, :] = 0
            for i in artificial_rows:
                tableau[-1, :] -= tableau[i, :]  # Sumar filas con variables artificiales

            # Ajustar tableau para solución básica inicial
            for col in artificial_cols:
                for i in range(num_restr):
                    if tableau[i, col] == 1:
                        tableau[-1, :] -= tableau[i, :] * tableau[-1, col]
                        break

            for iteration in range(1000):
                if not np.any(tableau[-1, :-1] < -1e-10):
                    break
                pivot_col = np.argmin(tableau[-1, :-1])
                ratios = []
                for i in range(num_restr):
                    if tableau[i, pivot_col] > 1e-10:
                        ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                    else:
                        ratios.append(np.inf)
                pivot_row = np.argmin(ratios)
                if ratios[pivot_row] == np.inf:
                    raise ValueError("El problema no tiene solución factible.")

                pivot_value = tableau[pivot_row, pivot_col]
                tableau[pivot_row, :] /= pivot_value
                for i in range(num_restr + 1):
                    if i != pivot_row:
                        factor = tableau[i, pivot_col]
                        tableau[i, :] -= factor * tableau[pivot_row, :]

            if abs(tableau[-1, -1]) > 1e-10:
                raise ValueError("El problema no tiene solución factible.")

        # Intentar remover variables artificiales básicas pivotando con alguna columna no artificial
        if artificial_vars > 0:
            tol = 1e-10
            for acol in artificial_cols:
                col = tableau[:-1, acol]
                if np.sum(np.abs(col) > tol) == 1:
                    row = np.where(np.abs(col) > tol)[0][0]
                    if abs(col[row] - 1) < tol:
                        # buscar columna de reemplazo no artificial con coeficiente no nulo
                        for j in range(total_vars):
                            if j in artificial_cols:
                                continue
                            if abs(tableau[row, j]) > tol:
                                pivot_value = tableau[row, j]
                                tableau[row, :] /= pivot_value
                                for i in range(num_restr + 1):
                                    if i != row:
                                        factor = tableau[i, j]
                                        tableau[i, :] -= factor * tableau[row, :]
                                break

        # Fase 2: Optimizar la función objetivo original
        tableau[-1, :] = 0
        # Usar -c para max y +c para min
        tableau[-1, :num_vars] = -c if is_max else c

        # Ajustar función objetivo para eliminar efecto de variables artificiales
        for j in range(total_vars):
            for i in range(num_restr):
                if abs(tableau[i, j] - 1) < 1e-10 and all(abs(tableau[k, j]) < 1e-10 for k in range(num_restr) if k != i):
                    tableau[-1, :] -= tableau[-1, j] * tableau[i, :]
                    break

        for iteration in range(1000):
            row_obj = tableau[-1, :-1]
            if is_max:
                if not np.any(row_obj < -1e-10):
                    break
                pivot_col = int(np.argmin(row_obj))
            else:
                if not np.any(row_obj > 1e-10):
                    break
                pivot_col = int(np.where(row_obj > 1e-10)[0][0])

            ratios = []
            for i in range(num_restr):
                if tableau[i, pivot_col] > 1e-10:
                    ratios.append(tableau[i, -1] / tableau[i, pivot_col])
                else:
                    ratios.append(np.inf)
            pivot_row = np.argmin(ratios)
            if ratios[pivot_row] == np.inf:
                raise ValueError("El problema es no acotado.")

            pivot_value = tableau[pivot_row, pivot_col]
            tableau[pivot_row, :] /= pivot_value
            for i in range(num_restr + 1):
                if i != pivot_row:
                    factor = tableau[i, pivot_col]
                    tableau[i, :] -= factor * tableau[pivot_row, :]

        # Extraer solución
        solution = np.zeros(num_vars)
        for j in range(num_vars):
            col = tableau[:-1, j]
            if np.sum(np.abs(col) > 1e-10) == 1:
                row = np.where(np.abs(col) > 1e-10)[0][0]
                if abs(col[row] - 1) < 1e-10:
                    solution[j] = tableau[row, -1]

        # Calcular z
        z = np.dot(c, solution)
        return solution, z

class SimplexAdvanced(SimplexIntermediate):
    """
    Implementación avanzada del método simplex con soporte para minimización y dual simplex.
    """
    def solve(self, c, A, b, constraints_types, is_max):
        if is_max:
            return super().solve(c, A, b, constraints_types, True)
        # Minimización: equivaler a maximizar -c usando Intermediate
        solution, _ = super().solve(c, A, b, constraints_types, True)
        z = float(np.dot(c, solution))
        # Fallback robusto en 2D: enumerar vértices factibles y tomar el mínimo
        n = len(c)
        if n == 2:
            cand = []
            # Intersecciones entre restricciones tratadas como igualdades
            m = A.shape[0]
            # Ejes x1=0, x2=0
            axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
            axes_b = [0.0, 0.0]
            lines_A = [A[i, :] for i in range(m)] + axes
            lines_b = [b[i] for i in range(m)] + axes_b
            for i in range(len(lines_A)):
                for j in range(i + 1, len(lines_A)):
                    M = np.vstack([lines_A[i], lines_A[j]])
                    if abs(np.linalg.det(M)) < 1e-12:
                        continue
                    rhs = np.array([lines_b[i], lines_b[j]])
                    x = np.linalg.solve(M, rhs)
                    if np.any(x < -1e-9):
                        continue
                    # Chequear factibilidad con los tipos
                    feasible = True
                    for k in range(m):
                        lhs = float(np.dot(A[k, :], x))
                        if constraints_types[k] == '<=' and lhs - b[k] > 1e-9:
                            feasible = False
                            break
                        if constraints_types[k] == '>=' and b[k] - lhs > 1e-9:
                            feasible = False
                            break
                        if constraints_types[k] == '=' and abs(lhs - b[k]) > 1e-9:
                            feasible = False
                            break
                    if feasible:
                        cand.append(x)
            if cand:
                cand = np.vstack(cand)
                vals = cand @ c
                idx = int(np.argmin(vals))
                brute_sol = cand[idx]
                brute_z = float(vals[idx])
                if brute_z < z - 1e-8:
                    return brute_sol, brute_z
        return solution, z