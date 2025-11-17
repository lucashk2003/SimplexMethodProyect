import os
from flask import Flask, render_template, request
import numpy as np
from services.simplex_methods import SimplexSimple, SimplexAdvanced

# app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

TOLERANCE = 1e-9


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/solve", methods=["POST"])
def solve():
    try:
        # 1. Objetivo
        objective = request.form.get("objective")   # "max" o "min"
        es_maximizacion = True if objective == "max" else False

        # 2. Dimensiones
        num_vars = int(request.form.get("num_vars"))
        num_restr = int(request.form.get("num_restr"))

        # 3. Coeficientes de Z
        c = np.array([float(x) for x in request.form.get("c").replace(",", " ").split()])

        # 4. Matriz A
        filas_A = request.form.get("A").split(",")
        A = np.zeros((num_restr, num_vars), dtype=float)
        for i, fila in enumerate(filas_A):
            A[i] = np.array([float(x) for x in fila.replace(",", " ").split()])

        # 5. Tipos de restricción
        constraints_types = [x.strip() for x in request.form.get("constraints_types").split(",")]

        # 6. Vector b
        b = np.array([float(x) for x in request.form.get("b").replace(",", " ").split()])

        # Validaciones rápidas
        if len(c) != num_vars:
            raise ValueError("El vector c no coincide con la cantidad de variables.")
        if len(b) != num_restr:
            raise ValueError("El vector b no coincide con la cantidad de restricciones.")
        if len(constraints_types) != num_restr:
            raise ValueError("Debe especificar un tipo de restricción por cada fila de A.")

        # 7. Selección automática de método
        is_simple_form = (
            es_maximizacion
            and all(t == "<=" for t in constraints_types)
            and np.all(b >= -TOLERANCE)
        )

        if is_simple_form:
            simplex = SimplexSimple()
            method_used = "Simplex Simple"
        else:
            simplex = SimplexAdvanced()
            method_used = "Simplex Avanzado"

        # 8. Resolver
        solution, z_value = simplex.solve(c, A, b, constraints_types, es_maximizacion)

        # preparar salida para template
        solution_data = {f"x{i+1}": float(solution[i]) for i in range(len(solution))}

        return render_template(
            "result.html",
            success=True,
            z_value=float(z_value),
            solution_data=solution_data,
            objective="Maximización" if es_maximizacion else "Minimización",
            method_used=method_used
        )

    except Exception as e:
        return render_template(
            "result.html",
            success=False,
            error_message=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)
