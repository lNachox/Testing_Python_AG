import numpy as np
import matplotlib.pyplot as plt

def triangular_superior(A: np.ndarray) -> bool:
    # Verifica si A es cuadrada y si es triangular superior (estrictamente bajo
    # la diagonal principal todos ceros). Imprime los mensajes y retorna True/False.
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        print("La entrada no es una matriz 2D de NumPy.")
        return False

    n_filas, n_cols = A.shape
    if n_filas == n_cols:
        print(f"La matriz es cuadrada de dimensión {n_filas}×{n_cols}.")
    else:
        print(f"La matriz no es cuadrada: dimensión {n_filas}×{n_cols}.")
        return False

    es_tri_sup = np.allclose(np.tril(A, k=-1), 0)
    if es_tri_sup:
        print("La matriz es triangular superior.")
    else:
        print("La matriz NO es triangular superior.")
    return es_tri_sup


if __name__ == "__main__":
    # 1) Sobre Numpy — pruebas con 2×2, 3×3 y 4×4
    print("\n== 2x2 ==")
    A2 = np.array([[1, 2],
                   [0, 3]])
    triangular_superior(A2)

    B2 = np.array([[1, 0],
                   [5, 3]])
    triangular_superior(B2)

    print("\n== 3x3 ==")
    A3 = np.array([[2, -1,  0.5],
                   [0,  3,  4.0],
                   [0,  0,  1.0]])
    triangular_superior(A3)

    B3 = np.array([[2, -1,  0.5],
                   [5,  3,  4.0],
                   [0,  0,  1.0]])
    triangular_superior(B3)

    print("\n== 4x4 ==")
    A4 = np.array([[1, 2, 3,  4],
                   [0, 5, 6,  7],
                   [0, 0, 8,  9],
                   [0, 0, 0, 10]])
    triangular_superior(A4)

    B4 = np.array([[1, 2, 3,  4],
                   [0, 5, 6,  7],
                   [0, 1, 8,  9],  # bajo la diagonal no es cero
                   [0, 0, 0, 10]])
    triangular_superior(B4)

    # 2) Sobre Ploteo — subplots 4×1
    a = np.linspace(0.1, 1.0, 100)  # 100 puntos entre 0.1 y 1

    fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
    colores = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    estilos = ["-", "--", "-.", ":"]

    for i in range(1, 5):
        ax = axes[i - 1]
        if i % 2 == 0:
            y = a ** i
            titulo = f"y = x^{i}"
        else:
            y = a ** (-i)
            titulo = f"y = x^(-{i})"

        ax.plot(a, y, color=colores[i - 1], linestyle=estilos[i - 1], linewidth=2)
        ax.set_title(titulo)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()