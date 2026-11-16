"""
Perceptrón simple: ¿aprueba el estudiante la experiencia curricular?
Reglas del ejercicio:
  a) 3 promedios (3 unidades)
  b) aprueba si tiene por lo menos 2 unidades aprobadas
  c) no se calcula promedio final, solo se aplica la regla b
  d) cada promedio va de 0 a 20

Fórmulas de las diapositivas:
  Net  = x1*w1 + x2*w2 + x3*w3 + 1*wbias
  y    = 1 si Net >= 0, 0 si Net < 0         (hardlim)
  error = yd - y
  wi(t+1) = wi(t) + fa * error * xi          (regla delta)
  wbias(t+1) = wbias(t) + fa * error * 1

Entradas del perceptrón: ui = 1 si nota_i >= NOTA_MIN, 0 en otro caso.
"""
import itertools
import json
import numpy as np

NOTA_MIN = 11.0  # cambiar a 10.5 si en tu curso se aprueba con 10.5


# ---------- utilidades ----------
def binarizar(notas, nota_min=NOTA_MIN):
    """Convierte cada nota (0-20) en 1 (unidad aprobada) o 0 (desaprobada)."""
    return (np.asarray(notas, dtype=float) >= nota_min).astype(int)


def regla_real(notas, nota_min=NOTA_MIN):
    """Regla b del ejercicio, para verificar el perceptrón."""
    return int(np.sum(np.asarray(notas) >= nota_min) >= 2)


def hardlim(net):
    return 1 if net >= 0 else 0


# 8 patrones de entrenamiento: todas las combinaciones de 3 unidades
PATRONES = np.array(list(itertools.product([0, 1], repeat=3)))
YD = (PATRONES.sum(axis=1) >= 2).astype(int)


# ---------- entrenamiento ----------
def entrenar(X, yd, fa=1.0, w0=None, wb0=0.0, max_epocas=100):
    """
    Entrena con la regla delta, patrón por patrón.
    Devuelve: pesos, bias, traza (una fila por patrón) y errores por época.
    """
    X = np.asarray(X, dtype=float)
    w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float)
    wb = float(wb0)
    traza, errores_epoca = [], []

    for epoca in range(1, max_epocas + 1):
        errores = 0
        for k, (x, t) in enumerate(zip(X, yd), start=1):
            w_ini, wb_ini = w.copy(), wb
            net = float(np.dot(x, w) + 1.0 * wb)
            y = hardlim(net)
            error = int(t - y)
            if error != 0:
                w = w + fa * error * x
                wb = wb + fa * error * 1.0
                errores += 1
            traza.append({
                "época": epoca, "patrón": k,
                "x1": x[0], "x2": x[1], "x3": x[2], "yd": int(t),
                "w1": w_ini[0], "w2": w_ini[1], "w3": w_ini[2], "wbias": wb_ini,
                "Net": net, "y": y, "error": error,
                "w1 nuevo": w[0], "w2 nuevo": w[1], "w3 nuevo": w[2],
                "wbias nuevo": wb,
            })
        errores_epoca.append(errores)
        if errores == 0:
            break
    return w, wb, traza, errores_epoca


def predecir(notas, w, wb, nota_min=NOTA_MIN):
    """Notas (0-20) -> binarización -> Net -> hardlim."""
    u = binarizar(notas, nota_min)
    net = float(np.dot(u, w) + wb)
    return hardlim(net), net, u


def evaluar(w, wb, nota_min=NOTA_MIN, n=5000, semilla=1):
    """Compara el perceptrón con la regla b en n estudiantes aleatorios."""
    rng = np.random.default_rng(semilla)
    notas = np.round(rng.uniform(0, 20, size=(n, 3)), 1)
    ok = sum(predecir(x, w, wb, nota_min)[0] == regla_real(x, nota_min) for x in notas)
    return ok / n


# ---------- experimento comparativo: notas crudas (no separable) ----------
def entrenar_crudo(fa=0.05, epocas=200, n=2000, semilla=42, nota_min=NOTA_MIN):
    """
    Perceptrón con notas crudas /20. NO converge: con entradas continuas la
    regla "al menos 2 de 3" no es linealmente separable.
    Ej.: (20,20,0)->1 y (0,20,20)->1, pero su punto medio (10,20,10)->0.
    """
    rng = np.random.default_rng(semilla)
    notas = np.round(rng.uniform(0, 20, size=(n, 3)), 1)
    yd = np.array([regla_real(x, nota_min) for x in notas])
    w, wb, _, errores = entrenar(notas / 20.0, yd, fa=fa,
                                 w0=rng.uniform(-0.5, 0.5, 3),
                                 wb0=rng.uniform(-0.5, 0.5), max_epocas=epocas)
    acc = np.mean([hardlim(np.dot(x / 20.0, w) + wb) == t for x, t in zip(notas, yd)])
    return errores, float(acc)


def guardar_json(w, wb, errores, ruta="pesos_perceptron.json", nota_min=NOTA_MIN, fa=1.0):
    datos = {"w1": float(w[0]), "w2": float(w[1]), "w3": float(w[2]),
             "wbias": float(wb), "nota_minima": nota_min, "fa": fa,
             "epocas_hasta_converger": len(errores), "errores_por_epoca": errores}
    with open(ruta, "w") as f:
        json.dump(datos, f, indent=4)


if __name__ == "__main__":
    import pandas as pd
    FA = 1.0
    w, wb, traza, errores = entrenar(PATRONES, YD, fa=FA)
    df = pd.DataFrame(traza)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(df.to_string(index=False))
    print(f"\nErrores por época: {errores}")
    print(f"Convergió en la época {len(errores) - 1} (la {len(errores)} confirma 0 errores)")
    print(f"Pesos finales: w1={w[0]}, w2={w[1]}, w3={w[2]}, wbias={wb}")
    print(f"Exactitud vs regla b (5000 estudiantes): {evaluar(w, wb) * 100:.2f}%")
    for ejemplo in ([11, 11, 10], [11, 10, 20], [20, 20, 0], [10.9, 10.9, 20], [8, 15, 12]):
        y, net, u = predecir(ejemplo, w, wb)
        print(f"{ejemplo} -> u={u.tolist()} Net={net:+.1f} -> {'APRUEBA' if y else 'DESAPRUEBA'}")
    df.to_csv("tabla_entrenamiento.csv", index=False)
    guardar_json(w, wb, errores, fa=FA)
    err_c, acc_c = entrenar_crudo()
    print(f"\n[Experimento notas crudas] errores última época={err_c[-1]}, exactitud={acc_c * 100:.2f}% (nunca llega a 0)")
