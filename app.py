import numpy as np
import pandas as pd
import streamlit as st

from perceptron import (PATRONES, YD, binarizar, entrenar, entrenar_crudo,
                        evaluar, predecir, regla_real)

st.set_page_config(page_title="Perceptrón - Aprobación", page_icon="🧠", layout="wide")
st.title("🧠 Perceptrón simple: ¿aprueba la experiencia curricular?")
st.markdown(
    "Regla del ejercicio: el estudiante aprueba si tiene **por lo menos 2 de 3 unidades aprobadas**. "
    "No se calcula promedio final. Cada promedio va de **0 a 20**."
)

# ---------------- barra lateral ----------------
st.sidebar.header("⚙️ Parámetros")
nota_min = st.sidebar.number_input("Nota mínima aprobatoria por unidad", 10.0, 14.0, 11.0, 0.5)
fa = st.sidebar.slider("Factor de aprendizaje (fa)", 0.1, 1.0, 1.0, 0.1)
w_init = st.sidebar.number_input("Peso inicial (w1=w2=w3=wbias)", -1.0, 1.0, 0.0, 0.1)

# El perceptrón se entrena con los 8 patrones binarios (0 = unidad desaprobada, 1 = aprobada)
w, wb, traza, errores = entrenar(PATRONES, YD, fa=fa, w0=[w_init] * 3, wb0=w_init)
convergio = errores[-1] == 0

tab1, tab2, tab3, tab4 = st.tabs(
    ["📝 Evaluar estudiante", "📋 Entrenamiento paso a paso", "📈 Convergencia y pesos", "🔬 ¿Por qué binarizar?"]
)

# ---------------- TAB 1: evaluar ----------------
with tab1:
    st.subheader("Ingrese los promedios de las 3 unidades (0 - 20)")
    c1, c2, c3 = st.columns(3)
    u1 = c1.number_input("Unidad 1 (x1)", 0.0, 20.0, 14.0, 0.5)
    u2 = c2.number_input("Unidad 2 (x2)", 0.0, 20.0, 12.0, 0.5)
    u3 = c3.number_input("Unidad 3 (x3)", 0.0, 20.0, 8.0, 0.5)

    if st.button("Evaluar estudiante", type="primary"):
        notas = [u1, u2, u3]
        y, net, u = predecir(notas, w, wb, nota_min)
        st.markdown("**1) Binarización** (unidad aprobada = 1 si nota ≥ %.1f)" % nota_min)
        st.table(pd.DataFrame([notas, u.tolist()], index=["Nota", "Entrada u"],
                              columns=["Unidad 1", "Unidad 2", "Unidad 3"]))
        st.markdown(
            f"**2) Net** = {u[0]}·{w[0]:g} + {u[1]}·{w[1]:g} + {u[2]}·{w[2]:g} + 1·({wb:g}) = **{net:g}**"
        )
        st.markdown(f"**3) Activación:** f({net:g}) = **{y}**")
        if y == 1:
            st.success("✅ APRUEBA la experiencia curricular (Y = 1)")
        else:
            st.error("❌ DESAPRUEBA la experiencia curricular (Y = 0)")
        st.caption(f"Unidades aprobadas: {int(u.sum())} de 3 · Regla del ejercicio: "
                   f"{'aprueba' if regla_real(notas, nota_min) else 'desaprueba'}")

# ---------------- TAB 2: traza ----------------
with tab2:
    st.subheader("Cálculo patrón por patrón (como en las diapositivas)")
    st.markdown(
        "`Net = x1·w1 + x2·w2 + x3·w3 + 1·wbias` → `y = 1 si Net ≥ 0, si no 0` → "
        "`error = yd − y` → `wi(t+1) = wi(t) + fa·error·xi`"
    )
    df = pd.DataFrame(traza)
    epoca_ver = st.selectbox("Época", sorted(df["época"].unique()))
    st.dataframe(df[df["época"] == epoca_ver].reset_index(drop=True))
    st.download_button("⬇️ Descargar tabla completa (CSV)", df.to_csv(index=False).encode(),
                       "tabla_entrenamiento.csv", "text/csv")

# ---------------- TAB 3: convergencia ----------------
with tab3:
    st.subheader("Resultado del entrenamiento")
    if convergio:
        st.success(f"Convergió: la época {len(errores)} tuvo 0 errores "
                   f"(los pesos quedaron fijos desde la época {len(errores) - 1}).")
    else:
        st.warning("No convergió con estos parámetros.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("w1", f"{w[0]:g}")
    m2.metric("w2", f"{w[1]:g}")
    m3.metric("w3", f"{w[2]:g}")
    m4.metric("wbias", f"{wb:g}")
    st.line_chart(pd.DataFrame({"Errores por época": errores}, index=range(1, len(errores) + 1)))
    acc = evaluar(w, wb, nota_min)
    st.metric("Exactitud en 5000 estudiantes aleatorios (notas 0–20)", f"{acc * 100:.2f}%")
    st.markdown(
        "Interpretación: con pesos (1, 1, 1) y bias −2, `Net = u1 + u2 + u3 − 2`. "
        "Net ≥ 0 solo cuando hay **2 o más** unidades aprobadas."
    )

# ---------------- TAB 4: experimento ----------------
with tab4:
    st.subheader("Experimento: perceptrón con notas crudas (sin binarizar)")
    st.markdown(
        "Con notas continuas de 0 a 20, la regla *al menos 2 de 3* **no es linealmente separable**. "
        "Ejemplo: (20, 20, 0) aprueba y (0, 20, 20) aprueba, pero su punto medio "
        "(10, 20, 10) desaprueba, así que ningún plano puede separarlos."
    )
    if st.button("Entrenar con notas crudas"):
        err_c, acc_c = entrenar_crudo(nota_min=nota_min)
        st.line_chart(pd.DataFrame({"Errores por época (notas crudas)": err_c}))
        st.error(f"Nunca llega a 0 errores. Exactitud final: {acc_c * 100:.2f}%")
        st.info("Por eso se binariza cada unidad (nota ≥ umbral → 1) antes de entrenar: "
                "así el problema sí es linealmente separable y el perceptrón converge al 100%.")
