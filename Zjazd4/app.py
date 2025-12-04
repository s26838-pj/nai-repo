import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from analysis import run_single_scenario, results_all

"""
Konfiguracja strony
"""
st.set_page_config(page_title="SVM vs Drzewa – Analiza", layout="wide")
st.title("Drzewa decyzyjne vs SVM")
st.markdown("### Porównanie klasyfikatorów na zbiorach: **Wine Quality** i **Pima Indians Diabetes**")
st.caption("Wybierz ziarno losowości → zobacz wyniki i wykresy z tego konkretnego podziału")

"""
ŹRÓDŁA DANYCH 
"""
st.markdown("---")
st.subheader("Źródła danych")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Wine Quality Dataset** 
    [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)  
    Link bezpośredni: [winequality-red.csv](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)
    """)

with col2:
    st.markdown("""
    **Pima Indians Diabetes Dataset**  
    [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/34/diabetes)    
    Link bezpośredni: [pima-indians-diabetes.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv)
    """)

"""
Sidebar
"""
st.sidebar.header("Ustawienia analizy")
random_state = st.sidebar.number_input("Ziarno losowości", min_value=0, max_value=99999, value=42, step=1)
run_button = st.sidebar.button("Uruchom analizę", type="primary", width="stretch")

if run_button:
    results_all.clear()
    os.makedirs("plots", exist_ok=True)

    with st.spinner("Trenuję modele i generuję wykresy..."):
        run_single_scenario(1, random_state)

    df = pd.DataFrame(results_all)

    """
    TABELA Z WYNIKAMI
    """
    st.subheader(f"Wyniki ziarna = {random_state}")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Wine Quality**")
        wine_df = df[df["Zbiór"].str.contains("Wine")][["Model", "Accuracy", "F1-weighted"]]
        st.dataframe(
            wine_df.style.highlight_max(axis=0, color='#90EE90'),
            width="stretch"
        )
        
    with col2:
        st.write("**Pima Indians Diabetes**")
        diab_df = df[df["Zbiór"].str.contains("Diabetes")][["Model", "Accuracy", "F1-weighted"]]
        st.dataframe(
            diab_df.style.highlight_max(axis=0, color='#90EE90'),
            width="stretch"
        )
        
    st.caption("Zielone pola = najlepszy wynik w danej kolumnie")
    """
    WYKRES SŁUPKOWY
    """
    st.subheader("Porównanie dokładności modeli")
    chart_df = df.pivot(index="Model", columns="Zbiór", values="Accuracy")
    st.bar_chart(chart_df, height=400, width="stretch")

    """
    WYKRESY Z FOLDERU
    """
    st.subheader("Wizualizacja danych z tego podziału")
    plot_files = sorted([f for f in os.listdir("plots") if f.endswith(".png") and f"rs{random_state}" in f])

    if len(plot_files) == 2:
        col1, col2 = st.columns(2)
        with col1:
            st.image(f"plots/{plot_files[0]}", caption=plot_files[0].replace("_", " ").replace(".png", ""), width="stretch")
        with col2:
            st.image(f"plots/{plot_files[1]}", caption=plot_files[1].replace("_", " ").replace(".png", ""), width="stretch")
    else:
        st.info("Wykresy są generowane... odśwież za chwilę")

    """
    WYNIKI
    """
    best_model = df.loc[df["Accuracy"].idxmax(), "Model"]
    st.success(f"Najlepszy model w tym uruchomieniu: **{best_model}**")





    