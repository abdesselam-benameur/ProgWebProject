import streamlit as st
import pandas as pd
import numpy as np
from fonctions_analyse_exploratoire import *

# un dataset aleatoire


# @st.cache
def load_data():
    # data = {'Nom': ['Alice', 'Bob', 'Charlie', 'David'],
    #         'Âge': [25, 30, 35, 40],
    #         'Genre': ['Femme', 'Homme', 'Homme', 'Homme'],
    #         'Salaire': [50500, 68000, 20000, 8000]}

    return pd.read_csv("./data/diabetes.csv")


df = load_data()

st.markdown("<center><h1>Analyse exploratoire</center>",
            unsafe_allow_html=True)
st.markdown("# 1.  Analyse unidimentionnelle :")

# st.markdown("### Mesures statistiques du dataset")
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<center><h3>Mesures statistiques du dataset</center>",
            unsafe_allow_html=True)

# un container pour les stats
with st.empty():
    st.table(statistics_for_numeric_variables(df))
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<center><h3>Histogramme</center>",
            unsafe_allow_html=True)
quantitatives_variables = list(df.select_dtypes(
    include=['int64', 'float64']).columns)
column1 = st.selectbox("Choisir une variable quantitative",
                       quantitatives_variables, key="histogramme")
# appeler la fonction qui affiche l'histogramme avec la variable choisie
histogramme(df[column1])
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<center><h3>Boxplot</center>", unsafe_allow_html=True)
# st.markdown("## Boxplot")
quantitatives_variables = list(df.select_dtypes(
    include=['int64', 'float64']).columns)
column2 = st.selectbox("Choisir une variable quantitative",
                       quantitatives_variables, key="boxplot")
# appeler la fonction qui affiche le boxplot avec la variable choisie
display_boxplot(df[column2])
st.markdown("<br>", unsafe_allow_html=True)


st.markdown("<center><h3>Diagramme circulaire</center>",
            unsafe_allow_html=True)
categorial_variables = list(df.select_dtypes(
    include=['object', 'category']).columns)
column1 = st.selectbox("Choisir une variable categorielle",
                       categorial_variables, key="camembert")
# appeler la fonction qui affiche le camembert avec la variable choisie
pie_plot_of_categories(df[column1])
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<center><h3>Diagramme Pareto</center>",
            unsafe_allow_html=True)
categorial_variables = list(df.select_dtypes(
    include=['object', 'category']).columns)
column2 = st.selectbox("Choisir une variable categorielle",
                       categorial_variables, key="pareto")
pareto_diagram(df[column2])
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("# 2. Analyse bidimentionnelle :")
st.markdown("<br>", unsafe_allow_html=True)

st.write("## 2.1. Quantitative vs Quantitative :")
st.markdown("<br>", unsafe_allow_html=True)


st.markdown("<center><h3>Heatmap</center>",
            unsafe_allow_html=True)
quantitatives_variables = list(df.select_dtypes(
    include=['int64', 'float64']).columns)
list_of_variables = st.multiselect("Choisir les variable quantitatives à afficher dans le heatmap",
                                   quantitatives_variables, key="heatmap2", default=quantitatives_variables)
heatmap_plot(df[list_of_variables])
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<center><h3>Scatter plot</center>",
            unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    quantitatives_variables = list(df.select_dtypes(
        include=['int64', 'float64']).columns)
    column21 = st.selectbox("Choisir une variable quantitative",
                            quantitatives_variables, key="scatter1")
with col2:
    quantitatives_variables = list(df.select_dtypes(
        include=['int64', 'float64']).columns)
    column22 = st.selectbox("Choisir une variable quantitative",
                            quantitatives_variables, key="scatter2")
scatter_plot(df[column21], df[column22])
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.write("## 2.2. Quantitative vs Categorielle :")
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<center><h3>Boxplot</center>",
            unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    quantitatives_variables = list(df.select_dtypes(
        include=['int64', 'float64']).columns)
    column11 = st.selectbox("Choisir une variable quantitative",
                            quantitatives_variables, key="boxplot1")
with col2:
    categorial_variables = list(df.select_dtypes(
        include=['object', 'category']).columns)
    column12 = st.selectbox("Choisir une variable categorielle",
                            categorial_variables, key="boxplot2")
box_plot_of_variable_according_to_categorical_variable(
    df[column11], df[column12], df)
st.markdown("<br>", unsafe_allow_html=True)


st.markdown("<center><h3>Mean & Std</center>",
            unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    quantitatives_variables = list(df.select_dtypes(
        include=['int64', 'float64']).columns)
    column21 = st.selectbox("Choisir une variable quantitative",
                            quantitatives_variables, key="barplot1")
with col2:
    categorial_variables = list(df.select_dtypes(
        include=['object', 'category']).columns)
    column22 = st.selectbox("Choisir une variable categorielle",
                            categorial_variables, key="barplot2")
st.table(mean_and_std_by_categorical_variable(
    column22, df[[column21, column22]]))


st.markdown("<br>", unsafe_allow_html=True)
st.markdown("## 2.3. Categorielle vs Categorielle :")
st.markdown("<br>", unsafe_allow_html=True)


st.markdown("<center><h3>Graphique en bâtons</center>",
            unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    categorial_variables = list(df.select_dtypes(
        include=['object', 'category']).columns)
    column11 = st.selectbox("Choisir une variable categorielle",
                            categorial_variables, key="diag1")
with col2:
    categorial_variables = list(df.select_dtypes(
        include=['object', 'category']).columns)
    column12 = st.selectbox("Choisir une variable categorielle",
                            categorial_variables, key="diag2")
fig, contingency_table = graphic_representation(df[column11], df[column12])
st.pyplot(fig)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<center><h3>Tableau contingence</center>",
            unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    categorial_variables = list(df.select_dtypes(
        include=['object', 'category']).columns)
    column21 = st.selectbox("Choisir une variable quantitative",
                            categorial_variables, key="cont1")
with col2:
    categorial_variables = list(df.select_dtypes(
        include=['object', 'category']).columns)
    column22 = st.selectbox("Choisir une variable categorielle",
                            categorial_variables, key="cont2")
st.table(contingency_table)
