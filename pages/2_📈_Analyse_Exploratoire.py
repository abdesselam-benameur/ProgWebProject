# from analyse_exploratoire import *
import streamlit as st
import pandas as pd
import numpy as np
from fonctions_analyse_exploratoire import *

st.title("Analyse exploratoire")

if "df" not in st.session_state:
    st.warning("Commencez d'abord par charger les données", icon="⚠️")
else:
    df = st.session_state.df

    # st.markdown("<center><h1>Analyse exploratoire</center>",
    #             unsafe_allow_html=True)
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
    quantitatives_variables = st.session_state.variable_quant
    column1 = st.selectbox("Choisir une variable quantitative",
                        quantitatives_variables, key="histogramme")
    # appeler la fonction qui affiche l'histogramme avec la variable choisie
    histogramme(df[column1])
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<center><h3>Boxplot</center>", unsafe_allow_html=True)
    # st.markdown("## Boxplot")
    column2 = st.selectbox("Choisir une variable quantitative",
                        quantitatives_variables, key="boxplot")
    # appeler la fonction qui affiche le boxplot avec la variable choisie
    display_boxplot(df[column2])
    st.markdown("<br>", unsafe_allow_html=True)


    
    if st.session_state.variable_categ:
        categorial_variables = st.session_state.variable_categ
        st.markdown("<center><h3>Diagramme circulaire</center>",
                unsafe_allow_html=True)
        column1 = st.selectbox("Choisir une variable categorielle", categorial_variables, key="camembert")
        # appeler la fonction qui affiche le camembert avec la variable choisie
        pie_plot_of_categories(df[column1])
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<center><h3>Diagramme Pareto</center>",
                    unsafe_allow_html=True)
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
    
    list_of_variables = st.multiselect("Choisir les variable quantitatives à afficher dans le heatmap",
                                    quantitatives_variables, key="heatmap2", default=quantitatives_variables)
    if list_of_variables:
        heatmap_plot(df[list_of_variables])
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<center><h3>Scatter plot</center>",
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        column21 = st.selectbox("Choisir une variable quantitative",
                                quantitatives_variables, key="scatter1")
    with col2:
        column22 = st.selectbox("Choisir une variable quantitative",
                                quantitatives_variables, key="scatter2")
    scatter_plot(df[column21], df[column22])
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.variable_categ:
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("## 2.2. Quantitative vs Categorielle :")
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<center><h3>Boxplot</center>",
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            column11 = st.selectbox("Choisir une variable quantitative",
                                    quantitatives_variables, key="boxplot1")
        with col2:
            column12 = st.selectbox("Choisir une variable categorielle",
                                    categorial_variables, key="boxplot2")
        st.table(mean_and_std_by_categorical_variable(
            column12, df[[column11, column12]]).T)
        
        box_plot_of_variable_according_to_categorical_variable(
            df[column11], df[column12], df)
        


        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 2.3. Categorielle vs Categorielle :")
        st.markdown("<br>", unsafe_allow_html=True)


        st.markdown("<center><h3>Graphique en bâtons</center>",
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            column11 = st.selectbox("Choisir une variable categorielle",
                                    categorial_variables, key="diag1")
        with col2:
            column12 = st.selectbox("Choisir une variable categorielle",
                                    categorial_variables, key="diag2")
        fig, contingency_table = graphic_representation(df[column11], df[column12])
        st.pyplot(fig)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<center><h3>Tableau contingence</center>",
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            column21 = st.selectbox("Choisir une variable quantitative",
                                    categorial_variables, key="cont1")
        with col2:
            column22 = st.selectbox("Choisir une variable categorielle",
                                    categorial_variables, key="cont2")
        st.table(contingency_table)
