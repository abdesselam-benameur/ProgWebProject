import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# interface pour charger, visualiser, traiter et analyser les données 
# et entrainer des modeles de machine learning avec ces données

file_uploader = st.file_uploader("Charger le dataset", type="csv")
if file_uploader is not None:
    # demander le séparateur si ce n'est pas ','
    seperator = st.text_input("Séparateur", value=";")
    if seperator != "":
        # afficher le dataset
        st.write("afficher le dataset")
        df = pd.read_csv(file_uploader, sep=seperator)
        st.write(df)

        options = [""] + df.columns.to_list()
        target = st.selectbox("Choisir la colonne target", options, index=0)

        # afficher le type des colonnes
        st.write("todo: afficher le type des colonnes")
        # todo: le type des colonnes
        
        # afficher les statistiques du dataset
        st.write("todo: afficher les statistiques du dataset")
        st.write(df.describe())

        # afficher la matrice de corrélation
        st.write("afficher la matrice de corrélation")
        st.write(df.corr())

        # afficher un graphe de la matrice de corrélation
        st.write("afficher un graphe de la matrice de corrélation")
        colormap = sns.color_palette("Greens")
        graph = sns.heatmap(df.corr(), annot=True, cmap=colormap)
        st.pyplot(graph.figure)

        if target != "":
            # afficher la colonne target
            st.write("afficher la colonne target")
            st.write(df[target])

            # afficher la distribution des classes
            st.write("afficher la distribution des classes")
            st.write(df[target].value_counts())

            # afficher un graphe de la distribution des classes
            st.write("afficher un graphe de la distribution des classes")
            st.bar_chart(df[target].value_counts())

            # afficher un graphe de la distribution des classes
            st.write("afficher un graphe de la distribution des classes")
            st.write(sns.countplot(x=target, data=df))

            # st.write("")
            # st.write(sns.countplot(x="class", hue="Age", data=df))