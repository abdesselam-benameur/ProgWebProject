import streamlit as st

st.title("Exploration de données et apprentissage de modèles supervisés")
st.write("""<p style="text-align: justify">
L’interface web représente un moyen de faire une analyse exploratoire et un apprentissage de modèles supervisés ainsi que son évaluation

L’interface est divisée en 3 parties principales : """, unsafe_allow_html=True)

st.markdown("""
### 1. Prétraitement des données :  
""")
st.markdown("""<p style="text-align: justify">
Dans cette partie on veille à ce que les données soient propres et prêtes à l’utilisation en effectuant plusieurs prétraitements qui sont les suivants :  

 

- Détection du type de chaque variable (ie., qualitative, quantitative)  

- Détection du nombre de catégories en cas de variables qualitatives

- Vérification de la présence possible d’<b>outliers</b> 

- Vérification des valeurs manquantes : faire un traitement par ligne en supprimant celles qui ont au moins une valeur manquante, ou par colonne en supprimant la colonne ou en remplaçant ses valeurs manquantes par la moyenne, médiane ou le mode) 

- Normalisation des données : un choix entre 2 méthodes (standard ou MinMax)

- Dummification des variables catégorielles : en utilisant One Hot Encodint ou  Label Encoding 

- Equilibrer des classes : en utilisant random undersampling, random oversampling, ADASYN ou SMOTE </p>
 """, unsafe_allow_html=True)


st.markdown('### 2. Analyse exploratoire des données : ')

st.markdown("""<p style="text-align: justify">
Dans cette partie on fait une visualisation de différents graphiques qui montrent une analyse unidimensionnelle et bidimensionnelle des différentes variables qualitatives et quantitatives. Parmi les graphes affichés, on a : <b>Box-plot</b>, <b>diagramme de Pareto</b>, <b>histogramme</b>, <b>heatmap</b>... 

Ensuite, on calcule différentes métriques en fonction des types des variables, ces métriques sont les suivantes : <b>Khi2</b>, <b>Fisher</b> ainsi que la <b>matrice de corrélation</b>.
</p>""", unsafe_allow_html=True)

st.markdown('### 3. Entraînement du modèle:')
st.markdown("""<p style="text-align: justify">
Nous avons effectué l’entraînement du modèle de machine Learning en ayant le choix entre 3 algorithmes <b>(KNN, LogisticRegression, DescisionTree)</b>. Pour cela on fait :  

- La division du dataset en train set et test set

- La sélection des <b>hyperparamètres</b> en utilisant soit la méthode <b>Grid Seash</b> ou <b>Randomized Search</b> 

- L'entraînement du modèle sur les données du train set

- Ensuite, on évalue le modèle en calculant différentes métriques telles que <b>accuracy</b>, <b>recall</b>, <b>f1-score</b> ainsi que la <b>matrice de confusion</b> et de la <b>courbe ROC</b> et <b>AUC</b></p> 
""", unsafe_allow_html=True)
