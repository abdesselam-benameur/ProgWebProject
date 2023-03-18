import streamlit as st

st.markdown("## Exploration de données et apprentissage de modèles supervisés")
st.write("""<p style="text-align: justify">
L’interface web represente un moyen de faire une analyse exploratoire et un apprentissage de modele supervisés ainsi que son evaluation avec streamlit 

L’interface est divisée en 3 parties Principales : """, unsafe_allow_html=True)

st.markdown("""
### 1. Prétraitement des données :  
""")
st.markdown("""<p style="text-align: justify">
Dans cette partie on veille à ce que les données soit propres et prête à l’utilisation en effectuant plusieurs prétraitements qui sont les suivant :  

 

- Détection du type de chaque variable (ie., qualitative, quantitative)  

- Détection du nombre de catégories en cas de variables qualitative 

- Vérification la présence possible d’<b>outliers</b> 

- Vérification des valeurs manquantes : faire un traitement par ligne en supprimant si celle qui a au moins une valeur manquante ou par colonne en supprimant la colonne ou en faisant la moyenne, mediane ou mode) 

- Faire la normalisation : un choix entre 2 méthodes normalisation standards ou normalisation minmax 

- Faire la dummification  : soit en utilisant onehotencodint ou  labelencoding 

- Enlever le désiquilibre des classes : soit utilisé random undersampling ou random oversampling</p>
 """, unsafe_allow_html=True)


st.markdown('### 2. Analyse exploratoire des données : ')

st.markdown("""<p style="text-align: justify">
Dans cette partie on fait une visualisation de différents graphiques qui montre une analyse unidimensionnelle et bidimensionnelle des différentes variables qualitatives et quantitatives. Parmi les graphes affichés on a : <b>Box-plot</b>, <b>diagramme de pareto</b>, <b>histogramme</b>, <b>heatmap</b>... 

Ensuite on a calculé avec différentes méthodes des métriques en fonction du type de variables pour faire des statistiques et l’analyse de données qui sont les suivantes : <b>Khi2</b>, <b>Fisher</b> ainsi que la <b>matrice de corrélation</b> et faire avec ça le features sélection. 
</p>""", unsafe_allow_html=True)

st.markdown('### 3. Entrainement du modèle:')
st.markdown("""<p style="text-align: justify">
Nous avons effectué l’entraînement modèle de machine Learning en ayant le choix entre 3 algorithmes <b>(KNN, LogisticRegression, DescisionTree)</b>. Pour cela on a fait :  

- La sélection des <b>hyperparamètres</b> en utilisant soit la fonction <b>gridSeashCV</b> ou <b>randomizedSearchCV</b> selon le choix de l’utilisateur 

- Traitement des 2 cas de classification binaire et multiclasses 

- Ensuite on a évalué le modèle en calculant différentes métriques et score tel que <b>accuracy</b>, <b>recall</b>, <b>F-score</b> ainsi que l’affichage de la <b>matrice de confusion</b> et de la <b>courbe ROC</b></p>
""", unsafe_allow_html=True)
# st.write("Loreum ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
