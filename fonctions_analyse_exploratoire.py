# !pip install pingouin
from functions import *
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
# import pingouin as pg
import scipy.stats as stats
import numpy as np
from sklearn.model_selection import train_test_split

# analyse unidimentionelle quantitative


def statistics_for_numeric_variables(data):
    """
        Params: a dataframe
        Return : generating a set of descriptive statistics for a DataFrame or a Series
    """

    return data.describe()

# def display_boxplot(col):
#     sns.boxplot(data=col).set(xlabel=col.name)
#     mean = col.mean()
#     plt.axhline(y=mean, color='r', linestyle='--', label='Mean')
#     plt.legend()
#     plt.show()


def display_boxplot(col):
    """
    The function displays the boxplot of a quantitative variable
    params : numeric variable of a dataframe
    """

    fig, ax = plt.subplots()
    sns.boxplot(data=col, ax=ax)
    mean = col.mean()
    median = col.median()
    ax.axhline(y=mean, color='r', linestyle='--', label='Moyenne')
    ax.axhline(y=median, color='g', linestyle='--', label='Mediane')
    ax.set(xlabel=col.name)
    ax.legend()
    st.pyplot(fig)


# def histogramme(numeric_column):
#     plt.hist(numeric_column, bins=10)
#     name = numeric_column
#     plt.title(f'Distribution de la variable {(name)}')
#     plt.xlabel(numeric_column)
#     plt.ylabel('Fréquence')
#     plt.show()

def histogramme(numeric_column):
    """
  La fonction affiche l'histogramme d'une variable numerique avec KDE plot
  params : numeric variable of a dataframe
    """
    fig, ax = plt.subplots()
    sns.histplot(numeric_column, ax=ax)
    # ax.hist(numeric_column)
    name = numeric_column.name
    plt.title(f'Histogramme de la variable {name}')
    plt.xlabel(str(name))
    # plt.ylabel('Fréquence')
    st.pyplot(fig)

# analyse unidimensionnelle qualitative:

# Identifier les catégories de la variable qualitative


def identify_categories(column):
    return column.unique()


# Calculer et representation graphique des pourcentages de chaque variable catégorielle

# def pie_plot_of_categories(column):
#     result = column.value_counts().apply(
#         lambda x: x*column.shape[0]/100).to_dict()
#     return plt.pie(result.values(), labels=result.keys(), autopct='%1.1f%%', shadow=True)

def pie_plot_of_categories(column):
    """
  The function returns a pie plot representing the percentage of each category of a qualitative variable
  params : column of categorical variable
  return : pie plot
    """
    result = column.value_counts().apply(
        lambda x: x*column.shape[0]/100).to_dict()
    fig, ax = plt.subplots()
    ax.pie(result.values(), labels=result.keys(),
           autopct='%1.1f%%', shadow=True)
    ax.set_title(f"Répartition de la colonne '{column.name}'")
    st.pyplot(fig)


# def frequency_table():
#     return df["color"].value_counts().to_frame()


# def pareto_diagram(categorical_column):
#     freq = categorical_column.value_counts()
#     freq = freq.sort_values(ascending=False)
#     cumfreq = freq.cumsum()
#     totalfreq = sum(freq)
#     percentcumfreq = (cumfreq/totalfreq)*100

#     fig, ax1 = plt.subplots()
#     ax1.bar(freq.index, freq.values, color='b')
#     ax1.set_xlabel('Modalité')
#     ax1.set_ylabel('Fréquence', color='b')
#     ax1.tick_params(axis='y', labelcolor='b')
#     # Création de la ligne de fréquence cumulée
#     ax2 = ax1.twinx()
#     ax2.plot(freq.index, percentcumfreq, color='r', marker='o')
#     ax2.set_ylabel('Pourcentage de fréquence cumulée', color='r')
#     ax2.tick_params(axis='y', labelcolor='r')

#     plt.title('Diagramme de Pareto')
#     plt.show()

def pareto_diagram(categorical_column):
    """
    The function calculates the frequencies of a categorical variable, sorts the modalities in decreasing order of frequency,
   then calculation of the cumulative frequencies and the percentages of cumulative frequencies to after display a graph
   representing the cumulative frequencies
   params : column of categorical variable
    """
    freq = categorical_column.value_counts()
    freq = freq.sort_values(ascending=False)
    cumfreq = freq.cumsum()
    totalfreq = sum(freq)
    percentcumfreq = (cumfreq/totalfreq)*100

    fig, ax1 = plt.subplots()
    ax1.bar(freq.index, freq.values, color='b')
    ax1.set_xlabel('Modalité')
    ax1.set_ylabel('Fréquence', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    # Création de la ligne de fréquence cumulée
    ax2 = ax1.twinx()
    ax2.plot(freq.index, percentcumfreq, color='r', marker='o')
    ax2.set_ylabel('Pourcentage de fréquence cumulée', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Diagramme de Pareto')
    st.pyplot(fig)

    # analyse bidimensionnelle
    # qualitative * qualitative


# def heatmap_plot(data):
#     colormap = sns.color_palette("Greens")
#     graph = sns.heatmap(data.corr(), annot=True, cmap=colormap)
#     plt.show(graph.figure)

# def heatmap_plot(data):
#     """
#   The function displays the heatmap between 2 or more quatitative variables which represents the level of collinearity between the variables
#   params : represents the dataframe with the quantitative variables
#     """
#     colormap = sns.color_palette("Greens")
#     fig, ax = plt.subplots()
#     graph = sns.heatmap(data.corr(), annot=True, cmap=colormap, ax=ax)
#     st.pyplot(fig)
def heatmap_plot(data):
    """
  The function displays the heatmap between 2 or more quantitative variables which represents the level of collinearity between the variables
  params : represents the dataframe with the quantitative variables
    """
    colormap = sns.color_palette("Greens")
    fig, ax = plt.subplots()
    corr_matrix = data.corr()
    graph = sns.heatmap(corr_matrix, annot=True, cmap=colormap, ax=ax)
    graph.set_xticklabels(graph.get_xticklabels(),
                          rotation=45, horizontalalignment='right')
    graph.set_yticklabels(graph.get_yticklabels(),
                          rotation=0, horizontalalignment='right')
    plt.title("Heatmap of Correlation Matrix")
    st.pyplot(fig)


# def scatter_pot(x, y):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     plt.plot(x, y, "ob")  # ob = type de points "o" ronds, "b" bleus
#     plt.title("Titre du graphique")
#     plt.show()

def scatter_plot(x, y):
    """
  The function displays a scatter plot of 2 quantitative variables that graphically shows whether there is a correlation between them or not
  params : quantitative variables representing columns of the dataset
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, "ob")  # ob = type de points "o" ronds, "b" bleus
    ax.set_title(f"Scatter plot de {x.name} et {y.name}")
    ax.set_xlabel(f"{x.name}")
    ax.set_ylabel(f"{y.name}")
    st.pyplot(fig)

  # quantitative * quantitative

# permet de faire la matrice de correlation
# La matrice de corrélation est un outil d'analyse de données qui permet d'analyser les relations linéaires entre deux ou plusieurs variables numériques
# Les coefficients de corrélation varient entre -1 et 1, où -1 indique une corrélation négative parfaite, 0 indique une absence de corrélation et 1 indique
# une corrélation positive parfaite. Les coefficients de corrélation peuvent être calculés à l'aide de la corrélation de Pearson,


def correlation_matrix(data):
    return data.corr()


def box_plot_of_variable_according_to_categorical_variable(numeric_variable, categorical_variable, data):
    """
  La fonction affiche un ou plusieurs boxplots representant la distribution de la variable quantitative en fonction des differentes categories d'une variable qualitative
  params :numeric_variable, categorical_variable : colonnes du dataframe

    """
    fig, ax = plt.subplots()
    sns.boxplot(x=categorical_variable, y=numeric_variable, data=data,
                showmeans=True, meanprops={"color": "green"}, ax=ax)
    ax.set_xlabel(categorical_variable.name)
    ax.set_ylabel(numeric_variable.name)
    st.pyplot(fig)

    # qualitative * quantitative

# calculer la moyenne et l'écart-type de la variable quantitative pour chaque catégorie de la variable catégorielle


# data est un dataframe avec une 2 colonnes une categorielle et lautre numerique
def mean_and_std_by_categorical_variable(categorical_variable, data):
    # categorical_variable c'est le nom de la variable seulemen
    means = data.groupby(categorical_variable).mean()
    means = means.rename(columns={means.columns[0]: "mean"})

    stds = data.groupby(categorical_variable).std()
    stds = stds.rename(columns={stds.columns[0]: "std"})

    return pd.concat([means, stds], axis=1)


# def graphic_representation(categorical_variable_1, categorical_variable_2):
#     """
#   The function builds the contingency table from the variables passed as parameters then displays the information returned by the latter in a bar plot
#   params : represents columns of the dataframe
#     """
#     contingency_table = pd.crosstab(
#         categorical_variable_1, categorical_variable_2)

#     # Création du diagramme en barres empilées
#     contingency_table.plot(kind='bar', stacked=True)
#     plt.title('Tableau de contingence entre',
#               categorical_variable_1, ' et ', categorical_variable_2)
#     plt.xlabel('Variable1')
#     plt.ylabel('Nombre d\'observations')
#     plt.show()

def graphic_representation(categorical_variable_1, categorical_variable_2):
    """
    The function builds the contingency table from the variables passed as parameters then displays the information returned by the latter in a bar plot
    params : represents columns of the dataframe
    """
    contingency_table = pd.crosstab(
        categorical_variable_1, categorical_variable_2)

    # Création du diagramme en barres empilées
    contingency_table.plot(kind='bar', stacked=True)
    plt.title('Tableau de contingence entre {} et {}'.format(
        categorical_variable_1.name, categorical_variable_2.name))
    plt.xlabel(categorical_variable_1.name)
    plt.ylabel('Nombre d\'observations')
    return plt.gcf(), contingency_table

# analyse bidimensionnelle sur les variables qualitatives

# test du chi-carré  et Le coefficient de contingence sont utiliséq pour 2 variables qualitatives

# Le test du chi-carré : est une méthode statistique utilisée pour évaluer l'indépendance entre deux variables qualitatives.
# Le test du chi-carré compare les fréquences observées dans un tableau de contingence avec les fréquences attendues si les deux
# variables étaient indépendantes. Si les fréquences observées sont significativement différentes des fréquences attendues, cela indique
# une relation entre les variables.

# Le coefficient de contingence  cramer : est une mesure de la force de la relation entre deux variables qualitatives. Le coefficient de contingence
# varie de 0 à 1, où 0 indique l'absence de relation et 1 indique une relation forte.


# Le Khi2 ici nous indique donc qu’il existe une liaison entre les deux variables ;
# le V de Cramer nous indique que cette liaison est très forte par sa valeur élevée.
# tableau de contingence entre 2 variable categorielles


def contingency_table_chi2_contengency_coefficent(categorical_variable_1, categorical_variable_2):
    """
    The function from the contingency table of the 2 variables passed as a parameter, calculates the test of χ2 which indicates the link between the two quantitative variables
    params : represents columns of the dataframe
    return : the contingency table as well as the value of the χ2 and the p-value
    """
    contingency_table = pd.crosstab(
        categorical_variable_1, categorical_variable_2)
    chi2, p_value, degres_liberte, _ = chi2_contingency(contingency_table)
    # contengency_coefficent=pg.cramers_v(contingency_table.values)

    return contingency_table, chi2, p_value

# χ2 = 0 si X et Y sont totalement indépendantes
# χ2 est d’autant plus grand que la liaison entre X et Y est forte
   # ou bien on verifie par rapport à la p-value si p-value <= 5% x a un effet significatif sur Y
   # si p-value > 5% x n'a pas d'effet sur Y


# data must have the categorical variable and the numeric variable that we want to calculate the fisher indicator
   # lien à utiliser pour (table de distribution): http://www.socr.ucla.edu/Applets.dir/F_Table.html avec df1= df_within et df2=df_within pour alpha=0,05
   # https://towardsdatascience.com/statistics-in-python-using-anova-for-feature-selection-b4dc876ef4f0
   # si l'indicateur de fisher < s5%(k ,n ), on conclura que x(categoriel) n'a pas d'effet significatif sur Y (on accepte l'hypothese nulle) => le x n'est pas inclu dans les features
   # si l'indicateur de fisher > s5%(k ,n ), on conclura que x(categoriel) a un effet significatif sur Y (on rejete l'hypothese nulle) => le x sera inclu dans les features
   # ou bien on verifie par rapport à la p-value si p-value <= 5% x a un effet significatif sur Y
   # si p-value > 5% x n'a pas d'effet sur Y


# data contient la variable categorielle et quantitative voulu et categorical_variable c'est le nom de la variable catégoriels
def fisher_indicator(categorical_variable, data, index):
    nb_categories = len(identify_categories(data[categorical_variable]))
    df = data.pivot(columns=categorical_variable, index=index)
    df_within = df.shape[0] - df.shape[1]
    df_between = df.shape[1] - 1
    fvalue, pvalue = stats.f_oneway(
        *df.iloc[:, 0:nb_categories].T.values)
    return pvalue


# def box_plot_of_variable_according_to_categorical_variable(numeric_variable, categorical_variable):
#     return sns.boxplot(x=categorical_variable, y=numeric_variable, data=data, showmeans=True, meanprops={"color": "green"})


def relation_between_variables(p_value, threshold):
    if abs(p_value) <= threshold:  # les variables sont dependantes
        return True
    else:
        return False
