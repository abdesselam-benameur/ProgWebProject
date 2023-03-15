import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

# class distribution
def get_class_distribuation(column):
    return column.value_counts()

# todo: implement some methods to deal with class distribution problem
def get_class_distribuation_percentages(column):
    return column.value_counts()/len(column)

# def get_class_distribuation_piechart(column):
#     return column.value_counts().plot(kind="pie", title="Distribution des classes")

# dummmification
def get_dummies(column):
    return pd.get_dummies(column)

# determine a type of a column
def variable_type(column):
    if column.dtype == 'int64' or column.dtype == 'float64':
        return 'quantitative'
    else:
        return 'catégorielle'

def variable_type_colored(column):
    if variable_type(column) == "quantitative":
        return ":green[quantitative]"
    else:
        return ":orange[catégorielle]"

# determine of categories in categorical columns
def categories(column):
    if variable_type(column) == 'catégorielle':
        return column.unique()
    else:
        return None

# find outliers in a column of a dataframe
def get_outliers(col):
    q1 = col.quantile(0.25)
    q3 = col.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    nb_lines_outliers=col[(col < lower_bound )| (col > upper_bound)].shape[0]
    outliers=col[(col < lower_bound )| (col > upper_bound)].unique()

    return nb_lines_outliers,outliers

def display_boxplot(col):
    return sns.boxplot(data=col).set(xlabel=col.name)

def handle_outliers(col,handle_type):
    if variable_type(col) == 'quantitavive':
        nb_lines_outliers,outliers=get_outliers(col)
        if nb_lines_outliers > 0:
            if handle_type=="median":
                new_value=col.median()
                col.replace(outliers,new_value,inplace=True)
            elif handle_type=="mean":
                new_value=col.mean()
                col.replace(outliers,new_value,inplace=True)

def get_missing_values_column(column, list_missing_values):
    """Return the missing values of a column"""
    return column[column.isin(list_missing_values)]
    
def get_missing_pourcentage_column(column : pd.Series):
    """Return the pourcentage of missing values of a column
            the missing values for this method is np.nan"""
    return column.isna().mean() * 100

def replace_missing_values_column_by_nan(column : pd.Series, list_missing_values):
    """Replace the missing values of a column by np.nan"""
    column.replace(list_missing_values, np.nan, inplace=True)
    # column = pd.Series(list(column.replace(list_missing_values, np.nan))).astype(np.float64)


def replace_missing_values_column_by_mode(column : pd.Series, list_missing_values):
    """Replace the missing values of a column by the mode"""
    mode = column.mode()[0]
    column.replace(list_missing_values, mode, inplace=True)

def replace_missing_values_column_by_mean(column : pd.Series, list_missing_values):
    """Replace the missing values of a column by the mean"""
    mean = column.mean()
    column.replace(list_missing_values, mean, inplace=True)
    
def replace_missing_values_column_by_mediane(column : pd.Series, list_missing_values):
    """Replace the missing values of a column by the mediane"""
    mediane = column.median()
    column.replace(list_missing_values, mediane, inplace=True)

def replace_missing_values_column_by_value(column : pd.Series, list_missing_values, value): 
    """Replace the missing values of a column by a value"""
    column.replace(list_missing_values, value, inplace=True)

#TODO: faire une fonction qui remplace les valeurs manquantes par prediction

def replace_missing_values_column_by_prediction(column : pd.Series, list_missing_values, model):
    """Replace the missing values of a column by the prediction of the model"""
    pass

def normalize_column_by_min_max(column: pd.Series):
    """Normalize a column by the min max method
        La normalisation Min-Max : cette méthode consiste à transformer les données 
        en une plage de valeurs entre 0 et 1. La formule de normalisation est la suivante :
        X_norm = (X - X_min) / (X_max - X_min)
        Où X est la valeur originale, X_min et X_max sont respectivement les valeurs minimale et maximale 
        de l'ensemble de données. Cette méthode est utile lorsque les données ont une plage de valeurs connue et délimitée.
    """

    

    # Exemple de données à normaliser
    # data = np.array([[10, 2], [5, 3], [8, 7]])

    # Créer un objet scaler
    scaler = MinMaxScaler()

    # Normaliser les données
    data_norm = scaler.fit_transform(column.values.reshape(-1, 1)).flatten()

    column.replace(column.values, data_norm, inplace=True)

def normalize_column_by_standardization(column : pd.Series):
    """Normalize a column by the standardization method (z score)
        La normalisation standard : cette méthode consiste à transformer les données en une distribution normale 
        avec une moyenne de 0 et un écart-type de 1. La formule de normalisation est la suivante :

        X_norm = (X - moyenne) / écart-type

        Où X est la valeur originale, la moyenne et l'écart-type sont calculés à partir de l'ensemble de données. 
        Cette méthode est utile lorsque les données ont une distribution normale ou presque normale.
    
    """
    # Exemple de données à normaliser
    # data = np.array([[10, 2], [5, 3], [8, 7]])

    # Créer un objet scaler
    scaler = StandardScaler()

    # Normaliser les données
    data_norm = scaler.fit_transform(column.values.reshape(-1, 1)).flatten()

    column.replace(column.values, data_norm, inplace=True)

def normalize_column_by_boxcox(column : pd.Series):
    """Normalize a column by the boxcox method
    La transformation de Box-Cox utilise une fonction puissance pour ajuster la distribution des données 
    à une distribution normale. La transformation est définie par une équation de la forme :

    y' = (y^lambda - 1) / lambda, si lambda différent de 0
    y' = log(y), si lambda = 0

    où y est la variable à transformer, y' est la variable transformée, et lambda est un paramètre 
    qui peut prendre n'importe quelle valeur réelle. La valeur optimale de lambda pour une variable 
    donnée est celle qui maximise la log-vraisemblance de la distribution transformée.
    """
    
    # Exemple de données à normaliser
    # data = np.array([[10, 2], [5, 3], [8, 7]])

    # Normaliser les données
    data_norm, _ = stats.boxcox(column.values)

    column.replace(column.values, data_norm, inplace=True)
