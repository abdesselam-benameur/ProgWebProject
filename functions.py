import pandas as pd

def get_class_distribuation(column):
    return column.value_counts()

def get_class_distribuation_percentages(column):
    return column.value_counts()/len(column)

# def get_class_distribuation_piechart(column):
#     return column.value_counts().plot(kind="pie", title="Distribution des classes")

def get_dummies(column):
    return pd.get_dummies(column)