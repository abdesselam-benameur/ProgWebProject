import pandas as pd
import seaborn as sns

filepath = "data/dermatology.data"
df = pd.read_csv(filepath, header=None)

# set the list of columns of df
df.columns = ["erythema", "scaling", "definite_borders", "itching", "koebner_phenomenon", "polygonal_papules", "follicular_papules", "oral_mucosal_involvement", "knee_and_elbow_involvement", "scalp_involvement", "family_history", "melanin_incontinence", "eosinophils_in_the_infiltrate", "PNL_infiltrate", "fibrosis_of_the_papillary_dermis", "exocytosis", "acanthosis", "hyperkeratosis", "parakeratosis", "clubbing_of_the_rete_ridges", "elongation_of_the_rete_ridges", "thinning_of_the_suprapapillary_epidermis", "spongiform_pustule", "munro_microabcess", "focal_hypergranulosis", "disappearance_of_the_granular_layer", "vacuolisation_and_damage_of_basal_layer", "spongiosis", "saw-tooth_appearance_of_retes", "follicular_horn_plug", "perifollicular_parakeratosis", "inflammatory_monoluclear_inflitrate", "band-like_infiltrate", "Age", "class"]

# determine a type of a column

def variable_type(column):
    if column.dtype == 'int64' or column.dtype == 'float64':
        return 'quantitavive'
    elif column.dtype == 'object':
        return 'qualitative'
    else:
        return None

# determine of categories in categorical columns
def categories(column):
    if variable_type(column) == 'qualitative':
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

