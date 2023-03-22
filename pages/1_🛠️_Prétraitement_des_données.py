import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from functions import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder



# button_style = """
#         <style>
#         .stButton button div p {
#             font-size: 0.9rem;
#         }
#         </style>
#         """
# st.markdown(button_style, unsafe_allow_html=True)

# interface pour charger, visualiser, traiter et analyser les données 
# et entrainer des modeles de machine learning avec ces données

st.title("Prétraitement des données")
st.markdown("# Charger les données")
st.sidebar.markdown("# [Charger les données](#charger-les-donn-es)", unsafe_allow_html=True)
file_uploader = st.file_uploader("Charger un fichier CSV avec un header valide", type="csv")

# add sidebar to navigate through markdown titles using hyperlinks but without a style

if file_uploader is not None:
    # demander le séparateur si ce n'est pas ','
    seperator = st.text_input("Séparateur", value=",")
    if seperator != "":
        # afficher le dataset
        st.write("Afficher le dataset")
        df = pd.read_csv(file_uploader, sep=seperator)
        df_copy = df.copy()
        st.write(df)
        st.write(f"Le dataset contient `{df.shape[0]}` lignes et `{df.shape[1]}` colonnes")

        st.write("### Choisir la colonne target")
        options = [""] + df.columns.to_list()
        target = st.selectbox("La colonne target", options, index=0)

        if target != "":
            Y = df.pop(target)
            st.write("---")
            st.markdown("# Prétraiter les données")
            st.sidebar.markdown("# [Prétraiter les données](#pr-traiter-les-donn-es)", unsafe_allow_html=True)


            # Types des varaiables
            st.markdown("## Types des variables")
            st.write("Vous avez le bouton **Switch** devant chaque type de colonne que vous pouvez utilisez pour corriger le type s'il est erroné")

            if "df_types" not in st.session_state:
                st.session_state.df_types = pd.DataFrame({'Type de la colonne': df.apply(variable_type_colored, axis=0)})
            if "switch_buttons_disabled" not in st.session_state:
                st.session_state.switch_buttons_disabled = False

            switcher = {":green[quantitative]": ":orange[catégorielle]", ":orange[catégorielle]": ":green[quantitative]"}
            for colName in st.session_state.df_types.index:
                col1, col2, col3 = st.columns(3, gap="large")
                with col1:
                    col1.write(colName)
                with col3:
                    if col3.button("Switch", use_container_width=True, key=colName+"button", disabled=st.session_state.switch_buttons_disabled):
                        st.session_state.df_types.loc[colName] = switcher[st.session_state.df_types.at[colName, 'Type de la colonne']]
                with col2:
                    col2.write(st.session_state.df_types.at[colName, "Type de la colonne"])

            if st.button("Valider les types des colonnes", use_container_width=True):
                st.session_state.switch_buttons_disabled = True
                st.experimental_rerun()
                # todo: make the necessary changes of df types
            # st.write("---")
            ################################################################################################################
            # Les valeurs manquantes
            if st.session_state.switch_buttons_disabled:
                st.markdown("## Traiter les valeurs manquantes")
                st.sidebar.markdown("## [Traiter les valeurs manquantes](#traiter-les-valeurs-manquantes)", unsafe_allow_html=True)
                # ask the user to give us the list of all possible values that represent missing values
                missing_values = st.text_area("Spécifiez tous les symboles, s'il y'en a, qui représentent des valeurs manquantes (**chaque valeur dans une ligne**)", value="")
                if missing_values != "":
                    # convert the string to a list
                    missing_values = missing_values.splitlines()
                    # replace the missing values with np.nan
                    df.replace(missing_values, np.nan, inplace=True)
                st.session_state.variable_categ = []
                st.session_state.variable_quant = []
                for col, colType in st.session_state.df_types.iterrows():
                    if colType[0] == ":green[quantitative]":
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        st.session_state.variable_quant.append(col)
                    else:
                        df[col] = df[col].astype("object") # think about "category"
                        st.session_state.variable_categ.append(col)
                        
                st.write(df)

                # delete the lines where the value of the target column is missing
                # test if there are missing values
                if Y.isnull().sum() != 0:
                    st.markdown(f":warning:  Il y a des valeurs manquantes dans la colonne target `{target}` donc on supprime leurs lignes correspondantes")
                    df = df[~Y.isna()]
                    Y.dropna(inplace=True)

                # Treat the lines where there are missing values
                # display the columns that have missing values
                suite = False
                cols_with_nan_values = df.columns[df.isnull().any()]
                if len(cols_with_nan_values) == 0:
                    st.markdown(":information_source: Il n'y a pas de valeurs manquantes dans le dataset")
                    suite = True
                else:
                    st.markdown("### Statistiques sur les valeurs manquantes")
                    st.write("#### Par ligne")
                    nb_lines_nan = df[cols_with_nan_values].isnull().any(axis=1).sum()
                    percentage_lines_nan = nb_lines_nan / len(df)
                    st.write(f"Nombre des lignes qui ont au moins une valeur manquante: `{nb_lines_nan}` (`{percentage_lines_nan*100} %`)")

                    st.write("#### Par colonne")
                    missing_values_stats = pd.DataFrame({
                                'Nombre de Valeurs Manquantes': df[cols_with_nan_values].isnull().sum(), 
                                'Pourcentage de Valeurs Manquantes': df[cols_with_nan_values].isnull().sum()/len(df)*100})
                    st.dataframe(missing_values_stats.T, use_container_width=True)

                    # choose the method to treat the missing values
                    st.markdown("### Choisir la méthode du traitement des valeurs manquantes")
                    # create a form
                    if "form_disabled" not in st.session_state:
                        st.session_state.form_disabled = False
                    tab1, tab2 = st.tabs(["par ligne", "par colonne"])
                    with tab1: # par ligne
                        with st.form(key='row_form'):
                            options = ["Supprimer les lignes"]
                            method = st.selectbox("Choisir la méthode", options, index=0, disabled=st.session_state.form_disabled)
                            col1, col2, col3 = st.columns(3, gap="large")
                            if col3.form_submit_button(label='Valider', use_container_width=True, disabled=st.session_state.form_disabled):
                                st.session_state.form_disabled = True
                                st.session_state.chosen_methods = method
                                st.experimental_rerun()
                    with tab2: # par colonne
                        options_quant = ["Supprimer la colonne", "Remplacer par la moyenne", "Remplacer par la médiane", "Remplacer par le mode"]
                        options_categ = ["Supprimer la colonne", "Remplacer par la catégorie la plus fréquente (le mode)"]
                        with st.form(key='col_form'):
                            for col in cols_with_nan_values:
                                if col in st.session_state.variable_quant:
                                    method = st.selectbox(col, options_quant, key=col+"selectbox", index=1, disabled=st.session_state.form_disabled)
                                else:
                                    method = st.selectbox(col, options_categ, key=col+"selectbox", index=1, disabled=st.session_state.form_disabled)
                            col1, col2, col3 = st.columns(3, gap="large")
                            if col3.form_submit_button(label='Valider', use_container_width=True, disabled=st.session_state.form_disabled):
                                st.session_state.form_disabled = True
                                st.session_state.chosen_methods = {}
                                for col in cols_with_nan_values:
                                    st.session_state.chosen_methods[col] = st.session_state[col+"selectbox"]
                                st.experimental_rerun()
                    
                    # Apply the chosen method(s)
                    if "chosen_methods" in st.session_state:
                        if type(st.session_state.chosen_methods) is str:
                            # delete the lines of df that contain at least one nan value and their corresponding ones in Y
                            df = df[~df.isna().any(axis=1)]
                            Y = Y[df.index]
                        else:
                            for col, method in st.session_state.chosen_methods.items():
                                if method.startswith("S"): # Supprimer la colonne
                                    df.drop(col, axis=1, inplace=True)
                                    st.session_state.variable_quant.remove(col)
                                elif method.endswith("moyenne"): # Remplacer par la moyenne
                                    df[col].fillna(df[col].mean(), inplace=True)
                                elif method.endswith("médiane"): # Remplacer par la médiane
                                    df[col].fillna(df[col].median(), inplace=True)
                                elif method.endswith("mode"): # Remplacer par le mode
                                    df[col].fillna(df[col].mode()[0], inplace=True)
                        suite = True
                        
                        st.write(f"Le nouveau dataset contient `{df.shape[0]}` lignes et `{df.shape[1]}` colonnes")
                        df.reset_index(inplace=True, drop=True)
                        st.write(df)


                ################################################################################################################
                if suite:
                    # Les valeurs aberrantes
                    st.markdown("## Les valeurs aberrantes")
                    st.sidebar.markdown("## [Les valeurs aberrantes](#les-valeurs-aberrantes)", unsafe_allow_html=True)

                    # create tabs and display a boxplot per col within each tab
                    tabs = st.tabs(st.session_state.variable_quant)
                    for tab, col in zip(tabs, st.session_state.variable_quant):
                        with tab:
                            nb_lines_outliers, outliers = get_outliers(df[col])
                            st.write("La liste des valeurs aberrantes trouvées:", str(list(outliers)))
                            st.write(f"Pourcentage des valeurs aberrantes dans la colonne: `{nb_lines_outliers/df.shape[0]*100} %`")
                    # todo: treat outliers
                    ################################################################################################################
                    # La Normalisation
                    st.markdown("## Normaliser les données")
                    st.sidebar.markdown("## [Normaliser les données](#normaliser-les-donn-es)", unsafe_allow_html=True)

                    if "norm_disabled" not in st.session_state:
                        st.session_state.norm_disabled = False
                    options = ["Oui", "Non"]
                    choice = st.radio("Voulez vous normaliser les données?", options, key="radio_norm", index=0, horizontal=True, disabled=st.session_state.norm_disabled)
                    # choisir la méthode de normalisation
                    options = ["StandardScaler", "MinMaxScaler"]
                    method = st.selectbox("Choisir la méthode de normalisation", options, index=0, disabled=st.session_state.radio_norm=="Non" or st.session_state.norm_disabled)

                    _, _, col3 = st.columns(3, gap="large")
                    if col3.button(label='Valider', use_container_width=True):
                        st.session_state.norm_disabled = True # disable the radio and the selectbox
                        st.experimental_rerun()
                    
                    if st.session_state.norm_disabled:
                        if choice == "Oui":
                            if method == "MinMaxScaler":
                                scaler = MinMaxScaler()
                            elif method == "StandardScaler":
                                scaler = StandardScaler()
                            df[st.session_state.variable_quant] = pd.DataFrame(scaler.fit_transform(df[st.session_state.variable_quant]), columns=st.session_state.variable_quant)
                            st.write(df)
                
                        ################################################################################################################
                        # Les variables catégorielles
                        suite = False
                        if st.session_state.variable_categ:
                            st.markdown("## Les variables catégorielles")
                            # display the number of categories in a table
                            st.dataframe(pd.DataFrame({"Nombre de catégories": df[st.session_state.variable_categ].nunique()}).T, use_container_width=True)
                            # treat the categorical variables
                            if "categ_disabled" not in st.session_state:
                                st.session_state.categ_disabled = False
                            options = ["Label Encoding", "One Hot Encoding"]
                            method = st.selectbox("Choisir la méthode de traitement des variables catégorielles", options, index=0, disabled=st.session_state.categ_disabled)
                            _, _, col3 = st.columns(3, gap="large")
                            if col3.button(label='Valider', key="st.session_state.variable_categ_button", use_container_width=True):
                                st.session_state.categ_disabled = True
                                st.experimental_rerun()
                            if st.session_state.categ_disabled:
                                st.session_state.dummification_method = method
                                suite = True
                        else:
                            suite = True

                        ################################################################################################################
                        if suite:
                            # Le déséquilibre des classes
                            st.markdown("## Distribution des classes")
                            st.sidebar.markdown("## [Distribution des classes](#distribution-des-classes)", unsafe_allow_html=True)
                            # display the number of classes in a table
                            st.write(f"Nombre de classes de la colonne target `{target}`: ", Y.nunique())
                            # display the distribution of the classes
                            st.write("Distribution des classes")
                            st.bar_chart(Y.value_counts())
                            # treat the target variable
                            if "target_disabled" not in st.session_state:
                                st.session_state.target_disabled = False
                            options = ["Oui", "Non"]
                            choice = st.radio("Voulez vous traiter le déséquilibre des classes?", options, key="radio_target", index=0, horizontal=True, disabled=st.session_state.target_disabled)
                            # choisir la méthode de traitement
                            options = ["Random Oversampling", "Random Undersampling", "SMOTE", "ADASYN"]
                            method = st.selectbox("Choisir la méthode de traitement", options, index=0, disabled=st.session_state.radio_target=="Non" or st.session_state.target_disabled)
                            _, _, col3 = st.columns(3, gap="large")
                            if col3.button(label='Valider', key="target_button", use_container_width=True):
                                st.session_state.target_disabled = True
                                st.experimental_rerun()
                            if st.session_state.target_disabled:
                                if choice == "Oui":
                                    df, Y = apply_imbalanced_data_method(df, Y, method)
                                    st.write("Nouvelle distribution des classes")
                                    st.bar_chart(Y.value_counts())
                                st.session_state.df = df
                                st.session_state.Y = Y if variable_type(Y) == "quantitative" else pd.Series(LabelEncoder().fit_transform(Y))

                        ################################################################################################################
                        
                        

                        # todo: feature selection 