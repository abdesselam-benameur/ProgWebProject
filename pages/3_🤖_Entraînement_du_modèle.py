from entrainement import *
import streamlit as st
import seaborn as sns

st.title("Entraînement du modèle")
st.set_option('deprecation.showPyplotGlobalUse', False)

if "df" not in st.session_state and "Y" not in st.session_state:
    st.warning("Commencez d'abord par charger les données et faire l'analyse exploratoire", icon="⚠️")
else:
    df = st.session_state.df.drop(columns=st.session_state.variable_categ)

    st.markdown("## Faites vos choix...")
    if "algo_choice" not in st.session_state:
        st.session_state.algo_choice = False
    
    c1, c2 = st.columns(2)
    with c1:
        # create a selectbox to get the value of the selected algorithm
        options = ["KNN", "Logistic Regression", "Decision Tree"]
        selected_algorithm = st.selectbox("L'algorithme", options, disabled=st.session_state.algo_choice)
    with c2:
        # create a selectbox to get the value of the selected hyperparameter method
        options = ["Grid Search", "Random Search"]
        hyper_params_method = st.selectbox("Méthode du finetuning", options, disabled=st.session_state.algo_choice)
    
    # create a slider to get the value of test size which can be between 10% and 50%
    test_size = st.slider("Pourcentage du :red[test set]", 10, 50, 20, 5, disabled=st.session_state.algo_choice)
    # create a dataframe to show the test size and train size
    st.table(pd.DataFrame({"Test set": [test_size], "Train set": [100 - test_size]}, index=["Pourcentage (%)"]))

    # add a button to validate the choice
    if st.button("Commencer l'entraînement !", key="valider_algo_choice", use_container_width=True):
        st.session_state.algo_choice = True
        st.experimental_rerun()

    if st.session_state.algo_choice:            
        # create a button to train the model
        with st.spinner('Entraînement en cours...'):
            X_train,X_test,y_train,y_test = load_dataset(df, st.session_state.Y, test_size/100)
            model = train_and_fine_tune(X_train, y_train, hyper_params_method, selected_algorithm)
        st.write("")
        st.success("L'entraînement est terminé avec succès", icon="✅")

        st.write("## Evaluation du modèle")
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        accuracy,f1,recall,precision,auc = evaluate(model,X_test, y_test)
        st.table(pd.DataFrame({"Accuracy": [accuracy], "Recall": [recall], "Precision": [precision], "AUC": [auc], "F1 Score": [f1],}, index=["Valeur"]))
        
        st.markdown("### Matrice de confusion")
        st.pyplot(confusion_matrix2(y_test, y_pred))

        # ROC Curve
        st.markdown("### Courbe ROC")
        st.pyplot(skplt.metrics.plot_roc(y_test, y_pred_proba, plot_micro=False, plot_macro=False, figsize=(10, 8)).figure)
