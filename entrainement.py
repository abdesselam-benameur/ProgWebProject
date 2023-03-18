from sklearn.metrics import recall_score,precision_score,accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
import scikitplot as skplt
import pandas as pd

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def load_dataset(features,Y,test_size):
  return train_test_split(features,Y,test_size=test_size,random_state=42, stratify=Y.values)

def model_with_params(algorithm):
    if algorithm == "KNN":
      params = {
        "n_neighbors": range(1, 21, 2),
        "weights": ['uniform', 'distance'],
        "metric": ['euclidean', 'manhattan', 'minkowski'],
        "algorithm": ['auto','ball_tree','kd_tree','brute'],
        "p": [1, 2]
      }
      return KNeighborsClassifier(), params
    elif algorithm == "Logistic Regression":
      params = {
        "C": [0.1, 1, 10],
        "penalty": ['l2'],
        "solver":['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
      }
      return LogisticRegression(), params
    elif algorithm == "Decision Tree":
      params = {
        "criterion":['gini','entropy',"log_loss"],
        "max_depth":list(range(1,16))
      }
      return DecisionTreeClassifier(), params

def train_and_fine_tune(X_train, y_train, finetuning_method, selected_algorithm):
  model, params = model_with_params(selected_algorithm)
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
  if finetuning_method=="GridSearchCV":
    model = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=cv, scoring="accuracy",error_score=0)
  elif finetuning_method == "RandomizedSearchCV":
    model = RandomizedSearchCV(estimator=model, param_distributions=params, scoring='accuracy', cv=cv)
  model.fit(X_train, y_train)
  return model

def prediction(model,X_test):
  y_pred = model.predict(X_test)
  return y_pred

# Ce score est une mesure de la performance du modèle, qui représente le coefficient de détermination R².
# Le coefficient de détermination R² est une mesure de l'adéquation du modèle aux données
def score(X_test, y_test,model):
  return model.score(X_test, y_test)

# def comparing_results(X_test,y_test,y_pred):
#   df = X_test.copy()
#   row_number=df.shape[1]+1
#   df.insert(df.shape[1],"Actual",y_test, True)
#   df.insert(row_number,"Predicted",y_pred, True)
#   return df

def confusion_matrix2(y_test,y_pred):
  skplt.metrics.plot_confusion_matrix(y_test,y_pred)

def rocCurve(X_test,y_test,fitted_data):
  #recall = TPR et specificity = FPR
  fpr, tpr, thresholds = roc_curve(y_test, fitted_data.predict_proba(X_test)[:,1])
  print("recall",tpr)
  print("specificity",1 - fpr)
  roc_df = pd.DataFrame({'recall': tpr, 'specificity': 1 - fpr})

  ax = roc_df.plot(x='specificity', y='recall', figsize=(4, 4), legend=False)
  ax.set_ylim(0, 1)
  ax.set_xlim(1, 0)
  # ax.plot((1, 0), (0, 1))
  ax.set_xlabel('specificity')
  ax.set_ylabel('recall')
  ax.fill_between(roc_df.specificity, 0, roc_df.recall, alpha=0.3)
  
  plt.tight_layout()
  plt.show()

# evaluation
def evaluate(model,X_test, y_test):
  y_pred = model.predict(X_test)
  y_predict_proba = model.predict_proba(X_test)
  accuracy=accuracy_score(y_test, y_pred)
  
  if y_test.nunique()==2:
    average = "binary"
    auc = roc_auc_score(y_test, y_predict_proba[:,1])
  else:
    average = "macro"
    #La macro-precision est une métrique utilisée pour évaluer les performances d'un modèle de classification multi-classes. 
    #Elle calcule la précision pour chaque classe individuellement et calcule ensuite la moyenne de ces précisions pour donner une 
    #mesure globale de la performance du modèle.
    # si on a un desiquilibre des classes vaut mieux utiliser micro-precision sinon macro
    # Une valeur de macro-precision élevée indique que le modèle est capable de classer correctement un grand nombre de classes différentes
    #le recall global en utilisant la moyenne macro, qui calcule la moyenne des recalls de chaque classe sans tenir compte de leur taille
    auc = roc_auc_score(y_test, y_predict_proba, multi_class='ovr')
  recall=recall_score(y_test, y_pred, average=average)
  precision = precision_score(y_test, y_pred, average=average)
  f1 = f1_score(y_test, y_pred, average=average)
  return accuracy,f1,recall,precision,auc

