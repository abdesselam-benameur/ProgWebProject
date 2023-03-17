from sklearn.metrics import recall_score,precision_score,accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
import scikitplot as skplt
import pandas as pd

from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def hyper_params_selection(method,algorithm,params):
  #cv = KFold(n_splits=10, shuffle=True, random_state=42)
  cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
  if method=="GridSearchCV": 
    grid_search = GridSearchCV(estimator=algorithm, param_grid=params, n_jobs=-1, cv=cv, scoring="accuracy",error_score=0)
  elif method=="RandomizedSearchCV":
    grid_search=RandomizedSearchCV(estimator=algorithm, param_grid=params, scoring='accuracy', cv=cv)   
  return grid_search
                        

def load_dataset(data,target,test_size):
  input=data.loc[:, data.columns != target.name]
  X_train,X_test,y_train,y_test = train_test_split(input,target,test_size=test_size,random_state=42, stratify=target.values)
  return X_train,X_test,y_train,y_test



def model_with_params(grid_result,algorithm):
  if algorithm == "knn":
    return KNeighborsClassifier(n_neighbors=grid_result.best_params_['n_neighbors'],weights=grid_result.best_params_["weights"],algorithm=grid_result.best_params_["algorithm"],metric=grid_result.best_params_["metric"],p=grid_result.best_params_["p"])
  elif algorithm == "logisticRegression":
    return LogisticRegression(penalty=grid_result.best_params_['penalty'],C=grid_result.best_params_['C'],solver=grid_result.best_params_['solver'])
  elif algorithm == "DecisionTree":
    return DecisionTreeClassifier(criterion=grid_result.best_params_['criterion'],max_depth=grid_result.best_params_['max_depth'])

def _params(algorithm):
  if algorithm == "knn":
    n_neighbors = range(1, 21, 2)
    weights = ['uniform', 'distance']
    metric = ['euclidean', 'manhattan', 'minkowski']
    algorithm = ['auto','ball_tree','kd_tree','brute']
    p= [1, 2]
    params = dict(n_neighbors=n_neighbors,weights=weights,metric=metric,algorithm=algorithm,p=p)
  if algorithm=="logisticRegression":
    params={'C': [0.1, 1, 10], 
     'penalty': ['l2'],
     'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
     }
  if algorithm == "DecisionTree":
    params={'criterion':['gini','entropy',"log_loss"],
            'max_depth':list(range(1,16))}

  return params

def model_without_params(algorithm):
    if algorithm == "knn":
      return KNeighborsClassifier()
    elif algorithm == "logisticRegression":
      return LogisticRegression()
    elif algorithm == "DecisionTree":
      return DecisionTreeClassifier()

def heuristic_method(train,method_name,X_train,y_train):
  if method_name=="OvO":
    ovo = OneVsOneClassifier(train)
    ovo.fit(X_train, y_train)
    return ovo
  elif method_name=="OvR":
    ovr = OneVsRestClassifier(train)
    ovr.fit(X_train, y_train)
    return ovr
  elif method_name=="":
    fitted_data=train.fit(X_train, y_train)
    return fitted_data

# data represente tout le dataset
# model represente l'appel de l'algorithme dans les parametre
# test_size represente la taille du du test set est c'est 0.89.. qlq chose
# heuristic_method represente la methode à utiliser pour diviser les données de la target 
# target represente la classe output le y 
# selected_algorithm represente le nom de l'algorithme à utiliser 
# hyper_params_method represente la methode utiliser pour faire la bonne selection des hyperparametre

def training(target,selected_algorithm,test_size,hyper_params_method,encoding_method,data):# encoding_method : si on est pas en multiclasse 
                                                                                          # on va mettre une chaine vide

  X_train,X_test,y_train,y_test=load_dataset(data,target,test_size)
  model=model_without_params(selected_algorithm)
  grid=_params(selected_algorithm)
  grid_search=hyper_params_selection(hyper_params_method,model,grid)
  grid_result = grid_search.fit(X_train, y_train)

  train=model_with_params(grid_result,selected_algorithm)
  fitted_data= heuristic_method(train,encoding_method,X_train,y_train)

  return fitted_data,X_test,y_test


def prediction(fitted_data,X_test):# ovr ou ovo + X_test
  y_pred = fitted_data.predict(X_test)
  return y_pred

# Ce score est une mesure de la performance du modèle, qui représente le coefficient de détermination R². Le coefficient de
# détermination R² est une mesure de l'adéquation du modèle aux données
def score(X_test, y_test,fitted_data):
  return fitted_data.score(X_test, y_test)

def comparing_results(X_test,y_test,y_pred):
  df = X_test.copy()
  row_number=df.shape[1]+1
  df.insert(df.shape[1],"Actual",y_test, True)
  df.insert(row_number,"Predicted",y_pred, True)
  return df

def confusion_matrix(y_test,y_pred):
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

# y a que l'accuracy qui est definie comme metrique pour le binaire et le mutliclasses
def evaluation(y_test,y_pred,X_test):
  accuracy=accuracy_score(y_test, y_pred)
  report_f1_recall_precision=classification_report(y_test, y_pred)
  if len(set(y_pred))<=2:
    curve_roc=rocCurve(X_test,y_test)
    recall=recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy,report_f1_recall_precision,curve_roc,recall,precision
  else:
    #La macro-precision est une métrique utilisée pour évaluer les performances d'un modèle de classification multi-classes. 
    #Elle calcule la précision pour chaque classe individuellement et calcule ensuite la moyenne de ces précisions pour donner une 
    #mesure globale de la performance du modèle.
    # si on a un desiquilibre des classes vaut mieux utiliser micro-precision sinon macro
    # Une valeur de macro-precision élevée indique que le modèle est capable de classer correctement un grand nombre de classes différentes
    precision=precision_score(y_test, y_pred, average="macro") 
    #le recall global en utilisant la moyenne macro, qui calcule la moyenne des recalls de chaque classe sans tenir compte de leur taille
    recall_macro = recall_score(y_test, y_pred, average='macro')
    return accuracy,report_f1_recall_precision,recall_macro,precision

