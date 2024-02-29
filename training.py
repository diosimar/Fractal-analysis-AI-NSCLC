# librerias implementadas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import  warnings 
warnings.filterwarnings("ignore")

# librerias  requeridas en el procesamiento de  datos
from sklearn.preprocessing import FunctionTransformer, StandardScaler,  OneHotEncoder,  OrdinalEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src import  utils as fn

# Libreria  modelos  implementados  para estimación de riesgo de supervivencia
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis , GradientBoostingSurvivalAnalysis, RandomSurvivalForest
from sksurv.svm import FastSurvivalSVM
from sklearn.model_selection import GridSearchCV
from sksurv.metrics import cumulative_dynamic_auc , concordance_index_ipcw, concordance_index_censored

# Cargue de datos a analizar( datos sin imputación)
df_sin_imputaciones = pd.read_csv('Data/df_sin_imputaciones.csv')  

# limpieza y control de valores perdidos
clean_transformer = FunctionTransformer(fn.limpiar_estadios_clinicos, validate=False)

### encoder para variables categoricas ordinales
Ordi_features = ['Overall.Stage']
Ordi_transformer = Pipeline(steps=[
    ('Ordi', OrdinalEncoder())
])

### encoder para variables categoricas no ordinales
NonO_features = ['Histology','gender']
NonO_transformer = Pipeline(steps=[
    ('Non-O', OneHotEncoder())
])

### escalar variables continuas
Scale_features = ['age', 'F.analysis']

Scale_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=3)),
        ('Scaling', StandardScaler())
        ]
)

Preprocessor = ColumnTransformer(transformers=[
    ('Ordinal',Ordi_transformer, Ordi_features ),
    ('Non-Ordinal', NonO_transformer, NonO_features),
    ('Scale', Scale_transformer, Scale_features)

], remainder = 'passthrough')

clf = Pipeline(steps=[
    ('clean',clean_transformer),
    ('preprocessor', Preprocessor)
     ])
# Aplicar el pipeline a tus datos ficticios
df_transformed = clf.fit_transform(df_sin_imputaciones)

### Ajuste  de DataFrame  final con procesamiento de datos 
Non_Ordinal_cols = clf.named_steps['preprocessor'].transformers_[1][1].named_steps['Non-O'].get_feature_names_out(NonO_features)
Non_Ordinal_cols = [x for x in Non_Ordinal_cols]
Non_processed_cols = ['PatientID', 'clinical.T.Stage', 'Clinical.N.Stage','Clinical.M.Stage', 'Survival.time', 'deadstatus.event']
cols = [y for x in [ Ordi_features, Non_Ordinal_cols , Scale_features, Non_processed_cols] for y in x]

transformed_x_train = pd.DataFrame(df_transformed,columns= cols)
transformed_x_train.columns = ['Overall.Stage', 'adenocarcinoma', 'large cell',
       'nos', 'squamous cell carcinoma', 'female',
       'male', 'age', 'F.analysis', 'PatientID', 'clinical.T.Stage',
       'Clinical.N.Stage', 'Clinical.M.Stage', 'Survival.time',
       'deadstatus.event']

#division de  la data para  el entrenamiento y prueba de  modelo
feature =['age', 'clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage', 'Overall.Stage', 'F.analysis', 'adenocarcinoma',
          'large cell', 'nos', 'squamous cell carcinoma', 'female', 'male']
answer  = ['deadstatus.event','Survival.time',]

xtrain_,xtest_,ytrain_,ytest_ = train_test_split(transformed_x_train[feature],transformed_x_train[answer],test_size=0.25,random_state=42)

ytrain_['deadstatus.event']=ytrain_['deadstatus.event'].astype('bool')
y_train_=ytrain_.to_records(index=False)

ytest_['deadstatus.event']=ytest_['deadstatus.event'].astype('bool')
y_test_= ytest_.to_records(index=False)


#############################################################################################################################
#----------------------------Proceso de entrenamiento de 4  modelos para analisis de supervivencia--------------------------# 
classifiers = [
    RandomSurvivalForest( random_state=42),
    GradientBoostingSurvivalAnalysis(random_state=0),
    ComponentwiseGradientBoostingSurvivalAnalysis(random_state=0),
    FastSurvivalSVM( random_state=0)
    ]

top_class = []
top_score = []
for classifier in classifiers:
    pipe = Pipeline(steps=[#('preprocessor', Preprocessor),
                      ('classifier', classifier)])

    # training model
    pipe.fit(xtrain_,y_train_)
    print(classifier)

    acc_score = pipe.score(xtest_, y_test_)
    print("model score: %.3f" % acc_score)

    top_class.append(classifier)
    top_score.append(acc_score)

dict_params = dict(zip(top_class, top_score))

# mejor modelo ajustado
topModels =  pd.DataFrame([[str(key).split('(')[0],dict_params[key] ] for key in dict_params.keys() ] , columns=['Model', 'acc_score'])
topModels['Model'] = ['Random Survival\n Forest', 'Gradient Boosting\n Survival' , 'Componen twise \nGradient Boosting Survival',  'Fast Survival SVM']
topModels.to_csv('output/Survival-analysis/Indice_concordancia_pull_models.txt', index = False, encoding='utf-8')
# Crear una lista de colores para cada barra
colors = ['green', 'blue', 'orange','red']
plt.figure(figsize=(12, 10))
plt.bar(topModels.Model, topModels.acc_score, color = colors )
# Agregar los valores sobre cada barra
plt.xticks(topModels.index, rotation=0, ha='center')
plt.grid(True,zorder=0)
plt.savefig(f'output/Survival-analysis/survival_models.jpg', bbox_inches ='tight')
plt.close()  
###############################################################################################################

n_estimators = [i * 5 for i in range(1, 100)]

estimators = {
    "no regularization": GradientBoostingSurvivalAnalysis(random_state=0),
    "learning rate": GradientBoostingSurvivalAnalysis(learning_rate=0.1, max_depth=1, random_state=0),
    "dropout": GradientBoostingSurvivalAnalysis(learning_rate=1.0, dropout_rate=0.1, max_depth=1, random_state=0),
    "subsample": GradientBoostingSurvivalAnalysis(learning_rate=1.0, subsample=0.5, max_depth=1, random_state=0),
}

scores_reg = {k: [] for k in estimators.keys()}
for n in n_estimators:
    for name, est in estimators.items():
        est.set_params(n_estimators=n)
        est.fit(xtrain_,y_train_)
        cindex = est.score(xtest_, y_test_)
        scores_reg[name].append(cindex)

scores_reg = pd.DataFrame(scores_reg, index=n_estimators)

plt.figure(figsize=(12, 10))
ax = scores_reg.plot(xlabel="n_estimators", ylabel="concordance index")
ax.grid(True)
plt.savefig(f'output/Survival-analysis/n_estimators_GBS.jpg', bbox_inches ='tight')
plt.close()  

scores_reg.max().to_csv('Indice_concordancia_variacion_hiperparametros_GBS.txt', index = False, encoding='utf-8')

columns = ['age', 'clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage',
       'Overall.Stage', 'F.analysis', 'adenocarcinoma', 'large cell', 'nos',
       'squamous cell carcinoma', 'female', 'male']
est_=GradientBoostingSurvivalAnalysis(n_estimators=150, learning_rate=1.0, subsample=0.5, max_depth=1, random_state=0)
est_.fit(xtrain_[columns],y_train_)
cindex_ = est_.score(xtest_[columns], y_test_)
print(cindex_)

############################

#------ busqueda  de hiperparametros  para optimizar el entrenamiento del modelo GBS

# Definir el modelo GradientBoostingSurvivalAnalysis
gbsa = GradientBoostingSurvivalAnalysis(random_state=0)

# Definir los hiperparámetros que deseas ajustar
param_grid = {
    'learning_rate': [0.01, 0.1, 0.5 , 1.0],
    'n_estimators': [150],
    'subsample': [0.8, 0.5, 1.0],
    'max_depth': [1, 4, 6],
    #'dropout_rate' : [.2 , .5,.8]

}

# Realizar la búsqueda de hiperparámetros mediante GridSearchCV
grid_search = GridSearchCV(gbsa, param_grid,  cv=5)
grid_search.fit(xtrain_[columns],y_train_)

best_model = grid_search.best_estimator_
# Evaluar el modelo en el conjunto de prueba
c_index = best_model.score(xtest_[columns], y_test_)
print(f"C-Index en el conjunto de prueba: {c_index}")


#######################################################################################
#--------- caracteristicas  importantes---------------# 
df = pd.DataFrame(est_.feature_importances_, columns=['Values'], index=xtest_.columns)

# Ordenar de mayor a menor
df_sorted = df.sort_values(by='Values', ascending=False)

# Convertir los valores a porcentajes
total_sum = df_sorted['Values'].sum()
df_sorted['Values'] = (df_sorted['Values'] / total_sum) * 100
df_sorted.to_csv('output/Survival-analysis/feature_importance_GBS_final.txt', index = False, encoding='utf-8')

#------------- evaluación time dependent auc-------------------#
y_events = y_train_[y_train_["deadstatus.event"]]

train_min, train_max = y_events["Survival.time"].min(), y_events["Survival.time"].max()

y_events = y_test_[y_test_["deadstatus.event"]]
test_min, test_max = y_events["Survival.time"].min(), y_events["Survival.time"].max()

assert (
    train_min <= test_min < test_max < train_max
), "time range or test data is not within time range of training data."

times = np.percentile(df_transformed["Survival.time"], np.linspace(5, 81, 15))
risk_score= est_.predict(xtest_)
df_transformed
#------------- evaluación time dependent auc por variable
def plot_cumulative_dynamic_auc(risk_score, label, color=None):
    auc, mean_auc = cumulative_dynamic_auc(y_train_,y_test_, risk_score, times)
    plt.figure(figsize=(12, 10))
    plt.plot(times, auc, marker="o", color=color, label=label)
    plt.xlabel("days from enrollment")
    plt.ylabel("time-dependent AUC")
    plt.axhline(mean_auc, color=color, linestyle="--")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'output/Survival-analysis/time-dependent_AUC_variables.jpg', bbox_inches ='tight')
    plt.close()  

for i, col in enumerate(['age','F.analysis','Overall.Stage','Clinical.N.Stage','clinical.T.Stage']):
  fn.plot_cumulative_dynamic_auc(xtest_.values[:, i], col, color=f"C{i}")


### evaluacion time dependent auc  general del modelo
auc, mean_auc = cumulative_dynamic_auc(y_train_,y_test_, risk_score, times)
fig, ax = plt.subplots()

ax.plot(times, auc, marker='o')
ax.axhline(mean_auc, ls='--')
ax.set_xlabel('Days after enrollment')
ax.set_ylabel('Time-dependent AUC')

plt.grid(True)
plt.tight_layout() 
plt.savefig(f'output/Survival-analysis/time-dependent_AUC_GBS.jpg', bbox_inches ='tight')
plt.close()  