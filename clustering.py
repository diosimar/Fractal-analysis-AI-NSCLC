import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import  silhouette_score, davies_bouldin_score
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.decomposition import PCA

import  warnings 
warnings.filterwarnings("ignore")

# Cargue de datos a analizar( datos sin imputación)
df_sin_imputaciones = pd.read_csv('Data/df_sin_imputaciones.csv')  
columnas_a_considerar = ['clinical.T.Stage', 'Histology']
df_sin_nan_especifico = df_sin_imputaciones.dropna(subset=columnas_a_considerar)
# dado que solo se  presentan valores  nulos  sobre la variable age,  se  realiza imputación sobre  el promedio
df_sin_nan_especifico['age'] = df_sin_nan_especifico['age'].fillna(df_sin_nan_especifico['age'].median())

# Ajustar  las  variables  categoricas  sobre  escalas numericas
df_encoded  = df_sin_nan_especifico.copy()
# transformación de columnas  nominales
df_encoded = pd.get_dummies(df_encoded , columns=['Histology', 'gender']) # geenrar  varaiables dummies para histology - gender

# Aplicación de Label Encoding a las  variables ordinales
le = LabelEncoder()
df_encoded['Overall.Stage'] = le.fit_transform(df_encoded['Overall.Stage'])

df_encoded.columns  = ['PatientID', 'age', 'clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage', 'Overall.Stage', 'Survival.time',
       'deadstatus.event', 'F.analysis', 'adenocarcinoma', 'large cell', 'nos', 'squamous cell carcinoma', 'female', 'male']
df_encoded.set_index('PatientID', inplace= True)

# Subconjunto del DataFrame con las columnas seleccionadas
cols =  ['age', 'clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage',
       'Overall.Stage',  'F.analysis',
       'adenocarcinoma', 'large cell', 'nos', 'squamous cell carcinoma',
       'female', 'male']

# Crea una instancia de PCA
pca = PCA()
# Ajusta y transforma los datos utilizando PCA
principal_components = pca.fit_transform(df_encoded[cols])
# Convierte los resultados en un DataFrame para mayor claridad
pca_df = pd.DataFrame( data = principal_components, columns = [f'PC{i+1}' for i in range(len(cols))])
# Visualiza la varianza explicada por cada componente principal
explained_variance_ratio = pca.explained_variance_ratio_
#print("Varianza explicada por cada componente principal:", explained_variance_ratio)
# Calcula la variabilidad total explicada
total_variability_explained = np.sum(explained_variance_ratio[:3])
print("Variabilidad total explicada:", total_variability_explained)
# Gráfico de la varianza acumulada
cumulative_explained_variance = explained_variance_ratio.cumsum()
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Número de Componentes Principales')
plt.ylabel('Varianza Acumulada Explicada')
plt.title('Varianza Acumulada Explicada por Componentes Principales')
plt.grid(True)
# Guardar la figura como un archivo JPG
#plt.savefig('output/PCA-Clustering/varianza_acumulada_explicada.jpg')
plt.close()

############################################################################################################
### escalar variables para obtener  la participación de  cada variable  sobre las componentes principales 
# Selecciona las columnas que deseas normalizar (puedes ajustar esto según tus necesidades)
columns_to_normalize = ['PatientID', 'age', 'clinical.T.Stage', 'Clinical.N.Stage', 'Clinical.M.Stage', 'Overall.Stage',
        'F.analysis', 'adenocarcinoma', 'large cell', 'nos', 'squamous cell carcinoma', 'female', 'male']

# Crea un nuevo DataFrame con las columnas normalizadas
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)

# Crea una instancia de PCA
pca = PCA()

# Ajusta y transforma los datos utilizando PCA
principal_components = pca.fit_transform(df_normalized[cols])

# Convierte los resultados en un DataFrame para mayor claridad
pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(len(cols))])

# Obtén los loadings de cada componente principal
loadings = pca.components_
# Crea un DataFrame para visualizar los loadings
loadings_df = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(len(cols))], index=cols)

# Visualiza las variables más importantes para cada componente principal
for i in range(len(cols)):
    important_variables = loadings_df[f'PC{i+1}'].sort_values(ascending=False).index
    
# Grafica los loadings para cada componente principal
plt.figure(figsize=(12, 10))
loadings_df.T.plot(kind='bar', width=1.5)
plt.title('Loadings de Variables en Componentes Principales')
plt.xlabel('Variable')
plt.ylabel('Loading')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
# Guardar la figura como un archivo JPG
#plt.savefig('output/PCA-Clustering/Loadings_Variables_CP1.jpg',  bbox_inches ='tight')
plt.close()

# Crear un biplot
plt.figure(figsize=(10, 8))
#
plt.scatter(pca_df['PC1']/4, pca_df['PC2']/4)

# Flechas para las cargas de las variables originales
feature_vectors = pca.components_.T
for i, v in enumerate(feature_vectors):
    plt.arrow(0, 0, v[0], v[1], color='black', alpha=0.5, linestyle='dotted', linewidth=1)
    plt.text(v[0], v[1],important_variables[i], color='black', ha='right', va='bottom', fontsize = "large")

plt.title('Biplot de PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.grid(True)
# Guardar la figura como un archivo JPG
plt.savefig('output/PCA-Clustering/Bitplot.jpg')
plt.close()
###### cunstrucción de  clusters por kmeans

#_______________________________Clustering K-means___________________________________________#
#_____ elbow methtod to identify the n_cluster to  use
X_ = df_encoded.copy()
kmeans = KMeans(random_state = 42)
visu = KElbowVisualizer(kmeans, k = (2,20), locate_elbow=False, metrics = 'silhouette',timings=False)
visu.fit(X_)
visu.show( outpath = 'output/PCA-Clustering/ElbowPlot.png')#plt.savefig(f'output/PCA-Clustering/ElbowPlot.jpg', bbox_inches ='tight')
plt.close()

distorsions = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_ )
    distorsions.append(kmeans.inertia_)

#fig = plt.figure(figsize=(15, 5))
#plt.plot(range(2, 20), distorsions)
#plt.grid(True)
#plt.title('Elbow curve')
#plt.savefig(f'output/PCA-Clustering/ElbowPlot1.jpg', bbox_inches ='tight')


#_______________
#### make clusters
def plot_metric(K, scores, metric_name):
  plt.figure(dpi=110, figsize=(9, 5))
  plt.plot(K, scores, 'bx-')
  plt.xticks(K); plt.xlabel('$k$', fontdict=dict(family = 'serif', size = 14))  
  plt.ylabel(metric_name, fontdict=dict(family = 'serif', size = 14))
  plt.title(f'K vs {metric_name}', fontdict=dict(family = 'serif', size = 18))
  plt.savefig(f'output/PCA-Clustering/{metric_name}.jpg', bbox_inches ='tight')
  plt.close()
  

silhouette_scores = []
K = range(2,22)
for k in K:
    km = KMeans(n_clusters = k, random_state = 42)
    silhouette_scores.append(silhouette_score(X_, km.fit_predict(X_)))

#plot_metric(K, silhouette_scores, 'silhouette_score ')

#_______________
davies_score = []
K = range(2,21)
for k in K:
    km = KMeans(n_clusters = k, random_state = 42)
    davies_score.append(davies_bouldin_score(X_, km.fit_predict(X_)))

#plot_metric(K, davies_score, 'davies_bouldin_score')

#### formación de  agrupaciones con kmeans

# Aplica K-Means
k =3# Número de clusters
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_.iloc[: , :6])

# Visualiza los resultados
plt.scatter(X_['Survival.time'], X_['F.analysis'], c=labels, cmap='viridis', marker='o', s=50)
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200, label='Centroides')
plt.title('Resultados de K-Means')
plt.xlabel('Survival.time')
plt.ylabel('F.analysis')
plt.legend()
plt.grid(True)
plt.savefig(f'output/PCA-Clustering/kmeans.jpg', bbox_inches ='tight')
plt.close()  