import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
plt.savefig('output/PCA-Clustering/varianza_acumulada_explicada.jpg')

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
plt.savefig('output/PCA-Clustering/Loadings_Variables_CP.jpg')

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