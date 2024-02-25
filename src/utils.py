# librerias necesarias 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro, kstest
import statsmodels

########## EDA Functions to use  #########

def manejo_tipo_dato(df_ ,cols_int , cols_float):
    '''
    Genera cambios  de tipo de  datos  sobre  las  variables  indicadas  que se encuentren en un dataFrame  pandas suministrado.

    Parameters:
    - df_ : El DataFrame que contiene  los datos.
    - cols_int : lista  de con columnas  a  modificar el tipo de  datos  a formato int.
    - cols_float : lista  de con columnas  a  modificar el tipo de  datos  a formato float.

    Returns:
    - df_ : DataFrame pandas con columnas  modificadas en tipo de datos  que fueron especificadas como parámetros . 
    '''
    df_[cols_int ] = df_[cols_int ].astype('int64')
    df_[cols_float] = df_[cols_float].astype('float64')
    return df_


def plot_histogram_(data_frame, column_name, color_, bins=10, figsize=(20, 15)):
    """
    Genera un histograma para una columna específica en un DataFrame.

    Parameters:
    - data_frame (pd.DataFrame): El DataFrame que contiene los datos.
    - column_name (str): El nombre de la columna para la cual se generará el histograma.
    - color_ (str): nombre del color  que se utilizara para pintar el histograma.
    - bins (int, optional): Número de bins en el histograma. Por defecto es 10.
    - figsize (tuple, optional): Tamaño de la figura. Por defecto es (20, 15).
    

    Returns:
    - None: Muestra el histograma en la pantalla.
    """
    plt.figure(figsize=figsize)
    plt.subplot(421)
    plt.hist(data_frame[column_name], bins= bins, color=color_, label=f'Histogram of {column_name}', edgecolor='black')
    plt.legend(loc='best')
    plt.grid(False)
    plt.show()

def plot_frequency(data_frame, column_name):
    """
    Genera un gráfico de barras que muestra la frecuencia de valores en una columna específica.

    Parameters:
    - data_frame (pd.DataFrame): El DataFrame que contiene los datos.
    - column_name (str): El nombre de la columna para la cual se generará el gráfico.

    Returns:
    - None: Muestra el histograma en la pantalla y  tabla con valores por categoria
    """
    # Contar y ordenar los valores
    data_frame[column_name].value_counts().sort_values(inplace=True)

    # Crear el gráfico
    sns.countplot(x=column_name, data=data_frame)

    # Personalizar el gráfico
    plt.title(f'Frecuencia de {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frecuencia')
    plt.grid(False)

    # Mostrar el gráfico
    plt.show()

    # Mostrar la cuenta de valores
    print(data_frame[column_name].value_counts())


def visualize_standardized_boxplot(data, numeric_columns):

    '''
    Genera cambios  sobre  la escala  de  las  variables numericas a análizar y grafica el boxplot de cada variable.

    Parameters:
    - data : El DataFrame que contiene  los datos.
    - numeric_columns: lista  de columnas  a  modificar  y graficar en el boxplot.
    
    Returns:
    - None: Muestra los boxplot en la pantalla.
    '''
    # Estandarizar las columnas numéricas
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data[numeric_columns])

    # Crear un DataFrame con las columnas estandarizadas
    standardized_df = pd.DataFrame(standardized_data, columns=numeric_columns)

    # Convertir el DataFrame a formato largo para seaborn
    standardized_df = pd.melt(standardized_df, var_name="Column", value_name="Value")

    # Configurar el tamaño de la figura
    plt.figure(figsize=(14, 8))

    # Asignar colores diferentes a cada boxplot
    palette = sns.color_palette("husl", len(numeric_columns))

    # Crear el boxplot estandarizado con colores diferentes
    sns.boxplot(x="Column", y="Value", data=standardized_df, palette=palette)

    plt.title("Boxplot de variables numéricas Estandarizadas")
    plt.xticks(rotation=45)
    plt.xlabel('Variables de  interés')
    plt.ylabel('Valores boxplot')
    plt.grid(False)
    plt.show()


def _plot_series(series, series_name, series_index=0):
  
  '''
    Genera grafico de serie temporar .S

    Parameters:
    - series : El DataFrame que contiene  los datos.
    - series_name:nombre de  la serie que se desplega en la legenda del plot.
    - series_index: indexación de  las  series para  graficar de diferente  color  cada tendencia  de las categorias de la lengenda del plot.
    Returns:
    - None: Muestra  de series temporales por categoria en la pantalla.
    '''
  
  palette = list(sns.palettes.mpl_palette('Dark2'))
  xs = series['Survival.time']
  ys = series['age'] 
  plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])
  plt.grid(False)  

def handle_outliers(data, cont_feature_col):
    '''
    Genera analisis de  datos outliers sobre  un conjunto de variables especificadas.

    Parameters:
    - data : El DataFrame que contiene  los datos.
    - cont_feature_col: lista  de columnas  a  analizar y graficar en el barplot con el porcentaje de datos faltantes por columna.
    
    Returns:
    - None: Muestra un barplot en la pantalla e imprime tabla con datos % de datos faltantes por variable analizada.
    '''

    # Subset del DataFrame con columnas continuas
    cont_df = data[cont_feature_col]

    # Calcular el rango intercuartílico (IQR)
    q1 = data[cont_feature_col].quantile(.25)
    q3 = data[cont_feature_col].quantile(.75)
    IQR = q3 - q1

    # Identificar outliers
    outliers_df = np.logical_or((data[cont_feature_col] < (q1 - 1.5 * IQR)), (data[cont_feature_col] > (q3 + 1.5 * IQR)))

    # Lista para almacenar información sobre outliers
    outlier_list = []
    total_outlier = []

    for col in list(outliers_df.columns):
        try:
            total_outlier.append(outliers_df[col].value_counts()[True])
            outlier_list.append((outliers_df[col].value_counts()[True] / outliers_df[col].value_counts().sum()) * 100)
        except:
            outlier_list.append(0)
            total_outlier.append(0)

    # Crear DataFrame con información sobre outliers
    outlier_df = pd.DataFrame(zip(list(outliers_df.columns), total_outlier, outlier_list), columns=['Variables de  interés', 'total', 'outlier(%)'])
    outlier_df.set_index('Variables de  interés', inplace=True)

    # Gráfico de barras
    plt.figure(figsize=(12, 8))
    outlier_df['total'].plot(kind='bar', color='blue', alpha=0.7, label='Total')
    outlier_df['outlier(%)'].plot(secondary_y=True, color='red', marker='o', label='Outlier(%)')

    plt.title('Análisis de outliers')
    plt.xlabel('Variables de  interés')
    plt.ylabel('Total de Outliers')
    plt.legend(loc='upper left', bbox_to_anchor=(0.6, 1.0))
    plt.grid(False) 
    plt.show()

    return outlier_df

def advanced_distribution_analysis(data, numeric_columns):
    '''
    Genera analisis el histograma de frecuencias sobre  un conjunto de variables especificadas.

    Parameters:
    - data : El DataFrame que contiene  los datos.
    - numeric_columns: lista  de columnas  a  analizar y graficar.
    
    Returns:
    - None: Muestra un histplot en la pantalla.
    '''
    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[col], kde=True)
        plt.title(f'Distribución de {col}')
        plt.grid(False) 
        plt.show()

def qq_plot_analysis(data, numeric_columns):
    '''
    Genera analisis el Q-Qplot sobre  un conjunto de variables especificadas.

    Parameters:
    - data : El DataFrame que contiene  los datos.
    - numeric_columns: lista  de columnas  a  analizar y graficar .
    
    Returns:
    - None: Muestra un Q-Qplot en la pantalla.
    '''
    
    for col in numeric_columns:
        plt.figure(figsize=(8, 8))
        qqplot(data[col], line='s')
        plt.title(f'Q-Q Plot de {col}')
        plt.grid(False) 
        plt.show()

def normality_tests(data, numeric_columns):
    '''
    Genera analisis de normalidadsobre  un conjunto de variables especificadas.

    Parameters:
    - data : El DataFrame que contiene  los datos.
    - numeric_columns: lista  de columnas  a  analizar.
    
    Returns:
    - None: Muestra el valor-p asociado a  Shapiro test y K-S test para chequedo de normalidad
    '''
    for col in numeric_columns:
        stat_shapiro, p_value_shapiro = shapiro(data[col])
        stat_kstest, p_value_kstest = kstest(data[col], 'norm')
        print(f'{col}: Shapiro p-value = {p_value_shapiro}, KS p-value = {p_value_kstest}')

def corrdot(*args, **kwargs):
    
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
                vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)        