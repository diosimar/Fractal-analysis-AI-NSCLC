# librerias necesarias 
import matplotlib.pyplot as plt
import pandas as pd


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
