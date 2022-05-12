# RoberTweet-Large

RoberTweet-Large es un modelo capaz de clasificar tweets en Inglés a 3 niveles: positive, negative y neutral. Para realizar predicciones con el, basta con realizar los siguientes pasos:


-->Descargar el modelo ".pt" que se encuentra almacenado en este enlace --> https://drive.google.com/drive/folders/1zJlPoYebKj4lps99X_eXVq1lVjD3Sguq?usp=sharing 

-->Descargar el archivo.ipynb de este repositorio que contiene el codigo necesario para realizar predicciones con el modelo y abrirlo con google colaboratory.

Es necesario modificar los siguientes parametros teniendo en cuenta lo siguiente:

-->El archivo que se le pase al modelo, debe ser un archivo ".csv" que contenga los tweets en una cabecera llamada "review".

-->En "file_path"--> Indicaremos la ruta donde hemos guardado el archivo .pt que nos hemos descargado desde este repositorio y que contiene el modelo.

-->En "Tweets"--> Indicaremos la ruta donde se encuentra el archivo .csv que contiene los tweets que se quieren clasificar.

Los resultados se encontraran en un array llamado "preds" que contendra la etiqueta asignada a cada tweet en el mismo orden que se encuentran los tweets en el archivo "Tweets".

Este array se puede postprocesar de la manera que se desee para trabajar con los resultados.
