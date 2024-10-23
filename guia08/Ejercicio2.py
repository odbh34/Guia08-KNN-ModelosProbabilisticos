from collections import Counter
import math
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from Ejercicio1 import *

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas

# Preparar el conjunto de datos para KNN
data = [list(X[i]) + [y[i]] for i in range(len(X))]  # Combina características y etiquetas

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lista para almacenar las precisiones
accuracies = []

# Probar diferentes valores de k
for k in range(1, 11):
    y_pred = []
    for query in X_test:
        _, predicted_label = knn(data, query.tolist() + [None], k, euclidean_distance, mode)
        y_pred.append(predicted_label)

    # Calcular la precisión
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Visualizar los resultados
plt.plot(range(1, 11), accuracies, marker='o')
plt.title('Precisión del modelo KNN para diferentes valores de k (Iris Dataset)')
plt.xlabel('Valor de k')
plt.ylabel('Precisión')
plt.xticks(range(1, 11))
plt.grid()
plt.show()

if __name__ == '__main__':
    # Ejemplo de predicción con un nuevo dato
    new_data = [[5.0, 3.5, 1.3, 0.3]]  # Ejemplo de características de una flor de iris
    # Usar el modelo KNN para predecir la especie basándose en las características proporcionadas
    predicted_species = knn(data, new_data[0] + [None], 3, euclidean_distance, mode)[1]
    print(f'La especie de iris predicha es: {iris.target_names[predicted_species]}')