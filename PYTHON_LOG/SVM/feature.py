import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# ---------- CONFIGURACIÓN ----------
# Ruta del archivo CSV
file_path = '/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/Datasets/acceleration/feature_extraction.csv'

# ---------- CARGAR DATOS ----------
# Cargar el CSV
data = pd.read_csv(file_path)

# Preparar los datos
X = data[['Softmax_Fall', 'Softmax_Idle', 'Acc_Betrag_mu', 'Acc_Betrag_sigma',
          'Ori_Betrag_mu', 'Ori_Betrag_sigma', 'Acc_Peak_Frequency']].values
y = data['Label'].apply(lambda x: 1 if x == 'FallEnlarged' else -1).values

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""
# ---------- PCA ----------
pca = PCA(n_components=3)  # Reducir a 3 dimensiones
X_pca = pca.fit_transform(X_scaled)
"""

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ---------- ENTRENAR SVM ----------
svm = SVC(kernel='rbf', C=1.0, gamma='auto', probability=True)
svm.fit(X_train, y_train)

# Evaluar el modelo
train_acc = svm.score(X_train, y_train)
test_acc = svm.score(X_test, y_test)
print(f'Precisión en entrenamiento: {train_acc:.2f}')
print(f'Precisión en prueba: {test_acc:.2f}')


import joblib

# Guardar el modelo SVM entrenado
svm_model_path = '/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/svm_model.pkl'
joblib.dump(svm, svm_model_path)
print(f"Modelo SVM guardado en: {svm_model_path}")

"""
# ---------- GRAFICAR 3D ----------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Colores según etiquetas reales
colors = ['red' if label == 1 else 'blue' for label in y]

# Formas según predicciones del SVM
predictions = svm.predict(X_pca)
shapes = ['X' if pred == 1 else 'o' for pred in predictions]


# Graficar puntos
for i in range(len(X_pca)):
    ax.scatter(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], c=colors[i], marker=shapes[i], alpha=0.6, edgecolors='k')

# ---------- GRAFICAR FRONTERA DE DECISIÓN ----------
# Crear malla en 2D para proyectar la frontera sobre un plano fijo (Z = 0)
x_range = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 30)
y_range = np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 30)
xx, yy = np.meshgrid(x_range, y_range)

# Evaluar modelo en cada punto de la malla
grid = np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]  # Proyectar en Z = 0
grid_predictions = svm.decision_function(grid).reshape(xx.shape)

# Dibujar contorno 2D proyectado en 3D
ax.contourf(xx, yy, grid_predictions, zdir='z', offset=X_pca[:, 2].min(), levels=20, cmap='coolwarm', alpha=0.3)

# ---------- ETIQUETAS ----------
ax.set_xlabel('Componente Principal 1')
ax.set_ylabel('Componente Principal 2')
ax.set_zlabel('Componente Principal 3')
ax.set_title('SVM Decision Boundary con PCA (3D)')

# Leyenda
scatter1 = ax.scatter([], [], [], c='red', marker='X', label='FallEnlarged (Predicted)')
scatter2 = ax.scatter([], [], [], c='blue', marker='o', label='IdleEnlarged (Predicted)')
scatter3 = ax.scatter([], [], [], c='red', marker='o', label='FallEnlarged (Real)')
scatter4 = ax.scatter([], [], [], c='blue', marker='o', label='IdleEnlarged (Real)')
ax.legend(handles=[scatter1, scatter2, scatter3, scatter4])

plt.tight_layout()
plt.show()
"""