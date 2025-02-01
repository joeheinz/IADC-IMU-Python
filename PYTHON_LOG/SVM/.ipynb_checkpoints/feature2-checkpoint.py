import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main1(): 
    # ---------- CONFIGURACIÓN ----------
    DATASET_PATH = '/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/Datasets/acceleration'
    LABELS = {'Fall': 1, 'Idle': -1}  # Directorios y etiquetas
    N_COMPONENTS = 2  # Número de componentes PCA
    #SVM_MODEL_PATH = os.path.join(DATASET_PATH, 'svm_model.pkl')  # Modelo guardado

    # ---------- CARGAR DATOS ----------
    def load_data():
        """Carga los datos desde los CSV en los directorios Fall e Idle."""
        data_list = []
        labels = []

        for label_name, label_value in LABELS.items():
            folder_path = os.path.join(DATASET_PATH, label_name)

            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    # Cargar CSV
                    file_path = os.path.join(folder_path, file)
                    print(f"Cargando: {file_path}")
                    df = pd.read_csv(file_path)

                    # Eliminar Timestamp
                    df = df.drop(columns=['Timestamp'])

                    # Añadir datos y etiquetas
                    data_list.append(df.values)
                    labels.append(np.full(df.shape[0], label_value))  # Etiqueta por fila

        # Combinar todos los datos
        X = np.vstack(data_list)
        y = np.concatenate(labels)
        return X, y

    # ---------- PROCESAR Y ENTRENAR ----------
    def train_svm():
        """Realiza PCA y entrena un modelo SVM."""
        # Cargar datos
        X, y = load_data()

        # Escalar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Aplicar PCA
        pca = PCA(n_components=N_COMPONENTS)
        X_pca = pca.fit_transform(X_scaled)

        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

        # Entrenar SVM
        svm = SVC(kernel='rbf', C=1.0, gamma=0.3)
        svm.fit(X_train, y_train)

        # Evaluar modelo
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del SVM: {accuracy:.2f}")

        # Guardar modelo
        #joblib.dump(svm, SVM_MODEL_PATH)
        #print(f"Modelo SVM guardado en: {SVM_MODEL_PATH}")

        # Devolver objetos para futuras predicciones

    # ---------- GRAFICAR 2D ----------
        plt.figure(figsize=(10, 8))

        # Colores según etiquetas reales
        colors = ['red' if label == 1 else 'blue' for label in y]

        # Formas según predicciones del SVM
        predictions = svm.predict(X_pca)
        shapes = ['X' if pred == 1 else 'o' for pred in predictions]

        # Graficar puntos
        for i in range(len(X_pca)):
            plt.scatter(X_pca[i, 0], X_pca[i, 1], c=colors[i], marker=shapes[i], alpha=0.6, edgecolors='k')

        # ---------- GRAFICAR FRONTERA DE DECISIÓN ----------
        # Crear malla en 2D
        x_range = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)
        y_range = np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 100)
        xx, yy = np.meshgrid(x_range, y_range)
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(grid).reshape(xx.shape)

        # Dibujar contornos de decisión
        plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.3)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')  # Frontera de decisión

        # Etiquetas y título
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.title('SVM Decision Boundary con PCA (2D)')

        # Leyenda
        plt.scatter([], [], c='red', marker='X', label='Fall (Predicted)')
        plt.scatter([], [], c='blue', marker='o', label='Idle (Predicted)')
        plt.scatter([], [], c='red', marker='o', label='Fall (Real)')
        plt.scatter([], [], c='blue', marker='o', label='Idle (Real)')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return pca, scaler, svm

    # ---------- EJECUTAR ENTRENAMIENTO ----------

    pca_model, scaler_model, svm_model = train_svm()
    return 0 

def main2(): 
        # ---------- CONFIGURACIÓN ----------
    DATASET_PATH = '/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/Datasets/acceleration'
    LABELS = {'Fall': 1, 'Idle': -1}  # Directorios y etiquetas
    N_COMPONENTS = 2  # Número de componentes PCA
    #SVM_MODEL_PATH = os.path.join(DATASET_PATH, 'svm_model.pkl')  # Modelo guardado

    # ---------- CARGAR DATOS ----------
    def load_data():
        """Carga los datos desde los CSV en los directorios Fall e Idle."""
        data_list = []
        labels = []

        for label_name, label_value in LABELS.items():
            folder_path = os.path.join(DATASET_PATH, label_name)

            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    # Cargar CSV
                    file_path = os.path.join(folder_path, file)
                    print(f"Cargando: {file_path}")
                    df = pd.read_csv(file_path)

                    # Eliminar Timestamp
                    df = df.drop(columns=['Timestamp'])

                    # Añadir datos y etiquetas
                    data_list.append(df.values)
                    labels.append(np.full(df.shape[0], label_value))  # Etiqueta por fila

        # Combinar todos los datos
        X = np.vstack(data_list)
        y = np.concatenate(labels)
        return X, y

    # ---------- PROCESAR Y ENTRENAR ----------
    def train_svm():
        """Realiza PCA y entrena un modelo SVM."""
        # Cargar datos
        X, y = load_data()

        # Escalar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Aplicar PCA
        pca = PCA(n_components=N_COMPONENTS)
        X_pca = pca.fit_transform(X_scaled)

        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

        # Entrenar SVM
        svm = SVC(kernel='rbf', C=1.0, gamma='auto')
        svm.fit(X_train, y_train)

        # Evaluar modelo
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del SVM: {accuracy:.2f}")

        # Guardar modelo
        #joblib.dump(svm, SVM_MODEL_PATH)
        #print(f"Modelo SVM guardado en: {SVM_MODEL_PATH}")

        # Devolver objetos para futuras predicciones

    # ---------- GRAFICAR 2D ----------
        plt.figure(figsize=(10, 8))

        # Colores según etiquetas reales
        colors = ['red' if label == 1 else 'blue' for label in y]

        # Formas según predicciones del SVM
        predictions = svm.predict(X_pca)
        shapes = ['X' if pred == 1 else 'o' for pred in predictions]

        # Graficar puntos
        for i in range(len(X_pca)):
            plt.scatter(X_pca[i, 0], X_pca[i, 1], c=colors[i], marker=shapes[i], alpha=0.6, edgecolors='k')

        # ---------- GRAFICAR FRONTERA DE DECISIÓN ----------
        # Crear malla en 2D
        x_range = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)
        y_range = np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 100)
        xx, yy = np.meshgrid(x_range, y_range)
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = svm.decision_function(grid).reshape(xx.shape)

        # Dibujar contornos de decisión
        plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.3)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')  # Frontera de decisión

        # Etiquetas y título
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.title('SVM Decision Boundary con PCA (2D)')

        # Leyenda
        plt.scatter([], [], c='red', marker='X', label='Fall (Predicted)')
        plt.scatter([], [], c='blue', marker='o', label='Idle (Predicted)')
        plt.scatter([], [], c='red', marker='o', label='Fall (Real)')
        plt.scatter([], [], c='blue', marker='o', label='Idle (Real)')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return pca, scaler, svm

    # ---------- EJECUTAR ENTRENAMIENTO ----------

    pca_model, scaler_model, svm_model = train_svm()
    
    
    
    
    

def main2(): 
    # ---------- CONFIGURACIÓN ----------
    DATASET_PATH = '/Users/joseheinz/Documents/Arbeit/G2/Silabs/PYTHON_LOG/SVM/Datasets/acceleration'
    LABELS = {'Fall': 1, 'Idle': -1}  # Directorios y etiquetas
    N_COMPONENTS = 3  # PCA con 3 componentes para entrenamiento
    TEST_SIZE = 0.3  # Tamaño de prueba

    # ---------- CARGAR DATOS ----------
    def load_data():
        """Carga los datos desde los CSV en los directorios Fall e Idle."""
        data_list = []
        labels = []

        for label_name, label_value in LABELS.items():
            folder_path = os.path.join(DATASET_PATH, label_name)

            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    # Cargar CSV
                    file_path = os.path.join(folder_path, file)
                    print(f"Cargando: {file_path}")
                    df = pd.read_csv(file_path)

                    # Eliminar Timestamp
                    df = df.drop(columns=['Timestamp'])

                    # Añadir datos y etiquetas
                    data_list.append(df.values)
                    labels.append(np.full(df.shape[0], label_value))  # Etiqueta por fila

        # Combinar todos los datos
        X = np.vstack(data_list)
        y = np.concatenate(labels)
        return X, y

    # ---------- PCA Y ENTRENAMIENTO ----------
    def train_svm():
        """Realiza PCA y entrena un modelo SVM."""
        # Cargar datos
        X, y = load_data()

        # Escalar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Aplicar PCA (3 componentes)
        pca = PCA(n_components=N_COMPONENTS)
        X_pca = pca.fit_transform(X_scaled)

        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=TEST_SIZE, random_state=42)

        # Entrenar SVM
        svm = SVC(kernel='rbf', C=1.0, gamma=10)
        svm.fit(X_train, y_train)

        # Evaluar modelo
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del SVM: {accuracy:.2f}")

        # ---------- GRAFICAR 2D ----------
        plt.figure(figsize=(10, 8))

        # Colores según etiquetas reales
        colors = ['red' if label == 1 else 'blue' for label in y]

        # Formas según predicciones del SVM
        predictions = svm.predict(X_pca)
        shapes = ['X' if pred == 1 else 'o' for pred in predictions]

        # Graficar puntos usando SOLO las primeras 2 componentes
        for i in range(len(X_pca)):
            plt.scatter(X_pca[i, 0], X_pca[i, 1], c=colors[i], marker=shapes[i], alpha=0.6, edgecolors='k')

        # ---------- GRAFICAR FRONTERA DE DECISIÓN ----------
        # Crear malla en 2D
        x_range = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)
        y_range = np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 100)
        xx, yy = np.meshgrid(x_range, y_range)
        grid = np.c_[xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())]  # Proyectar en Z = 0
        Z = svm.decision_function(grid).reshape(xx.shape)

        # Dibujar contornos de decisión
        plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.3)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')  # Frontera de decisión

        # Etiquetas y título
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.title('SVM Decision Boundary con PCA (2D)')

        # Leyenda
        plt.scatter([], [], c='red', marker='X', label='Fall (Predicted)')
        plt.scatter([], [], c='blue', marker='o', label='Idle (Predicted)')
        plt.scatter([], [], c='red', marker='o', label='Fall (Real)')
        plt.scatter([], [], c='blue', marker='o', label='Idle (Real)')
        plt.legend()

        plt.tight_layout()
        plt.show()

        return pca, scaler, svm

    # ---------- EJECUTAR ENTRENAMIENTO ----------
    pca_model, scaler_model, svm_model = train_svm()
    return 0


# ---------- EJECUTAR ENTRENAMIENTO ----------
if __name__ == "__main__":
    main1()
    #main2()