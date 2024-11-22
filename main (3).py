import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
import numpy as np

# Загрузка данных
df = pd.read_csv('books.csv')

# Проверка на пропущенные значения
print("Пропущенные значения перед обработкой:\n", df.isnull().sum())
df = df.dropna()

# Фильтрация категорий с >= 3 примерами
category_counts = df['category'].value_counts()
df = df[df['category'].isin(category_counts[category_counts >= 3].index)]
print("Распределение категорий после фильтрации:\n", df['category'].value_counts())

# Извлечение признаков: длина названия и количество слов в описании
df['title_length'] = df['title'].apply(lambda x: len(x.split()))
df['num_words_description'] = df['description'].apply(lambda x: len(x.split()))

# Выбор признаков
X = df[['num_words_description', 'title_length']]

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на тренировочную и тестовую выборки
X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=42)

# Baseline: все данные в одном кластере
print("Baseline: все данные в одном кластере.")
baseline_score = -1  # Символическое значение
print(f"Baseline score (символическое значение): {baseline_score}")

# Обучение модели k-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)

# Предсказание кластеров
train_clusters = kmeans.predict(X_train)
test_clusters = kmeans.predict(X_test)

# Оценка качества кластеризации
train_silhouette = silhouette_score(X_train, train_clusters)
train_davies_bouldin = davies_bouldin_score(X_train, train_clusters)
print(f"Silhouette score (train): {train_silhouette:.4f}")
print(f"Davies-Bouldin index (train): {train_davies_bouldin:.4f}")

# Визуализация кластеров с помощью PCA для тренировочной выборки
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=train_clusters, palette='Set1', s=100)
plt.title("Кластеры после K-means (PCA)")
plt.xlabel("Главная компонента 1")
plt.ylabel("Главная компонента 2")
plt.show()

# Подбор гиперпараметров
best_silhouette = -1
best_params = {}

for n_clusters in range(2, 10):
    kmeans_temp = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_temp.fit(X_train)
    silhouette = silhouette_score(X_train, kmeans_temp.labels_)
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_params = {'n_clusters': n_clusters}

print(f"Лучшие гиперпараметры: {best_params}")
print(f"Лучший silhouette score: {best_silhouette:.4f}")

# Кастомная реализация K-means
class CustomKMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(42)
        self.centroids = X[rng.choice(n_samples, self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

        self.labels_ = labels

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# Обучение кастомной модели
custom_kmeans = CustomKMeans(n_clusters=best_params['n_clusters'])
custom_kmeans.fit(X_train)

# Оценка кастомной модели
train_silhouette_custom = silhouette_score(X_train, custom_kmeans.labels_)
train_davies_bouldin_custom = davies_bouldin_score(X_train, custom_kmeans.labels_)

print(f"Silhouette score (custom train): {train_silhouette_custom:.4f}")
print(f"Davies-Bouldin index (custom train): {train_davies_bouldin_custom:.4f}")
