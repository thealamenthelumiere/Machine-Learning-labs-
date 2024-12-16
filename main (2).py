import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Загрузка данных
df = pd.read_csv('books.csv')

# Проверка на пропущенные значения
print("Пропущенные значения перед обработкой:\n", df.isnull().sum())
df = df.dropna()  # Удаление строк с пропущенными значениями

# Фильтрация: оставляем категории с >= 3 примерами
category_counts = df['category'].value_counts()
df = df[df['category'].isin(category_counts[category_counts >= 3].index)]
print("Распределение категорий после фильтрации:\n", df['category'].value_counts())

# Извлечение признаков: длина названия и количество слов в описании
df['title_length'] = df['title'].apply(lambda x: len(x.split()))  # Длина заголовка в словах
df['num_words_description'] = df['description'].apply(lambda x: len(x.split()))  # Количество слов в описании

# Baseline: предсказание на основе самой частой категории
baseline_category = df['category'].mode()[0]
baseline_accuracy = (df['category'] == baseline_category).mean()
print(f"Точность базовой модели: {baseline_accuracy:.4f}")

# График 1: Количество слов в описании и жанр
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='category', y='num_words_description')
plt.xticks(rotation=90)
plt.xlabel('Категория (Жанр)')
plt.ylabel('Количество слов в описании')
plt.title('Корреляция между количеством слов в описании и жанром')
plt.show()

# График 2: Количество слов в названии и жанр
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='category', y='title_length')
plt.xticks(rotation=90)
plt.xlabel('Категория (Жанр)')
plt.ylabel('Количество слов в названии')
plt.title('Корреляция между количеством слов в названии и жанром')
plt.show()

# Выбор признаков
X = df[['num_words_description', 'title_length']].values  # Используем количество слов в описании и длину заголовка

# Преобразование категорий в числовые значения (целевая переменная)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['category'])

# Проверка: достаточно ли категорий для стратифицированного разбиения
if len(set(y)) < 3:
    print("Недостаточно категорий для стратифицированного разбиения. Продолжаем без стратификации.")
    stratify_param = None
else:
    stratify_param = y

# Разделение данных на тренировочную, валидационную и тестовую выборки
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=stratify_param)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Инициализация и обучение модели RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, max_depth=5)  # Меньшее количество деревьев и глубина
model.fit(X_train, y_train)

# Подбор гиперпараметров
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Получение лучшей модели
best_model = grid_search.best_estimator_

# Оценка модели
y_train_pred = best_model.predict(X_train)
y_valid_pred = best_model.predict(X_valid)

# Вычисление метрик качества
train_accuracy = accuracy_score(y_train, y_train_pred)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
valid_precision = precision_score(y_valid, y_valid_pred, average='weighted', zero_division=0)
train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
valid_recall = recall_score(y_valid, y_valid_pred, average='weighted', zero_division=0)
train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
valid_f1 = f1_score(y_valid, y_valid_pred, average='weighted', zero_division=0)

print(f'Точность на тренировочной выборке: {train_accuracy}')
print(f'Точность на валидационной выборке: {valid_accuracy}')
print(f'Precision на тренировочной выборке: {train_precision}')
print(f'Precision на валидационной выборке: {valid_precision}')
print(f'Recall на тренировочной выборке: {train_recall}')
print(f'Recall на валидационной выборке: {valid_recall}')
print(f'F1-score на тренировочной выборке: {train_f1}')
print(f'F1-score на валидационной выборке: {valid_f1}')

# Создание и обучение кастомной модели (Simple Random Forest Classifier)
class SimpleRandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            y_sample = y[indices]

            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

# Инициализация и обучение кастомной модели
custom_model = SimpleRandomForestClassifier(n_estimators=20, max_depth=5, min_samples_split=5)
custom_model.fit(X_train, y_train)

# Оценка кастомной модели
y_train_pred_custom = custom_model.predict(X_train)
y_valid_pred_custom = custom_model.predict(X_valid)

train_accuracy_custom = accuracy_score(y_train, y_train_pred_custom)
valid_accuracy_custom = accuracy_score(y_valid, y_valid_pred_custom)

print(f"Точность кастомной модели на тренировочной выборке: {train_accuracy_custom}")
print(f"Точность кастомной модели на валидационной выборке: {valid_accuracy_custom}")
