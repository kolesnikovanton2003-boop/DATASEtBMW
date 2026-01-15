# -*- coding: utf-8 -*-
"""
Анализ рынка подержанных автомобилей BMW
Версия с сохранением всех графиков в файлы
"""

# ============================
# 1. ИМПОРТ БИБЛИОТЕК И НАСТРОЙКА
# ============================

import pandas as pd
import numpy as np
import matplotlib

# Используем бэкенд для сохранения без показа
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os
import json
import time

warnings.filterwarnings('ignore')

# Создаем директорию для сохранения графиков
output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)

print(f"Создана директория для графиков: {output_dir}")

# Настройка визуализаций
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['savefig.dpi'] = 150

# Библиотеки для машинного обучения
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix,
                             silhouette_score, davies_bouldin_score)
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

print("Все библиотеки успешно импортированы!")
print("Все графики будут сохраняться в папку 'output_plots'")

# ============================
# 2. СОЗДАНИЕ ТЕСТОВОГО ДАТАСЕТА
# ============================

print("\n" + "=" * 50)
print("СОЗДАНИЕ ТЕСТОВОГО ДАТАСЕТА BMW")
print("=" * 50)

# Создаем тестовый датасет с похожей структурой
np.random.seed(42)
n_samples = 5000

# Список моделей BMW
bmw_models = ['1 Series', '3 Series', '5 Series', '7 Series', 'X1', 'X3', 'X5', 'X6',
              '2 Series', '4 Series', '6 Series', 'X2', 'X4', 'X7', 'Z4', 'i3', 'i8']

# Создаем данные
data = {
    'model': np.random.choice(bmw_models, n_samples, p=[
        0.15, 0.20, 0.15, 0.05, 0.10, 0.08, 0.06, 0.04,
        0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01
    ]),
    'year': np.random.randint(2010, 2023, n_samples),
    'price': np.random.randint(5000, 80000, n_samples),
    'transmission': np.random.choice(['Automatic', 'Manual', 'Semi-Auto'], n_samples, p=[0.7, 0.2, 0.1]),
    'mileage': np.random.randint(1000, 150000, n_samples),
    'fuelType': np.random.choice(['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other'], n_samples,
                                 p=[0.4, 0.5, 0.05, 0.03, 0.02]),
    'tax': np.random.randint(0, 300, n_samples),
    'mpg': np.random.uniform(20, 60, n_samples).round(1),
    'engineSize': np.random.choice([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.4, 5.0, 6.0], n_samples,
                                   p=[0.1, 0.2, 0.25, 0.2, 0.1, 0.05, 0.05, 0.03, 0.02])
}

# Создаем DataFrame
df = pd.DataFrame(data)

# Добавляем реалистичные зависимости
df['price'] = df['price'] + (df['year'] - 2010) * 1500
df['price'] = df['price'] + df['engineSize'] * 2000
df['price'] = df['price'] - df['mileage'] / 1000 * 300
df.loc[df['transmission'] == 'Automatic', 'price'] = df.loc[df['transmission'] == 'Automatic', 'price'] * 1.15
df.loc[df['fuelType'] == 'Diesel', 'price'] = df.loc[df['fuelType'] == 'Diesel', 'price'] * 0.9
df['price'] = df['price'].clip(3000, 120000).astype(int)

print(f"Создан тестовый датасет с {n_samples} записями")

# ============================
# 3. ПРЕДОБРАБОТКА ДАННЫХ
# ============================

print("\n" + "=" * 50)
print("ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 50)

df_processed = df.copy()
current_year = 2023
df_processed['car_age'] = current_year - df_processed['year']

# Бинарные признаки для transmission
df_processed['is_automatic'] = df_processed['transmission'].apply(lambda x: 1 if 'Automatic' in str(x) else 0)
df_processed['is_manual'] = df_processed['transmission'].apply(lambda x: 1 if 'Manual' in str(x) else 0)
df_processed['is_semi_auto'] = df_processed['transmission'].apply(lambda x: 1 if 'Semi-Auto' in str(x) else 0)

# Группировка редких категорий в fuelType
fuel_counts = df_processed['fuelType'].value_counts()
rare_fuels = fuel_counts[fuel_counts < 50].index
df_processed['fuelType_grouped'] = df_processed['fuelType'].apply(
    lambda x: 'Other' if x in rare_fuels else x
)

# Кодирование категориальных переменных
label_encoders = {}
categorical_cols_to_encode = ['model', 'transmission', 'fuelType_grouped']

for col in categorical_cols_to_encode:
    le = LabelEncoder()
    df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

print(f"Размер датасета после обработки: {df_processed.shape}")

# ============================
# 4. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ (EDA)
# ============================

print("\n" + "=" * 50)
print("РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
print("=" * 50)

# 4.1. Распределение цен
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].hist(df_processed['price'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_title('Распределение цен автомобилей')
axes[0].set_xlabel('Цена (£)')
axes[0].set_ylabel('Количество')
axes[0].grid(True, alpha=0.3)

axes[1].boxplot(df_processed['price'], vert=False)
axes[1].set_title('Boxplot цен автомобилей')
axes[1].set_xlabel('Цена (£)')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/1_распределение_цен.png')
plt.close()

price_stats = df_processed['price'].describe()
print(f"\nСтатистика по ценам:")
print(f"  Минимум: £{price_stats['min']:,.0f}")
print(f"  Максимум: £{price_stats['max']:,.0f}")
print(f"  Среднее: £{price_stats['mean']:,.0f}")
print(f"  Медиана: £{price_stats['50%']:,.0f}")

# 4.2. Популярные модели
plt.figure(figsize=(12, 6))
model_counts = df_processed['model'].value_counts().head(10)
bars = plt.bar(model_counts.index, model_counts.values, edgecolor='black', alpha=0.7)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 5,
             f'{int(height)}', ha='center', va='bottom', fontsize=9)
plt.title('Топ-10 самых популярных моделей BMW', fontsize=14)
plt.xlabel('Модель', fontsize=12)
plt.ylabel('Количество автомобилей', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/2_топ_моделей.png')
plt.close()

# 4.3. Зависимость цены от года выпуска
plt.figure(figsize=(12, 6))
plt.scatter(df_processed['year'], df_processed['price'], alpha=0.5, s=20)
z = np.polyfit(df_processed['year'], df_processed['price'], 1)
p = np.poly1d(z)
plt.plot(df_processed['year'].sort_values(), p(df_processed['year'].sort_values()),
         "r--", linewidth=2, label='Линия тренда')
plt.title('Зависимость цены от года выпуска', fontsize=14)
plt.xlabel('Год выпуска', fontsize=12)
plt.ylabel('Цена (£)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/3_цена_от_года.png')
plt.close()

# 4.4. Зависимость цены от пробега
plt.figure(figsize=(12, 6))
plt.scatter(df_processed['mileage'], df_processed['price'], alpha=0.5, s=20)
z = np.polyfit(df_processed['mileage'], df_processed['price'], 1)
p = np.poly1d(z)
plt.plot(df_processed['mileage'].sort_values(), p(df_processed['mileage'].sort_values()),
         "r--", linewidth=2, label='Линия тренда')
plt.title('Зависимость цены от пробега', fontsize=14)
plt.xlabel('Пробег (миль)', fontsize=12)
plt.ylabel('Цена (£)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/4_цена_от_пробега.png')
plt.close()

correlation = df_processed[['mileage', 'price']].corr().iloc[0, 1]
print(f"Корреляция между пробегом и ценой: {correlation:.3f}")

# 4.5. Типы коробки передач
plt.figure(figsize=(10, 6))
transmission_counts = df_processed['transmission'].value_counts()
colors = plt.cm.Set3(np.arange(len(transmission_counts)) / len(transmission_counts))
patches, texts, autotexts = plt.pie(transmission_counts.values, labels=transmission_counts.index,
                                    autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Распределение по типам коробки передач', fontsize=14)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(10)
plt.axis('equal')
plt.tight_layout()
plt.savefig(f'{output_dir}/5_типы_коробки_передач.png')
plt.close()

# 4.6. Матрица корреляций
numeric_features = ['price', 'year', 'mileage', 'tax', 'mpg', 'engineSize', 'car_age']
correlation_matrix = df_processed[numeric_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Матрица корреляций числовых признаков', fontsize=14)
plt.tight_layout()
plt.savefig(f'{output_dir}/6_матрица_корреляций.png')
plt.close()

price_correlations = correlation_matrix['price'].sort_values(ascending=False)
print("\nКорреляция признаков с ценой:")
for feature, corr in price_correlations.items():
    print(f"  {feature:15}: {corr:.3f}")

# ============================
# 5. РЕГРЕССИОННЫЙ АНАЛИЗ
# ============================

print("\n" + "=" * 50)
print("РЕГРЕССИОННЫЙ АНАЛИЗ")
print("=" * 50)

# Подготовка данных
features = [
    'year', 'mileage', 'tax', 'mpg', 'engineSize', 'car_age',
    'is_automatic', 'is_manual', 'is_semi_auto',
    'model_encoded', 'fuelType_grouped_encoded'
]

available_features = [f for f in features if f in df_processed.columns]
X = df_processed[available_features]
y = df_processed['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Размеры выборок: обучающая - {X_train.shape[0]}, тестовая - {X_test.shape[0]}")

# Обучение и оценка моделей
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1, max_iter=10000),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
}

results = []

for name, model in models.items():
    if name in ['Decision Tree', 'Random Forest', 'Gradient Boosting']:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    results.append({
        'Модель': name,
        'R² Train': r2_train,
        'R² Test': r2_test,
        'RMSE Test': rmse_test
    })

    print(f"\n{name}:")
    print(f"  R² на обучении: {r2_train:.4f}")
    print(f"  R² на тесте: {r2_test:.4f}")
    print(f"  RMSE на тесте: £{rmse_test:,.2f}")

# Создаем DataFrame с результатами
results_df = pd.DataFrame(results).sort_values('R² Test', ascending=False)

print("\n" + "=" * 60)
print("СРАВНЕНИЕ МОДЕЛЕЙ (отсортировано по R² на тесте):")
print("=" * 60)
print(results_df.to_string(index=False))

# Визуализация сравнения моделей
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
x = range(len(results_df))
width = 0.35

axes[0].bar([i - width / 2 for i in x], results_df['R² Train'], width, label='Train', alpha=0.8)
axes[0].bar([i + width / 2 for i in x], results_df['R² Test'], width, label='Test', alpha=0.8)
axes[0].set_xlabel('Модель', fontsize=12)
axes[0].set_ylabel('R² Score', fontsize=12)
axes[0].set_title('Сравнение R² моделей регрессии', fontsize=14)
axes[0].set_xticks(x)
axes[0].set_xticklabels(results_df['Модель'], rotation=45, ha='right')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(x, results_df['RMSE Test'], width, alpha=0.8, color='orange')
axes[1].set_xlabel('Модель', fontsize=12)
axes[1].set_ylabel('RMSE (£)', fontsize=12)
axes[1].set_title('RMSE моделей на тестовой выборке', fontsize=14)
axes[1].set_xticks(x)
axes[1].set_xticklabels(results_df['Модель'], rotation=45, ha='right')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/7_сравнение_моделей_регрессии.png')
plt.close()

# Анализ важности признаков
best_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'Признак': available_features,
    'Важность': best_model.feature_importances_
}).sort_values('Важность', ascending=False)

print("\nВажность признаков (Random Forest):")
print(feature_importance.to_string(index=False))

plt.figure(figsize=(10, 6))
bars = plt.barh(feature_importance['Признак'][:10],
                feature_importance['Важность'][:10],
                edgecolor='black')
plt.xlabel('Важность признака', fontsize=12)
plt.title('Топ-10 самых важных признаков для прогнозирования цены', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2,
             f'{width:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/8_важность_признаков.png')
plt.close()

# ============================
# 6. КЛАССИФИКАЦИЯ
# ============================

print("\n" + "=" * 50)
print("КЛАССИФИКАЦИЯ")
print("=" * 50)

# Создание целевой переменной для классификации
price_percentiles = df_processed['price'].quantile([0.33, 0.67])


def price_to_class(price):
    if price <= price_percentiles[0.33]:
        return 0  # Дешевые
    elif price <= price_percentiles[0.67]:
        return 1  # Средние
    else:
        return 2  # Дорогие


df_processed['price_class'] = df_processed['price'].apply(price_to_class)
class_names = {0: 'Дешевые', 1: 'Средние', 2: 'Дорогие'}

# Распределение классов
plt.figure(figsize=(8, 5))
class_distribution = df_processed['price_class'].value_counts().sort_index()
colors = ['lightblue', 'lightgreen', 'salmon']
plt.bar(class_names.values(), class_distribution.values, color=colors, edgecolor='black', alpha=0.8)

for i, (name, count) in enumerate(zip(class_names.values(), class_distribution.values)):
    plt.text(i, count + 50, f'{count}\n({count / len(df_processed) * 100:.1f}%)',
             ha='center', va='bottom', fontsize=10)

plt.title('Распределение автомобилей по ценовым категориям', fontsize=14)
plt.xlabel('Ценовая категория', fontsize=12)
plt.ylabel('Количество автомобилей', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/9_распределение_классов.png')
plt.close()

# Разделение данных для классификации
X_clf = df_processed[available_features]
y_clf = df_processed['price_class']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

scaler_clf = StandardScaler()
X_train_scaled_clf = scaler_clf.fit_transform(X_train_clf)
X_test_scaled_clf = scaler_clf.transform(X_test_clf)

# Обучение моделей классификации
clf_models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_split=20, min_samples_leaf=10, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=9, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
}

results_clf = []

for name, model in clf_models.items():
    if name == 'KNN':
        model.fit(X_train_scaled_clf, y_train_clf)
        y_pred_train = model.predict(X_train_scaled_clf)
        y_pred_test = model.predict(X_test_scaled_clf)
    else:
        model.fit(X_train_clf, y_train_clf)
        y_pred_train = model.predict(X_train_clf)
        y_pred_test = model.predict(X_test_clf)

    train_accuracy = accuracy_score(y_train_clf, y_pred_train)
    test_accuracy = accuracy_score(y_test_clf, y_pred_test)

    results_clf.append({
        'Модель': name,
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy
    })

    print(f"\n{name}:")
    print(f"  Точность на обучении: {train_accuracy:.4f}")
    print(f"  Точность на тесте: {test_accuracy:.4f}")

# Сравнение моделей классификации
comparison_clf = pd.DataFrame(results_clf).sort_values('Test Accuracy', ascending=False)

print("\nСравнение моделей классификации:")
print(comparison_clf.to_string(index=False))

# Визуализация сравнения
plt.figure(figsize=(10, 5))
x = range(len(comparison_clf))
width = 0.35

plt.bar([i - width / 2 for i in x], comparison_clf['Train Accuracy'],
        width, label='Train Accuracy', alpha=0.8, edgecolor='black')
plt.bar([i + width / 2 for i in x], comparison_clf['Test Accuracy'],
        width, label='Test Accuracy', alpha=0.8, edgecolor='black')

plt.xlabel('Модель', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Сравнение точности моделей классификации', fontsize=14)
plt.xticks(x, comparison_clf['Модель'])
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

for i, (train_acc, test_acc) in enumerate(zip(comparison_clf['Train Accuracy'], comparison_clf['Test Accuracy'])):
    plt.text(i - width / 2, train_acc + 0.01, f'{train_acc:.3f}', ha='center', va='bottom', fontsize=9)
    plt.text(i + width / 2, test_acc + 0.01, f'{test_acc:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/10_сравнение_классификаторов.png')
plt.close()

# ============================
# 7. КЛАСТЕРНЫЙ АНАЛИЗ (ИСПРАВЛЕННЫЙ)
# ============================

print("\n" + "=" * 50)
print("КЛАСТЕРНЫЙ АНАЛИЗ: K-MEANS")
print("=" * 50)

# Подготовка данных для кластеризации
clustering_features = ['price', 'year', 'mileage', 'engineSize', 'mpg', 'car_age']
clustering_data = df_processed[clustering_features].copy()
scaler_cluster = StandardScaler()
clustering_scaled = scaler_cluster.fit_transform(clustering_data)

# Определение оптимального числа кластеров
inertia_values = []
silhouette_scores = []
k_range = range(2, 11)

print("Вычисление метрик для разных значений k...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(clustering_scaled)
    inertia_values.append(kmeans.inertia_)

    # Для силуэтного коэффициента нужно минимум 2 кластера
    if k > 1:
        silhouette_avg = silhouette_score(clustering_scaled, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        print(f"  k={k}: инерция={kmeans.inertia_:.2f}, силуэт={silhouette_avg:.4f}")

# Метод локтя и силуэтный анализ (ИСПРАВЛЕНО)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Метод локтя - используем все k от 2 до 10
axes[0].plot(k_range, inertia_values, 'bo-', linewidth=2)
axes[0].set_xlabel('Число кластеров (k)', fontsize=12)
axes[0].set_ylabel('Инерция', fontsize=12)
axes[0].set_title('Метод локтя для K-Means', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Силуэтный анализ - используем те же k (от 2 до 10)
axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2)
axes[1].set_xlabel('Число кластеров (k)', fontsize=12)
axes[1].set_ylabel('Силуэтный коэффициент', fontsize=12)
axes[1].set_title('Силуэтный анализ', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/11_определение_числа_кластеров.png')
plt.close()

# Находим оптимальное k по максимальному силуэтному коэффициенту
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"\nОптимальное число кластеров по силуэтному коэффициенту: {optimal_k_silhouette}")
print(f"Максимальный силуэтный коэффициент: {max(silhouette_scores):.4f}")

# Применение K-Means с оптимальным k
kmeans_final = KMeans(n_clusters=optimal_k_silhouette, random_state=42, n_init=10)
kmeans_final.fit(clustering_scaled)
df_processed['cluster'] = kmeans_final.labels_

print(f"Распределение по кластерам:")
cluster_distribution = pd.Series(kmeans_final.labels_).value_counts().sort_index()
for cluster_id, count in cluster_distribution.items():
    percentage = (count / len(clustering_data)) * 100
    print(f"  Кластер {cluster_id}: {count} авто ({percentage:.1f}%)")

# Визуализация кластеров с помощью PCA
pca = PCA(n_components=2)
clustering_pca = pca.fit_transform(clustering_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(clustering_pca[:, 0], clustering_pca[:, 1],
                      c=kmeans_final.labels_, cmap='tab10',
                      alpha=0.6, s=30, edgecolors='w', linewidth=0.5)
plt.xlabel('Первая главная компонента (PC1)', fontsize=12)
plt.ylabel('Вторая главная компонента (PC2)', fontsize=12)
plt.title(f'Визуализация {optimal_k_silhouette} кластеров с помощью PCA', fontsize=14)
plt.colorbar(scatter, label='Кластер')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/12_визуализация_кластеров.png')
plt.close()

print(f"Объясненная дисперсия PCA:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"  Суммарно: {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]):.2%}")

# Анализ характеристик кластеров
cluster_analysis = clustering_data.copy()
cluster_analysis['cluster'] = kmeans_final.labels_

print("\nСтатистика по кластерам:")
cluster_stats = cluster_analysis.groupby('cluster').agg({
    'price': ['mean', 'std', 'min', 'max'],
    'year': 'mean',
    'mileage': 'mean',
    'engineSize': 'mean',
    'car_age': 'mean'
}).round(2)

print(cluster_stats)

# Интерпретация кластеров
print("\nИНТЕРПРЕТАЦИЯ КЛАСТЕРОВ:")
for cluster_id in range(optimal_k_silhouette):
    cluster_data = cluster_analysis[cluster_analysis['cluster'] == cluster_id]

    avg_price = cluster_data['price'].mean()
    avg_year = cluster_data['year'].mean()
    avg_mileage = cluster_data['mileage'].mean()
    avg_engine = cluster_data['engineSize'].mean()
    avg_age = cluster_data['car_age'].mean()

    print(f"\nКластер {cluster_id} ({len(cluster_data)} авто):")
    print(f"  Средняя цена: £{avg_price:,.0f}")
    print(f"  Средний год выпуска: {avg_year:.1f}")
    print(f"  Средний пробег: {avg_mileage:,.0f} миль")
    print(f"  Средний объем двигателя: {avg_engine:.1f} л")
    print(f"  Средний возраст: {avg_age:.1f} лет")

    # Интерпретация
    if avg_price < 20000:
        price_cat = "бюджетные"
    elif avg_price < 40000:
        price_cat = "средние"
    else:
        price_cat = "премиум"

    if avg_age < 3:
        age_cat = "новые"
    elif avg_age < 7:
        age_cat = "среднего возраста"
    else:
        age_cat = "старые"

    print(f"  Характеристика: {age_cat} автомобили {price_cat} класса")

# ============================
# 8. ИТОГОВЫЕ ВЫВОДЫ И СОХРАНЕНИЕ
# ============================

print("\n" + "=" * 70)
print("ИТОГОВЫЕ ВЫВОДЫ И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 70)

# Создание итогового графика
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Сравнение моделей регрессии
axes[0, 0].bar(range(len(results_df)), results_df['R² Test'], alpha=0.7, edgecolor='black')
axes[0, 0].set_xticks(range(len(results_df)))
axes[0, 0].set_xticklabels(results_df['Модель'], rotation=45, ha='right')
axes[0, 0].set_ylabel('R² Score', fontsize=12)
axes[0, 0].set_title('Сравнение моделей регрессии', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)

# 2. Важность признаков
top_features = feature_importance.head(8)
axes[0, 1].barh(range(len(top_features)), top_features['Важность'], alpha=0.7, edgecolor='black')
axes[0, 1].set_yticks(range(len(top_features)))
axes[0, 1].set_yticklabels(top_features['Признак'])
axes[0, 1].set_xlabel('Важность признака', fontsize=12)
axes[0, 1].set_title('Топ-8 важных признаков', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# 3. Сравнение моделей классификации
axes[1, 0].bar(range(len(comparison_clf)), comparison_clf['Test Accuracy'],
               alpha=0.7, edgecolor='black', color='green')
axes[1, 0].set_xticks(range(len(comparison_clf)))
axes[1, 0].set_xticklabels(comparison_clf['Модель'], rotation=45, ha='right')
axes[1, 0].set_ylabel('Accuracy', fontsize=12)
axes[1, 0].set_title('Сравнение моделей классификации', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# 4. Визуализация кластеров
axes[1, 1].scatter(clustering_pca[:, 0], clustering_pca[:, 1],
                   c=kmeans_final.labels_, cmap='tab10', alpha=0.6, s=20)
axes[1, 1].set_xlabel('PC1', fontsize=12)
axes[1, 1].set_ylabel('PC2', fontsize=12)
axes[1, 1].set_title('Визуализация кластеров', fontsize=14)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Итоговые результаты анализа рынка подержанных BMW', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/13_итоговые_результаты.png', dpi=150, bbox_inches='tight')
plt.close()

# Сохранение данных
df_processed.to_csv('bmw_processed.csv', index=False)
print("✓ Обработанные данные сохранены в файл: bmw_processed.csv")

# Сохранение ключевых метрик
summary_stats = {
    'total_cars': len(df_processed),
    'avg_price': float(df_processed['price'].mean()),
    'avg_year': float(df_processed['year'].mean()),
    'avg_mileage': float(df_processed['mileage'].mean()),
    'best_regression_model': results_df.iloc[0]['Модель'],
    'best_regression_r2': float(results_df.iloc[0]['R² Test']),
    'best_classification_model': comparison_clf.iloc[0]['Модель'],
    'best_classification_accuracy': float(comparison_clf.iloc[0]['Test Accuracy']),
    'optimal_clusters': int(optimal_k_silhouette),
    'dataset_created': True,
    'graphs_saved_in': output_dir
}

with open('project_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, indent=2, ensure_ascii=False)

print("✓ Ключевые метрики сохранены в файл: project_summary.json")

# Вывод списка созданных файлов
print("\n" + "=" * 70)
print("СОЗДАННЫЕ ФАЙЛЫ:")
print("=" * 70)
print("1. bmw_processed.csv - обработанный датасет")
print("2. project_summary.json - ключевые метрики проекта")
print(f"3. Папка '{output_dir}/' содержит следующие графики:")

graph_files = os.listdir(output_dir)
for i, file in enumerate(sorted(graph_files), 1):
    print(f"   {i:2}. {file}")

print("\n" + "=" * 70)
print("ОСНОВНЫЕ ВЫВОДЫ:")
print("=" * 70)
print("1. Наиболее важные факторы, влияющие на цену:")
print("   - Пробег (корреляция: -0.466)")
print("   - Год выпуска (корреляция: 0.207)")
print("   - Возраст автомобиля (корреляция: -0.207)")

print("\n2. Лучшая модель для прогнозирования цен:")
print(f"   - {results_df.iloc[0]['Модель']} (R² на тесте: {results_df.iloc[0]['R² Test']:.4f})")

print("\n3. Лучшая модель для классификации:")
print(f"   - {comparison_clf.iloc[0]['Модель']} (Accuracy на тесте: {comparison_clf.iloc[0]['Test Accuracy']:.4f})")

print("\n4. Результаты кластеризации:")
print(f"   - Оптимальное число кластеров: {optimal_k_silhouette}")
print("   - Кластеры хорошо разделяют автомобили по цене, возрасту и пробегу")

print("\n" + "=" * 70)
print("ПРОЕКТ УСПЕШНО ЗАВЕРШЕН!")
print("=" * 70)
