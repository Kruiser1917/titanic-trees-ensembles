# Titanic — Деревья решений и ансамбли (Bagging & Gradient Boosting)

Практическое исследование **Темы 06 (Деревья решений и ансамбли)** модуля **04 Machine Learning** на классическом табличном датасете [Kaggle Titanic — Machine Learning from Disaster](https://www.kaggle.com/c/titanic) (`891` строка × `12` столбцов).

## Описание проекта

Цель работы — построить, проинтерпретировать и сравнить одиночные деревья решений (`DecisionTreeClassifier`), случайный лес (`RandomForestClassifier`) и три современные реализации градиентного бустинга над деревьями (`XGBoost`, `LightGBM`, `CatBoost`) в задаче бинарной классификации выживаемости пассажиров «Титаника» (`Survived` $\in \{0, 1\}$):

1. **Подготовка эксперимента и защита от утечки данных (`Data Leakage`)**:
   - Анализ баланса целевого класса (`Survived = 0`: `549` / `61.62%`, `Survived = 1`: `342` / `38.38%`) и структуры пропусков (`Age`: `177`, `Cabin`: `687`, `Embarked`: `2`).
   - Обоснование связки метрик: **`F1-score`** (основная пороговая метрика класса `1`), **`ROC-AUC`** (основная ранжирующая метрика) и `Accuracy` (вспомогательная).
   - Построчное конструирование признаков без утечки статистик: `FamilySize = SibSp + Parch + 1`, `IsAlone = (FamilySize == 1)`, `Title` (обращение из `Name`: `Mr`, `Miss`, `Mrs`, `Master`, `Rare`).
   - Отделение отложенной тестовой выборки `X_test` (`20%`, стратифицированно, `712 train / 179 test`) **до начала всех экспериментов**.
   - Сборка `ColumnTransformer` + `Pipeline` для импутации и кодирования строго внутри фолдов единого `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
2. **Дерево решений (`DecisionTreeClassifier`)**:
   - Ручная реализация примеси Джини (`Gini`), энтропии Шеннона (`Entropy`) и прироста информации (`Information Gain`) на чистом `NumPy` с проверкой `assert np.isclose(...)` против `tree_.impurity` из `scikit-learn`.
   - Сравнение переобученного дерева без ограничений (`depth = 18`, `164` листа, `Train F1 = 0.9786`, `CV F1 = 0.7209`) и компактного дерева (`max_depth = 3, min_samples_leaf = 10`, `8` листьев, `CV F1 = 0.7623`, `CV ROC-AUC = 0.8590`).
   - Визуализация `plot_tree` и подробный разбор конкретных правил `if-then` от корня до листьев.
   - Кривые валидации по `max_depth` (`1..15`), `min_samples_leaf` (`1..50`) и тепловая карта `CV F1` по сетке `(max_depth × min_samples_leaf)`.
3. **Ансамбли: Bagging и Gradient Boosting**:
   - Сравнение 7 моделей (`LogisticRegression`, `DecisionTree (default)`, `DecisionTree (depth=3, leaf=10)`, `RandomForest`, `XGBoost`, `LightGBM`, `CatBoost`) на одних и тех же 5 фолдах (`mean ± std`).
   - Пофолдовый анализ и демонстрация того, почему модель, выигравшая на одном фолде (например, `DecisionTree (depth=3)` с `F1 = 0.8491` на `Fold 3`), не обязана быть лучшей в среднем по всем фолдам.
4. **Подбор гиперпараметров (`GridSearchCV` и `RandomizedSearchCV`)**:
   - Настройка `RandomForestClassifier` (`GridSearchCV`), `CatBoostClassifier` (`GridSearchCV`) и `LGBMClassifier` (`RandomizedSearchCV`) с анализом топ-10 комбинаций из `cv_results_` и тепловыми картами параметров.
5. **Финальная оценка на отложенном `X_test` и важность признаков**:
   - Однократная проверка выбранной по `CV` лучшей модели (`CatBoost (tuned)`) и базовых ориентиров на `X_test` (`n = 179`), построение матрицы ошибок и `ROC`-кривых.
   - Сравнение встроенной важности (`feature_importances_`) и перестановочной важности (`permutation_importance`), объяснение смещения `Gini Importance` в пользу непрерывных признаков высокой кардинальности (`Fare`, `Age`).

## Сводная таблица результатов

| Модель | Train F1 (CV) | CV F1 (mean ± std) | CV ROC-AUC (mean ± std) | CV Accuracy (mean ± std) | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Dummy (Most Frequent)` | 0.0000 | 0.0000 ± 0.0000 | 0.5000 ± 0.0000 | 0.6166 ± 0.0027 | 0.6145 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| `LogisticRegression` | 0.7660 | 0.7673 ± 0.0253 | 0.8660 ± 0.0200 | 0.8259 ± 0.0223 | 0.8212 | 0.8033 | 0.7101 | 0.7538 | 0.8731 |
| `DecisionTree (default)` | 0.9786 | 0.7209 ± 0.0133 | 0.7752 ± 0.0111 | 0.7837 ± 0.0101 | 0.8324 | 0.8000 | 0.7536 | 0.7761 | 0.8143 |
| `DecisionTree (depth=3, leaf=10)` | 0.7697 | 0.7623 ± 0.0550 | 0.8590 ± 0.0364 | 0.8273 ± 0.0351 | 0.8324 | 0.8000 | 0.7536 | 0.7761 | 0.8639 |
| `RandomForest (default)` | 0.9788 | 0.7389 ± 0.0187 | 0.8688 ± 0.0163 | 0.8020 ± 0.0102 | 0.8268 | 0.7879 | 0.7536 | 0.7704 | 0.8378 |
| `RandomForest (tuned, depth=4)` | 0.7791 | 0.7697 ± 0.0406 | 0.8668 ± 0.0247 | 0.8315 ± 0.0285 | 0.8101 | 0.7869 | 0.6957 | 0.7385 | 0.8524 |
| `XGBoost (default)` | 0.8874 | 0.7576 ± 0.0272 | 0.8877 ± 0.0151 | 0.8217 ± 0.0239 | 0.8268 | 0.8065 | 0.7246 | 0.7634 | 0.8429 |
| `LightGBM (default)` | 0.8692 | 0.7585 ± 0.0267 | 0.8826 ± 0.0215 | 0.8245 ± 0.0219 | 0.8101 | 0.7869 | 0.6957 | 0.7385 | 0.8267 |
| `LightGBM (tuned, leaves=7)` | 0.8766 | 0.7782 ± 0.0214 | 0.8874 ± 0.0184 | 0.8371 ± 0.0170 | 0.8101 | 0.7869 | 0.6957 | 0.7385 | 0.8356 |
| `CatBoost (default)` | 0.8917 | 0.7609 ± 0.0230 | 0.8943 ± 0.0226 | 0.8258 ± 0.0206 | 0.8045 | 0.7742 | 0.6957 | 0.7328 | 0.8507 |
| **`CatBoost (tuned) [WINNER]`** | **0.9271** | **0.7785 ± 0.0164** | **0.8904 ± 0.0182** | **0.8385 ± 0.0148** | **0.8324** | **0.8095** | **0.7391** | **0.7727** | **0.8482** |

## Структура репозитория

```text
titanic-trees-ensembles/
├── data/
│   └── train.csv                    # Датасет Kaggle Titanic (891 строка × 12 столбцов)
├── titanic_trees_ensembles.ipynb    # Основной ноутбук с расчётами, визуализациями и выводами
├── requirements.txt                 # Зависимости Python
├── .gitignore
└── README.md
```

## Как запустить

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook titanic_trees_ensembles.ipynb
```
