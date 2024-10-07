from proceccing import test, train, test_id
from create_file import make_submission_file
from calculate_MAE import calculate_MAE
# train & params
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# Boosting
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
import pandas as pd

# Cross-validation
X = train.drop(['Transported'], axis=1)  # Признаки
y = train['Transported']  # Целевая переменная

# Разделяем данные на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=len(train) / (len(train) + len(test)),
                                                    random_state=0)

# Создаем объект классификатора
# RandomForestClassifier
rdf_clf = RandomForestClassifier()
rdf_clf.fit(x_train, y_train)

# XGBoost Classifier
xgb_clf = XGBClassifier(eval_metric=["auc", "logloss"])
xgb_clf.fit(x_train, y_train, verbose=True)
xgb_val_prob = xgb_clf.predict_proba(x_test)

# LightGBM Classifier
lgb_clf = LGBMClassifier()
lgb_clf.fit(x_train, y_train)
lgb_val_prob = lgb_clf.predict_proba(x_test)

# CatBoost Classifier
cat_clf = CatBoostClassifier(verbose=0)

cat_clf.fit(x_train, y_train)
cat_val_prob = cat_clf.predict_proba(x_test)

# DecisionTree Classifier
dec_tree_clf = DecisionTreeClassifier(random_state=42)
dec_tree_clf.fit(x_train, y_train)

# Подсчет MAE
MAE_metric_value = calculate_MAE(cat_val_prob, lgb_val_prob, xgb_val_prob, y_test)
print(max(MAE_metric_value, key=lambda x:x[3]))

# Подсчет точности для каждой модели
models = [lgb_clf, dec_tree_clf, xgb_clf, cat_clf, rdf_clf]
for model in models:
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Точность модели: {accuracy:.5f}')

#подсчет вероятности для каждой модели
lgb_prob = lgb_clf.predict_proba(test)
lgb_prob = pd.DataFrame(lgb_prob)[1]

xgb_prob = xgb_clf.predict_proba(test)
xgb_prob = pd.DataFrame(xgb_prob)[1]

cat_prob = cat_clf.predict_proba(test)
cat_prob = pd.DataFrame(cat_prob)[1]

# Отправка файла
ens_prob = 0.8 * cat_prob + 0.2 * lgb_prob + 0 * xgb_prob
ens_sub = make_submission_file("ans.csv", ens_prob, test_id, 'PassengerId', 'Transported', threshold=0.5)
