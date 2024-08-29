import pandas as pd
import numpy as np
from skops.io import dump, get_untrusted_types
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import missingno as msno
import skops.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def create_csv(b):
    test_2 = pd.read_csv("Kosmo_titan/spaceship-titanic/test.csv")
    data = {'PassengerId': test_2["PassengerId"][:len(b)], 'Transported': b}
    df = pd.DataFrame(data)
    df.to_csv('Kosmo_titan/spaceship-titanic/ans.csv', index=False)

# открываем файлы
test = pd.read_csv('Kosmo_titan/spaceship-titanic/test.csv')
train = pd.read_csv("Kosmo_titan/spaceship-titanic/train.csv")

# обработка пустых значений
categorical_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Name', 'Cabin']
for col in categorical_cols:
    train[col].fillna(train[col].mode()[0], inplace=True)
    test[col].fillna(test[col].mode()[0], inplace=True)

# PassengerId хранит в себе информацию о группе пассажира,
# которую можно использовать для определения статуса семьи
train.drop('PassengerId', inplace=True, axis=1)
test.drop('PassengerId', inplace=True, axis=1)

# обработка секции Cabin
train[['Deck', 'Num', 'Side']] = train['Cabin'].str.split('/', expand=True)
train['Num'] = train['Num'].astype(int)
train.drop('Cabin', axis=1, inplace=True)

test[['Deck', 'Num', 'Side']] = test['Cabin'].str.split('/', expand=True)
test['Num'] = test['Num'].astype(int)
test.drop('Cabin', axis=1, inplace=True)


# Заполняем пустые значения
number_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for col in number_cols:
    train[col].fillna(train[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

train["Age"].fillna(train["Age"].median(), inplace=True)
test["Age"].fillna(test["Age"].median(), inplace=True)

train[['CryoSleep', 'VIP', 'Transported']] = train[['CryoSleep', 'VIP', 'Transported']].astype(int)
test[['CryoSleep', 'VIP', ]] = test[['CryoSleep', 'VIP']].astype(int)

# вывод информации о полях датасета
print(train.isnull().sum())
print(train.info())
print(train.describe().T)
print(train.hist(bins=20, figsize=(10, 10)))

# построение графиков
features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
plt.figure(figsize=(10, 10))

for index, feature in enumerate(features):
    plt.subplot(2, 3, index + 1)
    sns.distplot(train[feature])
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()

HomePlanet = train['HomePlanet'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(HomePlanet, labels=HomePlanet.index, autopct='%1.1f%%', textprops={'fontsize': 10, 'rotation': 30})
plt.title('Distribution of HomePlanet')
plt.show()

# Конвертируем string в int
HomePlanet_map = {'Europa': 0, 'Earth': 1, 'Mars': 2}
train['HomePlanet'] = train['HomePlanet'].map(HomePlanet_map)
test['HomePlanet'] = test['HomePlanet'].map(HomePlanet_map)

Destination_map = {'TRAPPIST-1e': 0, '55 Cancri e': 1, 'PSO J318.5-22': 2}
train['Destination'] = train['Destination'].map(Destination_map)
test['Destination'] = test['Destination'].map(Destination_map)

Side_map = {'P': 0, 'S': 1}
train['Side'] = train['Side'].map(Side_map)
test['Side'] = test['Side'].map(Side_map)

Deck_map = {'D': 0, 'B': 1, 'C': 2, 'A': 3, 'E': 4, 'F': 5, 'T': 6, 'G': 7}
train['Deck'] = train['Deck'].map(Deck_map)
test['Deck'] = test['Deck'].map(Deck_map)

train.drop(['Num','Side', 'Name'],axis=1,inplace=True)
test.drop(['Num','Side', 'Name'],axis=1,inplace=True)


# обучение моделей
X = train.drop(['Transported'], axis=1)
y = train['Transported']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Определение параметров и их значений для перебора
param_grid = {
    'gamma': np.arange(0.9, 1.5, 0.1),
    'max_depth': np.arange(2,12,2),
    'n_estimators': np.arange(40, 160, 20)
}

models = [XGBClassifier(learning_rate=0.12, n_estimators=10,objective="binary:logistic",nthread=3, tree_method="hist"),
          CatBoostClassifier(n_estimators=50, learning_rate=0.051), RandomForestClassifier(),
          LGBMClassifier(lambda_l2=6.5, max_depth=15, num_leaves=25, learning_rate=0.051, bagging_fraction=0.5),
          GradientBoostingClassifier()]

# просмотр accuracy моделей
acc = list()
for mod in models:
    mod.fit(X_train, y_train)
    acc.append([mod.score(X_test, y_test), mod.__str__()[:6]])
print(acc, max(acc), acc.index(max(acc)))

# Итоговая модель
model = XGBClassifier(gamma=1.2, max_depth=6, min_child_weight=6, n_estimators=50, objective='binary:logistic', tree_method="hist")
model.fit(X,y)

# сохранение модели
dump(model, "my-model.skops")
acccuracy = model.score(X_test, y_test)
print(acccuracy)
print(X.info())
model.predict(test)

# конвертация массва 0 1 в True False
b = model.predict(test)
b = b > 0
create_csv(b)

