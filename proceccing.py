import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()  # for LabelEncoder
dicts = {}

test = pd.read_csv("Kosmo_titan/spaceship-titanic/test.csv")
train = pd.read_csv("Kosmo_titan/spaceship-titanic/train.csv")
test_id = test["PassengerId"]
df = pd.concat([train, test], ignore_index=True)

print(df.info())

# PassengerId section
train.drop(["PassengerId", "Name"], axis=1, inplace=True)
test.drop(["PassengerId", "Name"], axis=1, inplace=True)


# HomePlanet section
mask = train["HomePlanet"].isna()  # Получаем распредление вероятности для HomePlanet
P = train["HomePlanet"].value_counts() / len(train["HomePlanet"].dropna())
# заполняем пропуски с вероятностью `p`
train.loc[mask, "HomePlanet"] = np.random.choice(P.index.to_list(),
                                                 size=mask.sum(),
                                                 p=P.to_list())

mask = test["HomePlanet"].isna()  # Получаем распредление вероятности для HomePlanet
P = test["HomePlanet"].value_counts() / len(test["HomePlanet"].dropna())
# Заполняем пропуски с вероятностью `p`
test.loc[mask, "HomePlanet"] = np.random.choice(P.index.to_list(),
                                                size=mask.sum(),
                                                p=P.to_list())

# Меняем строковые значения на числовые
label.fit(train.HomePlanet.drop_duplicates())  # задаем список значений для кодирования
dicts['HomePlanet'] = list(label.classes_)
train.HomePlanet = label.transform(train.HomePlanet)  # заменяем значения из списка кодами закодированных элементов

label.fit(test.HomePlanet.drop_duplicates())  # задаем список значений для кодирования
dicts['HomePlanet'] = list(label.classes_)
test.HomePlanet = label.transform(test.HomePlanet)  # заменяем значения из списка кодами закодированных элементов

# CryoSleep
CryoSleep_mapping = {False: 1, True: 2}


mask = train["CryoSleep"].isna()  # Получаем распредление вероятности для CryoSleep
P = train["CryoSleep"].value_counts() / len(train["CryoSleep"].dropna())
# заполняем пропуски с вероятностью `p`
train.loc[mask, "CryoSleep"] = np.random.choice(P.index.to_list(),
                                                size=mask.sum(),
                                                p=P.to_list())

mask = train["CryoSleep"].isna()  # Получаем распредление вероятности для CryoSleep
P = train["CryoSleep"].value_counts() / len(train["CryoSleep"].dropna())
# заполняем пропуски с вероятностью `p`
train.loc[mask, "CryoSleep"] = np.random.choice(P.index.to_list(),
                                                size=mask.sum(),
                                                p=P.to_list())

train['CryoSleep'] = train["CryoSleep"].map(CryoSleep_mapping)
test['CryoSleep'] = test["CryoSleep"].map(CryoSleep_mapping)

# Cabin section
train[['Cabin_1', 'Cabin_2', 'Cabin_3']] = train['Cabin'].str.split('/', expand=True)
train.drop('Cabin', axis=1, inplace=True)
cabin_1_mapping = {"B": 1, "F": 2, "A": 3, "G": 4, "E": 5, "D": 6, "C": 7, "T": 8}
train['Cabin_1'] = train['Cabin_1'].map(cabin_1_mapping)
train['Cabin_1'] = train['Cabin_1'].fillna(0)

cabin_3_mapping = {"P": 1, "S": 2}
train['Cabin_3'] = train['Cabin_3'].map(cabin_3_mapping)
train['Cabin_3'] = train['Cabin_3'].fillna(0)

train['Cabin_2'] = train['Cabin_2'].fillna(9999)
train['Cabin_2'] = train['Cabin_2'].astype(int) + 1
train['Cabin_2'] = train['Cabin_2'].replace(10000, 0)

test[['Cabin_1', 'Cabin_2', 'Cabin_3']] = test['Cabin'].str.split('/', expand=True)
test.drop('Cabin', axis=1, inplace=True)
cabin_1_mapping = {"B": 1, "F": 2, "A": 3, "G": 4, "E": 5, "D": 6, "C": 7, "T": 8}
test['Cabin_1'] = train['Cabin_1'].map(cabin_1_mapping)
train['Cabin_1'] = train['Cabin_1'].fillna(0)

cabin_3_mapping = {"P": 1, "S": 2}
test['Cabin_3'] = test['Cabin_3'].map(cabin_3_mapping)
test['Cabin_3'] = test['Cabin_3'].fillna(0)

test['Cabin_2'] = test['Cabin_2'].fillna(9999)
test['Cabin_2'] = test['Cabin_2'].astype(int) + 1
test['Cabin_2'] = test['Cabin_2'].replace(10000, 0)

# Destination section
mask = train["Destination"].isna()  # Получаем распредление вероятности для Destination
P = train["Destination"].value_counts() / len(train["Destination"].dropna())
# заполняем пропуски с вероятностью `p`
train.loc[mask, "Destination"] = np.random.choice(P.index.to_list(),
                                                 size=mask.sum(),
                                                 p=P.to_list())

mask = test["Destination"].isna()  # Получаем распредление вероятности для Destination
P = test["Destination"].value_counts() / len(test["Destination"].dropna())
# заполняем пропуски с вероятностью `p`
test.loc[mask, "Destination"] = np.random.choice(P.index.to_list(),
                                                 size=mask.sum(),
                                                 p=P.to_list())

label.fit(train.Destination.drop_duplicates())  # задаем список значений для кодирования
dicts['Destination'] = list(label.classes_)
train.Destination = label.transform(train.Destination)  # заменяем значения из списка кодами закодированных элементов

label.fit(test.Destination.drop_duplicates())  # задаем список значений для кодирования
dicts['Destination'] = list(label.classes_)
test.Destination = label.transform(test.Destination)  # заменяем значения из списка кодами закодированных элементов

# Age section
train.loc[train["Transported"] == 1, "Age"] = (
    train[train["Transported"] == 1]["Age"]
    .fillna(train[train["Transported"] == 1]["Age"].mean()))
train.loc[train["Transported"] == 0, "Age"] = (
    train[train["Transported"] == 0]["Age"]
    .fillna(train[train["Transported"] == 0]["Age"].mean()))

test['Age'] = test['Age'].fillna(28.827968418500845)

# VIP section
vip_mapping = {False: 1, True: 2}
train["VIP"] = train["VIP"].map(vip_mapping)
test["VIP"] = test["VIP"].map(vip_mapping)

train['VIP'] = train['VIP'].fillna(0)
test['VIP'] = test['VIP'].fillna(0)

# Servise section
cols = ['FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']

train['Fare'] = train[['FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
test['Fare'] = test[['FoodCourt', 'RoomService', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)

for col in cols:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(test[col].median())

print(train.info())

df = pd.concat([train, test], ignore_index= True)


