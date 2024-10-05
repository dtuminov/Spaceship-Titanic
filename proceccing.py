import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()  # for LabelEncoder
dicts = {}

test = pd.read_csv("Kosmo_titan/spaceship-titanic/test.csv")
train = pd.read_csv("Kosmo_titan/spaceship-titanic/train.csv")

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
# Cabin section
train.drop("Cabin", axis=1, inplace=True)
test.drop("Cabin", axis=1, inplace=True)

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
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

# VIP section
print(train["VIP"])
mask = train["VIP"].isna()  # Получаем распредление вероятности для VIP
P = train["VIP"].value_counts() / len(train["VIP"].dropna())
# заполняем пропуски с вероятностью `p`
train.loc[mask, "CryoSleep"] = np.random.choice(P.index.to_list(),
                                                size=mask.sum(),
                                                p=P.to_list())

mask = train["CryoSleep"].isna()  # Получаем распредление вероятности для VIP
P = train["VIP"].value_counts() / len(train["VIP"].dropna())
# заполняем пропуски с вероятностью `p`
train.loc[mask, "VIP"] = np.random.choice(P.index.to_list(),
                                                size=mask.sum(),
                                                p=P.to_list())

# Servise section

train.drop(["RoomService", "FoodCourt", "ShoppingMall",
           "Spa", "VRDeck"], axis=1, inplace=True)

test.drop(['RoomService', "FoodCourt", "ShoppingMall",
           "Spa", "VRDeck"], axis=1, inplace=True)

df = pd.concat([train, test], ignore_index= True)


