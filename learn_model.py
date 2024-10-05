from proceccing import test, train
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cross-validation
X = train.drop(['Transported'], axis=1) # Признаки
y = train['Transported'] # Целевая переменная

# Разделяем данные на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = len(train)/(len(train)+len(test)))

# Создаем объект классификатора
model = DecisionTreeClassifier(random_state=42)

# Обучаем модель на обучающих данных
model.fit(x_train, y_train)

# Прогнозируем на тестовых данных
y_pred = model.predict(x_test)

# Оцениваем точность
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.2f}')
