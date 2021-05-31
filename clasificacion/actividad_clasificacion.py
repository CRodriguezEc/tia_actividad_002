# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style="darkgrid")


# %%
print(tf.__version__)


# %%
cols = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety','output']
cars = pd.read_csv(r'https://raw.githubusercontent.com/CRodriguezEc/tia_actividad_002/master/clasificacion/car_evaluation.csv', header=None, names=cols)


# %%
cars.shape


# %%
cars.info()


# %%
#   Muestro los 10 primeros registros del archivo
cars.head(10)


# %%
#   Resumen estadistico de los datos
cars.describe()


# %%
cars['price'].value_counts().sort_index()


# %%
plot_size = plt.rcParams["figure.figsize"]
plot_size [0] = 8
plot_size [1] = 6
plt.rcParams["figure.figsize"] = plot_size


# %%
cars.output.value_counts().plot(kind='pie', autopct='%0.05f%%', colors=['lightblue', 'lightgreen', 'orange', 'pink'], explode=(0.05, 0.05, 0.05,0.05))


# %%
#   CONVIERTE SOLO 
#   Normalizar es poner a todas las variables numericas (solo) en un mismo rango (0 y 1), media de la variable sea cero y 
#   Desviacion estandar sea 1 
price = pd.get_dummies(cars.price, prefix='price')
maint = pd.get_dummies(cars.maint, prefix='maint')

doors = pd.get_dummies(cars.doors, prefix='doors')
persons = pd.get_dummies(cars.persons, prefix='persons')

lug_capacity = pd.get_dummies(cars.lug_capacity, prefix='lug_capacity')
safety = pd.get_dummies(cars.safety, prefix='safety')

labels = pd.get_dummies(cars.output, prefix='condition')


# %%
#X = pd.concat([price, maint, doors, persons, lug_capacity, safety] , axis=1)
X = pd.concat([maint, doors, persons, lug_capacity, safety] , axis=1)
print(price)


# %%
labels.head()


# %%
#y = cars.output
y = labels.values
#print(y)
#   price, maint, doors, persons, lug_capacity, safety
#maint.shape


# %%
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.20, random_state=42)


# %%
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model


# %%
input_layer = Input(shape=(X.shape[1],))
dense_layer_1 = Dense(15, activation='relu')(input_layer)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
output = Dense(y.shape[1], activation='softmax')(dense_layer_2)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


# %%
print(model.summary())


# %%
history = model.fit(X_train, y_train, batch_size=8, epochs=50, verbose=1, validation_split=0.2 )


# %%
#Predecidos correctamente, 73% de las fueron correctas, error del 27%
#   score = model.score(X_test, y_test)
score = model.evaluate(X_test, y_test, verbose=1)
print("Test score: ", score[0])
print("Test Accuracy: ", score[1])


# %%



