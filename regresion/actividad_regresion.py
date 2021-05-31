# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from tensorflow import keras
from tensorflow.keras import layers

import seaborn as sns


# %%
print(tf.__version__)


# %%
cols = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Revenue']
dsPC = pd.read_csv(r'https://raw.githubusercontent.com/CRodriguezEc/tia_actividad_002/master/online_shoppers_intention.csv'
                    , names=cols
                    , header=None)


# %%
dsPC.shape


# %%
dsPC.info()


# %%
dsPC.head(5)


# %%
dsPC.describe()


# %%
dsPC.tail()


# %%
X = dsPC.iloc[:, 0:4].values
y = dsPC.iloc[:, 4].values


# %%
sns.pairplot(dsPC[['Administrative', 'Informational', 'ProductRelated', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']], diag_kind="kde")


# %%
dsPC.describe()


# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# %%
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# %%
input_layer = tf.keras.Input(shape=(X.shape[1],))
dense_layer_1 = tf.keras.layers.Dense(100, activation='relu')(input_layer)
dense_layer_2 = tf.keras.layers.Dense(50, activation='relu')(dense_layer_1)
dense_layer_3 = tf.keras.layers.Dense(25, activation='relu')(dense_layer_2)
output = tf.keras.layers.Dense(1)(dense_layer_3)

model = tf.keras.Model(inputs=input_layer, outputs=output)
model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])


# %%
history = model.fit(X_train, y_train, batch_size=2, epochs=50, verbose=1, validation_split=0.2)


# %%
from sklearn.metrics import mean_squared_error
from math import sqrt

pred_train = model.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train)))

pred = model.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred)))


# %%
test_predictions = model.predict(X_train).flatten()
a = plt.axes(aspect='equal')
plt.scatter(pred_train, test_predictions)
plt.xlabel('prediccion')
plt.ylabel('valores reales')
_ = plt.plot(y_train, y_train, color="orange")


# %%



