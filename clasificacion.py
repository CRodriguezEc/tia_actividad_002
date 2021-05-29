import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pltmatplotlib
import seaborn as sns

cols = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Revenue']
cars = pd.read_csv(r'https://raw.githubusercontent.com/CRodriguezEc/tia_actividad_002/master/online_shoppers_intention.csv'
                    , names=cols
                    , header=None)

cars.head()

#   version
print(tf.__version__)

