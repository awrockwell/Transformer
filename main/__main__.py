import os
from load_df import load_data
from measure import measure
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from sklearn import linear_model, metrics
import time
from functools import wraps
from time import time

DataLoaderClass = load_data("TestData.csv")
# print(DataLoaderClass.df_loader())
dfY, dfXs = DataLoaderClass.splitYX()

reg = linear_model.LinearRegression()

# all first column
# print(dfXs.iloc[:,0])

dfYX = pd.DataFrame([dfY, dfXs.iloc[:,0]])
dfYX = dfYX.transpose()
print(dfYX)

@measure
def runtimeReg():
    for x in range(1):
        reg.fit(dfXs*(1+.3**x), dfY)
        print(metrics.r2_score(dfY, reg.predict(dfXs*(1+.01*x))))

# runtimeReg()



pt = PowerTransformer()

data = dfYX
print(pt.fit(data))
print(pt.lambdas_)
print(pt.fit_transform(data))
# print(pt.transform(data))
np.savetxt("foo.csv", pt.lambdas_, delimiter=",")



# toconvert = (pt.transform(data))

# df['newcol'] = arr.toarray().tolist()

# df.to_csv('file1.csv')
# print(pd.DataFrame.to_csv(dfYX))


# df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
# print(df)
#
# array = np.array([7, 8, 9])
# df['c'] = array
# print(df)


# This applies a function to all the columns, apply is powerful!

# import numpy as np
# import pandas as pd
#
# my_array = np.array([[10,2,13],
#                      [21,22,23],
#                      [31,32,33],
#                      [10,57,20],
#                      [20,20,20],
#                      [101,91,10]])
#
#
# def my_function(x):
#     position = np.argmax(x) #<which row has max
#     return position
#
# print (np.apply_along_axis(my_function, axis=0, arr=my_array))