import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import power_transform
from load_df import load_data
from pandas import read_csv
from pandas import DataFrame
from scipy.stats import boxcox
from matplotlib import pyplot
from sklearn.preprocessing import PowerTransformer
from sklearn import linear_model, metrics

DATA_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/"

class transformer:
    def __init__(self, dfY, dfXs, dfAll, dfXn):
        self.dfY = dfY
        self.dfXs = dfXs
        self.dfAll = dfAll
        self.dfXn = dfXn

    def iterate_columns(self):
        '''
        Used to append final Transformed Results to end of dfAll
        :return: Pandas DataFrame
        '''
        return ...

    def BoxCox(self):
        '''
        Used to do analysis, Column by Column, appends each column
        transformed at the end, e.g., X1.BoxCox.Lambda=0.05
        :return: Pandas Dataframe
        '''
        data = self.dfAll
        # do several estimates of the BoxCox, like 2 to 2, then half whats closer, then half etc...
        # use correl as what's being more successful
        X1 = self.dfXn

        Lambda = 5
        transX1 = (X1**Lambda-1)/Lambda
        first = [Lambda, np.corrcoef(self.dfY, transX1)[1,0]]

        Lambda = -5
        transX1 = (X1**Lambda-1)/Lambda
        first = np.vstack((first, [Lambda, np.corrcoef(self.dfY, transX1)[1,0]]))

        Lambda = 0
        transX1 = np.log(X1)
        first = np.vstack((first, [Lambda, np.corrcoef(self.dfY, transX1)[1,0]]))

        for x in range(13):
            Lambda = ((first[first[:, 1].argsort()][-1,0]) + (first[first[:, 1].argsort()][-2,0])) / 2
            transX1 = (X1**Lambda-1)/Lambda
            first = np.vstack((first, [Lambda, np.corrcoef(self.dfY, transX1)[1,0]]))

        BestLambda = (first[first[:, 1].argsort()][-1, 0])

        if BestLambda == 0:
            dfOut = np.log(X1)
        else:
            dfOut = (X1**BestLambda)

        return dfOut

    def GroupBoxCox(self):
        '''
        Used to do analysis, All Columns by Column
        :return: Pandas Dataframe
        '''
        data = self.dfAll
        X1 = self.dfXs

        reg = linear_model.LinearRegression()
        reg.fit(X1, self.dfY)
        metrics.r2_score(self.dfY, reg.predict(X1))

        Lambda = 5
        transX1 = (X1**Lambda-1)/Lambda
        reg.fit(transX1, self.dfY)
        first = [Lambda, metrics.r2_score(self.dfY, reg.predict(transX1))]

        Lambda = -5
        transX1 = (X1**Lambda-1)/Lambda
        reg.fit(transX1, self.dfY)
        first = np.vstack((first, [Lambda, metrics.r2_score(self.dfY, reg.predict(transX1))]))

        Lambda = 0
        transX1 = np.log(X1)
        reg.fit(transX1, self.dfY)
        first = np.vstack((first, [Lambda, metrics.r2_score(self.dfY, reg.predict(transX1))]))

        for x in range(13):
            Lambda = ((first[first[:, 1].argsort()][-1,0]) + (first[first[:, 1].argsort()][-2,0])) / 2
            transX1 = (X1**Lambda-1)/Lambda
            reg.fit(transX1, self.dfY)
            first = np.vstack((first, [Lambda, metrics.r2_score(self.dfY, reg.predict(transX1))]))

        BestLambda = (first[first[:, 1].argsort()][-1, 0])

        if BestLambda == 0:
            dfOut = np.log(X1)
        else:
            dfOut = (X1**BestLambda)

        return dfOut

    def Normalize01(self):
        '''
        Normalize between 0 and 1 using: (x_i - min) / (max - min)
        :return: Pandas Dataframe
        '''
        data = self.dfAll
        X1 = self.dfXn

        transX1 = (X1 - min(X1)) / (max(X1)-min(X1))

        correl = np.corrcoef(self.dfY, transX1)[1,0]

        return transX1


DataLoaderClass = load_data("TestData.csv")
dfAll = DataLoaderClass.df_loader()
dfY, dfXs = DataLoaderClass.splitYX()

# asdf = transformer(dfY, dfXs, dfAll, dfXs.iloc[:, 0])
asdf = transformer(dfY, dfXs, dfAll, dfXs.iloc[:, 0])


# print(asdf.GroupBoxCox())
print(asdf.Normalize01())

#print(dfXs.iloc[:, 0])

# # Slow Coding to test all columns
# runData = asdf.BoxCox()
# asdf.dfXn = dfXs.iloc[:, 1]
# runData = np.vstack((runData, asdf.BoxCox()))
# asdf.dfXn = dfXs.iloc[:, 2]
# runData = np.vstack((runData, asdf.BoxCox()))
# asdf.dfXn = dfXs.iloc[:, 3]
# runData = np.vstack((runData, asdf.BoxCox()))
#
# print(runData)
# np.savetxt("foo.csv", runData, delimiter=",")

#(X^Lambda-1)/(Lambda)