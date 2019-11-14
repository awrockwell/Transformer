import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import power_transform
from load_df import load_data
from pandas import read_csv
from pandas import DataFrame
import scipy.stats
from matplotlib import pyplot
from sklearn.preprocessing import PowerTransformer
from sklearn import linear_model, metrics
from math import *

pd.set_option('display.float_format', lambda x: '%.6f' % x)
DATA_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/"

class transformer:
    def __init__(self, dfY, dfXs, dfAll, dfXn, TransClass):
        self.dfY = dfY
        self.dfXs = dfXs
        self.dfAll = dfAll
        self.dfXn = dfXn
        self.TransClass = TransClass


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


    def BestLambda(self):
        '''
        Finds optimal lambda for functions
        :return: Pandas Dataframe
        '''
        data = self.dfAll
        Xs = self.dfXs
        Y = self.dfY
        reg = linear_model.LinearRegression()
        TransformClass = self.TransClass()

        df = pd.DataFrame({'Lambda': [],'Correlation': []})

        starters = TransformClass.starters
        rotations = TransformClass.rotations

        for x in range(rotations):
            if x < len(starters):
                Lambda = starters[x]
            else:
                Lambda = df['Lambda'][(df['Correlation'].nlargest(2)).index].mean()

            transX1 = TransformClass.equation(Xs=Xs, Lambda=Lambda)
            reg.fit(transX1, Y)
            df = df.append(pd.DataFrame({'Lambda': [Lambda],
                                         'Correlation': [metrics.r2_score(Y, abs(reg.predict(transX1)))]}), ignore_index=True)

        BestLambda = float(df['Lambda'][(df['Correlation'].nlargest(1)).index])
        BestCorrel = float(df['Correlation'].nlargest(1))

        dfOut = TransformClass.equation(Xs=Xs, Lambda=BestLambda)

        # Rename fdOut title names to include correlation and transform type and lambda
        dfOut.columns = dfOut.columns + "|" + TransformClass.__class__.__name__ + "|" + str(BestLambda)

        return data.join(dfOut), df


class funcHolding:
    def __init__(self):
        pass


class BoxCox(funcHolding):
    starters = [-5, 0.00001, 5]
    rotations = 15

    def equation(self, Xs, Lambda):
        return (Xs ** Lambda - 1) / Lambda


class Inverse(funcHolding):
    starters = [-10000, 0, 10000]
    rotations = 15

    def equation(self, Xs, Lambda):
        return 1 / (Xs + Lambda)


class Normalize01(funcHolding):
    starters = []
    rotations = 1

    def equation(self, Xs, Lambda):
        return (Xs - Xs.min()) / (Xs.max() - Xs.min())


class NormalizeStdDev(funcHolding):
    starters = []
    rotations = 1

    def equation(self, Xs, Lambda):
        return (Xs - Xs.mean()) / Xs.std()

class NormalDistCDF(funcHolding):
    starters = []
    rotations = 1

    def equation(self, Xs, Lambda):
        columnSave = Xs.columns
        npArray = scipy.stats.norm(Xs.mean(), Xs.std()).cdf(Xs)
        return pd.DataFrame(data=npArray[0:, 0:], columns = columnSave)


DataLoaderClass = load_data("TestData.csv")
dfAll = DataLoaderClass.df_loader()
dfY, dfXs = DataLoaderClass.splitYX()

asdf = transformer(dfY, dfXs, dfAll, dfXs.iloc[:, 0],Inverse)


print(asdf.BestLambda())
asdf.BestLambda().to_csv("normalized.csv")


#print(asdf.GroupBoxCox())
# print(asdf.Normalize01())

# print(asdf.BoxCox())
# print(asdf.Inverse())

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


# mdoel = BoxCox()
# print(mdoel.starters)

# if TransformClass.rotations == 0:
#     BestLambda = ""
#     transX1 = TransformClass.equation(Xs=Xs, Lambda=1)
#     reg.fit(transX1, Y)
#     BestCorrel = [metrics.r2_score(Y, abs(reg.predict(transX1)))]
# else:

#
# # this can be more sophisticated, largest compared to closest, largest neighbor...
# # maybe just a function elsewhere that really digs deep into what the next number should bee
# # I need the largest and closest biggest and closest smallest.. hmm
# # rank feature! need its rank, its rank - 1 and its rank + 1
# largestCorIndx = (df['Correlation'].nlargest(1)).index
# df['Lambda'].min()
# # df['Lambda'][largestCorIndx]
#
# if float(df['Lambda'][(df['Correlation'].nlargest(1)).index]) == df['Lambda'].min():
#     Lambda = df['Lambda'].min() - (df['Lambda'].max() - df['Lambda'].min())
# elif float(df['Lambda'][(df['Correlation'].nlargest(1)).index]) == df['Lambda'].max():
#     Lambda = df['Lambda'].max() + (df['Lambda'].max() - df['Lambda'].min())
# else: