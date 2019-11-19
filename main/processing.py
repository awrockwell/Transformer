import pandas as pd
import os
import numpy as np
from sklearn import linear_model, metrics
from math import *
from main.transformers import *

# Make non-scientific display
pd.set_option('display.float_format', lambda x: '%.6f' % x)

DATA_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/"

class main_processor:
    def __init__(self, fileToLoad, TransClass):
        dfAll = fileToLoad
        dfYheader = str(dfAll.columns[0])
        dfY = dfAll[dfYheader]
        dfXs = dfAll.loc[:, dfAll.columns != dfYheader]
        self.dfY = dfY
        self.dfXs = dfXs
        self.dfAll = dfAll
        self.TransClass = TransClass

    def BestLambda(self):
        '''
        Finds optimal lambda for functions
        :return: Pandas Dataframe
        '''

        # this\/ will later be an input \/
        Xs = self.dfXs
        reg = linear_model.LinearRegression()

        TransformClass = self.TransClass()

        df = pd.DataFrame({'Lambda': [], 'Correlation': []})

        starters = TransformClass.starters
        rotations = TransformClass.rotations

        for x in range(rotations):
            if x < len(starters):
                Lambda = starters[x]
            else:
                # Changed how calculations for optimal, from if min and max are best correl, if gets stuck
                if x % 2 == 0:
                    Lambda = df['Lambda'][(df['Correlation'].nlargest(2)).index].mean()
                else:
                    Lambda = df['Lambda'][(df['Correlation'].nlargest(1)).index].mean() - df['Lambda'].std()/x
            transX1 = TransformClass.equation(Xs=Xs, Lambda=Lambda)
            reg.fit(transX1, self.dfY)
            df = df.append(pd.DataFrame({'Lambda': [Lambda],
                                         'Correlation': [metrics.r2_score(self.dfY, abs(reg.predict(transX1)))]}), ignore_index=True)

        best_lambda = float(df['Lambda'][(df['Correlation'].nlargest(1)).index])
        best_correl = float(df['Correlation'].nlargest(1))

        dfOut = TransformClass.equation(Xs=Xs, Lambda=best_lambda)

        # Rename fdOut title names to include correlation and transform type and lambda
        dfOut.columns = dfOut.columns + "|" + TransformClass.__class__.__name__ + "|" + str(best_lambda)

        #want to return a non joined table, that should be another function
        return self.dfAll.join(dfOut), df

    def oneAtaTime(self):
        pass


dfAll = pd.read_csv(DATA_PATH + "TestData.csv")

asdf = main_processor(dfAll, BoxCox)

print(asdf.BestLambda())
# asdf.BestLambda().to_csv("normalized.csv")


#dfYheader = str(dfAll.columns[0])
#for a in dfAll.columns:
#    print(a)
#    print("here")