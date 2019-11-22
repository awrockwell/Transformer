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
        self.dfAllAdditional = dfAll
        self.TransClass = TransClass

    def BestLambda(self, Xvalue):
        '''
        Finds optimal lambda for transformation functions
        :return: Pandas Dataframe
        '''
        Xs = Xvalue
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

            transX1 = np.nan_to_num(pd.DataFrame(TransformClass.equation(Xs=Xs, Lambda=Lambda)))
            stayDataFrame = pd.DataFrame(TransformClass.equation(Xs=Xs, Lambda=Lambda))
            #print(self.dfY.as_matrix())

            #print(type(stayDataFrame.iloc[0,:]))
            #print(TransformClass.equation(Xs=Xs, Lambda=Lambda))
            #print(type(self.dfY))
            #print(self.dfY)
            #print(stayDataFrame.iloc[:,0])
            print("----------------------------------------")
            print(np.corrcoef(self.dfY,stayDataFrame.iloc[:,0])[1,0]) # could square for r^2
            reg.fit(transX1, self.dfY)
            df = df.append(pd.DataFrame({'Lambda': [Lambda],
                                         'Correlation': [metrics.r2_score(self.dfY, abs(reg.predict(transX1)))]}), ignore_index=True)
            print(metrics.r2_score(self.dfY, abs(reg.predict(transX1))))
        best_lambda = float(df['Lambda'][(df['Correlation'].nlargest(1)).index])
        best_correl = float(df['Correlation'].nlargest(1))


        dfOut = pd.DataFrame(TransformClass.equation(Xs=Xs, Lambda=best_lambda))
        # print(df)
        # print(dfOut)

        # Rename fdOut title names to include correlation and transform type and lambda
        dfOut.columns = dfOut.columns + "|" + TransformClass.__class__.__name__ + "|" + str(best_lambda) + "|" + str(best_correl)

        #want to return a non joined table, that should be another function
        return dfOut  #, df


    def trnsfrm1ataTime(self, TransClass):
        """
        Takes a Series or DataFrame, runs through a series of optimizations.
        Returns all the best Lambda transformations.
        :return: Pandas DataFrame
        """
        self.TransClass = TransClass
        dfXs = self.dfXs
        df = pd.DataFrame([])
        x = 1
        for column in dfXs.columns:
            dfX = pd.DataFrame(dfXs[column])
            #first time df setup
            if x == 1:
                df = self.BestLambda(dfX)
                x += 1
            else:
                df = df.join(self.BestLambda(dfX))
        return df


    def allTransformers(self):
        # for classes in (BoxCox, Inverse, NormalDistCDF):
        #     self.dfAllAdditional = self.dfAllAdditional.join(self.trnsfrm1ataTime(classes))
        self.dfAllAdditional = self.dfAllAdditional.join(self.trnsfrm1ataTime(BoxCox))
        self.dfAllAdditional = self.dfAllAdditional.join(self.trnsfrm1ataTime(Inverse))
        self.dfAllAdditional = self.dfAllAdditional.join(self.trnsfrm1ataTime(NormalDistCDF))
        return self.dfAllAdditional



#dfAll = pd.read_csv(DATA_PATH + "TestData.csv")
#dfAll = pd.read_csv(DATA_PATH + "TestData.csv")
dfAll = pd.read_csv(DATA_PATH + "TeamReg.csv")

asdf = main_processor(dfAll, Inverse) #BoxCox)

#best ,93 all together = .8751
#print(asdf.allTransformers())
asdf.allTransformers().to_csv("normalized.csv", index=False)
#asdf.BestLambda(dfAll).to_csv("normalized.csv", index=False)
#print(asdf.trnsfrm1ataTime(BoxCox))   #.BestLambda(dfAll))



#print(asdf.BestLambda(dfAll))


# asdf.BestLambda().to_csv("normalized.csv")


#dfYheader = str(dfAll.columns[0])
#for a in dfAll.columns:
#    print(a)
#    print("here")
