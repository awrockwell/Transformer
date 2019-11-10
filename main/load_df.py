import pandas as pd
import os

DATA_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/"
TRANSFORMERS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/transformers/"

# print(os.path.abspath(__file__))
class load_data:
    def __init__(self, fileToLoad):
        self.fileToLoad = fileToLoad

    def df_loader(self):
        '''
        Used to append final Transformed Results
        :return: Pandas DataFrame
        '''
        return pd.read_csv(DATA_PATH + self.fileToLoad)

    def splitYX(self):
        '''
        Used to do analysis, Column by Column
        :return: Two Numpy Arrays, One with Just the Y and all the others as Xs
        '''
        dfY = pd.read_csv(DATA_PATH + self.fileToLoad)["Y"]
        dfXs = pd.read_csv(DATA_PATH + self.fileToLoad)
        dfXs = dfXs.loc[:, dfXs.columns != 'Y']
        return dfY, dfXs

DataLoaderClass = load_data("TestData.csv")
print(DataLoaderClass.df_loader())
print(DataLoaderClass.splitYX())
