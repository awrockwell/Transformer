import pandas as pd
import os

DATA_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/"
# print(os.path.abspath(__file__))
#def df_loader():

df = pd.read_csv(DATA_PATH + "TestData.csv")
print(df)