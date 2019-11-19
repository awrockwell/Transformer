
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
#
# if prevLambda == Lambda:
#     Lambda = (float(df['Lambda'][(df['Correlation'].nlargest(1)).index]) / x)
# # Lambda = Lambda / 2

# # correl:
# Lambda = 5
# transX1 = (X1 ** Lambda - 1) / Lambda
# first = [Lambda, np.corrcoef(self.dfY, transX1)[1, 0]]


#
# DataLoaderClass = load_data("TestData.csv")
# dfAll = DataLoaderClass.df_loader()
# dfY, dfXs, dfAll = DataLoaderClass.splitYX()