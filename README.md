# Transformer
Takes in a database and finds the best fitting linear transformations in Python

# To do list:
[ ]	User inputs dataframe file
[ ]	Python Reads File and Relays Column Headers to User
[ ]	User inputs which column is to by the Y
[ ]	Project runs each transformation (Create Transformation Class)
[ ]		•	Runs correlation and other metrics inside class
[ ]		•	Make it capable of easily integrating new transformations 
[ ]	Creates a readable output to include:
[ ]		•	Table of all Transformations ordered by best to least
[ ]		•	Graphs 	of top 4 transformations	
[ ]		•	Dataframe with top transformation

# Project Proposal: 
I would like to build an application that takes in a dataframe and runs it through several transformation options to pick the best transformation for linear fit (most likely using correlation, maybe several selection criteria, e.g., train/test fit, excluding outliers). After the transformations have been processed, the application will output the information in a presentable format. 

# Class Learning Applied:
•	Robust Testing
•	Decorators
•	Descriptors
•	Functional Programming
•	Parquet Output
•	Atomic Write

# Types of Transformations:
•	Boxcox
•	Modified Box
•	Manly Exponential
•	Yeo-Johnson
•	Ordered Quantile
•	Arcsin
•	Log
•	Min-Max Normalization

# Needed Modules:
	pandas, pyarrow
	statsmodels, numpy, scipy, matplotlib
	python-pptx

# Tests:
	Loading dataframes with errors (#NULL, Negative Numbers, Strings)
	Output consistency (Output is in the correct range)
	Atomic Writing
	Individual Transformations 
	Speed to Process


