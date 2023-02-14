import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

dataset = pd.read_csv("./avocado.csv")

#dataset = dataset.drop(columns=["ocean_proximity"])
#print(dataset.info())
#dataset.dropna(inplace=True)

predictiontype = 'conventional'
dataset = dataset[dataset.type == predictiontype]
dataset['Date'] = pd.to_datetime(dataset['Date'])

regions = dataset.groupby(dataset.region)
predictfor = "TotalUS"
dateprice = regions.get_group(predictfor)[['Date', 'AveragePrice']].reset_index(drop=True)
hi = dateprice.plot(x='Date', y='AveragePrice', kind="line", color="r").get_figure()
hi.savefig(fname="aliner.png")
