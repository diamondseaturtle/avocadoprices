import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import datetime as dt

#%matplotlib inline

#load a new data set !
dataset = pd.read_csv("./avocado.csv")
dataset["Date"] = pd.to_datetime(dataset["Date"])
dataset["Date"]=dataset["Date"].map(dt.datetime.toordinal)
print(dataset.info())
#print(dataset.columns)

#stuff (entry byebye) !
#dataset = dataset.drop(columns=["ocean_proximity"])
#print(dataset.info())
#dataset.dropna(inplace=True)

#multiplot all !
#graph = sns.pairplot(dataset)
#graph.savefig(fname="aplot.png")

#singleplot heatmap pop !
#fig = plt.figure()
#subplot = sns.distplot(dataset["Small Bags"])
#subplot = sns.heatmap(dataset.corr())
#fig.add_subplot(subplot)
#fig.savefig(fname="adist.png")
#fig.savefig(fname="aheat.png")

#trains regression !
# model, input and output datasets

X = dataset[["Date"]]
Y = dataset[["AveragePrice"]]
lr_model = LinearRegression()

#print(len(X))
#print(len(Y))

# traning and testing data 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
 
lr_model.fit(X_train, Y_train)
#print(lr_model.intercept_)
#print(lr_model.score(X_train, Y_train))
#print(lr_model.score(X_test, Y_test))

# trnaspose coeff and build DataFrame
#coeff = lr_model.coef_.transpose()

# Plot DataFrame and change png
#coeff_df = pd.DataFrame(coeff,X.columns,columns=["Coefficient"])
#fi = coeff_df.plot().get_figure()
#fi.savefig(fname="aput.png")

predictions = lr_model.predict(X_test)
fi2 = plt.scatter(Y_test, predictions, color = "black").get_figure()
actuals = Y_test.values.tolist() 
idline = np.linspace(max(min(actuals), min(predictions)), min(max(actuals), max(predictions)))
plt.plot(idline, idline, color="red", linestyle="dashed", linewidth=2.5)

plt.title("Scatterplot of Avocado Price Prediciton")
plt.xlabel("Integer Dates")
plt.ylabel("Predicted Avocado Prices")
fi2.savefig(fname="apricef.png")
plt.show()
#not important here hEe ehe !
#print(dataset.columns)

#print(dataset.head())
#print(dataset.head(8))

#print(dataset.info())

#print(dataset.describe())
#print(dataset.describe([.2, .4, .6, .8]))






#print(dataset.head())
#print(dataset.info())
#print(dataset.describe())
