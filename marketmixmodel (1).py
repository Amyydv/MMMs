# https://practicaldatascience.co.uk/machine-learning/how-to-create-a-basic-marketing-mix-model-in-scikit-learn
# https://medium.com/analytics-vidhya/ordinary-least-squared-ols-regression-90942a2fdad5
# Market Mix Modeling 
import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sns
import matplotlib.pyplot as plt
# Sales Data  
#To understand how much each marketing input contributes to sales, and
# how much to spend on each marketing input.
df = pd.read_csv("F:/MY FILE/Machine_Learning/Market Mix Modelling/Advertising.csv")
df.head()
#remove extra 'Unnamed' column
df_clean = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df_clean.head()
# examine the data
df_clean.describe()
#Correlation b/w variables
plt.figure(figsize=(8,5))
heatmap = sns.heatmap(df_clean.corr(), annot=True, cmap="Blues")
#Labels and features
labels = df_clean['sales']
features = df_clean.drop(['sales'], axis=1)
features
#Scatter graph b/w response and features
for x in features:
    plt.plot(labels, features[x], 'ro', color = 'blue')  # arguments are passed to np.histogram
    plt.title("Sales vs " + x)
    plt.xlabel(x)
    plt.ylabel("sales")
    plt.show()
    
# Data Distribution 
for x in features:
    plt.hist(features[x], bins='auto')
    plt.title(x)
    plt.show()

#As from the above histogram graph, the data distribution for the newspaper is skewness towards left.
#Lets correct it using Box Cox which helps in removing the data skewness.
from scipy import stats
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(211)
x = df_clean['newspaper']
prob = stats.probplot(x, dist=stats.norm, plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Probplot against normal distribution')
# We now use boxcox to transform the data so it’s closest to normal:
ax2 = fig.add_subplot(212)
df_clean['newspaper'], _ = stats.boxcox(x)
prob = stats.probplot(df_clean['newspaper'], dist=stats.norm, plot=ax2)
ax2.set_title('Probplot after Box-Cox transformation')
plt.show()
#========

plt.hist(df_clean['newspaper'], bins='auto')  # arguments are passed to np.histogram
plt.title("Newspaper after Box cox transformation")
plt.show()

plt.plot(df_clean['sales'], df_clean['newspaper'], 'ro')  # arguments are passed to np.histogram
plt.title("Scatter plot b/w sales and newspaper")
plt.xlabel("Newspaper")
plt.ylabel("Sales")
plt.show()

# As from the above graph it is clear that newspaper do not have any relationship with the Sales.
# Lets build 2 algorithm with and without newspaper to get more clear picture.

import statsmodels.formula.api as sm
model1 = sm.ols(formula="sales~TV+radio+newspaper", data=df_clean).fit()
model2 = sm.ols(formula="sales~TV+radio", data=df_clean).fit()
model3 = sm.ols(formula="sales~TV", data=df_clean).fit()
#sales~TV+radio+newspaper
print(model1.summary())
print(model2.summary())
print(model3.summary())

# From the above results it is clear that the 'model 2' with feature 'radio' and 'TV' 
# is having the lowest AIC & BIC

#Model 2 Parameters, error, and r square
print('Parameters: ', model2.params)
print('R2: ', model2.rsquared)
print('Standard errors: ', model2.bse)

#Actual and predicted values
y_pred = model2.predict()
df1 = pd.DataFrame({'Actual': labels, 'Predicted': y_pred})  
df1.head(10)

# model accuracy check
df1.corr()
df1['Residuals'] = df1['Actual'] - df1['Predicted'] # residuals

from sklearn import metrics
print('MAE:', np.round_(metrics.mean_absolute_error(labels, y_pred), 3))
print('MSE:', np.round_(metrics.mean_squared_error(labels, y_pred), 3))
print('RMSE:', np.round_(np.sqrt(metrics.mean_squared_error(labels, y_pred)),3))


#From the above values it is clear that newspaper maketing is not affecting sales by any chance.
#High P-value(>0.05) is always fail to reject null hypothesis.
#That means there is no relationship between the newspaper marketing and sales.


