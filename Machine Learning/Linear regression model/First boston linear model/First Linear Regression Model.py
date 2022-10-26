#!/usr/bin/env python
# coding: utf-8

# # First Linear Regression Model

# Importing the basic libraries

# In[59]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning) 


# Importing the dataset from sklearn 
# The name of the dataset is boston

# In[2]:


from sklearn.datasets import load_boston


# In[60]:


df=load_boston()


# Checking the keys of dataset

# In[6]:


df.keys()


# Description of the dataset

# In[8]:


print(df.DESCR)


# In[12]:


print(df.data)


# In[11]:


print(df.target)


# Getting the name of the features

# In[13]:


print(df.feature_names)


# Creating the dataset from array of boston data frame 

# In[18]:


data=pd.DataFrame(df.data,columns=df.feature_names)


# Adding the price columns to our dataset

# In[20]:


data['Price']=df.target


# Viewing the top 5 rows of dataset

# In[21]:


data.head()


# Checking the dytpes of the columns

# In[24]:


data.info()


# Checking the statistics of the numerical columns 

# In[25]:


data.describe()


# In[34]:


data.shape


# # EDA

# Checking the null values in the dataset

# In[29]:


data.isna().sum()


# Checking the correlation of the dataset

# In[37]:


data.corr()


# In[42]:


sns.set(rc={'figure.figsize':(10,10)})
sns.heatmap(data.corr(),annot=True,cmap="YlGnBu",linewidths=.5,cbar=False)


# In[61]:


sns.set(rc={'figure.figsize':(7,7)})
sns.distplot(x=data["AGE"],hist=True,kde=True,color="GREEN")


# In[57]:


sns.scatterplot(data=data,x="CRIM",y="Price")


# In[67]:


sns.regplot(data=data,x="DIS",y="Price",scatter=True,marker="+",color="green")


# In[72]:


sns.lmplot(data=data,x="RM",y="Price")


# In[82]:


sns.jointplot(data=data,x="CRIM",y="Price",kind="reg",color="green",height=10,ratio=8,marginal_ticks=True)


# In[84]:


sns.set(rc={'figure.figsize':(7,5)})
sns.boxplot(data['TAX'],color="green",linewidth=2,palette="Set3")


# In[90]:


sns.violinplot(x=data["RM"],palette="Set1")


# ### Spliting the features and target in x and y variables 

# In[92]:


x=data.iloc[:,:-1]
y=data.iloc[:,-1]


# In[93]:


x


# In[96]:


y


# ### Spliting the dataset into train and test part

# In[97]:


from sklearn.model_selection import train_test_split


# In[125]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=40)


# In[152]:


x_train


# In[101]:


x_train.shape


# In[102]:


x_test.shape


# In[105]:


y_test.shape


# In[106]:


y_train.shape


# ### Standardize or feature scaling the dataset

# In[107]:


from sklearn.preprocessing import StandardScaler


# In[122]:


scaler=StandardScaler()


# In[126]:


x_train=scaler.fit_transform(x_train)


# In[127]:


x_test=scaler.transform(x_test)


# In[128]:


x_train


# In[129]:


x_test


# ### Model Building

# In[130]:


from sklearn.linear_model import LinearRegression


# In[131]:


lr=LinearRegression()


# In[132]:


lr


# In[134]:


lr_model=lr.fit(x_train,y_train)


# In[135]:


lr_model.score(x_train,y_train)


# In[157]:


Linear_regression_coefficent=lr_model.coef_


# In[158]:


Linear_regression_coefficent.transpose


# In[159]:


Linear_regression_coefficent


# In[181]:


Lr_coefficent=pd.DataFrame(data=df.feature_names,columns=["Independent factors"])


# In[182]:


Lr_coefficent["coefficent"]=Linear_regression_coefficent


# In[183]:


Lr_coefficent


# In[137]:


lr_model.intercept_


# In[184]:


predicted_values=lr_model.predict(x_test)


# In[185]:


predicted_values


# In[187]:


Comparsion_table=pd.DataFrame(data=predicted_values,columns=['Predicted values'])


# In[189]:


Comparsion_table["Actual Value"]=y_test


# In[191]:


Comparsion_table["Difference"]=Comparsion_table["Actual Value"]-Comparsion_table["Predicted values"]


# In[200]:


Comparsion_table.dropna().head()


# ### Performance Metrics

# In[210]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
print("Mean squared error:",mean_squared_error(y_test,predicted_values))
print("Mean absolute error:",mean_absolute_error(y_test,predicted_values))
print("Root mean squared error:",np.sqrt(mean_squared_error(y_test,predicted_values)))


# ### Assumptions Of Linear Regression

# In[211]:


sns.regplot(x=y_test,y=predicted_values)


# In[212]:


residuals=y_test-predicted_values


# In[216]:


sns.displot(residuals,kind="kde",palette="Set1",color="green")


# In[220]:


sns.scatterplot(x=predicted_values,y=residuals,color='green',)


# ### R square and adjusted R square
# 

# In[221]:


from sklearn.metrics import r2_score
R_score=r2_score(y_test,predicted_values)
print("R score value is",R_score)


# In[223]:


Adjusted_r_score=1 - (1-R_score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print("Adjusted R score value is",Adjusted_r_score)


# ### Ridge regression model

# In[225]:


from sklearn.linear_model import Ridge
ridge=Ridge()


# In[229]:


ridge_model=ridge.fit(X_train,y_train)


# In[228]:


ridge_predict=ridge.predict(X_test)


# In[230]:


ridge_model.coef_


# In[231]:


ridge_model.intercept_


# In[232]:


sns.regplot(x=y_test,y=ridge_predict)


# In[233]:


ridge_resuidals=y_test-ridge_predict
sns.displot(ridge_resuidals,kind="kde",palette="Set1",color="green")


# In[236]:


sns.scatterplot(x=ridge_predict,y=ridge_resuidals,color='green',)


# ### Performance Metrics

# In[237]:


print("Mean squared error:",mean_squared_error(y_test,ridge_predict))
print("Mean absolute error:",mean_absolute_error(y_test,ridge_predict))
print("Root mean squared error:",np.sqrt(mean_squared_error(y_test,ridge_predict)))


# ### Lasso regression model

# In[238]:


from sklearn.linear_model import Lasso


# In[239]:


lasso_model=Lasso()


# In[242]:


lasso_model=lasso_model.fit(x_train,y_train)


# In[243]:


lasso_predict=lasso_model.predict(x_test)


# In[244]:


lasso_predict


# In[245]:


lasso_model.coef_


# In[246]:


lasso_model.intercept_


# In[256]:


Lasso_model_coefficent=pd.DataFrame(data=df.feature_names,columns=["Independent factors"])


# In[257]:


Lasso_model_coefficent[" lasso coefficent"]=lasso_model.coef_


# In[258]:


Lasso_model_coefficent


# In[259]:


sns.regplot(x=y_test,y=lasso_predict)


# ### Performance Metrics

# In[260]:


print("Mean squared error:",mean_squared_error(y_test,lasso_predict))
print("Mean absolute error:",mean_absolute_error(y_test,lasso_predict))
print("Root mean squared error:",np.sqrt(mean_squared_error(y_test,lasso_predict)))


# ### Elastic Net regression model

# In[267]:


from sklearn.linear_model import ElasticNet


# In[269]:


en=ElasticNet()
Elasticnet_model=en.fit(x_train,y_train)
Elasticnet_predict=en.predict(x_test)


# In[272]:


sns.regplot(x=y_test,y=Elasticnet_predict)


# ### Performance Metrics

# In[273]:


print("Mean squared error:",mean_squared_error(y_test,Elasticnet_predict))
print("Mean absolute error:",mean_absolute_error(y_test,Elasticnet_predict))
print("Root mean squared error:",np.sqrt(mean_squared_error(y_test,Elasticnet_predict)))


# ### Comparsion of coeffiecent between linear,Ridge,Lasso resgression

# In[264]:


Comparsion_of_coeffiecent=pd.DataFrame(data=df.feature_names,columns=["Independent factors"])


# In[270]:


Comparsion_of_coeffiecent["Linear regreesion coefficient"]=Linear_regression_coefficent
Comparsion_of_coeffiecent["Ridge regression coefficient"]=ridge_model.coef_
Comparsion_of_coeffiecent["Lasso regression coefficient"]=lasso_model.coef_
Comparsion_of_coeffiecent["ElasticNet regression coefficient"]=Elasticnet_model.coef_


# In[271]:


Comparsion_of_coeffiecent


# From this table we can conclude that how overfitting and underfitting can be reduce with ridge and lasso regreesion.
# In ridge regression in cost function we take lamba and square of slope in plus of linear regreesion but
# In Lasso regreesion in cost function we take lamba and absolute  of slope in plus of linear regreesion
# In Elastic net regression in cost function we take addition of linear,ridge and lasso cost function 

# In[ ]:




