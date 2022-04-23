#!/usr/bin/env python
# coding: utf-8

# # GRIP : The Sparks Foundation
# 
# # Data Science And Business Analytics Internship
# 
# # Author- Abhishek

# # Task-1 Prediction Using Supervised ML
#  
# In This Linear-Regression Task,We will Predict The Percentage of Marks that a student is expected to score based upon  the number 
# of Hours they Studied.
# 
# Here We use the Data Available at http://bit.ly/w-data

# In[32]:


##Step1 : Importing Modules:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[34]:


## Step2: Reading And Preparing the Data..
data=pd.read_csv("http://bit.ly/w-data")
print("Total Rows and columns in this Dataset is ",data_set.shape)
print("Top 10 Rows of this Dataset is as Below:")
data_set.head(10)


# In[35]:


## Step 3: Now ,We will check for Missing Values in this Dataset..

data.isnull().sum()


# In[36]:


## Step 4: Finding More info of this dataset given by using info()
data.describe()


# In[37]:


data.info()


# In[39]:


## Step 5: Plotting scatter graph b/w  Scores vs No.of Hours Studied
data.plot(kind='scatter', x='Hours', y='Scores')
plt.show()


# In[42]:


## Step 6: Relationship Between  No.of Hours Studied and Scored Marks

data.corr()


# In[ ]:


##  From Step 6 We can say that ,There is Positive Linear Relation Between two variables i.e: Hours & Score
    ## Means if No.of Hours increases, then Marks scored Will also Increases


# In[44]:


## Step 7: Distrinution Plots of 2 Variables
Hour=data['Hours']
Marks=data['Scores']
sns.distplot(Hour)


# In[45]:


sns.distplot(Marks)


# # Implementing Linear Regression Model:

# In[46]:


x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In[49]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)


# In[50]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)


# In[52]:


m=reg.coef_
c=reg.intercept_
line=m*x+c
plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# In[53]:


y_pred=reg.predict(x_test)


# In[54]:


ac_predicted=pd.DataFrame({'Target':y_test,'Predicted':y_pred})
print("Predicted Values are :")
ac_predicted


# In[55]:


## plot showing the target value and the predicted value by Model
sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# In[56]:


### What would be the predicted score if a student studies for 9.25 hours/day?
hour=9.25
score=reg.predict([[hour]])
print("If a student studies  for {} hours per day ,then he/she will  score {} % in exams ".format(hour,score))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




