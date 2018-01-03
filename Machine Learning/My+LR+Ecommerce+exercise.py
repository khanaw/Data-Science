
# coding: utf-8

# In[1]:


#import basic data packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:


#import ecommerce data from csv into pandas DataFrame callec 'customers'
customers=pd.read_csv('Ecommerce Customers')
#return info on new DataFram
customers.info()


# In[5]:


#check the dataframe header
customers.head(2)


# In[6]:


#obtain basic statistical measures on customers
customers.describe()


# In[17]:


#making a jointplot to explore 'time on website' vs 'yearly amount spend', but first setting the style
sns.set(style='darkgrid')
sns.set_palette('BuGn_r')
sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data=customers)

#not mutch correlation between the two


# In[18]:


#comparing time on app and amount spent
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)

#seems that there is positive correlation, we can explore later


# In[19]:


#what about time on app and length of membership
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
#nothing standing out here


# In[20]:


#lets just compare all the columns
sns.pairplot(customers)

#We can see some correlations here, length of membership and amount spent seems to be the one that stands out the most.


# In[21]:


#quick lmplot to see in detail
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


# In[24]:


#time for some linear regressing based training and testing data
#first we will set a variable X to be our input data set and y to be our predicted variable. 
#Here x will be all numerical columns but price, and y will be price


X=customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y=customers['Yearly Amount Spent']


# In[25]:


#import from sklearn
from sklearn.model_selection import train_test_split


# In[26]:


#set our test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[27]:


#import Linear Regression
from sklearn.linear_model import LinearRegression


# In[28]:


#set a blank linear regression and train/fit it
lm=LinearRegression()
lm.fit(X_train,y_train)


# In[32]:


#view our coefficients from the model

pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[34]:


#test our model by using our test set 
predictions=lm.predict(X_test)
plt.scatter(y_test,predictions)


#We can see our model works well in this instance


# In[36]:


#lets evaluate our model with Mean Absolute Error, Mean Squared Error, and Root Means Squared Error
#first we need to import metrics from sklearn
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[38]:


#plotting the residuals
sns.distplot((y_test-predictions),bins=30);


# In[39]:


#lets start evaluating what impacts sales the most
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# In[ ]:


#we can see that time on the app and length of membership increase the total spent the most. 
#To guide this company we would need more info
#but we could potentially say develop the website to either catch up to the app, 
#or develop the app more to foster continued sale
#or develop a way to increase continued membership

