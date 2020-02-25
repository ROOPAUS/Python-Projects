#!/usr/bin/env python
# coding: utf-8

# In[238]:


import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt


# In[240]:


# Importing the dataset
sales = pd.read_csv('MonthlySales.csv',index_col=0)
sales.head()


# In[113]:


sales.tail()


# In[115]:


sales.plot()


# In[ ]:


# From the above plot it is evident that there is some trend. But it is more towards a stationary time series
# A stationary time series is one with constant mean and standard deviation
# Now let us transform this to a moving avaerage(method to smooth the time series)


# In[117]:


sales_mean=sales.rolling(window=10).mean()
sales_mean


# In[ ]:





# In[121]:


sales.plot()
sales_mean.plot(color='red')


# In[ ]:


#Baseline model
# Assumption- recent history is the past reflection of the future


# In[132]:


series_value=sales.values
series_value


# In[136]:


value=pd.DataFrame(series_value)
value


# In[137]:


sales.df=pd.concat([value,value.shift(1)],axis=1)
sales.df


# In[143]:


sales.df.columns=['Actual_sales','Forecast_sales']
sales.df


# In[146]:


# Finding error
from sklearn.metrics import mean_squared_error
import numpy as np


# In[148]:


# Use head() to find any NaN values
sales.df.head()


# In[149]:


# Use tail() to find any NaN values
sales.df.tail()


# In[150]:


#Removing NaN values
sales.df=sales.df[1:47]
sales.df.head()


# In[151]:


sales_error=mean_squared_error(sales.df.Actual_sales, sales.df.Forecast_sales)
sales_error


# In[152]:


np.sqrt(sales_error)


# In[ ]:


#Using ARIMA
#ARIMA - Auto Regressive(p) Integrated(d) Moving Average(q)


# In[154]:


from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# to predict value of q
plot_acf(sales)


# In[ ]:


# At 0, correlation between actual and predicted value is 1
# Here 0,1 are above crtical limits
#Therefore, we can take q=1,2


# In[155]:


#To predict value of p
plot_pacf(sales)


# In[ ]:


# Here also, we can p=1(above critical level)
# Also d=0, as the graph is stationary


# In[157]:


# AR Model
X = sales.values
sales.shape
train = X[0:40] # 40 data as train data
test = X[40:]  # 8 data as test data


# In[180]:


train.size


# In[181]:


test.size


# In[158]:


from statsmodels.tsa.ar_model import AR

model_ar = AR(train)
model_ar_fit = model_ar.fit()


# In[184]:


predictions = model_ar_fit.predict(start=40,end=47)
predictions.size


# In[185]:


plt.plot(test)
plt.plot(predictions,color='red')


# In[91]:


sales.plot()


# In[186]:


from statsmodels.tsa.arima_model import ARIMA


# In[234]:


#p,d,q  p = periods taken for autoregressive model
#d -> Integrated order, difference
# q periods in moving average model
# lesser value of aic , the better
model_arima = ARIMA(train,order=(4,0,4))
model_arima_fit = model_arima.fit()
print(model_arima_fit.aic)


# In[235]:


predictions= model_arima_fit.forecast(steps=8)[0]
predictions


# In[236]:


plt.plot(test)
plt.plot(predictions,color='red')


# In[209]:


mean_squared_error(test,predictions)


# In[210]:


np.sqrt(mean_squared_error(test,predictions))


# In[211]:


import itertools
p=d=q=range(0,7)
pdq = list(itertools.product(p,d,q))
pdq


# In[212]:


import warnings
warnings.filterwarnings('ignore')
for param in pdq:
    try:
        model_arima = ARIMA(train,order=param)
        model_arima_fit = model_arima.fit()
        print(param,model_arima_fit.aic)
    except:
        continue

