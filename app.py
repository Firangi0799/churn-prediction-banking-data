#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
df = pd.read_csv('archive/Churn_Modelling.csv')


# In[5]:


df.head()


# In[7]:


df.drop(['RowNumber', 'CustomerId', 'Surname'], axis='columns', inplace=True)


# In[8]:


df.head()


# In[38]:


cols_to_scale = ['CreditScore', 'Tenure', 'Balance', 'EstimatedSalary', 'NumOfProducts']


# In[39]:


from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()


# In[40]:


df[cols_to_scale] = scale.fit_transform(df[cols_to_scale])


# In[41]:


df.head()


# In[17]:


df.dtypes


# In[20]:


df['Age'].max()


# In[24]:


import numpy as np

condition = df['Age'] > 30
condition1 = (df['Age'] >= 30) & (df['Age'] <= 59)
condition2 = df['Age'] >= 60

df['YoungAdults'] = np.where(condition, '0', '1')
df['MiddleAged'] = np.where(condition1, '1', '0')
df['OldAged'] = np.where(condition2, '1', '0')

df.head()


# In[25]:


df['Geography'].unique()


# In[27]:


df.drop(['Age'], axis='columns', inplace=True)


# In[28]:


df.head()


# In[29]:


df.replace({'Female': 1, 'Male': 0}, inplace=True)


# In[30]:


df.head


# In[31]:


df.head()


# In[43]:


df1 = pd.get_dummies(data=df, columns=['Geography'])


# In[44]:


df1.head()


# In[54]:


df1.dtypes


# In[55]:


df1[['YoungAdults', 'MiddleAged', 'OldAged']] = df1[['YoungAdults', 'MiddleAged', 'OldAged']].astype(int)


# In[62]:


len(df1.columns)


# In[63]:


from sklearn.model_selection import train_test_split


# In[65]:


X = df1.drop('Exited', axis='columns')
y = df1['Exited']


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=5)


# In[67]:


import tensorflow as tf
from tensorflow import keras

# Define your model architecture
model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(14,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

model.fit(X_train, y_train, epochs=10)


# In[68]:


model.evaluate(X_test, y_test)


# In[69]:


yp = model.predict(X_test)


# In[70]:


y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[71]:


y_pred[:10]


# In[72]:


y_test[:10]


# In[73]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))


# In[75]:


import seaborn as sn
import matplotlib.pyplot as plt
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[76]:


accurate = round((1584+32)/(1584+32+373+11) * 100, 2)
print(accurate)


# In[94]:


tenure_churn_no = df5[df5.Exited==0].Geography
tenure_churn_yes = df5[df5.Exited==1].Geography

plt.xlabel("tenure")
plt.ylabel("Number Of Customers")
plt.title("Customer Churn Prediction Visualiztion")

plt.hist([tenure_churn_yes, tenure_churn_no], rwidth=0.95, color=['green','red'],label=['Churn=1','Churn=0'])
plt.legend()


# In[89]:


df5 = pd.read_csv('archive/Churn_Modelling.csv')


# In[92]:


x = df5['Tenure']
y = df5['Exited']

# Perform linear regression
slope, intercept = np.polyfit(x, y, 1)
regression_line = slope * x + intercept

# Plot the data points and the regression line
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, regression_line, color='red', label='Linear Regression')

# Add labels and title
plt.xlabel('Tenure')
plt.ylabel('Exited')
plt.title('Linear Regression: Tenure vs. Exited')

# Add legend
plt.legend()

# Show the plot
plt.show()


# In[ ]:




