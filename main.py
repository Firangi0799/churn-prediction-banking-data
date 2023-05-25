#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')


# In[2]:


df.head()


# In[3]:


df.drop('customerID',axis='columns', inplace=True)


# In[4]:


df[pd.to_numeric(df.TotalCharges, errors = 'coerce').isnull()]


# In[5]:


df1 = df[df.TotalCharges!=' ']
df1.shape


# In[6]:


pd.to_numeric(df1.TotalCharges)


# In[7]:


df1.TotalCharges = pd.to_numeric(df1.TotalCharges)


# In[8]:


df1.dtypes


# In[9]:


for col in df1:
    print(f'{col} : {df1[col].unique()}')


# In[10]:


df1.replace("No phone service", "No", inplace=True)
df1.replace("No internet service", "No", inplace=True)


# In[11]:


df1.replace({'Yes': 1, 'No': 0}, inplace=True)


# In[12]:


df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'])


# In[13]:


df2.replace({'Female': 1, 'Male': 0}, inplace=True)


# In[14]:


cols_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']


# In[15]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


# In[16]:


df2[cols_scale] = scaler.fit_transform(df2[cols_scale])


# In[17]:


df2.head()


# In[18]:


df2.dtypes


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X = df2.drop('Churn', axis='columns')
y = df2['Churn']


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=5)


# In[22]:


import tensorflow as tf
from tensorflow import keras

# Define your model architecture
model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(26,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

model.fit(X_train, y_train, epochs=5)


# In[23]:


model.evaluate(X_test, y_test)


# In[24]:


yp = model.predict(X_test)


# In[25]:


y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[26]:


y_pred[:10]


# In[27]:


y_test[:10]


# In[28]:


from sklearn.metrics import confusion_matrix , classification_report

print(classification_report(y_test,y_pred))


# In[29]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[30]:


accurate = round((877+232)/(877+122+176+232) * 100, 2)
print(accurate)


# In[31]:


model.save('churn_model.h5')


# In[1]:


jupyter nbconvert --to python main.ipynb


# In[ ]:




