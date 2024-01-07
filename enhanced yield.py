#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pwd run #print working directory


# In[3]:


df=pd.read_csv("C:/Users/kunis/Downloads/archive (1)/crop_yield.csv")
df.head()


# In[4]:


df.isnull().sum()


# In[5]:


df.duplicated().sum()


# In[6]:


df.describe()


# In[7]:


df['Annual_Rainfall']


# In[8]:


def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True


# In[9]:


to_drop=df[df['Annual_Rainfall'].apply(isStr)].index 


# In[10]:


df=df.drop(to_drop)


# In[11]:


df


# In[12]:


df.info()


# In[13]:


plt.figure(figsize=(10,20))
sns.countplot(y=df['State'])


# In[14]:


country=(df['State'].unique())
print(country)


# In[15]:


yield_per_country=[]
for state in country:
    yield_per_country.append(df[df['State']==state]['Yield'].sum())


# In[16]:


yield_per_country


# In[17]:


plt.figure(figsize=(10,20))
sns.barplot(y=country,x=yield_per_country)


# In[18]:


plt.figure(figsize=(10,20))
sns.countplot(y=df['Crop'])


# In[19]:


crops=(df['Crop'].unique())


# In[20]:


yield_per_item=[]
yield_per_crop=[]
for crop in crops:
    yield_per_crop.append(df[df['Crop']==crop]['Yield'].sum())
yield_per_crop


# In[21]:


col=['Crop_Year','Area','Production','Annual_Rainfall','Fertilizer','Pesticide','Crop','Season','State','Yield' ]
df=df[col]
df


# In[22]:


x=df.drop('Yield',axis=1)
x1=x.drop('Season',axis=1)
y=df['Yield']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=42)
X_train.shape
X_train


# In[23]:


#optional
X_test.shape


# In[24]:


X_train


# In[25]:


from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
# Define your transformers
ohe = OneHotEncoder(drop='first')
scaler = StandardScaler()

preprocesser = ColumnTransformer(
    transformers=[
        ('onehotencoder', ohe, [6,7]),  
        ('standardization', scaler, [0,1,2,3,4,5])  
],
    
remainder='passthrough'
)
X_train_dummy = preprocesser.fit_transform(X_train)
X_test_dummy = preprocesser.transform(X_test)
X_train_dummy


# In[26]:


preprocesser


# In[27]:


from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.metrics import mean_squared_error
models={
    'lr':LinearRegression(),
    'lss':Lasso(),
    'rg':Ridge(),
    'Knr':KNeighborsRegressor(),
    'dtr':DecisionTreeRegressor()   

}

for name,mod in models.items():
    mod.fit(X_train_dummy,y_train)
    y_pred=mod.predict(X_test_dummy)
    print(f"{name} MSE : {mean_squared_error(y_test,y_pred)} Score {r2_score(y_test,y_pred)}")


# In[28]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming you have already run the code for model training and evaluation

# Lists to store the metrics for plotting
model_names = []
mse_values = []
r2_scores = []

# Iterate through the models and calculate metrics
for name, model in models.items():
    model.fit(X_train_dummy, y_train)
    y_pred = model.predict(X_test_dummy)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_names.append(name)
    mse_values.append(mse)
    r2_scores.append(r2)

# Plotting
x = np.arange(len(model_names))

fig, ax1 = plt.subplots()

# Bar plots for MSE
ax1.bar(x - 0.2, mse_values, 0.4, label='MSE', color='b')
ax1.set_xlabel('Models')
ax1.set_ylabel('MSE', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Bar plots for R2 score
ax2 = ax1.twinx()
ax2.bar(x + 0.2, r2_scores, 0.4, label='R2 Score', color='r')
ax2.set_ylabel('R2 Score', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Model names as x-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(model_names)

# Title and legend
plt.title('Model Comparison - MSE and R2 Score')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

plt.show()


# In[34]:


from sklearn.metrics import accuracy_score, precision_score, f1_score, log_loss

# Define a threshold for classification (e.g., 0.5)
threshold = 0.5

for name, mod in models.items():
    mod.fit(X_train_dummy, y_train)
    y_pred = mod.predict(X_test_dummy)

    # Convert regression predictions into binary classifications
    y_pred_binary = [1 if y >= threshold else 0 for y in y_pred]
    y_test_binary = [1 if y >= threshold else 0 for y in y_test]

    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    precision = precision_score(y_test_binary, y_pred_binary)
    f1 = f1_score(y_test_binary, y_pred_binary)
    
    # Calculate log loss (cross-entropy loss) for probabilistic models
    if hasattr(mod, 'predict_proba'):
        y_prob = mod.predict_proba(X_test_dummy)  # Probability predictions
        loss = log_loss(y_test_binary, y_prob)
    else:
        loss = None
    
    print(f"{name} Accuracy: {accuracy}")
    print(f"{name} Precision: {precision}")
    print(f"{name} F1 Score: {f1}")
    print(f"{name} Log Loss: {loss}")
    if loss is not None:
        print(f"{name} Log Loss: {loss}")
    print()


# In[45]:


import matplotlib.pyplot as plt

# Initialize empty lists to store metrics for each model
model_names = list(models.keys())
accuracy_scores = {name: [] for name in model_names}
precision_scores = {name: [] for name in model_names}
f1_scores = {name: [] for name in model_names}
log_losses = {name: [] for name in model_names}

epochs = 10

for epoch in range(epochs):
    for name, mod in models.items():
        mod.fit(X_train_dummy, y_train)
        y_pred = mod.predict(X_test_dummy)

        # Convert regression predictions into binary classifications
        y_pred_binary = [1 if y >= threshold else 0 for y in y_pred]
        y_test_binary = [1 if y >= threshold else 0 for y in y_test]

        accuracy = accuracy_score(y_test_binary, y_pred_binary)
        precision = precision_score(y_test_binary, y_pred_binary)
        f1 = f1_score(y_test_binary, y_pred_binary)

        if hasattr(mod, 'predict_proba'):
            y_prob = mod.predict_proba(X_test_dummy)  # Probability predictions
            loss = log_loss(y_test_binary, y_prob)
        else:
            loss = None

        accuracy_scores[name].append(accuracy)
        precision_scores[name].append(precision)
        f1_scores[name].append(f1)
        

# Create separate plots for each metric for each model
for name in model_names:
    plt.figure(figsize=(10, 6))

    # Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(range(epochs), accuracy_scores[name], label='Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{name} - Accuracy')
    plt.grid(True)

    # Precision
    plt.subplot(2, 2, 2)
    plt.plot(range(epochs), precision_scores[name], label='Precision', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title(f'{name} - Precision')
    plt.grid(True)

    # F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(range(epochs), f1_scores[name], label='F1 Score', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'{name} - F1 Score')
    plt.grid(True)

    # Log Loss
    


# In[29]:


import matplotlib.pyplot as plt

# Initialize empty lists to store metrics for each epoch
accuracy_scores = []

epochs = 10

for epoch in range(epochs):
    for name, mod in models.items():
        mod.fit(X_train_dummy, y_train)
        y_pred = mod.predict(X_test_dummy)

        # Convert regression predictions into binary classifications
        y_pred_binary = [1 if y >= threshold else 0 for y in y_pred]
        y_test_binary = [1 if y >= threshold else 0 for y in y_test]

        accuracy = accuracy_score(y_test_binary, y_pred_binary)

        accuracy_scores.append(accuracy)

# Create a plot for accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(epochs * len(models)), accuracy_scores, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.grid(True)
plt.show()


# In[47]:


dtr=DecisionTreeRegressor()
dtr.fit(X_train_dummy,y_train)
dtr.predict(X_test_dummy)


# In[48]:


def prediction(Crop_Year,Area,Production,Annual_Rainfall,Fertilizer,Pesticide,Crop,State):
    features=np.array([[Crop_Year,Area,Production,Annual_Rainfall,Fertilizer,Pesticide,Crop,State]])
    transformed_features=preprocesser.transform(features)
    predicted_value=dtr.predict(transformed_features).reshape(1,-1)
    return predicted_value[0]
Crop_Year=2007
Area=75.0
Production=196
Annual_Rainfall=500000
Fertilizer=10005.38
Pesticide=12.00
Crop='Barley'
State='Delhi'

result=prediction(Crop_Year,Area,Production,Annual_Rainfall,Fertilizer,Pesticide,Crop,State)
result


# In[49]:


import pickle
with open('dtrr.pkl','wb') as file:
    pickle.dump(dtr,file)
with open('preprocesserr.pkl','wb') as file:
    pickle.dump(preprocesser,file)


# In[1]:


import sklearn
print(sklearn.__version__)


# In[ ]:




