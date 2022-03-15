#!/usr/bin/env python
# coding: utf-8

# ## A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.
# 
# ## In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits
# 
# ## The company wants to know:
# 
# ### 1. Which variables are significant in predicting the demand for shared bikes.
# ### 2. How well those variables describe the bike demands
# 

# In[739]:


## import libraries

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[740]:


## Load dataset 

path= 'D:/Mahadev/upgrad/Assignments/day.csv'
df = pd.read_csv(path)


# In[741]:


df.head()


# In[742]:


df.describe()


# In[743]:


df.info()


# In[744]:


df.drop(['dteday', 'casual', 'registered', 'instant'], axis=1, inplace= True)


# In[745]:


df.head()


# In[746]:


df.holiday.unique()

# season = 1,2,3,4
# yr = 0,1
# mnth = 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12
# weekday = 6, 0, 1, 2, 3, 4, 5
# workingday = 0, 1
# weathersit = 1,2,3
# holiday = 0,1 


# In[747]:


df.corr()


# In[748]:


# plot heatmap 

#fig, ax = plt.subplots(figsize=(10,10)) 
sns.heatmap(df[['temp','atemp','hum','windspeed','cnt']].corr(), cmap="YlGnBu", annot=True)
plt.show()


# In[749]:


df.head()


# ### Univariate and Bivariate Analysis 

# In[750]:


sns.boxplot( y=df["cnt"] )
plt.ylabel('Demand for Bikes')
plt.show()


# In[751]:


## Bar plot to show demand as per each month

sns.barplot(y='cnt', x='mnth', data=df)
plt.xticks(rotation=90)
plt.show()


# In[752]:


## Bar plot to show demand as per Temp

sns.displot(df['temp'], bins=25)
plt.xlabel('Temperature')
plt.show()


# In[753]:


sns.barplot(y='cnt', x='weekday', data=df)
plt.xticks()
plt.show()


# In[754]:


sns.catplot(x="yr", y="cnt", kind="box", data=df)


# In[756]:


sns.boxplot(x="season", y="cnt", data=df)


# In[738]:


sns.catplot(x="workingday", y="cnt", kind="box", data=df)


# In[570]:


sns.catplot(x="weekday", y="cnt", kind="box", data=df)


# In[571]:


sns.catplot(x="mnth", y="cnt", kind="box", data=df)


# In[572]:


plt.figure(figsize = (15,30))
sns.pairplot(data=df,vars=['cnt', 'temp', 'atemp', 'hum','windspeed'])
plt.show()


# In[704]:


# season = 1,2,3,4 1:spring, 2:summer, 3:fall, 4:winter
# yr = 0,1
# mnth = 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12
# weekday = 6, 0, 1, 2, 3, 4, 5
# workingday = 0, 1
# weathersit = 1,2,3
# holiday = 0,1 

##########

df.season.replace({1:"spring", 2:"summer", 3:"fall", 4:"winter"},inplace = True)

df.weathersit.replace({1:'good',2:'moderate',3:'bad'},inplace = True)

df.mnth.replace({1: 'jan',2: 'feb',3: 'mar',4: 'apr',5: 'may',6: 'jun',
                  7: 'jul',8: 'aug',9: 'sept',10: 'oct',11: 'nov',12: 'dec'},inplace = True)

df.weekday.replace({0: 'sun',1: 'mon',2: 'tue',3: 'wed',4: 'thu',5: 'fri',6: 'sat'},inplace = True)

df.head()


# In[705]:


df.weathersit.unique()


# In[706]:


df = pd.get_dummies(data= df,columns=["season","mnth","weekday"],drop_first=True)
df = pd.get_dummies(data= df,columns=["weathersit"], drop_first= True)


# In[702]:


df.head()


# In[707]:


# Droping instant column# Droping instant column

df.drop(['instant'], axis = 1, inplace = True)

# Dropping dteday
df.drop(['dteday'], axis = 1, inplace = True)

# Dropping casual and registered column

df.drop(['casual'], axis = 1, inplace = True)
df.drop(['registered'], axis = 1, inplace = True)


# In[708]:


#y to contain only target variable
y= df['cnt']

#X is all remainign variable also our independent variables
X= df.drop(['cnt'], axis= 1)


# In[709]:


# Test train split 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print(X_train.shape)
X_test.shape


# In[710]:


from sklearn.preprocessing import MinMaxScaler

#Use Normalized scaler to scale
scaler = MinMaxScaler()


#Fit and transform training set only
X_train[['temp', 'atemp', 'hum', 'windspeed']] = scaler.fit_transform(X_train[['temp', 'atemp', 'hum', 'windspeed']])


# In[711]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

lr= LinearRegression()
lr.fit(X_train, y_train)


# In[712]:


#Cut down number of features to 15 using automated approach
rfe = RFE(lr, n_features_to_select=15)
rfe.fit(X_train,y_train)


# In[672]:


#Columns selected by RFE and their weights
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[720]:


def model_lr(cols):
    X_train_sm = sm.add_constant(X_train[cols])
    lm = sm.OLS(y_train, X_train_sm).fit()
    print(lm.summary())
    return lm

def get_vif(cols):
    df1 = X_train[cols]
    #print(df1.head())
    #print(df1.shape)
    vif = pd.DataFrame()
    vif['Features'] = df1.columns
    vif['VIF'] = [variance_inflation_factor(df1.values, i) for i in range(df1.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)
    print(vif.sort_values(by='VIF',ascending=False))


# In[721]:


X_train.columns[rfe.support_]


# In[722]:


X_train_rfe = X_train[['yr', 'holiday', 'workingday', 'atemp', 'hum', 'windspeed',
                    'season_spring', 'season_winter', 'mnth_dec', 'mnth_jul', 'mnth_nov',
                    'weekday_sat', 'weekday_sun', 'weathersit_good', 'weathersit_moderate']]


# In[723]:


cols = ['yr', 'holiday', 'workingday', 'atemp', 'hum', 'windspeed',
       'season_spring', 'season_winter', 'mnth_dec', 'mnth_jul', 'mnth_nov',
       'weekday_sat', 'weekday_sun', 'weathersit_good', 'weathersit_moderate']

model_lr(cols)
get_vif(cols)


# In[759]:


plt.figure(figsize = (15,10))
sns.heatmap(df.corr(), cmap='GnBu', annot=True)
plt.show()


# In[728]:


def build_model_final(X,y):
    lr1 = LinearRegression()
    lr1.fit(X,y)
    return lr1


columns= ['yr', 'holiday', 'atemp', 'hum', 'windspeed',
       'season_spring', 'season_winter', 'mnth_dec', 'mnth_jul', 'mnth_nov']
        
lr_final = build_model_final(X_train[columns],y_train)
print(lr_final.intercept_,lr_final.coef_)


# In[730]:


y_train_pred = lr_final.predict(X_train[columns])

def plot_res_dist(act, pred):
    sns.distplot(act-pred)
    plt.title('Error Terms')
    plt.xlabel('Errors')


# In[731]:


plot_res_dist(y_train, y_train_pred)


# In[732]:


residual = (y_train - y_train_pred)
plt.scatter(y_train,residual)
plt.ylabel("y_train")
plt.xlabel("Residual")
plt.show()


# In[733]:


# scale test variables 

#Scale variables in X_test
vars_ = ['temp','atemp','hum','windspeed']

#Test data to be transformed only, no fitting
X_test[vars_] = scaler.transform(X_test[vars_])


# In[735]:


#Predict the values for test data
y_test_pred = lr_final.predict(X_test[columns])


# In[736]:


# find R2 value 

r2_score(y_test,y_test_pred)
'yr', 'holiday', 'atemp', 'hum', 'windspeed',
       'season_spring', 'season_winter', 'mnth_dec', 'mnth_jul', 'mnth_nov'


# ### Significant variables which predict the demand for shared bikes are as per LR
# 
# ##### holiday
# ##### atemp
# ##### hum
# ##### windspeed
# ##### Season
# ##### months(July, November, December)
# ##### Year 
# 

# In[ ]:




