import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import metrics
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge


df = pd.read_csv('train.csv')
df.head()
df.describe()
df.rename(columns={'X1': 'Item_ID', 'X2': 'Item_Weight','X3':'Item_Fat_Amount','X4':'Item_Store_Allocation','X5':'Item_Category','X6':'Item_Price','X7':'Store_ID','X8':'Store_Establishment_Year','X9':'Store_Size','X10':'Store_Location_Type','X11':'Store_Type'}, inplace=True)
df.head()
#Preprocessing
#checking if there's null values in the dataset
df.isnull().sum()
#Filling Null Values
def impute_Item_Weight(df):
    # #Determine the average weight per item:
    item_avg_weight = df.groupby(["Item_ID"])["Item_Weight"].mean()
    item_avg_weight

    #Get a boolean variable specifying missing Item_Weight values
    miss_bool = df['Item_Weight'].isnull()

    #Impute data and check #missing values before and after imputation to confirm
    print('Orignal #missing: %d'% sum(miss_bool))
    df.loc[miss_bool,'Item_Weight'] = df.loc[miss_bool,'Item_ID'].apply(lambda x: item_avg_weight.loc[x])
    print('Final #missing: %d'% sum(df['Item_Weight'].isnull()))
#filling gaps in item weight by mean
impute_Item_Weight(df)

df=df.dropna(axis=0)
df.isnull().sum()
from scipy.stats import mode

def impute_Store_size(df):
    #Determing the mode for each
    outlet_size_mode = df.pivot_table(values='Store_Size', columns='Store_Type',aggfunc=(lambda x:mode(x).mode[0]) )
    print('Mode for each Store_Type:')
    print(outlet_size_mode)

    #Get a boolean variable specifying missing Item_Weight values
    miss_bool = df['Store_Size'].isnull()

    #Impute data and check #missing values before and after imputation to confirm
    print('\nOrignal #missing: %d'% sum(miss_bool))
    df.loc[miss_bool,'Store_Size'] = df.loc[miss_bool,'Store_Type'].apply(lambda x: outlet_size_mode[x])
    print('\nFinal #missing: %d'%sum(df['Store_Size'].isnull()))
#impute_Store_size(df)

df['Store_Size'].fillna(method ='ffill', inplace = True)
#There are some zero's in item store allocation , so we'll be filling them by mean
df["Item_Store_Allocation"]
#as we can observe from the above, that there are zero value in item store allocation , so we will preform pre-processing on them
visibility_avg = df.pivot_table(values='Item_Store_Allocation', index='Item_ID')
print(visibility_avg)
miss_bool = (df['Item_Store_Allocation'] == 0)
print(miss_bool)
df.loc[miss_bool,'Item_Store_Allocation'] = df.loc[miss_bool,'Item_ID'].apply(lambda x: visibility_avg.Item_Store_Allocation[x])
df['Combined_Categories'] = df['Item_ID'].apply(lambda x:x[0:2])

df['Combined_Categories'] = df['Combined_Categories'].map({'FD':'Food',
                                                         'DR':'Drinks',
                                                         'NC':'Non-consumables'})

from scipy import stats

F, p = stats.f_oneway(df[df.Combined_Categories=='Food'].Y,
                      df[df.Combined_Categories=='Drinks'].Y,
                      df[df.Combined_Categories=='Non-consumables'].Y)

print("F value = ",F," P value = ",p)

df['Store_Years'] = 2013 - df.Store_Establishment_Year

df.replace(to_replace ="LF",value ="Low Fat",inplace=True)
df.replace(to_replace ="low fat",value ="Low Fat",inplace=True)
df.replace(to_replace ="reg",value ="Regular",inplace=True)
df.Item_Fat_Amount.value_counts()

df.loc[df.Combined_Categories=='Non-consumables','Item_Fat_Amount']='indigestible'

import numpy as np
import matplotlib.pyplot as plt
plt.bar(df['Item_Fat_Amount'].unique(), df['Item_Fat_Amount'].value_counts(), width=0.5, bottom=None, align='center', data=df)
plt.title('Item_Fat_Amount Distribution')
plt.xlabel('Item_Fat_Amount')
plt.ylabel('Frequency')
print('Item_Fat_Amount:\n',df['Item_Fat_Amount'].value_counts())


import numpy as np
import matplotlib.pyplot as plt
plt.bar(df['Store_Size'].value_counts().index, df['Store_Size'].value_counts(), width=0.5, bottom=None, align='center', data=df)
plt.title('Store_Size')
#plt.xticks(rotation='vertical')
plt.xlabel('Store_Size')
plt.ylabel('Frequency')

df['Store_Type'].value_counts()
import numpy as np
import matplotlib.pyplot as plt
plt.bar(df['Store_Type'].value_counts().index, df['Store_Type'].value_counts(), width=0.5, bottom=None, align='center', data=df)
plt.title('Store_Type Distribution')
#plt.xticks(rotation='vertical')
plt.xlabel('Store_Type')
plt.ylabel('Frequency')

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
#df['Item_ID'] = encoder.fit_transform(df['Item_ID'])

df['Item_Fat_Amount'] = encoder.fit_transform(df['Item_Fat_Amount'])

df['Combined_Categories'] = encoder.fit_transform(df['Combined_Categories'])

df['Item_Category'] = encoder.fit_transform(df['Item_Category'])


#df['Store_ID'] = encoder.fit_transform(df['Store_ID'])

df['Store_Size'] = encoder.fit_transform(df['Store_Size'])

df['Store_Location_Type'] = encoder.fit_transform(df['Store_Location_Type'])

df['Store_Type'] = encoder.fit_transform(df['Store_Type'])

df.head()

#Normalization on Item Price
from sklearn.preprocessing import MinMaxScaler

norm=MinMaxScaler().fit(df[["Item_Price"]])
df["Item_Price"]=norm.transform(df[["Item_Price"]])

norm=MinMaxScaler().fit(df[["Item_Weight"]])
df["Item_Weight"]=norm.transform(df[["Item_Weight"]])

norm=MinMaxScaler().fit(df[["Store_Years"]])
df["Store_Years"]=norm.transform(df[["Store_Years"]])

norm=MinMaxScaler().fit(df[["Item_Store_Allocation"]])
df["Item_Store_Allocation"]=norm.transform(df[["Item_Store_Allocation"]])

norm=MinMaxScaler().fit(df[["Combined_Categories"]])
df["Combined_Categories"]=norm.transform(df[["Combined_Categories"]])

norm=MinMaxScaler().fit(df[["Item_Category"]])
df["Item_Category"]=norm.transform(df[["Item_Category"]])


norm=MinMaxScaler().fit(df[["Store_Size"]])
df["Store_Size"]=norm.transform(df[["Store_Size"]])


norm=MinMaxScaler().fit(df[["Store_Location_Type"]])
df["Store_Location_Type"]=norm.transform(df[["Store_Location_Type"]])


norm=MinMaxScaler().fit(df[["Store_Type"]])
df["Store_Type"]=norm.transform(df[["Store_Type"]])


norm=MinMaxScaler().fit(df[["Store_Establishment_Year"]])
df["Store_Establishment_Year"]=norm.transform(df[["Store_Establishment_Year"]])
df["Item_Weight"]


#Viewing the coorelations
corrmat = df.corr()
f,ax = plt.subplots(figsize = (15,10))
sns.heatmap(corrmat,annot=True,ax=ax,cmap="YlGnBu",linewidths=0.1,fmt=".2f",square=True)
plt.show()


#Filtering the unneeded columns before proceeding into model training¶
from xgboost import XGBRegressor
train=df
X=train.drop(['Y','Store_ID','Item_ID'],axis=1)
y = df['Y']
xgb = XGBRegressor(n_estimators=100)
xgb.fit(X, y)


from sklearn.model_selection import train_test_split
train=df
#train.drop(['Item_ID','Item_Fat_Amount','Store_Years','Combined_Categories','Item_Category','Store_ID','Store_Establishment_Year','Store_Size', 'Store_Location_Type',
      # 'Store_Type'],axis=1,inplace=True)
#X=train.drop(['Y','Item_Category','Item_Fat_Amount','Item_ID','Store_Size','Store_ID','Store_Establishment_Year'],axis=1)
X=train.drop(['Y','Combined_Categories','Item_ID','Store_ID','Store_Years'],axis=1)

y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
df.isnull().sum()


#Performing all the pre-processing done on train data on test
df_test = pd.read_csv('test.csv')
df_test.rename(columns={'X1': 'Item_ID', 'X2': 'Item_Weight','X3':'Item_Fat_Amount','X4':'Item_Store_Allocation','X5':'Item_Category','X6':'Item_Price','X7':'Store_ID','X8':'Store_Establishment_Year','X9':'Store_Size','X10':'Store_Location_Type','X11':'Store_Type'}, inplace=True)
df_test['Item_Weight'].fillna(df_test['Item_Weight'].mean(), inplace=True)

Store_Size=df_test.pivot_table(values='Store_Size', columns='Store_Type', aggfunc=(lambda x: x.mode()[0]))
df_test.loc[df_test['Store_Size'].isnull(), 'Store_Size'] = df_test.loc[df_test['Store_Size'].isnull(), 'Store_Type'].apply(lambda x: Store_Size[x])

dic = {'Grocery Store':'Small'}
s = df_test.Store_Type.map(dic)
df_test.Store_Size= df_test.Store_Size.combine_first(s)

dic = {"Tier 2":"Small"}
s = df.Store_Location_Type.map(dic)
df_test.Store_Size = df_test.Store_Size.combine_first(s)
df_test.Store_Size.value_counts()


visibility_avg = df_test.pivot_table(values='Item_Store_Allocation', index='Item_ID')
miss_bool = (df_test['Item_Store_Allocation'] == 0)
df_test.loc[miss_bool,'Item_Store_Allocation'] = df_test.loc[miss_bool,'Item_ID'].apply(lambda x: visibility_avg.Item_Store_Allocation[x])

df_test['Combined_Categories'] = df_test['Item_ID'].apply(lambda x:x[0:2])
df_test['Combined_Categories'] = df_test['Combined_Categories'].map({'FD':'Food',
                                                         'DR':'Drinks',
                                                         'NC':'Non-consumables'})

df_test.replace(to_replace ="LF",value ="Low Fat",inplace=True)
df_test.replace(to_replace ="low fat",value ="Low Fat",inplace=True)
df_test.replace(to_replace ="reg",value ="Regular",inplace=True)

df_test['Store_Years'] = 2013 - df_test.Store_Establishment_Year
df_test.loc[df_test.Combined_Categories=='Non-consumables','Item_Fat_Amount']='indigestible'

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df_test['Item_Fat_Amount'] = encoder.fit_transform(df_test['Item_Fat_Amount'])

df_test['Combined_Categories'] = encoder.fit_transform(df_test['Combined_Categories'])

df_test['Item_Category'] = encoder.fit_transform(df_test['Item_Category'])

df_test['Store_Size'] = encoder.fit_transform(df_test['Store_Size'])

df_test['Store_Location_Type'] = encoder.fit_transform(df_test['Store_Location_Type'])

df_test['Store_Type'] = encoder.fit_transform(df_test['Store_Type'])




norm=MinMaxScaler().fit(df_test[["Item_Price"]])
df_test["Item_Price"]=norm.transform(df_test[["Item_Price"]])

norm=MinMaxScaler().fit(df[["Item_Weight"]])
df_test["Item_Weight"]=norm.transform(df_test[["Item_Weight"]])

norm=MinMaxScaler().fit(df_test[["Store_Years"]])
df_test["Store_Years"]=norm.transform(df_test[["Store_Years"]])

norm=MinMaxScaler().fit(df_test[["Item_Store_Allocation"]])
df_test["Item_Store_Allocation"]=norm.transform(df_test[["Item_Store_Allocation"]])

norm=MinMaxScaler().fit(df_test[["Combined_Categories"]])
df_test["Combined_Categories"]=norm.transform(df_test[["Combined_Categories"]])

norm=MinMaxScaler().fit(df_test[["Item_Category"]])
df_test["Item_Category"]=norm.transform(df_test[["Item_Category"]])


norm=MinMaxScaler().fit(df_test[["Store_Size"]])
df_test["Store_Size"]=norm.transform(df_test[["Store_Size"]])


norm=MinMaxScaler().fit(df_test[["Store_Location_Type"]])
df_test["Store_Location_Type"]=norm.transform(df_test[["Store_Location_Type"]])


norm=MinMaxScaler().fit(df_test[["Store_Type"]])
df_test["Store_Type"]=norm.transform(df_test[["Store_Type"]])


norm=MinMaxScaler().fit(df_test[["Store_Establishment_Year"]])
df_test["Store_Establishment_Year"]=norm.transform(df_test[["Store_Establishment_Year"]])


MyTest=df_test
MyTest.drop(['Combined_Categories','Item_ID','Store_ID','Store_Years'],axis=1,inplace=True)


#2- Model training & testing
#Model 1:
#Linear Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
model = LinearRegression(normalize=True)
#train(model, X_train, X_test)
model.fit(X_train,y_train)
pred=model.predict(X_test)
print("MSE:",mean_squared_error(y_test,pred))
print("Mean Absolute Error:",mean_absolute_error(y_test,pred))

#Model 2 :¶
#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(max_depth=30, min_samples_leaf=120)
model.fit(X_train,y_train)
pred=model.predict(X_test)
#Test_Pred=model.predict(df_test)
#pd.DataFrame(Test_Pred, columns=['predictions']).to_csv('prediction2.csv')

print("MSE:",mean_squared_error(y_test,pred))
print("Mean Absolute Error:",mean_absolute_error(y_test,pred))


#Model 3:
#Random Forest Regressor
import math
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=100, max_depth=50
)
model.fit(X_train,y_train)
pred=model.predict(X_test)

Test_Pred=model.predict(MyTest)
#pd.DataFrame(Test_Pred, columns=['Y']).to_csv('predictionzwhythis.csv')
print("MSE:",math.sqrt(mean_squared_error(y_test,pred)))
print("Mean Absolute Error:",mean_absolute_error(y_test,pred))


import xgboost

model =  XGBRegressor()
model.fit(X_train,y_train)
pred=model.predict(X_test)

#Test_Pred=model.predict(MyTest)
#pd.DataFrame(Test_Pred, columns=['Y']).to_csv('prediction3.csv')
print("MSE:",mean_squared_error(y_test,pred))
print("Mean Absolute Error:",mean_absolute_error(y_test,pred))


from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=3,min_samples_leaf=100,subsample=0.7,alpha=0.9)
model.fit(X_train,y_train)
pred=model.predict(X_test)

Test_Pred=model.predict(MyTest)
pd.DataFrame(Test_Pred, columns=['Y']).to_csv('predictio99.csv')
print("MSE:",np.sqrt(mean_squared_error(y_test,pred)))
print("Mean Absolute Error:",mean_absolute_error(y_test,pred))


