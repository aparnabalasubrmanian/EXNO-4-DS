# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1: Read the given Data.
STEP 2: Clean the Data Set using Data Cleaning Process.
STEP 3: Apply Feature Scaling for the feature in the data set.
STEP 4: Apply Feature Selection for the feature in the data set.
STEP 5: Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:

1. Filter Method
2. Wrapper Method
3. Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![322341265-ff7cccdb-bd60-45dd-9dc6-47b49a563538](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/06370a43-1c87-4cd3-9606-8e7c10c78411)

```
data.isnull().sum()
```
![322341376-59b6ad1a-bf12-48c2-a294-5aac15886114](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/cd89e221-9dd3-49b6-9ac8-1d3223a0dd1e)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![322341457-33a8312b-2d89-4abd-bb0d-35cdfa2b7b44](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/5db36bae-0913-4737-8bbf-96790ac19b8d)
```
data2=data.dropna(axis=0)
data2
```
![322341553-84785d2a-5779-49fa-b399-8edbd5892b9e](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/cce26f48-37f8-462a-ad9a-749d1fa85c30)
```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![322341702-dcbc0758-336f-4d51-8fe8-e685afb2ed93](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/08492127-b333-4d1f-a9ee-c71b55502e53)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![322341813-0f52b3b1-2144-4e94-ba27-03ca586215f0](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/7fe1a749-0f06-4201-a687-fa706dadac03)

```
data2
```
![322341909-ffb9c16e-40c4-4439-a29f-b201581a12c8](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/680ad939-4f34-4b7a-aa34-05f8083e9712)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![322342080-de84fef7-7332-46d8-ab50-471d0169a901](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/37080519-8873-435a-ba6f-175c2e152385)

```
columns_list=list(new_data.columns)
print(columns_list)
```
![321852182-ed5ee91b-8bf1-4e8c-a8cf-3d9d96a5589d](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/e96ea321-eea6-427d-84f3-2bd96ad242a8)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![321852508-5ddd2e17-6819-4f8b-acba-90cf4897797a](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/6da28c47-8a09-4f11-b9db-96cbf479d9d3)
```
y=new_data['SalStat'].values
print(y)
```
![321852572-ba4ffd91-efc5-4987-8486-49b620e17f41](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/d06fbce5-75e5-452c-9159-ea6aaf4f3380)
```
x=new_data[features].values
print(x)
```
![321852650-e0f0561b-97cf-4176-bbb5-19614b52c408](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/9550ea60-c8f9-4ae2-9303-21cc709e624c)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![322342343-4a77fca3-3948-4c11-bdc0-514a61afceab](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/5092362a-bc87-41bf-9893-b2d4acf4d501)

```

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![321852840-3f22074f-9d4d-4758-962c-57f12b70146b](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/c07483b0-bfba-4058-996e-f20a2ce46f3c)
```

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![322336946-e31a4e64-7fca-4531-a188-48a5ff07266e](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/c7d45d47-7602-4806-8b48-52eb459dc3dc)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![322337033-d7291f25-f68a-4c7b-b781-a745058b2770](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/fa862143-fc3f-4723-afac-a9c163afb1fb)
```
data.shape
```
![322337115-bcaaa675-3cb4-477f-83b5-5d4fdea4d996](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/8afde3c9-98a9-4404-b892-f9df67443dab)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![322337195-9263244e-6532-4827-8413-9a0633efbf7d](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/606e53bc-7221-42e9-870d-34909c6a11b7)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![322342594-f6ad3642-5ec2-4c93-88c0-0019e3127b90](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/d02459b6-a30f-404a-8e63-c80e5cf119f8)
```
tips.time.unique()
```
![322337526-f4b72a8c-b35a-40df-8649-9123983f7704](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/e18500e3-7f4c-410b-be19-fd3ce9b259d3)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![322342713-9cd46b28-eda2-44b7-82c2-277dfac3bcc5](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/2ba6c24c-25b5-4db3-bb6d-85dd8a007679)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![322342804-2fff30ed-ac94-411c-9256-51bf5928d9b9](https://github.com/aparnabalasubrmanian/EXNO-4-DS/assets/123351172/a2350ffd-8c06-4779-b72a-bc7d1a0126ce)


# RESULT:
Thus,both Feature selection and Feature scaling has been used and executed in the given dataset.

