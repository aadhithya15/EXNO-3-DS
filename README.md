## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
  ## FEATURE ENCODING
  ```
  import pandas as pd
  df=pd.read_csv("/content/Encoding Data.csv")
  df
  ```
  ![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/c6689464-1019-49fc-9f47-4a93e9640306)

  ## ORDINAL ENCODER:
  ```
  from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
  pm=["Hot","Warm",'Cold']
  e1=OrdinalEncoder(categories=[pm])
  e1.fit_transform(df[["ord_2"]])
  ```
  ![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/3a23c128-58d5-4d04-b168-c1d6b39ee817)
  ```
  df['bo2']=e1.fit_transform(df[["ord_2"]])
  df
  ```
  ![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/e7e973b4-af40-439e-acdb-7fed44bd0ecd)

  ## LABEL ENCODER:
  ```
  le=LabelEncoder()
  dfc=df.copy()
  dfc['ord_2']=le.fit_transform(dfc['ord_2'])
  dfc
  ```
  ![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/823f3927-d59e-46b7-a39c-95d4315ed149)

  ## ONEHOT ENCODER:
  ```
  from sklearn.preprocessing import OneHotEncoder
  ohe=OneHotEncoder(sparse=False)
  df2=df.copy()
  enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
  df2=pd.concat([df2,enc],axis=1)
  pd.get_dummies(df2,columns=["nom_0"])
  ```
  ![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/b057cf9b-a3b1-4893-85cd-f58dad2e0466)
  
  ## BINARY ENCODER:
  ```
  from category_encoders import BinaryEncoder
  import pandas as pd
  df=pd.read_csv("/content/data.csv")
  be=BinaryEncoder()
  nd=be.fit_transform(df['Ord_2'])
  dfb=pd.concat([df,nd],axis=1)
  dfb
  ```
  ![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/9d8a661e-9d42-4e73-9457-54fa1bd31a1c)

  ## TARGET ENCODER
  ```
  from category_encoders import TargetEncoder
  te=TargetEncoder()
  cc=df.copy()
  new=te.fit_transform(X=cc["City"],y=cc["Target"])
  cc
  ```
  ![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/3dbd6a29-ee1f-430e-b866-27c84ab13acc)

# DATA TRANSFORMATION:
```
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv('/content/Data_to_Transform.csv')
df

```
![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/4294bc06-850e-4ec1-8af7-d95702e8a1b8)
```
df.skew()
```
![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/ebd5c776-8115-4640-b363-a60d166be890)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/f40bfb73-e1cc-4aa7-9134-bfee1877a017)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/ca0ef93a-de6e-42ea-b108-dd5d88b63c72)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/04f31da2-07e2-4950-9004-c50e5fd94c4e)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/9ffdc419-6a96-4370-985a-45de0ee0bc8c)
```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/e73b83d5-0fa1-446f-b3b1-723507415ef7)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df.skew()
```
![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/4786e948-d30e-4bd4-9135-5119ad29878e)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/4985bae8-87b9-4c02-84b3-cb784b77a384)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/e65d12dd-8d94-4739-8ece-47e7babe75e0)
```
sm.qqplot(np.log(df["Moderate Negative Skew"]),line='45')
```
![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/ca6514c6-7622-4fd1-a7c5-460864ccb9c0)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/ShanmathiShanmugam/EXNO-3-DS/assets/121243595/1252aab3-27f9-49df-9b7b-0d53b2104cdb)

# RESULT:
The given data has been performed Feature Encoding and Transformation process successfully.


       
