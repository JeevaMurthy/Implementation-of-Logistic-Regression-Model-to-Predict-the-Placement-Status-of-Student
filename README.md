# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Start
2. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).
3. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.
4. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.
5. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.
6. Stop

## Program & Output:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JEEVA K
RegisterNumber:  212223230090
*/
```
```python
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
```

![image](https://github.com/user-attachments/assets/490a56a2-4c9a-480b-9857-6fdad7085c7a)

```python
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

```

![image](https://github.com/user-attachments/assets/a1984bf0-fa73-4c45-a072-f9b829880e39)

```python
data1.isnull().sum()
```

![image](https://github.com/user-attachments/assets/32686c90-cf04-4f22-9a76-b41f1e77707f)

```python
data1.duplicated().sum()
```

![image](https://github.com/user-attachments/assets/ae09249b-100d-4be1-af0f-a9e86cfccce0)

```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 
```

![image](https://github.com/user-attachments/assets/67524388-af9e-4e27-b973-4278029335e8)

```python
x=data1.iloc[:,:-1]
x
```

![image](https://github.com/user-attachments/assets/584d8ad4-093c-43ea-88c8-9d1c6add9cc7)


```python
y=data1["status"]
y

```

![image](https://github.com/user-attachments/assets/24f3a626-d593-462b-aad9-8299c1bdb411)


```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```

![image](https://github.com/user-attachments/assets/c8ab2b70-7972-4bf1-b3ec-7fccedfb46a4)

```python
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```

![image](https://github.com/user-attachments/assets/3c157556-0622-4f12-aa6e-53b628acad06)

```python
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```

![image](https://github.com/user-attachments/assets/959838f4-8f21-4781-8815-333ed375a16d)

```python
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```

![image](https://github.com/user-attachments/assets/d0f25f58-b84e-4058-bed0-33d5a152b759)

```python
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

![image](https://github.com/user-attachments/assets/2d435be3-d60e-4b59-8605-119c29ba5da2)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
