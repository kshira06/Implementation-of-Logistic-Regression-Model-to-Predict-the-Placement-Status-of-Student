# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices. 

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Kshira K
RegisterNumber: 212224040166
```
import pandas as pd
data = pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)
data.head()
data1.isnull().sum()
data1.duplicated().sum()
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
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
### Placement data

![Screenshot 2024-09-05 091903](https://github.com/user-attachments/assets/8d046e9f-dd45-4d6a-bcde-56a98f63a7db)

### Salary

![Screenshot 2024-09-05 091912](https://github.com/user-attachments/assets/73536031-4677-487e-bbf9-374f31da3c7c)

### Status data

![Screenshot 2024-09-05 091922](https://github.com/user-attachments/assets/8ba60292-578d-4f3a-8021-6ba72a88da1e)

### Duplicate data

![Screenshot 2024-09-05 091931](https://github.com/user-attachments/assets/a4d4b9a5-ac14-48b7-bb33-0fe9e81a07d8)

### Data

![Screenshot 2024-09-05 092026](https://github.com/user-attachments/assets/3f1527fa-9322-4cc6-8542-f41a2665c8d1)

### X Data

![Screenshot 2024-09-05 092036](https://github.com/user-attachments/assets/354761a3-2c60-4077-b23b-993d2c7fe0f0)

### Y Status

![Screenshot 2024-09-05 092044](https://github.com/user-attachments/assets/d9c01f8e-56b2-4c4f-a021-05c33f028ecf)

### Y Prediction

![Screenshot 2024-09-05 092050](https://github.com/user-attachments/assets/48209989-2585-44e4-abd8-feeda7007160)

### Accuracy

![Screenshot 2024-09-05 092057](https://github.com/user-attachments/assets/46457403-e427-4199-8a6c-f64d1568b4be)

### Confusion Data

![Screenshot 2024-09-05 092121](https://github.com/user-attachments/assets/9ebafaed-8f18-485f-bbd1-42132e51d313)

### Classification Data

![Screenshot 2024-09-05 092138](https://github.com/user-attachments/assets/00024e6c-cf34-4d76-bdb0-12c031cce006)

### Predicted data

![Screenshot 2024-09-05 092147](https://github.com/user-attachments/assets/b6799f67-f449-49b5-9293-93550545788c)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
