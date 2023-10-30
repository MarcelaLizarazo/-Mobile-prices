# Mobile-prices

### Table of Contents:
1. Introduction
2. Data
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Training and Evaluating Models
    1. Dimensional Reduction (PCA)
6. Conclusion

## 1. Introduction

This project is based on the Mobile prices dataset taken from Kaggle, this dataset contains several factors among which we find the brand, size, weight, image quality, RAM, battery, etc. that interfere in the sales price of a Mobile.  With this data set, I want to estimate a price range that indicates how high the price is in relation to the mentioned features. For these we will apply the logistic regression and KNeighbors Classifier models.

## 2. Data 

This dataset contains all the information related to different characteristics that interfere with the prices of a Mobile, such as:

- **battery_power:** Total energy a battery can store in one time measured in mAh
- **blue:** Has bluetooth or not
- **clock_speed:** speed at which microprocessor executes instructions
- **dual_sim:** Has dual sim support or not
- **fc:** Front Camera mega pixels
- **four_g:** Has 4G or not
- **int_memory:** Internal Memory in Gigabytes
- **m_dep:** Mobile Depth in cm
- **mobile_wt:** Weight of mobile phone
- **n_cores:** Number of cores of processor
- **pc:** Primary Camera mega pixels
- **px_height:** Pixel Resolution Height
- **px_width:** Pixel Resolution Width
- **ram:** Random Access Memory in Mega Bytes
- **sc_h:** Screen Height of mobile in cm
- **sc_w:** Screen Width of mobile in cm
- **talk_time:** longest time that a single battery charge will last when you are
- **three_g:** Has 3G or not
- **touch_screen:** Has touch screen or not
- **wifi:** Has wifi or not
- **price_range:** This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).

**First I am going to import all the libraries that I will use in the project and open the dataset:**

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

# 3. Exploratory Data Analysis (EDA)

data=pd.read_csv('mobile_prices.csv')
data.head()

data.shape
print('The dataset has {} instances (rows) and {} features (columns).'.format(data.shape[0],data.shape[1]))

**With the following code I want to identify if the dataset has null values. In addition, I want to identify the data types that the dataset has:**

data.info()

**The dataset has no null values, so I don't need to do the imputation. Also, the dataset has 19 integer data and 2 float data. Since we do not have categorical data, it is not necessary to encode our variables.**

**With the following code I want to see the main statistical data of the dataset:**

data.describe()

**With the following code I want which data contain integer values and which have floating data values:**

data_categorical=data.select_dtypes(include='int64')
data_categorical.head()

data_categorical=data.select_dtypes(include='float64')
data_categorical.head()

**The following graph shows the participation by price range:**

labels = data["price_range"].value_counts().index
sizes = data["price_range"].value_counts()
colors = sns.color_palette('pastel')[0:10]
plt.figure(figsize = (5,5))
plt.pie(sizes, labels=labels, rotatelabels=False, autopct='%1.1f%%',colors=colors,shadow=True, startangle=90)
plt.title('Participation by price range',color = 'green',fontsize = 15)
plt.show()

**With the following graph we can see the participation by price range and by mobile features:**

pd.DataFrame(data = [data.groupby('price_range')['blue'].value_counts(), 
                     data.groupby('price_range')['dual_sim'].value_counts(),
                     data.groupby('price_range')['four_g'].value_counts(),
                     data.groupby('price_range')['three_g'].value_counts(),
                     data.groupby('price_range')['touch_screen'].value_counts(),
                     data.groupby('price_range')['wifi'].value_counts()],  

             index=["blue", "dual_sim","four_g","Three_g","touch_screen","wifi"]).T.style.background_gradient(cmap='coolwarm')

**The following graph shows the distribution for each feature of the dataset:**

fig, axes = plt.subplots(3, 7,figsize=(20,10))
axe = axes.flatten()
color_palette = sns.color_palette("pastel") + sns.color_palette("Set2") + sns.color_palette("husl", 25)


for i,feature in enumerate(data.columns):
    sns.histplot(data=data, x=feature, kde=True, ax=axe[i], color=color_palette[i])    
plt.show()

**The following graph shows the boxplot for each feature of the dataset:**

fig, axes = plt.subplots(3, 7,figsize=(20,10))
axe = axes.flatten()

for i,feature in enumerate(data.columns):
    sns.boxplot(data=data, x=feature, ax=axe[i], color=color_palette[i])    
plt.show()

# 4. Feature Engineering

Once we have analyzed the information in our dataset, we proceed to prepare the data to apply the prediction models.

data[data.duplicated()]

**I haven't duplicate values in the data set. Now we gonna see the relation with the target variable:**

data.corr()["price_range"].sort_values(ascending=False)

plt.figure(figsize=(13,9))
sns.heatmap(data.corr(), vmax=0.8, linewidth=0.1, cmap='vlag')
plt.show()

**Price appears to be highly correlated with RAM. In addition, clock_speed, mobile_wt and touch_screen appear to be negatively correlated, indicating that there is a relationship between them, such that as the value of one variable increases, the value of the other decreases.**

# 5. Training and Evaluating Models

X = data.drop('price_range',axis=1)
y = data['price_range']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

**Train and Fit Models**

Model1 = LogisticRegression()
Model1.fit(X_train, y_train)
predicts1 = Model1.predict(X_test)

print("Logistic Regression")
Model1_acc = accuracy_score(y_test, predicts1)*100

print("\nAccuracy test: ", round(Model1_acc,2))

print("\nclassification report\n")
report1 = classification_report(y_test, predicts1)
print(report1)

print("Confusion Matrix")
confusionmatrix1 = confusion_matrix(y_test, predicts1)
p = sns.heatmap(pd.DataFrame(confusionmatrix1), annot=True,fmt='g')
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

Model2 = KNeighborsClassifier()
Model2.fit(X_train, y_train)
predicts2 = Model2.predict(X_test)

print("KNeighbors Classifier_ Metrics")
Model2_acc = accuracy_score(y_test, predicts2)*100

print("\nAccuracy test: ", round(Model2_acc,2))

print("\nclassification report\n")
report2 = classification_report(y_test, predicts2)
print(report2)

print("Confusion Matrix")
confusionmatrix2 = confusion_matrix(y_test, predicts2)
p = sns.heatmap(pd.DataFrame(confusionmatrix2), annot=True,fmt='g')
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

**Dimensional Reduction** (Principal Component Analysis: PCA)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data)
principalDf = pd.DataFrame(data = principalComponents, columns=['Component1','Component2'])
print('\nThe Importance of each column is explained by %: ',pca.explained_variance_ratio_)
finaldata=principalDf.join(data['price_range'])
print('\nFinal DataFrame')
finaldata.head()

X = finaldata.drop('price_range',axis=1)
y = finaldata['price_range']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

Model1_pca = LogisticRegression()
Model1_pca.fit(X_train, y_train)
predicts1_pca = Model1_pca.predict(X_test)

print("Logistic Regression_PCA")
Model1_pca_acc = accuracy_score(y_test, predicts1_pca)*100

print("\nAccuracy test: ", round(Model1_pca_acc,2))

print("\nclassification report\n")
report3 = classification_report(y_test, predicts1_pca)
print(report3)

print("\nConfusion Matrix")
confusionmatrix3 = confusion_matrix(y_test, predicts1_pca)
p = sns.heatmap(pd.DataFrame(confusionmatrix3), annot=True,fmt='g')
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

Model2_pca = KNeighborsClassifier()
Model2_pca.fit(X_train, y_train)
predicts2_pca = Model2_pca.predict(X_test)

print("KNN_PCA\n")
Model2_pca_acc = accuracy_score(y_test, predicts2_pca)*100
print("Accuracy test: ", round(Model2_pca_acc,2))

print("\nclassification report\n")
report4 = classification_report(y_test, predicts2_pca)
print(report4)

print("\nConfusion Matrix")
confusionmatrix4 = confusion_matrix(y_test, predicts2_pca)
p = sns.heatmap(pd.DataFrame(confusionmatrix4), annot=True,fmt='g')
plt.title('Confusion matrix')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# 6. Conclusion

conclusion = pd.DataFrame({
    'Model': ['Logistic Regression','Logistic Regression_PCA', 'KNeighbors Classifier', 'KNeighbors Classifier_PCA'],
    'Accuracy': [Model1_acc,Model1_pca_acc,Model2_acc,Model2_pca_acc,]})

conclusion.head()

The Mobile prices dataset is a dataset to which the logistic regression and KNeighbors Classifier models were applied. It uses PCA; however, from the table above it is clear that this technique does not benefit the KNN model, as it reduces the accuracy, and logistic regression increases it; however, it is not as high as the accuracy for the KNN model without PCA. In conclusion, the model that best fits the data set and has a higher accuracy is the KNN model without applying the PCA technique.
