# Titanic-Machine-Learning-from-Disaster
In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

Titanic: Machine Learning from Disaster
Predict survival on the Titanic
 * 	Defining the problem statement
 *	Collect the data
 *	Feature engineering
 *	Modeling
 *  Testing


## 1)	Defining the problem statement

Complete the analysis of what sorts of people were likely to survive  

## 2)	Collect the data  

Training data set and testing set are given by Kaggle   
### *Load train, test data set using Pandas.*  
Pandas is an open source, a library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language.  

```Import pandas as pd
Train = pd.read_csv(‘train.csv’)
Test = pd.read_csv(‘test.csv’)
``` 
Printing first 10 rows of the train dataset.  
```train.head(10)```  
![train head 10](https://user-images.githubusercontent.com/25092397/35195787-357ba6ee-fed1-11e7-8d83-dad3bdbe5a53.png)

### Data Description:

* Survived: Survived (1) or died (0)
* Pclass: Passenger's class
* Name: Passenger's name
* Sex: Passenger's sex
* Age: Passenger's age
* SibSp: Number of siblings/spouses aboard
* Parch: Number of parents/children aboard
* Ticket: Ticket number
* Fare: Fare
* Cabin: Cabin
* Embarked: Port of embarkation
### Visualize  
***matplotlib.pyplot*** provides plotting framework.  
***Seaborn***  is a Python visualization library based on ***matplotlib***.  
It provides a high-level interface for drawing attractive statistical graphics.

```
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
```
To switch to seaborn defaults, simply call the `set()` function.  
Defining a bar_chart function:
```
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts(normalize = True)
    dead = train[train['Survived']==0][feature].value_counts(normalize = True)
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked = True, figsize = (10,5))
```  
Example:

`bar_chart('Sex')`

![bar_chart_sex](https://user-images.githubusercontent.com/25092397/35195737-810c3e44-fed0-11e7-9047-cdd46cdbc631.png)





## 3)	Feature engineering  

Feature engineering is the process of using domain knowledge of the data to create features (feature vectors) to make machine learning algorithms work. 
Feature vector is an ***n-dimensional vector*** of numerical features that represent some object.
Many algorithms in machine learning require a numerical representation of objects, since such representations facilitate processing and statistical analysis. 

## 4)  Feature selection


## 5)  Modelling
