# Data-Science
- Binary Logistic Regression

```ruby
# logistic regression predicts the probability of an event or class that is dependent on other factors.
# Thus the output of logistic regression always lies between 0 and 1.
# Because of this property it is commonly used for classification purpose.

#The Sigmoid Function is given by: S(x) = 1 / (1 + e ^ (- bo - b1X1 - b2X2 .... bnXn))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


plt.rcParams["figure.figsize"] = (10, 6)

data = pd.read_csv("datasets_228_482_diabetes.csv")
data.head()
data.info()

#checking  for Multicollinearity among the Xs
data.corr()
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.
sns.pairplot(data)

plt.figure(figsize=(20,20))
sns.heatmap(data.corr(), annot=True,cmap='RdYlGn',square=True)

#assign Xs and y
feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age','Glucose','BloodPressure','DiabetesPedigreeFunction']
X = data[feature_cols] # Features
y = data.Outcome # Target variable

#check for imbalanced classes
y.value_counts()

sns.countplot(x='Outcome',data=data,palette='hls')
plt.show()

count_no_sub = len(data[data['Outcome']==0])
count_sub = len(data[data['Outcome']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of 0s", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of 1s", pct_of_sub*100)

#Implementing the model (before that we could also balance the classes or use RFE https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8 )
logit_model=sm.Logit(y,sm.add_constant(X))
result=logit_model.fit()
print(result.summary2())
#The variables with p-values that are smaller than 0.05, need to remain as Xs.

cols = ['Pregnancies','Glucose','BloodPressure',]
X=data[cols]

logit_model=sm.Logit(y,sm.add_constant(X))
result=logit_model.fit()
print(result.summary2())
"""
- disable sklearn regularization LogisticRegression(C=1e9)

- add statsmodels intercept sm.Logit(y,sm.add_constant(X)) OR disable sklearn intercept LogisticRegression(C=1e9,fit_intercept=False)

- sklearn returns probability for each class so model_sklearn.predict_proba(X)[:,1] == model_statsmodel.predict(X)

- Use of predict fucntion model_sklearn.predict(X) == (model_statsmodel.predict(X)>0.5).astype(int)

"""
# the odds
np.exp(result.params)

# Convert odds to probability e.g.4 pregnancies, 100 glucose, 85 blood pressure
1/(1+np.exp(-(0.0951*4 + 0.0122*100 - 0.0335*85)))

# Divide the data to training set and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0,stratify=y) #75% data will be used for model training and 25% for model testing.In case of an imbalanced dataset, be sure to set stratify=y so that class proportions are preserved when splitting.


# Create an instance and fit the model 
logreg = LogisticRegression() #(penalty="l2", C=1e42, solver='liblinear') set penalty=l2 and C=1e42 to avoid regularization
logreg.fit(X_train,y_train)

print('intercept ', logreg.intercept_[0])
print(pd.DataFrame({'coeff': logreg.coef_[0]}, index=X.columns))

# Making predictions
y_pred=logreg.predict(X_test)

#Model Evaluation using Confusion Matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True,cmap="YlGnBu", fmt='g')
plt.show()

# Accuracy, Precision, Recall, Specificity
TP = confusion_matrix.iloc[1, 1]
TN = confusion_matrix.iloc[0, 0]
FP = confusion_matrix.iloc[0, 1]
FN = confusion_matrix.iloc[1, 0]

print('Accuracy: ',metrics.accuracy_score(y_test, y_pred)) #(tp + tn) / (tp + tn + fp + fn)
print("Precision:",metrics.precision_score(y_test, y_pred)) #tp / (tp + fp)
print("Recall or Sensitivity:",metrics.recall_score(y_test, y_pred)) #tp / (tp + fn)
print("Specificity:", TN / (TN + FP)) 
print("F1:",metrics.f1_score(y_test, y_pred)) #The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0. F1= 2 * [(presicion * recall) / (pecision + recall)]
print(classification_report(y_test, y_pred))
print("Misclassification Rate:",1 - metrics.accuracy_score(y_test, y_pred)) 


#ROC Curve (AUC score 1 represents perfect classifier, and 0.5 represents a worthless classifier)
#Generally, as we decrease the threshold, we move to the right and upwards along the curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))  #AUC is the percentage of the ROC plot that is underneath the curve
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()

####################################
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point
####################################

i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plot tpr vs 1-fpr
fig, ax = plt.subplots()
plt.plot(roc['tpr'])
plt.plot(roc['1-fpr'], color = 'red')
plt.xlabel('1-False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.set_xticklabels([])


# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])

evaluate_threshold(0.6)


y_pred_new_threshold = (logreg.predict_proba(X_test)[:,1]>=0.60).astype(int)
print('Accuracy: ',metrics.accuracy_score(y_test, y_pred_new_threshold))
confusion_matrix = pd.crosstab(y_test, y_pred_new_threshold, rownames=['Actual'], colnames=['Predicted'])

def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

#make prediction on new test data
#create new df similar to X_test
#y_pred=logreg.predict(df2)
```
