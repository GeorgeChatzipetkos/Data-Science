# Data-Science

- **Permutations and Combinations **

```python
import pandas as pd    
import itertools

#x = [1, 2, 3, 4, 5, 6]
x = [*range(1,6)]

resultList = [value for value in itertools.product(x, repeat=3)] # Permutations: order matters, Repeats allowed,  formula =  n**r , AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD
resultList = [value for value in itertools.permutations(x, r=3)] # Permutations: order matters, no Repeats, formula = n!/(n-r)! , 	AB AC AD BA BC BD CA CB CD DA DB DC
resultList = [value for value in itertools.combinations_with_replacement(x, r=3)] # Combinations: order doesn't matter, Repeats allowed, formula = (r+n-1)!/r!(n-1)!, AA AB AC AD BB BC BD CC CD DD
resultList = [value for value in itertools.combinations(x, r=3)] # Combinations: order doesn't matter, no Repeats, formula = n!/r!(n-r)!, AB AC AD BC BD CD

df = pd.DataFrame(resultList)
```

- **outliers**

```python
# checking the number of outliers in the dataset
def number_of_outliers(df):
    
    df = df.select_dtypes(exclude = 'object')
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    return ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    
# treat outliers by making them equal to Q3 or Q1
def lower_upper_range(datacolumn):
    sorted(datacolumn)
    Q1,Q3 = np.percentile(datacolumn , [25,75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range,upper_range
  
for col in columns:  
    lowerbound,upperbound = lower_upper_range(df[col])
    df[col]=np.clip(df[col],a_min=lowerbound,a_max=upperbound)
```

- **Binary Logistic Regression**

```python
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

- **Multiple Linear Regression**

```python
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from dmba import regressionSummary
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from scipy.stats import probplot

#%matplotlib inline

dataset = pd.read_csv('datasets_4458_8204_winequality-red.csv')

dataset.shape

dataset.describe()

#check which are the columns that contain NaN values
dataset.isnull().any()

#divide the data into “attributes” and “labels”
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', \
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
y = dataset['quality'].values

#check the average value of the “quality” column
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(dataset['quality'])

#split 80% of the data to the training set while 20% of the data to test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#train our model
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

#slope
regressor.intercept_

#get the coefficients of the attributes
columns_ = dataset.iloc[:,:-1].columns
coeff_df = pd.DataFrame(regressor.coef_, index = columns_, columns=['Coefficient'])  
coeff_df
#This means that for a unit increase in “density”, there is a decrease of 31.51 units in the quality of the wine. Similarly, a unit decrease in “Chlorides“ results in an increase of 1.87 units in the quality of the wine. We can see that the rest of the features have very little effect on the quality of the wine.

#do prediction on test data
y_pred = regressor.predict(X_test)

#R-Squared (variance explained by the model)
r2_score(y_test,y_pred)

#Check the difference between the actual value and predicted value.
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
df1

#plot the comparison of Actual and Predicted values
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

# evaluate the performance of the algorithm ( MAE, MSE, and RMSE)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  #Variance of error values(How widely dispersed errors are)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) #Standard Deviation of errors
regressionSummary(y_test, y_pred) #Does not include MSE

###################################################################################################
#Reduce the Number of Predictors (then change X accordingly)
""" There are 3 methods namely, 
    -exhaustive search (code included in this script)
    -Backward elimination
    -Forward selection 
    -Stepwise regression 
   the code is included in: 'Data Mining for Business Analytics Concepts, Techniques and Applications in Python by Galit Shmueli Peter C. Bruce Peter Gedeck Nitin R. Patel (z-lib.org)) '
"""
#Exhaustive search
import itertools
def exhaustive_search(variables, train_model, score_model):
    """ Variable selection using backward elimination
    Input:
    variables: complete list of variables to consider in
    model building
    train_model: function that returns a fitted model for a
    given set of variables
    score_model: function that returns the score of a
    model; better models have lower
    scores
    Returns:
    List of best subset models for increasing number of
    variables
    """
# create models of increasing size and determine the best models in each case
    result = []
    for nvariables in range(1, len(variables) + 1):
        best_subset = None
        best_score = None
        best_model = None
        for subset in itertools.combinations(variables,nvariables):
            subset = list(subset)
            subset_model = train_model(subset)
            subset_score = score_model(subset_model, subset)
            if best_subset is None or best_score > subset_score:
                best_subset = subset
                best_score = subset_score
                best_model = subset_model
        result.append({
            'n': nvariables,
            'variables': best_subset,
            'score': best_score,
            'model': best_model,
            })
    return result

#AIC_score, BIC_score
import math
import numpy as np
def AIC_score(y_true, y_pred, model=None, df=None):
    """ calculate Akaike Information Criterion (AIC)
    Input:
    y_true: actual values
    y_pred: predicted values
    model (optional): predictive model
    df (optional): degrees of freedom of model
    One of model or df is requried
    """
    if df is None and model is None:
        raise ValueError('You need to provide either model ordf')
    n = len(y_pred)
    p = len(model.coef_) + 1 if df is None else df
    resid = np.array(y_true) - np.array(y_pred)
    sse = np.sum(resid ** 2)
    constant = n + n * np.log(2*np.pi)
    return n * math.log(sse / n) + constant + 2 * (p + 1)
def BIC_score(y_true, y_pred, model=None, df=None):
    """ calculate Schwartz's Bayesian Information Criterion (AIC)
    Input:
    y_true: actual values
    y_pred: predicted values
    model: predictive model
    df (optional): degrees of freedom of model"""
    aic = AIC_score(y_true, y_pred, model=model, df=df)
    p = len(model.coef_) + 1 if df is None else df
    n = len(y_pred)
    return aic - 2 * (p + 1) + math.log(n) * (p + 1)

#adjusted_r2_score
from sklearn.metrics import regression
def adjusted_r2_score(y_true, y_pred, model):
    """ calculate adjusted R2
    Input:
    y_true: actual values
    y_pred: predicted values
    model: predictive model
    """
    n = len(y_pred)
    p = len(model.coef_)
    r2 = regression.r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) /(n - p - 1)

def train_model(variables):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor
def score_model(model, variables):
    y_pred = regressor.predict(X_train)
    # we negate as score is optimized to be as low as possible
    return -adjusted_r2_score(y_train, y_pred, regressor)
allVariables = dataset.iloc[:,:-1].columns
results = exhaustive_search(allVariables, train_model, score_model)
data = []
for result in results:
    regressor = result['model']
    variables = list(result['variables'])
    AIC = AIC_score(y_train, regressor.predict(X_train), regressor)
    d = {'n': result['n'], 'r2adj': -result['score'], 'AIC': AIC}
    d.update({var: var in result['variables'] for var in allVariables})
    data.append(d)
ExSearch = pd.DataFrame(data, columns=('n', 'r2adj', 'AIC') + tuple(sorted(allVariables)))
'The code reports the best model with a single predictor, two predictors, and so on. The higher R2 adj and lowest AIC, the better the combination'
#####################################################################################################################

#Assumption of Linearity
# visualize the relationship between the features and the response using scatterplots
sns.pairplot(dataset, x_vars=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', \
             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol'],\
             y_vars='quality', size=7, aspect=0.7)

#Assumption of Multicollinearity
dataset.corr()
plt.figure(figsize=(20,20))  # on this line I just set the size of figure to 12 by 10.
sns.pairplot(dataset)

plt.figure(figsize=(20,20))
sns.heatmap(dataset.corr(), annot=True,cmap='RdYlGn',square=True)  # seaborn has very simple solution for heatmap

#Mean of Residuals
residuals = y_test-y_pred
mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))

# Assumption of Homoscedasticity (there should not be any pattern in the error terms)
p = sns.scatterplot(y_pred,residuals)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-10,10)
plt.xlim(0,26)
p = sns.lineplot([0,26],[0,0],color='blue')
p = plt.title('Residuals vs fitted values plot for homoscedasticity check')
#for Goldfeld Quandt we have to run the regression using the sm.OLS (https://www.listendata.com/2018/01/linear-regression-in-python.html#Assumptions-of-linear-regression)

# Bartlett’s test tests the null hypothesis that all input samples are from populations with equal variances.

#Null Hypothesis: Error terms are homoscedastic
#Alternative Hypothesis: Error terms are heteroscedastic.

#If p value is less than 0.05 , it's null hypothesis that error terms are homoscedastic gets rejected, that's not good for a regression.

from scipy.stats import bartlett
bstat, bp = bartlett( y_pred,residuals)
print("test statistic = {}, p-value = {}".format(bstat,bp))

#Normality of error terms/residuals
sns.distplot(residuals,kde=True)
plt.title('Normality of error terms/residuals')

_ = probplot(residuals, plot=plt) #QQ plot of residual terms. The errors should follow the red line.

# We use Shapiro Wilk test from scipy library to check the normality of residuals.
# Null Hypothesis: The residuals are normally distributed.
# Alternative Hypothesis: The residuals are not normally distributed.
#If p value is less than 0.05 we reject the null hypothesis that error terms are nomally distributed
from scipy import stats
sstat, sp = stats.shapiro(residuals)
print("test statistic = {}, p-value = {}".format(sstat, sp))

#No autocorrelation of residuals
plt.figure(figsize=(10,5))
p = sns.lineplot(y_pred,residuals,marker='o',color='blue')
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')
plt.ylim(-10,10)
plt.xlim(0,26)
p = sns.lineplot([0,26],[0,0],color='red')
p = plt.title('Residuals vs fitted values plot for autocorrelation check')

# Checking for autocorrelation To ensure the absence of autocorrelation we use Ljungbox test.
# Null Hypothesis: Autocorrelation is absent.
# Alternative Hypothesis: Autocorrelation is present.
#If p value is less than 0.05 we reject the null hypothesis that error terms are not autocorrelated

min(diag.acorr_ljungbox(residuals , lags = 40)[1])

# autocorrelation
# If The results show signs of autocorelation, there are spikes outside the red confidence interval region. This could be a factor of seasonality in the data.
sm.graphics.tsa.plot_acf(residuals, lags=40)
plt.show()

# prediction with sklearn (give potential values for all variables in the equation)
#Var1 = 2.75
#Var2 = 5.3
#print ('Predicted y: \n', regressor.predict([[Var1 ,var2]]))
regressor.predict([[6.8, 0.5, 0.3, 2.1, 0.058, 20, 45, 0.9947, 3.34, 0.7, 10.0]])
```

- **K-means clustering**

```python
import pandas as pd
#import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel (r'CN Deposits 2020.xlsx',sheet_name = 'deposits')

df.info()

#k-means clustering
from sklearn.cluster import KMeans

#subset with numerical columns
x = df.iloc[:, [1,4,6,7]].values

#########Determine the optimal number of clusters############

#The Elbow Method
Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)

#choose the k for which the Within-Cluster-Sum of Squared Errors (WSS) first starts to diminish
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()

#The Silhouette Method
from sklearn.metrics import silhouette_score

sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(x)
  labels = kmeans.labels_
  sil.append(silhouette_score(x, labels, metric = 'euclidean'))
  
# The Silhouette Score reaches its global maximum at the optimal k. This should ideally appear as a peak in the Silhouette Value-versus-k plot.
plt.plot(range(2, kmax+1), sil)
plt.title('Silhouette method')
plt.xlabel('No of clusters')
plt.ylabel('Silhouette Score')
plt.show()

#the data can be optimally clustered into k clusters (in this dataset in 2)
kmeans2 = KMeans(n_clusters=2)
y_kmeans2 = kmeans2.fit_predict(x)
print(y_kmeans2)

kmeans2.cluster_centers_
kmeans2.labels_

plt.scatter(x[:,1],x[:,3],c=y_kmeans2,cmap='rainbow')
plt.scatter(kmeans2.cluster_centers_[:,1] ,kmeans2.cluster_centers_[:,3], color='black')
plt.title('K-means Clustering')
plt.xlabel('Gateway ID')
plt.ylabel('Deposit Amount USD')
```

- **PCA with K-means clustering**

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # Standardize the data

# Load in the data
df = pd.read_csv('protein.csv')

#visualize raw data

plt.figure(figsize = (20,20))
df.corr()
sns.pairplot(df)

# Standardize the data to have a mean of ~0 and a variance of 1
X_std = StandardScaler().fit_transform(df.iloc[:,1:])
X_std= pd.DataFrame(X_std,columns=df.iloc[:,1:].columns.values)

# Create a PCA instance: pca
pca = PCA()
scores_pca = pca.fit_transform(X_std)

#see how much percentage of variance is explained by each of the 9 individual components
pca.explained_variance_ratio_

# Plot the explained variances
#features = range(pca.n_components_) 
plt.bar(range(1,10), pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(range(1,10))

#cumulative variance plot (A rule of thumb is to preserve around 80 % of the variance)
plt.figure(figsize = (10,8))
plt.plot(range(1,10),pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle='--')
plt.title('Explained variance by Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained variance')

# Save components to a DataFrame
PCA_components = pd.DataFrame(scores_pca,columns=range(1,10))

#check if there are any clear clusters (change components each time)
plt.scatter(PCA_components[1], PCA_components[2], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

#we choose 4 components
pca = PCA(n_components=4)
scores_pca = pca.fit_transform(X_std)

loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_),columns=range(1,5),index=df.iloc[:,1:].columns.values)

#PCA Biplot
def myplot(score,coeff,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
 
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

myplot(scores_pca[:,0:2],np.transpose(pca.components_[0:2, :]),list(X_std.columns))
plt.show()

##################################################################################################################
#We’ll incorporate the newly obtained PCA scores in the K-means algorithm.
#That’s how we can perform segmentation based on principal components scores instead of the original features.
##################################################################################################################

#find optimal number of clusters (elbow method)
wcss = []
for i in range(1,21):
    kmeans_pca=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans_pca.fit(scores_pca)
    wcss.append(kmeans_pca.inertia_)

plt.figure(figsize = (10,8))
plt.plot(range(1,21), wcss, '-o', linestyle='--', color='black')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.xticks(range(1,21))
plt.title('K-means with PCA Elbow method')
plt.show()

#implement k-means
kmeans_pca=KMeans(n_clusters=5,init='k-means++',random_state=42)
kmeans_pca.fit(scores_pca)

##################################################################################################################
#K-means with PCA results
##################################################################################################################

#create new df including PCA scores and assigned cluster
df_PCA_Kmeans = pd.concat([df.reset_index(drop=True),pd.DataFrame(scores_pca)],axis=1)
df_PCA_Kmeans.columns.values[-4:]=['Component 1', 'Component 2', 'Component 3', 'Component 4']
df_PCA_Kmeans['Assigned Cluster']=kmeans_pca.labels_


#superimpose the clusters with respect to the first two components
x_axis = df_PCA_Kmeans['Component 1']
y_axis = df_PCA_Kmeans['Component 2']
plt.figure(figsize = (10,8))
p1 = sns.scatterplot(x_axis,y_axis,hue=df_PCA_Kmeans['Assigned Cluster'], palette=['g','r','c','m','y'])
#the loop adds labels next to dots
for line in range(0,df_PCA_Kmeans.shape[0]):
     p1.text(df_PCA_Kmeans['Component 1'][line]+0.01, df_PCA_Kmeans['Component 2'][line], 
     df_PCA_Kmeans['Country'][line], horizontalalignment='left', 
     size='medium', color='black', weight='regular') #or semibold weight

plt.title('Clusters by PCA Components')
plt.show()
```
