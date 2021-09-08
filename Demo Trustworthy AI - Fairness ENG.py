#!/usr/bin/env python
# coding: utf-8

# # Demo Trustworthy AI - Fairness

# This notebook demonstrates how to use Fairlearn and the Fairness dashboard to generate predictions for the Census dataset.
# This dataset features a classification problem: given a range of data on 32.000 individuals predicting whether their
# annual income is greater than or less than $ 50.000 a year. For the purposes of this notebook, we will treat it as a
# loan decision problem: the label indicates whether or not each individual has repaid a loan in the past. We will use
# the data to train a predictor that will tell us whether or not previously unseen individuals will repay a loan.
# The hypothesis is that the model's predictions are used to decide whether an individual should be granted a loan.
# We will first train an unaware predictor of equity and show that it leads to unfair decisions under a specific
# notion of equity called * demographic parity *. Then we mitigate the iniquity by applying the GridSearch algorithm from the Fairlearn package.

# In[1]:


from fairlearn.widget import FairlearnDashboard
from fairlearn.reductions import GridSearch
from fairlearn.reductions import DemographicParity, ErrorRate

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import pandas as pd
from IPython.display import Markdown, display


# Upload and inspect the data

# In[2]:


data = fetch_openml(data_id=1590, as_frame=True)
X_raw = data.data
Y = (data.target == '>50K') * 1
X_raw.head(10)


# We treat the gender of each individual as a sensitive characteristic (where 0 indicates female and 1 male); we separate this feature and eliminate it from the main dataframe. We then go through some standard data preprocessing steps to convert the data into a format suitable for ML algorithms. 

# In[ ]:


print(shape())


# In[ ]:


A = X_raw["sex"]
X = X_raw.drop(labels=['sex'], axis=1)
A.head(10)


# In[ ]:


X.head(10)


# Converting categorical variables into dummy variables

# In[ ]:


X = pd.get_dummies(X)


# In[ ]:


sc = StandardScaler()
X_scaled = sc.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)


# In[ ]:


le = LabelEncoder()
Y = le.fit_transform(Y)


# Split the data into training and testing sets.

# In[ ]:


X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X_scaled,
                                                                     Y,
                                                                     A,
                                                                     test_size=0.2,
                                                                     random_state=0,
                                                                     stratify=Y)


# In[ ]:


display(Markdown("#### Training Dataset shape"))
print(X_train.shape)

display(Markdown("#### Test Dataset shape"))
print(X_test.shape)


# Work around the indexing error.

# In[ ]:


X_train = X_train.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)


# Formation of an unaware predictor of fairness. To show the Fairlearn effect, we will first train a standard ML predictor that does not incorporate the fairness. 

# In[ ]:


unmitigated_predictor = LogisticRegression(solver='liblinear', fit_intercept=True)

unmitigated_predictor.fit(X_train, Y_train)


# In[ ]:


#list(Y_train) ##[:10]
## stampare dataset con colonna Y_train alla fine

Y_df=pd.DataFrame(Y_train, columns = ['Prediction'])
X_train.join(Y_df).head(10)


# Load this predictor into the Fairness dashboard and evaluate the fairness. 

# In[ ]:


FairlearnDashboard(sensitive_features=A_test, sensitive_feature_names=['sex'],
                   y_true=Y_test,
                   y_pred={"unmitigated": unmitigated_predictor.predict(X_test)})


# Looking at the disparity in accuracy, we see that males have about three times as much error as females.
# 
# More interesting, is the inequality of opportunities: males are granted loans at a rate three times higher than that of females.
# 
# Despite the fact that we have removed the feature from the training data, our predictor still discriminates on the basis of gender.
# This shows that simply ignoring a sensitive feature when fitting a predictor rarely eliminates the disadvantage.
# There will generally be enough other variables related to the removed feature to lead to a meaningful disparate impact. 

# ### GridSearch Mitigation
# 
# The user provides a standard ML estimator, which is treated as a black box. GridSearch functions by generating a sequence of re-labels and re-weightings and trains a predictor for each one. 

# For this example, we specify the Demographic Parity (on the sensitive sex characteristic) as a fairness metric. Demographic parity requires individuals to be offered the opportunity (loan approvation) regardless of belonging to the sensitive class (i.e., women and men should be offered loans at the same rate). In general, the appropriate fairness metric will not be obvious. 

# In[ ]:


sweep = GridSearch(LogisticRegression(solver='liblinear', fit_intercept=True),
                   constraints=DemographicParity(),
                   grid_size=71)


# In[ ]:


sweep.fit(X_train, Y_train,
          sensitive_features=A_train)

predictors = sweep.predictors_


# In[ ]:


errors, disparities = [], []
for m in predictors:
    def classifier(X): return m.predict(X)

    error = ErrorRate()
    error.load_data(X_train, pd.Series(Y_train), sensitive_features=A_train)
    disparity = DemographicParity()
    disparity.load_data(X_train, pd.Series(Y_train), sensitive_features=A_train)

    errors.append(error.gamma(classifier)[0])
    disparities.append(disparity.gamma(classifier).max())

all_results = pd.DataFrame({"predictor": predictors, "error": errors, "disparity": disparities})

non_dominated = []
for row in all_results.itertuples():
    errors_for_lower_or_eq_disparity = all_results["error"][all_results["disparity"] <= row.disparity]
    if row.error <= errors_for_lower_or_eq_disparity.min():
        non_dominated.append(row.predictor)


# We can put the dominant models in the Fairness dashboard, along with the non-attenuated model.

# In[ ]:


dashboard_predicted = {"unmitigated": unmitigated_predictor.predict(X_test)}
for i in range(len(non_dominated)):
    key = "dominant_model_{0}".format(i)
    value = non_dominated[i].predict(X_test)
    dashboard_predicted[key] = value


FairlearnDashboard(sensitive_features=A_test, sensitive_feature_names=['sex'],
                   y_true=Y_test,
                   y_pred=dashboard_predicted)


# We see a Pareto front forming: the set of predictors representing the optimal trade-offs between accuracy and disparity in predictions. In the ideal case, we would have a predictor in (1,0), perfectly accurate and without any injustice under the conditions of Demographic Equality (with respect to the sensitive characteristic "sex"). The Pareto front represents how close we can get to this ideal based on our data and the choice of estimator. Note the range of the axes: the disparity axis covers multiple accuracy values, so we can substantially reduce the disparity for a small loss of accuracy. By clicking on the individual models in the chart, we can inspect their metrics for disparity and accuracy in greater detail. In a real-world example, we will then choose the model that represents the best compromise between accuracy and disparity, given the relevant business constraints.

# In[ ]:




