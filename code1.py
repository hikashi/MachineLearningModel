import os
import tarfile
from six.moves import urllib
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
# import library from sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split, GridSearchCV
from future_encoders import OneHotEncoder
from scipy.stats import randint, expon, reciprocal
from scipy import stats
from skopt import BayesSearchCV
# pip install scikit-optimize for other computer who want to use the code of bayesian optimizantion
from skopt.space import Real, Categorical, Integer
import time
#from sklearn.preprocessing import OneHotEncoder

# column index 
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

# global paramter
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH  = os.path.join("datasets", "housing")
HOUSING_URL   = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# function for downloading data and save as tgz and extract as csv
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# function for loading in the housing data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# function for splitting data into training and testing datasets
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# for reporting the result of the data
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
# display score for the data
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
def minmax(val_list):
    min_val = min(val_list)
    max_val = max(val_list)

    return (min_val, max_val)


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

########################################################################################################################
# download the file from the server
fetch_housing_data()
housing = load_housing_data()

# USE TO CHECK THE SANITY OF THE DATA WHETHER IT IS POSSIBLE OR NOT.
# checking the information of the data
# print(housing.head())
# print(housing.info())
# print categorical data
# print(housing["ocean_proximity"].value_counts())

# plot out the data distribion iin the housing data structure
#housing.hist(bins=50, figsize=(20,15))
#plt.show()
# train_set, test_set = split_train_test(housing, 0.2)
# train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = )
# print(len(train_set), "train +", len(test_set), "test")

# attempt to make the categorical data for income data,
# housing['income_cat'] = np.ceil(housing["median_income"] / 1.5)
# housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#save_fig("bad_visualization_plot")

# add one more attribute about he income category
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

########################################################################################################################
# start splitting data
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

########################################################################################################################
# housing label
housing        = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
housing_num    = housing.drop('ocean_proximity', axis=1)

########################################################################################################################
# pipe line for preprocessing the data
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])
# concate the data
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
housing_prepared = full_pipeline.fit_transform(housing)


######################################################################################################################## 
# perform grid search on the data
######################################### grid search CV

dataExtract  = housing_prepared[1:100]
labelExtract = housing_labels[1:100]

# grid search
param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

start_time_GRID = time.time()
svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=1)
grid_search.fit(dataExtract, labelExtract)
elapsed_time_GRID = time.time() - start_time_GRID

# random search
param_rand = {
         'C': expon(scale=10000),
         'gamma': expon(scale=.1),
         'kernel': ['rbf', 'linear']
        }

start_time_RAND = time.time()
svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_rand,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error', random_state=42, verbose=2, n_jobs=1)
rnd_search.fit(dataExtract, labelExtract)
elapsed_time_RAND = time.time() - start_time_RAND

# bayesian search
#param_bayes = [
#         {'C': expon(scale=100)},
#         {'gamma': expon(scale=.1)},
#         {'kernel': ['rbf', 'linear']}
#        ]
#
#param_bayes

svc_search = {
#    'model': Categorical([SVR()]),
    'model__C': Real(1000, 100000, prior='log-uniform'),
    'model__gamma': Real(0.01, 3.0, prior='log-uniform'),
    'model__kernel': Categorical(['rbf', 'linear']),
}
pipe = Pipeline([
    ('model', SVR())
])

start_time_BAYES = time.time()
svm_reg = SVR()
#bayes_search = BayesSearchCV(svm_reg, param_distributions=param_bayes,
#                                n_iter=20, cv=5, scoring='neg_mean_squared_error', random_state=42, verbose=2, n_jobs=1)
# search_spaces=[{'C': expon(scale=100)},{'gamma': expon(scale=.1)},{'kernel': ['rbf', 'linear']}]
bayes_search = BayesSearchCV(pipe, search_spaces=svc_search,scoring='neg_mean_squared_error', verbose=2, n_jobs=1, n_iter=50)
bayes_search.fit(dataExtract, labelExtract)

elapsed_time_BAYES = time.time() - start_time_BAYES
########################################################################################################################
#final_model = grid_search.best_estimator_
final_model_GRID = grid_search.best_estimator_
final_model_RAND = rnd_search.best_estimator_
final_model_BAYES = bayes_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions_GRID  = final_model_GRID.predict(X_test_prepared)
final_predictions_RAND  = final_model_RAND.predict(X_test_prepared)
final_predictions_BAYES = final_model_BAYES.predict(X_test_prepared)

final_mse_GRID  = mean_squared_error(y_test, final_predictions_GRID)
final_mse_RAND  = mean_squared_error(y_test, final_predictions_RAND)
final_mse_BAYES = mean_squared_error(y_test, final_predictions_BAYES)

final_rmse_GRID = np.sqrt(final_mse_GRID)
final_rmse_RAND = np.sqrt(final_mse_RAND)
final_rmse_RAND = np.sqrt(final_mse_BAYES)

#######################################################################################################################
minvalue, maxvalue  = minmax(y_test)

print("===========================================")
print("MIN TEST VALUE : %f      MAX TEST VALUE: %f" %(minvalue, maxvalue))
print("====== GRID SEARCH ======")
print(final_model_GRID)
print("TIME FOR GRID SEARCH: %f" %(elapsed_time_GRID))
print("BEST SCORE          : %f" %(grid_search.best_score_))
print("RMSE IS             : %f " %(final_rmse_GRID))
print("====== RANDOM SEARCH ======")
print(final_model_RAND)
print("TIME FOR RANS SEARCH: %f" %(elapsed_time_RAND))
print("BEST SCORE          : %f" %(rnd_search.best_score_))
print("RMSE IS             : %f " %(final_rmse_RAND))
print("====== BAYES SEARCH ======")
print(final_model_BAYES)
print("TIME FOR BAYES SEARCH: %f" %(elapsed_time_BAYES))
print("BEST SCORE           : %f" %(bayes_search.best_score_))
print("RMSE IS              : %f " %(final_rmse_RAND))
print("===========================================")