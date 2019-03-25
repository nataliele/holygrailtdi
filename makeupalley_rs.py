#######################import packages###########################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import re
import math
import string
import collections
import dill
import uuid
from scipy.sparse import csr_matrix
import random
from pandas import Series
import sklearn.cluster
# import editdistance
import Levenshtein
from sklearn.cluster import AffinityPropagation
from collections import OrderedDict, Counter

from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn import base
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RidgeCV, LinearRegression, SGDRegressor, Ridge
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from lightfm import LightFM, data, cross_validation
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.feature_extraction import DictVectorizer

# get working directory to save files
working_dir = os.getcwd()

# Category names and ids under skincare-face
cat_name = ['Treatments (Face)', 'Toners', 'Cleansers'
            , 'Moisturizers', 'Masks', 'Eye Makeup Remover', 'Neck/Decollete Cream', 'Scrubs', 'Treatments (Eye)', ]
cat_id = ['705', '702', '701', '707', '703', '708', '709', '704', '706']
cat_dict = dict(zip(cat_id, cat_name))

# df_clean is 300MB, not loaded to github
# df = dill.load(open('data/df_clean.pkd', 'rb'))

df.shape

select_cols = ['pro_ids', 'ratings', 'repurchases', 'pkg_quals', 'prices'
       ,'ingredients', 'brand_ids', 'users', 'reviews', 'lipies', 'age', 'eyes'
       ,'skin_type', 'skin_tone', 'skin_undertone', 'hair_color', 'hair_type'
       ,'hair_texture']

colab_filt_cols = ['pro_ids', 'users', 'lipies']

content_cols = ['names', 'pro_ids', 'ratings', 'repurchases', 'pkg_quals', 'prices', 'ingredients', 'brands']

df_cf = df[colab_filt_cols]


################################################################

### create unique id for users
users = df_cf['users'].unique()
len(users)
# 98840

# create random sample of 98840, from range(98840) without replacement,
random.seed(42)
id = Series(random.sample(range(len(users)), len(users)))
len(id.unique())

# join unique users and their unique ids
df_users = pd.DataFrame({'users': users, 'uid': id})

# join uid to main df by 'users' column
df_uid = pd.merge(df, df_users, on='users')

# join uid to main df_cf by 'users' column
df_cf_uid = pd.merge(df_cf, df_users, on='users')

# export to csv file to use in spark
# df_cf_uid[['pro_ids', 'uid', 'lipies']].to_csv('df_cf.csv', index=False)

prod_ratings = df_uid[['pro_ids', 'lipies', 'uid', 'reviews']]
prod_ratings_no_rev = df_uid[['pro_ids', 'lipies', 'uid']]

dill.dump(prod_ratings, open('data/prod_ratings.pkd', 'wb'))
dill.dump(prod_ratings_no_rev, open('data/prod_ratings_no_rev.pkd', 'wb'))


################################################################
### group ingredients

# create new df of unique product contents
df_content = df[content_cols]
# drop duplicates
df_content_uniq = df_content.drop_duplicates()
# using .loc will turn pro_ids into index
# df_content_uniq = df_content.loc[~df_content.pro_ids.duplicated(), :]
df_content_uniq.shape
# (5298, 8)
# table of product id and names
prod_df = df_content_uniq.copy()
# dill.dump(prod_df, open('data/prod_df.pkd', 'wb'))
# prod_df = dill.load(open('data/prod_df.pkd', 'rb'))

df_content_uniq['pro_ids'] = df_content_uniq['pro_ids'].astype('int64')
df_content_uniq = df_content_uniq.set_index('pro_ids')


# match searches the beginning of the string
# search searches the first appearance of the pattern
# findall searches all non-overlapping patterns

# if the character has a special meaning (in regex, have to escape with '\'
# in the var exploration window, text is displayed in regex style with escape char '\'
# when printing out, it's in unicode
def rm_delimiters(ingr_row):
    """
    split ingredients column into a list of string instead of strings
    :param ingr_row: a string row (of dataframe) of ingredients
    :return: list of (str) ingredients
    """
    # remove numbers
    # add 'and' and '-' and '.', ':', '...', '/' as splitter
    delim = [',', 'and', '\.+', ':', '/', '\s-+\s', '\*+', '\[\]', '\a', '\n', '\t', '\s\s+', 'â¢', ';', '-+', '_+'
        , 'active\singredients', 'also\scontains', 'rapid\sactivation\sgel', 'step\s1', 'step\s2', 'step\s3'
        , 'active\singredient', 'inactive\singredients', 'inactive\singredient', 'other\singredients'
        , 'ingredients', 'ingredient', 'previous\singredients', 'others', 'other']

    return re.split('|'.join(delim), ingr_row.lower())


def water_parse(ingr_list):
    """
    find and keep ingredients with word 'water-binding'; the rest if it has the word 'water', just keep
    'water'
    :param ingr_list: list of string ingredients
    :return: ingredients 'water' and 'water-binding'
    """
    water = ['water' if ('water' in ingr and 'water-binding' not in ingr) else ingr for ingr in ingr_list]
    # regex = re.compile(r'(water)')
    # water = [match for match in regex.findall(ingr) else ingr for ingr in ingr_list]
    return water

def oil_parse(ingr_list):
    """
    if ingredient has the word 'oil', just keep 'oil'; if ingredient is 'mineral oi', keep 'mineral oi'
    :param ingr_list: list of string ingredients
    :return: ingredients 'oil'
    """
    oil = ['oil' if (' oil' in ingr and 'mineral oil' not in ingr) else ingr for ingr in ingr_list]
    # regex = re.compile(r'(water)')
    # water = [match for match in regex.findall(ingr) else ingr for ingr in ingr_list]
    return oil


def extract_parse(ingr_list):
    """
    if ingredient has the word 'extract', just keep 'extract'
    :param ingr_list: list of string ingredients
    :return: ingredients 'extract'
    """
    extract = ['extract' if ' extract' in ingr else ingr for ingr in ingr_list]
    # regex = re.compile(r'(water)')
    # water = [match for match in regex.findall(ingr) else ingr for ingr in ingr_list]
    return extract



# remove words with 'ingredient' in it
# do parsing on popular words
# remove words with less than 2 char
# remove numbers

# split ingredients column into a list of string instead of strings
df_content_uniq['ingredients_n'] = df_content_uniq['ingredients'].apply(lambda x: rm_delimiters(x))

# turn lower case, strip leading and trailing spaces, remove empty lists
df_content_uniq['ingredients_n'] = df_content_uniq['ingredients_n'].apply(lambda x: list(filter(None, [i.lower().strip() for i in x])))

# parse water ingredients
df_content_uniq['ingredients_n'] = df_content_uniq['ingredients_n'].apply(lambda x: water_parse(x))

# parse oil ingredients
df_content_uniq['ingredients_n'] = df_content_uniq['ingredients_n'].apply(lambda x: oil_parse(x))

# parse extracts ingredients
df_content_uniq['ingredients_n'] = df_content_uniq['ingredients_n'].apply(lambda x: extract_parse(x))


# keep words with more than 3 characters
df_content_uniq['ingredients_n'] = df_content_uniq['ingredients_n'].apply(lambda x: [word for word in x if len(word)>3])
# now we have empty lists for rows with no ingredients

# list of ingredients of each product
words = list(df_content_uniq.ingredients_n)
# 5298


words_flat = [item for word in words for item in word]
words_flat_uniq = list(OrderedDict.fromkeys(words_flat))
# 9435 ingredients
words_arr = np.asarray(words_flat_uniq) #So that indexing with a list will work


lev_similarity = -1*np.array([[Levenshtein.distance(w1,w2) for w1 in words_arr] for w2 in words_arr])

affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.99)
affprop.fit(lev_similarity)


affprop_labels = affprop.labels_
affprop_clusters = affprop.cluster_centers_indices_


# affprop.labels_
# is the cluster label for each word
# np.unique(affprop.labels_)
# is the ordered array of cluster labels
# affprop.cluster_centers_indices_[0]
# the index of the word that other words will cluster around
# words_arr[2]


dill.dump(affprop_labels, open('data/affprop_labels.pkd', 'wb'))
dill.dump(affprop_clusters, open('data/affprop_clusters.pkd', 'wb'))

affprop_labels = dill.load(open('data/affprop_labels.pkd', 'rb'))
affprop_clusters = dill.load(open('data/affprop_clusters.pkd', 'rb'))

# there are 261 clusters for ingredients
len(affprop_clusters)
len(affprop_labels)

# create dictionary of unique ingredients and their cluster number/index
ingred_labels = dict(zip(words_flat_uniq, affprop_labels))
# ingred_labels = dict(zip(affprop_labels, words_flat_uniq))
ingred_labels[0]


for cluster_id in np.unique(affprop_labels[5]):
    exemplar = words_arr[affprop_clusters[cluster_id]]
    cluster = np.unique(words_arr[np.nonzero(affprop_labels == cluster_id)])
    cluster_str = ", ".join(cluster)
    # print(exemplar)
    print(" - *%s:* %s" % (exemplar, cluster_str))

###############################################################################
### onehot encoding features

one_hot_transformer = OneHotEncoder(sparse=False)

# map ingredients into groups and make into dictionary
df_content_uniq['words_mapped'] = df_content_uniq['ingredients_n'].apply(lambda x: [ingred_labels[ingredient] for ingredient in x])\
                                                                .apply(lambda x: np.unique(x))\
                                                                .apply(lambda x: Counter(x))

dict_vec = DictVectorizer(sparse=False)
ingred_ohe = dict_vec.fit_transform(df_content_uniq['words_mapped'])

ingred_features = np.hstack([ingred_ohe, df_content_uniq[['ratings', 'repurchases', 'pkg_quals', 'prices']]])

# fit knn to identify products that are similar
ingred_nn = NearestNeighbors(n_neighbors=11).fit(ingred_features)

# use dill to save nearest neighbor model
# dill.dump(ingred_nn, open('data/nn_ingred.pkd', 'wb'))
# ingred_nn = dill.load(open('data/nn_ingred.pkd', 'rb'))


# ohe matrix to find nn products
cbf_matrix = pd.DataFrame(ingred_features).set_index(df_content_uniq['names'])
# dill.dump(cbf_matrix, open('data/cbf_matrix.pkd', 'wb'))
# cbf_matrix = dill.load(open('data/cbf_matrix.pkd', 'rb'))


# example
# find 10 products similar to the test product
# indices are indices of the 10 most similar products, not the actual product id
ingred_dists, ingred_indices = ingred_nn.kneighbors(ingred_features[0].reshape(1, -1))

# list of 10 products similar to test product, excluding the first one, which is the test product
prod_list = prod_df.iloc[ingred_indices[0]]



###############################################################################

# ALS in Spark
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from datetime import datetime
from lxml import etree
import bz2
import os


# get current location for this main.py file, then add path to SE (StackExchange) files
def localpath(path):
    return 'file://' + str(os.path.abspath(os.path.curdir)) + '/' + path

# initialize pyspark and sql
sc = SparkContext("local[*]", "demo")
print(sc.version)

sqlContext = SQLContext(sc)

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

tot_df = spark.read.csv("df_cf.csv",header=True, inferSchema=True)

# Create ALS model
als_cv = ALS(userCol="uid", itemCol="pro_ids", ratingCol="lipies", nonnegative = True, implicitPrefs = False)
# Confirm that a model called "als" was created
type(als_cv)


# Add hyperparameters and their respective values to param_grid
# rank is the number of latent features
# regparam: regularization lambda
# coldstartstrategy: for train/test when all user's ratings are in the test set
# so if train set doesnt have rating for that user, dont calculate the rmse
param_grid = ParamGridBuilder() \
           .addGrid(als_cv.rank, [10, 20, 50]) \
           .addGrid(als_cv.regParam, [.05, .1, .15]) \
           .build()

# Define evaluator as RMSE and print length of evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="lipies", predictionCol="prediction")
print("Num models to be tested: ", len(param_grid))

# Build cross validation using CrossValidaator
cv = CrossValidator(estimator=als_cv, estimatorParamMaps=param_grid, evaluator=evaluator,numFolds=5)

# Confirm cv was built
print(cv)

#Fit cross validator to the 'train' dataset
model = cv.fit(tot_df)

#Extract best model from the cv model above
best_model = model.bestModel

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
userRecs_pd = userRecs.toPandas()
# need to convert to string for split below to work
userRecs_pd['new_rec'] = userRecs_pd['recommendations'].apply(lambda x: [str(pair[0]) for pair in x])
# create dataframe with rec columns
userRecs_pd[['rec1', 'rec2', 'rec3', 'rec4', 'rec5', 'rec6', 'rec7', 'rec8', 'rec9', 'rec10']] = pd.DataFrame(userRecs_pd.new_rec.values.tolist(), index= userRecs_pd.index)
userRecs_pd.drop(['recommendations', 'new_rec'], axis=1, inplace=True)
# dill.dump(userRecs_pd, open('userRecs_pd.pkd', 'wb'))

##############################################################################

# load recommendation table by ALS
df_als = dill.load(open('data/userRecs_pd.pkd', 'rb'))
df_als.index = df_als['uid']
len(df_als)


##############################################################################
# find users similar to current user

df_uid.columns
user_features = ['age', 'eyes', 'skin_type', 'skin_tone', 'skin_undertone'
    , 'hair_color', 'hair_type', 'hair_texture']
df_features = df_uid[user_features]
# have to have uid in the df so when dropping dont drop users with same chars
df_features['uid'] = df_uid['uid']
df_features.index = df_uid['uid']

# dataframe of unique users and their chars
df_features = df_features.drop_duplicates()
len(df_features)
# 111,820

# dill.dump(df_features['uid'], open('data/df_features.pkd', 'wb'))
# df_features = dill.load(open('data/df_features.pkd', 'rb'))

one_hot_transformer = OneHotEncoder(sparse=False)

# one hot encoding of characteristics
user_features_ohe = one_hot_transformer.fit_transform(df_features[user_features])
len(user_features_ohe)
# 111,820


# get feature names for each onehot encoder
feat_names = one_hot_transformer.get_feature_names()
for i, feat in enumerate(feat_names):
    print(i, feat)

# fit knn to identify users who are similar
nn = NearestNeighbors(n_neighbors=10).fit(user_features_ohe)

# use dill to save nearest neighbor model
# dill.dump(nn, open('data/nn_users.pkd', 'wb'))
nn = dill.load(open('data/nn_users.pkd', 'rb'))

# example
u_test = np.array([0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0])

# find 10 users similar to the test user
# indices are indices of the 10 most similar users, not the actual uid
# dists, indices = nn.kneighbors(user_features_ohe[0].reshape(1, -1))
dists, indices = nn.kneighbors(u_test.reshape(1, -1))

# list of users similar to our test user
user_list = df_features.iloc[indices[0]]

# merge list of similar user with list of users' recommendations
# rec is a list of product ids
rec = pd.merge(user_list, df_als, left_on=user_list.index, right_on=df_als.index, how='left')[['uid_x', 'rec1']]

prod_rec = pd.merge(rec, prod_df, left_on=rec.rec1, right_on=prod_df.pro_ids, how='left')

prod_rec_dedup = prod_rec.drop(['key_0', 'uid_x', 'rec1'], axis=1).drop_duplicates()

top10_df = pd.DataFrame(top10)
top10_rec = pd.merge(top10_df, prod_df, left_on=top10_df.index, right_on=prod_df.names, how='left')

tot_rec = prod_rec_dedup[['names', 'ratings', 'repurchases', 'pkg_quals', 'prices',
       'ingredients', 'brands']].append(top10_rec[['names', 'ratings', 'repurchases', 'pkg_quals', 'prices',
       'ingredients', 'brands']])

test = tot_rec.sample(10, random_state=123)



#########################product similarities using cosine#############################
# uid is number between 0 and 98839. pro_ids are id from the source and it's between 524 to 199302
# index error because the csr_matrix function looks at the index and assume there are rows with
# index 0-523 even though there arent. So hypothetically if i convert this to a dense matrix i'll
# have those 0-523 rows with 0s in it.
# but down the line this doesnt really work because the product column and new features get converted
# back into dense matrix.
col_num = len(df_cf_uid['uid'].unique())
df_cf_uid['pro_ids'] = df_cf_uid['pro_ids'].astype('int')
product_review_matrix = csr_matrix((df_cf_uid['lipies'], (df_cf_uid['pro_ids'], df_cf_uid['uid'])), shape=((df_cf_uid['pro_ids'].max()+1), col_num))
# shape is 199303, 98840

# sample view
product_review_pivot = df_cf.iloc[2000:2500].pivot_table(index='pro_ids', columns='users', values='lipies').fillna(0)

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()
# MaxAbsScaler, transforms the data so that all users have the same influence on the model,
# regardless of how many products they used.


# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()


# Create a pipeline: pipeline
# pipeline = make_pipeline(scaler, nmf, normalizer)
pipeline = make_pipeline(nmf, normalizer)

# Apply fit_transform to product review matrix: norm_features
norm_features = pipeline.fit_transform(product_review_matrix)
# note we have 199303 rows/products and 20 features now; NMF effective reduced dimensions
norm_features.shape

# index is implied to be index=range(df_cf['pro_ids'].max()+1)
df_nmf = pd.DataFrame(norm_features)
# df_nmf = pd.DataFrame(norm_features, index=range(df_cf['pro_ids'].max()+1))
df_nmf_t = pd.merge(df_nmf, prod_df, left_on=df_nmf.index, right_on='pro_ids', how='left')
df_nmf_prod = pd.DataFrame(norm_features, index=df_nmf_t.names)

# save nmf table
dill.dump(df_nmf_prod, open('data/df_nmf.pkd', 'wb'))
df_nmf_prod = dill.load(open('data/df_nmf.pkd', 'rb'))

# example: find products similar to "Cetaphil Gentle Skin Cleanser", 'Retin A'
product = df_nmf_prod.loc['Retin A']

# Compute cosine similarities: similarities
similarities = df_nmf_prod.dot(product)

# Display those with highest cosine similarity
print(similarities.nlargest(10))

t = similarities.nlargest(10)[1:]
s = '\n'.join([prod for prod in t.index[1:]])

top10 = similarities.nlargest(11)[1:]
