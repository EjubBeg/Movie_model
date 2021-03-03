# -*- coding: utf-8 -*-
"""MLProject.ipynb

---

# ML Project (Movie revenue prediction model)

Problem: For given dataset which includes wide informations of a movie , we have to build a model which will precisely predict the worldwide revenue for a movie.

## Introduction
"""

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer(sparse_output=True)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler
standardScalerX = StandardScaler()

! pip install ann_visualizer
from ann_visualizer.visualize import ann_viz;

from sklearn.metrics import r2_score ,mean_squared_error

import warnings
warnings.filterwarnings('ignore')

from scipy.stats import pearsonr

import matplotlib.pyplot as plt

import pandas_profiling

import seaborn as sns 

from sklearn.tree import export_graphviz

from keras.models import Sequential

from keras.layers import Dense

import pandas as pd 

import numpy as np

import collections

import statistics

import copy

import math

import json 

import ast

import seaborn as sns
sns.set_theme(style="ticks", color_codes=True)


# %matplotlib inline
plt.style.use('seaborn')
pd.set_option('max_columns', None)

movie_dataset = pd.read_csv('drive/MyDrive/new.csv')

"""# Analyzing dataset"""

# Prints information about dataset 
movie_dataset.info()

# We print first 5 rows of dataset
movie_dataset.head()

# prints the number of missing data for all rows in a descanding order

movie_dataset.isnull().sum().sort_values(ascending=False)

# We drop rows with missing values for the next features 

movie_dataset = movie_dataset[movie_dataset.budget.notnull()]
movie_dataset = movie_dataset[movie_dataset.genres.notnull()]
movie_dataset = movie_dataset[movie_dataset.id.notnull()]
movie_dataset = movie_dataset[movie_dataset.imdb_id.notnull()]
movie_dataset = movie_dataset[movie_dataset.original_language.notnull()]
movie_dataset = movie_dataset[movie_dataset.popularity.notnull()]
movie_dataset = movie_dataset[movie_dataset.production_companies.notnull()]
movie_dataset = movie_dataset[movie_dataset.production_countries.notnull()]
movie_dataset = movie_dataset[movie_dataset.release_date.notnull()]
movie_dataset = movie_dataset[movie_dataset.revenue.notnull()]
movie_dataset = movie_dataset[movie_dataset.runtime.notnull()]
movie_dataset = movie_dataset[movie_dataset.title.notnull()]
movie_dataset = movie_dataset[movie_dataset.vote_count.notnull()]
movie_dataset = movie_dataset[movie_dataset.vote_average.notnull()]
movie_dataset = movie_dataset[movie_dataset.cast.notnull()]
movie_dataset = movie_dataset[movie_dataset.crew.notnull()]
movie_dataset = movie_dataset[movie_dataset.keywords.notnull()]

# We drop rows with zero value for revenue
movie_dataset = movie_dataset[movie_dataset.revenue != 0]

# We reset the index for dataset and drop useless features 

movie_dataset = movie_dataset.reset_index(drop=True)
movie_dataset.reset_index(drop=True, inplace=True) 
movie_dataset= movie_dataset.drop(columns = ['id' , 'imdb_id', 'poster_path','overview','tagline',
                  'original_title','title','status','adult','video'])

# prints the number of missing data for all rows in a descanding order

movie_dataset.isnull().sum().sort_values(ascending=False)

"""# Analyzing and exploring data"""

# We convert the budget and popularity feature to a numeric type , it was string previously  
movie_dataset[['budget']] = movie_dataset[['budget']].apply(pd.to_numeric) 
movie_dataset[['popularity']] = movie_dataset[['popularity']].apply(pd.to_numeric) 


# For these features we change datatype to a dictionary instead of a string type
text_cols = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'keywords', 'cast', 'crew']

def text_to_dict(df):
    for col in text_cols:
        df[col] = df[col].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x))
    return df

movie_dataset = text_to_dict(movie_dataset)

# The number of outliners in each numerical feature

Q1 = movie_dataset.quantile(0.25)
Q3 = movie_dataset.quantile(0.75)
IQR = Q3 - Q1
((movie_dataset < (Q1 - 1.5 * IQR)) | (movie_dataset > (Q3 + 1.5 * IQR))).sum()[((movie_dataset < (Q1 - 1.5 * IQR)) | (movie_dataset > (Q3 + 1.5 * IQR))).sum() != 0]

"""### Belongs to collection"""

# A example of the belongs to colection feature 
movie_dataset['belongs_to_collection'][0]

# We made a new list which stores 1 if movie belongs to collection and 0 if not
part_of_collection = movie_dataset['belongs_to_collection'].apply(lambda x: 1 if len(x) > 0 else 0)

part_of_collection

# We print the correleation and the number of movies which belong to a collection
pearsonr(part_of_collection, movie_dataset['revenue']) , sum(part_of_collection)

# Scatter plot of part of collection list compared to revenue
plt.scatter(part_of_collection, movie_dataset['revenue'])
plt.show()

# print median revenue of movies which do not belong to a collection
movie_dataset.revenue[part_of_collection==1].median() , sum(part_of_collection)

# print median revenue of movies which do belong to a collection
movie_dataset['revenue'][part_of_collection==0].median()

# Now the column will store 1 and 0 instead of a dictionary
movie_dataset.belongs_to_collection = part_of_collection

"""### Keywords"""

# This is how extracting values from this variable work , we have a list of lists filled with dictionaries
print(movie_dataset['keywords'][0])
print("\n")
print(movie_dataset['keywords'][0][0])
print("\n" + movie_dataset['keywords'][0][0]['name'])

# Here we extract all keywords in vector and extract them in a 2d matrix
allkeywords=[]

def getkey(row):
    x = []
    for i in row:
        allkeywords.append(i['name'])
        x.append(i['name'])
    return x

keywords = movie_dataset['keywords'].apply(lambda x: getkey(x))

# Printing maximal number of keyword values in a single movie
key_num = movie_dataset['keywords'].apply(lambda x: len(x) if x != {} else 0)
print(max(key_num))

# We put all keywords in a set , so we dont have any duplicate
set1 = set(allkeywords);

# We print the number of keyword classes 
len(set1)

# Histogram for number of keywords in a movie 
n, bins, patches = plt.hist(x=key_num, bins='auto', color='#0504aa')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Number of Keywords')
plt.ylabel('Frequency')
plt.title('Hist for number of keywords')

# describing number of keywords 
key_num.describe()

# Scatterplot for keyword number and revenue
plt.scatter(key_num, movie_dataset['revenue'])
plt.show()

# correlation between keyword number and revenue
pearsonr(key_num , movie_dataset['revenue'] )

# We will drop the keywords
movie_dataset = movie_dataset.drop(columns=['keywords'])

"""### homepage"""

# printing first values and the type
movie_dataset['homepage'][0] , type(movie_dataset['homepage'][0])

# Because we have a lot of missing data mostly because not all movies have a homepage and
# thats why we will make a new variable which defines does a movie contains/has a homepage

contain_homepage = movie_dataset['homepage'].isnull().apply(lambda x: 0 if x else 1 )

contain_homepage.head(5)

# number of movies which contain a homepage
sum(contain_homepage)

# Correlation
pearsonr(contain_homepage, movie_dataset['revenue'])

# Scatterplot
plt.scatter(part_of_collection, movie_dataset['revenue'])
plt.show()

# Median revenue of movies which have a homepage
movie_dataset['revenue'][contain_homepage==1].median()

# Median revenue of movies which do not have a homepage
movie_dataset['revenue'][contain_homepage==0].median()

# The binary representation will be used as a new feature instead of homepage links 
movie_dataset.homepage = contain_homepage

"""### Genres"""

# Example of genre values
movie_dataset.genres[1]

# Here we made 3 new columns , one which will store number of genres in a movie ,
# one which will store all genre names and one which will store all names in a list

allgenres =[]
def getgenres(row):
    x = []
    for i in row:
        allgenres.append(i['name'])
        x.append(i['name'])    
    return x
    
genr = movie_dataset['genres'].apply(lambda x: getgenres(x))
gen_num = movie_dataset.genres.apply(lambda x: len(x))

setgen = set(allgenres)
print('We have ' + str(len(setgen)) + ' different genres \n') 
setgen

genres_count = collections.Counter([i for j in genr for i in j]).most_common()
fig = plt.figure(figsize=(8, 5))
sns.barplot([val[1] for val in genres_count],[val[0] for val in genres_count])
plt.xlabel('Count')
plt.title('Top 20 Genres')
plt.show()

# Summary of number of genres in a movie
gen_num.describe()

# Scatter plot of genres number to revenue

plt.scatter(gen_num, movie_dataset['revenue'])
plt.show()

# Correlation
pearsonr(gen_num , movie_dataset.revenue)

# Top 5 examples of filtered genre columns
genr.head()

# We will use the filtered version now
movie_dataset.genres = genr

"""### Crew 

"""

# Example of crew value and how to use dictionarie in python
print(movie_dataset['crew'][0])
print("\n")
print(movie_dataset['crew'][0][0])
print("\n" + movie_dataset['crew'][0][0]['name'])

# Here we get the number a crew from every movie
def getcrewnum(row):
    return len(row)

crewnumber = movie_dataset['crew'].apply(lambda x: getcrewnum(x))

# Summary
crewnumber.describe()

# Correlation
pearsonr(crewnumber,movie_dataset['revenue'])

# Scatterplot 
plt.scatter(crewnumber, movie_dataset['revenue'])
plt.show()

plt.boxplot(crewnumber)  # We can clearly see a lot of outliners in out boxplot
plt.show()

#Next i will take the directors name and make a category from it
def getdir(row):
    for i in row:
        if i['job'] == 'Director':
            return i['name']

directors = movie_dataset['crew'].apply(lambda x: getdir(x))

directors.head(5)

setd = set(directors);
len(setd) # We have 3358 different directors

# We us ethe filtered version and add cresnumber as a new column in our dataset
movie_dataset.crew = directors
movie_dataset['crew_number'] = crewnumber

"""### Cast """

# Example of our feature
print(movie_dataset['cast'][0])
print("\n")
print(movie_dataset['cast'][0][0])
print("\n" + movie_dataset['cast'][0][0]['name'])

# get number of cast in a movie
def getcastnum(row):
    return len(row)

castnumber = movie_dataset['cast'].apply(lambda x: getcastnum(x))

# Summary of cast number in  am movie 
castnumber.describe()

# Correlation between cats number and revenue of a movie 
pearsonr(castnumber , movie_dataset['revenue'])

# scatter plot 
plt.scatter(castnumber, movie_dataset['revenue'])
plt.show()
len(castnumber[castnumber == 0])

# Boxplot of castnumber 
plt.boxplot(castnumber)
plt.show()
sum(castnumber==0)

# Here we will get top 3 actors from every movie and all those names in a vector

def getactorsformovie(row):
    x = []
    for i in row:
        if i['order'] in (0,1,2):
            x.append(i['name'])
    return x

castpeople =[]

for row in movie_dataset['cast']:
    for single in row:
        if single['order'] in (0,1,2):
            castpeople.append(single['name'])
        
castnames = movie_dataset['cast'].apply(lambda x: getactorsformovie(x))

castnames.head(5)

setact = set(castpeople)

len(setact) # number of different names

cast_count = collections.Counter([i for j in castnames for i in j]).most_common(20)
fig = plt.figure(figsize=(8, 5))
sns.barplot([val[1] for val in cast_count],[val[0] for val in cast_count])
plt.xlabel('Count')
plt.title('Top 20 Cast names')
plt.show()

# Put cleaned data and add new column to dataset
movie_dataset.cast = castnames
movie_dataset['cast_number'] = castnumber

"""### Popularity """

# Top 10 values
movie_dataset['popularity'].head(10)

# Summary of popularity
movie_dataset['popularity'].describe()

# Boxplot , histogram and scatterplot
plt.boxplot(movie_dataset['popularity']) 
plt.show()
plt.hist(movie_dataset['popularity'])
plt.show()
plt.scatter(movie_dataset['popularity'] , movie_dataset['revenue'])
plt.show()

# Correlation 
pearsonr(movie_dataset['popularity'] , movie_dataset['revenue'])

"""### Release date """

# Change from string to date time type
movie_dataset.release_date = pd.to_datetime(movie_dataset.release_date,  errors='coerce')

# We extract the year , month and day of the week from date 
# And we drop the release date feature
 
release_year = pd.DatetimeIndex(movie_dataset['release_date']).year
release_month = pd.DatetimeIndex(movie_dataset['release_date']).month
release_dow = pd.DatetimeIndex(movie_dataset['release_date']).dayofweek

movie_dataset = movie_dataset.drop(columns=['release_date'])

movie_dataset['release_dow']= release_dow
movie_dataset['release_year'] = release_year
movie_dataset['release_month'] = release_month

# Summary of new added columns
print('For day of the week release :\n')
print('Median is :' + str(movie_dataset['release_dow'].median()))
print(movie_dataset['release_dow'].describe())
print('\nFor month release :\n')
print('Median is :' + str(movie_dataset['release_month'].median()))
print(movie_dataset['release_month'].describe())
print('\nFor year release :\n')
print('Median is :' + str(movie_dataset['release_year'].median()))
print(movie_dataset['release_year'].describe())

# Crosstable for day of the week and month
days = pd.crosstab(index=movie_dataset["release_dow"], columns="count") 
months = pd.crosstab(index=movie_dataset["release_month"], columns="count") 
days ,months

# scatter plots for new columns 
plt.scatter(movie_dataset.release_dow , movie_dataset.revenue)
plt.title('Day of the week')
plt.show()
plt.scatter(movie_dataset.release_month , movie_dataset.revenue)
plt.title('Month')
plt.show()
plt.scatter(movie_dataset.release_year , movie_dataset.revenue)
plt.title('Year')
plt.show()

# We will make the day of the week and month categorical
movie_dataset.release_dow = movie_dataset.release_dow.astype('category')
movie_dataset.release_month = movie_dataset.release_month.astype('category')

"""We have three new columns but we will use only the year because the other two are uniformal

### Runtime
"""

# top 5 examples
movie_dataset['runtime'].head()

# Scatter plot 
plt.scatter(movie_dataset['runtime'] , movie_dataset['revenue'])
plt.show()

# Boxplot
plt.boxplot(movie_dataset['runtime'])
plt.show()

# Summary of runtime 
movie_dataset['runtime'].describe()

# Correlation
pearsonr(movie_dataset.runtime, movie_dataset['revenue'])

# We add the median where runtime is zero
runtime_med = movie_dataset.runtime.median()
movie_dataset.runtime[movie_dataset.runtime == 0] = runtime_med

"""### Budget """

# Summary of budget 
movie_dataset['budget'].describe()

# Boxplot
plt.boxplot(movie_dataset['budget'])
plt.show()

# Scatterplot
plt.scatter(movie_dataset['budget'] , movie_dataset['revenue'])
plt.show()

# Correlation 
pearsonr(movie_dataset.budget , movie_dataset.revenue)

# Number of examples with 0 value
sum(movie_dataset.budget == 0)

movie_dataset.info()

# We will predict values for budget where budget value is 0

train_bud = copy.copy(movie_dataset[movie_dataset.budget != 0])
unknown_bud = copy.copy(movie_dataset[movie_dataset.budget == 0])
Y = train_bud.budget

X = train_bud[[ 'vote_count'  ,'popularity' , 'homepage' , 'revenue' , 
               'crew_number' , 'cast_number'  ,'release_year']]

newX = unknown_bud[['vote_count'  ,'popularity' , 'homepage' , 'revenue' , 
                    'crew_number' , 'cast_number'  ,'release_year' ]]



rfr = RandomForestRegressor(n_estimators= 100)
rfr.fit(X, Y)

predicted = rfr.predict(newX)
movie_dataset.budget[movie_dataset.budget == 0] = predicted

# New correlation
pearsonr(movie_dataset.budget , movie_dataset.revenue)

# New scatterplot
plt.scatter(movie_dataset['budget'] , movie_dataset['revenue'])
plt.show()

"""### Production companies"""

# Example of production companies example
movie_dataset['production_companies'][0][0]['name']

# Here we get number of production companies in movie
def getprodcomnum(row):
    return len(row)

prod_com_number = movie_dataset['production_companies'].apply(lambda x: getprodcomnum(x))

# Summary of number of production companies
print('Median : ' + str(prod_com_number.median())) , prod_com_number.describe()

# Scatterplot of number of production companies and revenue
plt.scatter(prod_com_number, movie_dataset['revenue'])
plt.show()

# correlation of number of production companies
pearsonr(prod_com_number , movie_dataset['revenue'])

# Now we get the production companies names 
com = []
def getcomp(row):
    x = []
    for i in row:
        com.append(i['name'])
        x.append(i['name'])
    return x
  
companies = movie_dataset['production_companies'].apply(lambda x: getcomp(x))

companies_count = collections.Counter([i for j in companies for i in j]).most_common(20)
fig = plt.figure(figsize=(8, 5))
sns.barplot([val[1] for val in companies_count],[val[0] for val in companies_count])
plt.xlabel('Count')
plt.title('Top 20 Production Company')
plt.show()

# We print number of classes for production companies 
setcom = set(com)
len(setcom)

# Extracted names of companies 
companies.head()

# We put the cleaned data insted of the old one 
movie_dataset.production_companies = companies

"""### Production countries """

# Example of a production country value 
movie_dataset['production_countries'][0]

# We get the number of production countries involved in a movie
def getprodcouum(row):
    return len(row)

prod_cou_number = movie_dataset['production_countries'].apply(lambda x: getprodcouum(x))

# Summary of number of production countries
print('Median : ' + str(prod_cou_number.median())) , prod_cou_number.describe()

# Scatter plot 
plt.scatter(prod_cou_number, movie_dataset['revenue'])
plt.show()

# Correlation
pearsonr(prod_cou_number , movie_dataset['revenue'])

# Now we get the nanes of production countries 
cou = []
def getnat(row):
    x = []
    for i in row:
        cou.append(i['iso_3166_1'])
        x.append(i['iso_3166_1'])
    return x
  
nations = movie_dataset.production_countries.apply(lambda x: getnat(x))

countries_count = collections.Counter([i for j in nations for i in j]).most_common(20)
fig = plt.figure(figsize=(8, 5))
sns.barplot([val[1] for val in countries_count],[val[0] for val in countries_count])
plt.xlabel('Count')
plt.title('Top 20 Production Country')
plt.show()

# Number of classes of production countries
setnat = set(cou)
len(setnat)

# We use the filtered names now 
movie_dataset.production_countries = nations

"""### Original and Spoken language"""

# We change the type to categorical instead of string  
movie_dataset['original_language'] = movie_dataset['original_language'].astype('category')

# Summary of original language 
movie_dataset.original_language.describe()

# Histogram of original language
movie_dataset.original_language = movie_dataset.original_language.astype('category')
sns.countplot(movie_dataset.original_language).set_title("Original languages")

# Number of spoken languages in a movie 
num_spoken = movie_dataset.spoken_languages.apply(lambda x : len(x) if len(x) > 0 else 0)

# Scatterplot 
plt.scatter(num_spoken , movie_dataset.revenue)
plt.show()

# Correlation
pearsonr(num_spoken , movie_dataset.revenue)

# We drop the spoken language feature 
movie_dataset = movie_dataset.drop(columns=['spoken_languages'])

"""We will use categorical original language , spoken we will not use since it is not important

###Votes
"""

# Summary of vote average 
movie_dataset.vote_average.describe()

# Boxplot 
plt.boxplot(movie_dataset.vote_average)
plt.show()

# Scatterplot 
plt.scatter(movie_dataset.vote_average , movie_dataset.revenue)
plt.show()

# Correlation
pearsonr(movie_dataset.vote_average,movie_dataset.revenue)

# Summary of vote count 
movie_dataset.vote_count.describe()

# Boxplot 
plt.boxplot(movie_dataset.vote_count)
plt.show()

# Scatterplot 
plt.scatter(movie_dataset.vote_count , movie_dataset.revenue)
plt.show()

# Correlation
pearsonr(movie_dataset.vote_count,movie_dataset.revenue)

# We will fill all values with 0 with the median
vote_ave_med = movie_dataset.vote_average.median()
vote_count_med = movie_dataset.vote_count.median()

movie_dataset.vote_average[movie_dataset.vote_average == 0] = vote_ave_med
movie_dataset.vote_count[movie_dataset.vote_count == 0] = vote_count_med

"""###Revenue """

# Summary of the rvenue
movie_dataset.revenue.describe()

# Boxplot 
plt.boxplot(movie_dataset.revenue )
plt.show()

"""# Preparing dataset for the model

## Encoding categorical data

Now i will make two datasets :
1. With all columns after encoding 
2. With just the most important columns after encoding
"""

large_dataset = copy.copy(movie_dataset) 
large_dataset.info()

"""### Genre 

"""

large_dataset = large_dataset.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(large_dataset.pop('genres')),
                index=large_dataset.index,
                columns=mlb.classes_))

"""### Production countries"""

large_dataset = large_dataset.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(large_dataset.pop('production_countries')),
                index=large_dataset.index,
                columns=mlb.classes_))

"""### Production companies"""

large_dataset = large_dataset.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(large_dataset.pop('production_companies')),
                index=large_dataset.index,
                columns=mlb.classes_))

"""### Cast"""

# Because those names exist also as production companies 
# i changed their names before encoding
large_dataset.cast[561][1] = 'Walter Elias Disney' 
large_dataset.cast[4709][2] = 'Don Blutch'
large_dataset.cast[5941][0] = 'Coco '

large_dataset = large_dataset.join(
            pd.DataFrame.sparse.from_spmatrix(
                mlb.fit_transform(large_dataset.pop('cast')),
                index=large_dataset.index,
                columns=mlb.classes_))

"""### Crew """

# Because a high number of actors also direct movies ,we have a problem of 
# pair column names se we decided to ad _dir to the end of every actor who is in directors column

directors = pd.get_dummies(directors, sparse=True)

dir_col = directors.columns
data_col =   large_dataset.columns

cols = []
index = 0

for d in dir_col:
    for m in data_col:
        if d == m :
            cols.append(d)
    index = index +1

for i in directors.columns:
    if i in cols:
        directors.rename(columns = {i: i + '_dir' }, inplace = True) 

large_dataset = large_dataset.join(directors)

"""### Original language"""

org_lan = pd.get_dummies(movie_dataset.original_language, sparse=True)
large_dataset = large_dataset.join(org_lan)

"""### Release day of the week 

"""

release_dow_dummy = pd.get_dummies(large_dataset.release_dow, sparse=True)
release_dow_dummy.columns

# We rename the columns to days names
release_dow_dummy.rename(columns = {0:'Monday'}, inplace = True) 
release_dow_dummy.rename(columns = {1:'Tuesday'}, inplace = True) 
release_dow_dummy.rename(columns = {2:'Wednesday' }, inplace = True) 
release_dow_dummy.rename(columns = {3: 'Thursday' }, inplace = True) 
release_dow_dummy.rename(columns = {4:'Friday' }, inplace = True) 
release_dow_dummy.rename(columns = {5:'Saturday'}, inplace = True) 
release_dow_dummy.rename(columns = {6:'Sunday'}, inplace = True) 
release_dow_dummy.columns

# An example of how it looks at the end 
release_dow_dummy.head()

# We add the columns to our dataset 
large_dataset = large_dataset.join(release_dow_dummy)

"""### Release month"""

release_month_dummy = pd.get_dummies(large_dataset.release_month, sparse=True)
large_dataset = large_dataset.join(release_month_dummy)

"""## Making small dataset and removing encoded features"""

# We drop all data that is not numeric
datatype = large_dataset.dtypes
useless_columns = datatype[(datatype == 'object') | (datatype == 'category')].index.tolist()
large_dataset= large_dataset.drop(columns = useless_columns)

# We print the shape of our large dataset
large_dataset.shape

# Here we extract all column which have a higher correlation that 15 % 
# and add them to our new small dataset
corr= []
for col in large_dataset.columns:
    corr.append(pearsonr(large_dataset[col], large_dataset.revenue)[0])

ind = 0
indexes = []
for i in corr:
    if i > 0.15 or i < - 0.15 :
        indexes.append(ind)
    ind = ind + 1 

small_dataset = large_dataset.iloc[:, indexes]
# Info about new dataset
small_dataset.info()

"""# Model training and evaluation"""

large_dataset.info()

"""### Splitting data"""

Y_s = small_dataset.revenue
X_s = small_dataset.drop(columns= ['revenue']) 

X_train_small , X_test_small , Y_train_small , Y_test_small = train_test_split(X_s , Y_s , test_size = 0.2 , random_state = 0)

Y_h = large_dataset.revenue
X_h = large_dataset.drop(columns= ['revenue']) 

X_train_large , X_test_large , Y_train_large , Y_test_large = train_test_split(X_h , Y_h , test_size = 0.2 , random_state = 0)

"""## Linear regression

Large dataset
"""

# Training linear regression for large dataset 
linear_regressor = LinearRegression()
linear_regressor.fit(X_train_large, Y_train_large)
accuracy = linear_regressor.score(X_test_large, Y_test_large)
print('Large dataset :')
print('Accuracy: %.2f' % (accuracy*100))

"""Encoded Dataset"""

# Encoding 
standardScalerX = StandardScaler()
X_train_std = standardScalerX.fit_transform(X_train_large)
X_test_std = standardScalerX.fit_transform(X_test_large)

norm = MinMaxScaler().fit(X_train_large)
X_train_norm = norm.transform(X_train_large)
X_test_norm = norm.transform(X_test_large)

# Training linear regression for large scaled dataset 
linear_regressor = LinearRegression()
linear_regressor.fit(X_train_std, Y_train_large)
accuracy = linear_regressor.score(X_test_std, Y_test_large)
print('Large standardized dataset :')
print('Accuracy: %.2f' % (accuracy*100))

linear_regressor = LinearRegression()
linear_regressor.fit(X_train_norm, Y_train_large)
accuracy = linear_regressor.score(X_test_norm, Y_test_large)
print('Large normalized dataset :')
print('Accuracy: %.2f' % (accuracy*100))

"""Small dataset"""

# Training linear regression for small dataset 
linear_regressor.fit(X_train_small, Y_train_small)
accuracy = linear_regressor.score(X_test_small, Y_test_small)
print('Small dataset :')
print('Accuracy: %.2f' % (accuracy*100))
print(pd.DataFrame(linear_regressor.coef_, X_train_small.columns, columns = ['Coeff']))

# Predicted vs actual values reveue
Y_test_small = Y_test_small.values
print(np.concatenate((predicted.reshape(len(predicted),1), Y_test_small.reshape(len(Y_test_small),1)),1)[0 :10 ,])

# Scaling small dataset
X_train_std = standardScalerX.fit_transform(X_train_small)
X_test_std = standardScalerX.fit_transform(X_test_small)

norm = MinMaxScaler().fit(X_train_small)
X_train_norm = norm.transform(X_train_small)
X_test_norm = norm.transform(X_test_small)

# training linear regression model with small standardized 
linear_regressor.fit(X_train_std, Y_train_small)
accuracy = linear_regressor.score(X_test_std, Y_test_small)
print('Small standardized dataset :')
print('Accuracy: %.2f' % (accuracy*100))
print(pd.DataFrame(linear_regressor.coef_, X_train_small.columns, columns = ['Coeff']))

# training linear regression model with small normalized 
linear_regressor.fit(X_train_norm, Y_train_small)
accuracy = linear_regressor.score(X_test_small, Y_test_small)
print('Small normalized dataset :')
print('Accuracy: %.2f' % (accuracy*100))
print(pd.DataFrame(linear_regressor.coef_, X_train_small.columns, columns = ['Coeff']))

"""## Random Forest Regression

Large dataset
"""

# Training RFR model with large unscaled dataset 
regressor = RandomForestRegressor(n_estimators= 1000 , random_state= 0 )
regressor.fit(X_train_large,Y_train_large)
predicted = regressor.predict(X_test_large)
r2_score(Y_test_large , predicted)

regressor

# Predicted vs actual values reveue
Y_test_large = Y_test_large.values
print(np.concatenate((predicted.reshape(len(predicted),1), Y_test_large.reshape(len(Y_test_large),1)),1)[0 : 15 , ])

# Scaling large dataset
from sklearn.preprocessing import StandardScaler
standardScalerX = StandardScaler()
X_train_std = standardScalerX.fit_transform(X_train_large)
X_test_std = standardScalerX.fit_transform(X_test_large)

from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler().fit(X_train_large)
X_train_norm = norm.transform(X_train_large)
X_test_norm = norm.transform(X_test_large)

# Training linear regression for large normalized dataset 
regressor = RandomForestRegressor(n_estimators= 1000 , random_state= 0 )
regressor.fit(X_train_norm,Y_train_large)
predicted = regressor.predict(X_test_norm)
r2_score(Y_test_large , predicted) # 80.1

# Training linear regression for large standardized  dataset 

regressor = RandomForestRegressor(n_estimators= 1000 , random_state= 0 )
regressor.fit(X_train_std,Y_train_large)
predicted = regressor.predict(X_test_std)
r2_score(Y_test_large , predicted)# 78.5

"""Small dataset """

# Training linear regression for small unscled dataset 

regressor = RandomForestRegressor(n_estimators= 100 , random_state= 0 )
regressor.fit(X_train_small,Y_train_small)
predicted = regressor.predict(X_test_small)
r2_score(Y_test_small , predicted) # 77.4

# Drawing the RFR

estimator = regressor.estimators_[5]

# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = X_train_small.columns,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')

regressor

# Predicted vs actual values reveue
Y_test_small = Y_test_small.values
print(np.concatenate((predicted.reshape(len(predicted),1), Y_test_small.reshape(len(Y_test_small),1)),1)[0 : 15 , ])

# Scaling the small dataset
X_train_std = standardScalerX.fit_transform(X_train_small)
X_test_std = standardScalerX.fit_transform(X_test_small)

norm = MinMaxScaler().fit(X_train_small)
X_train_norm = norm.transform(X_train_small)
X_test_norm = norm.transform(X_test_small)

# Training small normalized dataset
regressor = RandomForestRegressor(n_estimators= 100 , random_state= 0 )
regressor.fit(X_train_norm,Y_train_small)
predicted = regressor.predict(X_test_norm)
r2_score(Y_test_small , predicted) # 77.3

# Training small standardized dataset
regressor = RandomForestRegressor(n_estimators= 100 , random_state= 0 )
regressor.fit(X_train_std,Y_train_small)
predicted = regressor.predict(X_test_std)
r2_score(Y_test_small , predicted) # 74.9

"""## ANN"""

# Scaling the large dataset

from sklearn.preprocessing import StandardScaler
standardScalerX = StandardScaler()
X_train_std = standardScalerX.fit_transform(X_train_large)
X_test_std = standardScalerX.fit_transform(X_test_large)

from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler().fit(X_train_large)
X_train_norm = norm.transform(X_train_large)
X_test_norm = norm.transform(X_test_large)

"""Large dataset """

# We need the input number to input for ANN
X_train_large.shape

# Training the ANN with standardized large dataset

model = Sequential()
model.add(Dense(100, input_dim=18949, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train_std, Y_train_large , epochs=1000, verbose=0)
predicted = model.predict(X_test_std)
r2_score(Y_test_large , predicted) # 48.5

# Training the ANN with normalized large dataset

model = Sequential()
model.add(Dense(100, input_dim=18949, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train_norm, Y_train_large , epochs=1000, verbose=0)
predicted = model.predict(X_test_norm)
r2_score(Y_test_large , predicted) # 43.5

"""Small dataset """

# Scalling the small dataset
from sklearn.preprocessing import StandardScaler
standardScalerX = StandardScaler()
X_train_std = standardScalerX.fit_transform(X_train_small)
X_test_std = standardScalerX.fit_transform(X_test_small)

from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler().fit(X_train_small)
X_train_norm = norm.transform(X_train_small)
X_test_norm = norm.transform(X_test_small)

# Training ANN model with small normalized model
# and drawing the neural network

model = Sequential()
model.add(Dense(30, input_dim=27, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu')) 
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam' , metrics=['mse','mae'])
model.fit(X_train_norm, Y_train_small , epochs=1000, verbose=0)
predicted = model.predict(X_test_norm) 
r2_score(Y_test_small, predicted) # 77.2 for small dataset

# Vizualizing our ANN model
ann_viz(model, view=True, title="Movie predictor ANN")

# Training ANN model with small normalized model

model = Sequential()
model.add(Dense(40, input_dim=27, 
                activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu')) 
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='adam')
model.fit(X_train_std, Y_train_small, 
          epochs=500, verbose=0)
predicted = model.predict(X_test_std)
r2_score(Y_test_small, predicted)

# We will run different ANN model with different layer numbers 
# Also we changed th enodes number 

print('### Small node number ### \n')

for i in range(1,10):
    model = Sequential()
    model.add(Dense(100, input_dim=27, activation='relu'))
    for layers in range(0,i):
        model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train_std, Y_train_small, epochs=500, verbose=0)
    predicted = model.predict(X_test_std)
    r2 = r2_score(Y_test_small, predicted)
    print('Number : ' + str(i) + ' r2score : ' + str(r2))

print()
print('### Large node number ###\n')

for i in range(1,10):
    model = Sequential()    
    model.add(Dense(100, input_dim=27, activation='relu'))
    for layers in range(0,i):
        model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train_std, Y_train_small, epochs=500, verbose=0)
    predicted = model.predict(X_test_std)
    r2 = r2_score(Y_test_small, predicted)
    print('Number : ' + str(i) + ' r2score : ' + str(r2))

"""## Improving the best model"""

# To better understand each parameter of the model we will train a model with variety of different parameters

regressor = RandomForestRegressor(n_estimators= 100)
regressor.fit(X_train_norm,Y_train_small)
predicted = regressor.predict(X_test_norm)
print('Less number of trees in the forest : %.2f'  % (100*r2_score(Y_test_small , predicted))) 

regressor = RandomForestRegressor(n_estimators= 1000)
regressor.fit(X_train_norm,Y_train_small)
predicted = regressor.predict(X_test_norm)
print('\nMore number of trees in the forest : %.2f'  % (100*r2_score(Y_test_small , predicted))) 

regressor = RandomForestRegressor(n_estimators= 1000 , min_samples_leaf=1 )
regressor.fit(X_train_norm,Y_train_small)
predicted = regressor.predict(X_test_norm)
print('\nSmaller number for minimum of samples required to be at a leaf node : %.2f'  % (100*r2_score(Y_test_small , predicted))) 

regressor = RandomForestRegressor(n_estimators= 1000 , min_samples_leaf=10 )
regressor.fit(X_train_norm,Y_train_small)
predicted = regressor.predict(X_test_norm)
print('\nLarger number for minimum of samples required to be at a leaf node : %.2f'  % (100*r2_score(Y_test_small , predicted))) 

regressor = RandomForestRegressor(n_estimators= 1000 , max_features=0.5 )
regressor.fit(X_train_norm,Y_train_small)
predicted = regressor.predict(X_test_norm)
print('\nNumber of features to consider when looking for the best split is 0.5 : %.2f'  % (100*r2_score(Y_test_small , predicted))) 

regressor = RandomForestRegressor(n_estimators= 1000 , max_features=1 )
regressor.fit(X_train_norm,Y_train_small)
predicted = regressor.predict(X_test_norm)
print('\nNumber of features to consider when looking for the best split is 1 : %.2f'  % (100*r2_score(Y_test_small , predicted)))

regressor = RandomForestRegressor(n_estimators= 1000 ,  max_features='sqrt' )
regressor.fit(X_train_norm,Y_train_small)
predicted = regressor.predict(X_test_norm)
print('\nNumber of features to consider when looking for the best split is sqrt : %.2f'  % (100*r2_score(Y_test_small , predicted)))

regressor = RandomForestRegressor(n_estimators= 1000 ,  max_features='log2')
regressor.fit(X_train_norm,Y_train_small)
predicted = regressor.predict(X_test_norm)
print('\nNumber of features to consider when looking for the best split is log2 : %.2f'  % (100*r2_score(Y_test_small , predicted)))

regressor = RandomForestRegressor(n_estimators= 1000 , max_depth = 5 )
regressor.fit(X_train_norm,Y_train_small)
predicted = regressor.predict(X_test_norm)
print('\nMaximal depth of tree is 5 : %.2f'  % (100*r2_score(Y_test_small , predicted))) 

regressor = RandomForestRegressor(n_estimators= 1000 , max_depth = 20 )
regressor.fit(X_train_norm,Y_train_small)
predicted = regressor.predict(X_test_norm)
print('\nMaximal depth of tree is 20 : %.2f'  % (100*r2_score(Y_test_small , predicted))) 

regressor = RandomForestRegressor(n_estimators= 1000 , max_depth = 100 )
regressor.fit(X_train_norm,Y_train_small)
predicted = regressor.predict(X_test_norm)
print('\nMaximal depth of tree is 100 : %.2f'  % (100*r2_score(Y_test_small , predicted)))

# Commented out IPython magic to ensure Python compatibility.
# Applying from the prvious output we traind a new model on the large dataset
regressor = RandomForestRegressor(n_estimators= 1000 ,
                                  max_features=0.5)

regressor.fit(X_train_large,Y_train_large)
predicted = regressor.predict(X_test_large)
print('The improved model for RFR : %.2f' 
#       % (100*r2_score(Y_test_large , predicted)))