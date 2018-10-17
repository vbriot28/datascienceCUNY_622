import pandas as pd
import numpy as np


from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from sklearn import svm

from sklearn.externals import joblib

from sklearn.base import BaseEstimator, TransformerMixin

# Import scripts that pulls data to expose functions
import pull_data

# Custom Transformers
# 1. to select type of variables
# https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696
class TypeSelector(BaseEstimator, TransformerMixin):
    """ Custom Transformer to select features by types"""

    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return (X.select_dtypes(include=[self.dtype]))

# 2. to dummy variables
# https://www.youtube.com/watch?v=KLPtEBokqQ0&app=desktop
class DummyEncoder(TransformerMixin):
    """ Custom Transformer to dummy up categorical variables with drop first"""

    def fit(self, X, y=None):
        return(self)

    def transform(self, X, y=None):
        return(np.asarray(pd.get_dummies(X, drop_first=True)))

# Helper Functions
# Classify age into category
def categorize_age (Age):
    """ Function to return a category according to the input age as follows;
        if Age < 16 --> 'Child', else if Age <= 60 --> 'Adult', else --> 'Senior'"""
    if(Age < 16):
        person_cat = 'Child'
    elif (Age <= 60):
        person_cat = 'Adult'
    else:
        person_cat = 'Senior'

    return(person_cat)

# Perform some basic transformation on data set
def transform_data(df, values, type = 'train'):
    """ This function will perform the necessary transformation for input dataframe
        1. Dropping of unwanted columns
        2. Imputation of missing values
        3. Add column 'Person_category'
        4. Convert categorical columns to 'Category' dtype
        """

    # 1. Drop Column = 'Cabin' from dataframe
    df.drop('Cabin', axis = 1, inplace=True)

    # 2. Fil in missing data
    df.fillna(value=values, inplace=True)

    # 3. Add Person_category based on Age
    df['Person_category'] = df['Age'].apply(categorize_age)

    # 4. Convert object type to Category for categorical variables
    for col in ['Pclass', 'Sex', 'Embarked', 'Person_category']:
        df[col] = df[col].astype('category')

    return(df)

# Through away functions for debugging and unit testing
def num_missing(x):
  return sum(x.isnull())

# If this script is executed stand alone (as Main program), call function and save data set to file.
if (__name__ == "__main__"):
    #Load files from local directory (output of pull_data script)
    train_url = 'train.csv'
    train_data = pull_data.load_data(train_url, 'train.csv')
    test_url = 'test.csv'
    test_data = pull_data.load_data(test_url, 'test.csv')

    # Check format of files by calling validation functions
    is_train_data_valid = pull_data.is_data_dimmenion_valid(train_data, 'train.csv', (891, 12)) and pull_data.is_data_columns_valid(train_data, 'train.csv', 'train')
    is_test_data_valid = pull_data.is_data_dimmenion_valid(test_data, 'test.csv', (418, 11)) and pull_data.is_data_columns_valid(test_data, 'test.csv', 'test')

    if (is_train_data_valid and is_test_data_valid):

        # I. Proceed with transformation

        #1. determine values for imputation based on trainning data set and build values dictionary
        median_age = train_data['Age'].dropna().median()
        median_fare = train_data['Fare'].dropna().median()
        most_used =  train_data['Embarked'].mode().iloc[0]

        values = {'Age' : median_age, 'Embarked' : most_used, 'Fare' : median_fare}

        #2. Transform both data set
        train_data = transform_data(train_data, values, 'train')
        test_data = transform_data(test_data, values, 'test')

        # II. Build pipeline
        # 1. Build one pipeline each for transformation steps for each variables type: Numerical and Categorical

        # Construct transformation steps for numerical type variables; select, scale
        steps_numerical = [('selector', TypeSelector(np.number)), ('scaler', StandardScaler())]

        # create pipeline
        p_trans_numerical = Pipeline(steps_numerical)

        # Construct transformation steps for categorical type variables; select, label, encode
        #steps_categorical = [('selector', TypeSelector('category')), ('labelEncoder', LabelEncoder()),
        #                     ('encoder', OneHotEncoder())]
        steps_categorical = [('selector', TypeSelector('category')), ('encoder', DummyEncoder())]

        # create pipeline
        p_trans_categorical = Pipeline(steps_categorical)

        #2. Combine the transformation by feature union & create tranformation pipeline
        FU_transform = FeatureUnion([('numerical', p_trans_numerical),
                             ('categorical', p_trans_categorical)])

        pipe_transform = Pipeline([('features', FU_transform)])

        #3. Create final pipeline by adding classifier
        # SVM model
        clf = svm.SVC(kernel='linear', C=1)
        pipe = Pipeline([('transformation', pipe_transform),
                         ('classifier', clf)])

        #4. create X,y data set and train model
        X=train_data.drop('Survived', axis = 1)
        y=train_data['Survived']

        #cv_scores = cross_val_score(pipe, X, y, cv=5)
        #print(cv_scores)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        #5. Fit Model
        pipe.fit(X_train, y_train)

        #6. Persist model as .pkl file
        try:
            joblib.dump(pipe, 'model_pipe.pkl')
            print("\n\nModel save to local drive as model_pipe.pkl")
        except:
            print("\n\nUnable to save model to local file - Contact Support")
