import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from sklearn.externals import joblib

# Import scripts that pulls data to expose functions
import pull_data
import train_model
from train_model import TypeSelector, DummyEncoder




# If this script is executed stand alone (as Main program)
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
        train_data = train_model.transform_data(train_data, values, 'train')
        test_data = train_model.transform_data(test_data, values, 'test')

        #4. create X,y data set and train model
        X=train_data.drop('Survived', axis = 1)
        y=train_data['Survived']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        #Load previously persisted pipeline
        try:
            pipe2 = joblib.load('model_pipe.pkl')
            print("\n\nModel model_pipe.pkl successfully loaded from local drive")

            #5. Test for accuracy of model
            y_predict = pipe2.predict(X_test)
            score = accuracy_score(y_test, y_predict)
            report = classification_report(y_test, y_predict)
            # print accuracy score
            print("\n\nprint accuracy of model: %s" %score)

            print("\n\n" + report)

            y_final_predict = pipe2.predict(test_data)

            try:
                f = open('classification_report.txt','w')
                f.write(report)
                f.close()

                try:
                    np.savetxt("prediction.csv", y_final_predict, delimiter=",")
                    print("\n\nFinal Predictions saved to prediction.csv on local drive")
                except:
                    print("\n\nFinal Predictions could not be saved to prediction.csv on local drive - Contact Support")
            except:
                print("\n\n Classification Report could not be written to local drive - Contact Support")

        except:
            print("\n\nUnable to load model from local drive - Contact Support")
