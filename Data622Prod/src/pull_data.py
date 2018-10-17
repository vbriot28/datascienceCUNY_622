import pandas as pd

# Function to load data from a given source
# For now, we will load data from public Github User Account
def load_data(url, filename):
    """ This function will retrieve a .csv file from input url
        and return a dataframes.
        Expected format of file; .csv with header"""

    try:
        df = pd.read_csv(url)
        msg_success = "File: %s was loaded successfully" % filename
        print(msg_success)
        return(df)
    except:
        msg_failure = "Error loading file: %s - Contact Support" % filename
        print(msg_failure)

#Funtion to perform some "sanity check" on downloaded file
def is_data_dimmenion_valid(df, filename, r_c):
    """ This function will validate that the dimension of input dataframe
        matched input (r,c) parameter and return boolien"""
    (r,c) = r_c
    if (df.shape == (r,c)):
        msg_to_print = "Loaded file: %s passed dimension validation" % filename
        result = True
    else:
        msg_to_print = "Loaded file: %s failed dimention validation - Contact Support" % filename
        result = False

    print (msg_to_print)

    return(result)

#Function to perform some "sanity check" on downloaded file to verify columns
def is_data_columns_valid(df, filename, type = 'train'):
    """ This function will check the columns heading of an input dataframe
        to ensure that match what is expected """
    if (type == 'train'):
        columns_header = ['PassengerId', 'Survived', 'Pclass', 'Name',
                          'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
                          'Cabin', 'Embarked']
    else:
        columns_header = ['PassengerId', 'Pclass', 'Name',
                          'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare',
                          'Cabin', 'Embarked']

    # Identify columns header for input dataframe
    df_columns = df.columns.values.tolist()

    # Validate columns columns columns_header
    if (df_columns == columns_header):
        msg_to_print = "Loaded file: %s passed columns validation" % filename
        result = True
    else:
        msg_to_print = "Loaded file: %s failed columns validation - Contact Support" % filename
        result = False

    print(msg_to_print)

    return(result)

#Function to write downloaded data sets to .csv file on local drive,
#only to be used when this script is executed as stand alone (as main program)

def write_to_local_drive (df, filename):
    """ This function will take a dataframe and a string as input and write a .csv file to local drive"""
    result = False
    try:
        df.to_csv(filename, index=False)
        msg_success = "File: %s was successfully written to local drive as .csv" % filename
        print(msg_success)
        result = True
        return(result)
    except:
        msg_failure = "Error writting file: %s to local drive as .csv - Contact Support" % filename
        print(msg_failure)

# If this script is executed stand alone (as Main program), call function and save data set to file.
if (__name__ == "__main__"):
    train_url = 'https://raw.githubusercontent.com/vbriot28/datascienceCUNY_622/master/train.csv'
    train_data = load_data(train_url, 'train.csv')
    test_url = 'https://raw.githubusercontent.com/vbriot28/datascienceCUNY_622/master/test.csv'
    test_data = load_data(test_url, 'test.csv')

    # Check format of files by calling validation functions
    is_train_data_valid = is_data_dimmenion_valid(train_data, 'train.csv', (891, 12)) and is_data_columns_valid(train_data, 'train.csv', 'train')
    is_test_data_valid = is_data_dimmenion_valid(test_data, 'test.csv', (418, 11)) and is_data_columns_valid(test_data, 'test.csv', 'test')

    # Debug statement
    #print(train_data.columns.values.tolist())

    if (is_train_data_valid and is_test_data_valid):
        write_successful = write_to_local_drive(train_data, 'train.csv')
        if write_successful:
            write_successful = write_to_local_drive(test_data, 'test.csv')
