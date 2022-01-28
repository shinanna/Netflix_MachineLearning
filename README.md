# Project7_Netflix

Code for data split:

data.drop('Unnamed: 0', axis=1, inplace=True)

# split dataset into independent and dependent variables
X = data.drop('rating', axis=1)
y = data['rating']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
