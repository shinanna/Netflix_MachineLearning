# Project7_Netflix

Code for data split:

data.drop('Unnamed: 0', axis=1, inplace=True)

# split dataset into independent and dependent variables
X = data.drop('rating', axis=1)
y = data['rating']

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

GCP hypertuning

# define model
GPC = GaussianProcessClassifier()

# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define grid
grid = dict()
grid['kernel'] = [1*RBF(), 1*DotProduct(), 1*Matern(), 1*RationalQuadratic(), 1*WhiteKernel()]

# define search
search = GridSearchCV(GPC, grid, scoring='accuracy', cv=cv, n_jobs=-1)

# perform the search
results = search.fit(x_train, y_train)

# summarize best
print('Best Mean Accuracy: %.3f' % results.best_score_)
print('Best Config: %s' % results.best_params_)

# summarize all
means = results.cv_results_['mean_test_score']
params = results.cv_results_['params']
for mean, param in zip(means, params):
    print(">%.3f with: %r" % (mean, param))
