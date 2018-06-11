import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Code from https://www.jianshu.com/p/1110f485880f
train_df=pd.read_csv('train.csv',header=0)
test_df=pd.read_csv('test.csv',header=0)


#We will impute missing values using the median for numeric
#columns and the most common value for string columns
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self,X,y=None):
        self.fill=pd.Series( [ X[c].value_counts().index[0]
                if X[c].dtype == np.dtype('O')  else X[c].median() for c in X    ] )

        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)

feature_columns_to_use=['Pclass','Sex','Age','Fare','Parch']
nonnumeric_columns=['Sex']

#Join the features  from train and test toghther before imputing missing values,
# in case theire distribution is slighty different
big_X=train_df[ feature_columns_to_use ].append( test_df[feature_columns_to_use] )
big_X_imputed=DataFrameImputer().fit_transform(big_X)

#XGBoos doesn't (yet) handle categorial features automaticall , so we need to
#change them to intgear values
#See https://scikit-learn.org/stable/modules/preprocessing.html
#details and output
le=LabelEncoder()
for feature  in nonnumeric_columns:
    big_X_imputed[feature] = le.fit_transform( big_X_imputed[ feature ] )

#Prepare the inputs for the model
train_X=big_X_imputed[ 0:train_df.shape[ 0 ] ].as_matrix()
test_X = big_X_imputed[ train_df.shape[0] :: ].as_matrix()
train_y= train_df[ 'Survived' ]

#You can experiment with many other options here, using the same fit() and .predict()
#methods; see https://scikit-learn.org
#This example use the current build of XGBoost
gbm=xgb.XGBClassifier(max_depth=3,n_estimators=300,learning_rate=0.05).fit(train_X,train_y)
predictions= gbm.predict(test_X)

#Kaggle need the submission to have a certian format
# see https://www.kaggle.com/c/titanic-gettingStarted/download/gendermodel.csv
#for an example of what its supposed to look like
submission = pd.DataFrame( { 'PssengerId': test_df['PassengerId'],
                            'Survived':predictions} )
submission.to_csv("submission.csv",index=False)
