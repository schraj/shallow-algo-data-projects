import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.evaluation import Evaluation

class Modeling:
  df = None
  X = None
  Y = None
  target_variable = ''
  X_train = None
  X_test = None
  y_train = None
  y_test = None
  evaluation = None
  
  def __init__(self, df, target_variable):
    self.df = df
    self.target_variable = target_variable
    self.X = df.drop(columns=[target_variable])
    self.y = df[target_variable]
    self.evaluation = Evaluation()

  def one_hot_encode(self, exclude_columns = [], drop_first = True):
    columns = [c for c in self.df.select_dtypes(include=['object']) if c not in exclude_columns]
    new_cols = pd.get_dummies(self.X[columns], drop_first=drop_first)
    self.X = pd.concat([self.X, new_cols], axis=1)
    self.X.drop(columns=columns, inplace=True)

  def set_train_test_split(self):
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

  def scale_attributes(self, columns = []):
    for column in columns:
      # use a separate scaler for each attribute
      scaler = StandardScaler()
      scaler.fit(self.X_train[[column]])
      self.X_train[column] = scaler.transform(self.X_train[[column]])
      self.X_test[column] =scaler.transform(self.X_test[[column]])

  def classify_logistic_regression(self, max_iter=1000):
    lr = LogisticRegression(max_iter=max_iter)
    lr.fit(self.X_train, self.y_train)
    return lr.predict(self.X_test)
  
  def classify_logistic_regression_cv_classifier(self, max_iter=1000):
    lr = LogisticRegressionCV(max_iter=max_iter)
    lr.fit(self.X_train, self.y_train)
    preds = lr.predict(self.X_test)
    print('Classify Logistic Regression')
    self.evaluation.accuracy_evaluation(preds, self.y_test)

  def classify_decision_tree(self):
    tree_model = DecisionTreeClassifier()
    tree_model.fit(self.X_train, self.y_train)
    preds = tree_model.predict(self.X_test)
    print('Classify Decision Tree')
    self.evaluation.accuracy_evaluation(preds, self.y_test)
  
  def classify_random_forest(self):
    rf_model = RandomForestClassifier()
    rf_model.fit(self.X_train, self.y_train)
    preds = rf_model.predict(self.X_test)
    print('Classify Random Forest')
    self.evaluation.accuracy_evaluation(preds, self.y_test)
