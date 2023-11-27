import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class Evaluation:
  def __init__(self) -> None:
    pass

  def F1_score(predictions, targets):
    zipped = np.column_stack((predictions, np.array(targets)))
    df_z = pd.DataFrame(zipped)
    TP = df_z[(df_z[0] == 1) & (df_z[1] == 1)]
    FP = df_z[(df_z[0] == 1) & (df_z[1] == 0)]
    FN = df_z[(df_z[0] == 0) & (df_z[1] == 1)]

    tp_c = TP.shape[0]
    fp_c = FP.shape[0]
    fn_c = FN.shape[0]

    print('TP:', TP.shape[0])
    print('FP:', FP.shape[0])
    print('FN:', FN.shape[0])

    precision = tp_c/(fp_c + tp_c)
    recall = tp_c/(tp_c + fn_c)
    f1_score = 2 * (precision * recall) / (precision + recall)  
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1_score) 

  def accuracy_evaluation(self, predictions, targets):
    print("predictions:")
    predictions = pd.DataFrame(predictions)
    print(predictions.value_counts())
    print('')
    print("targets:", targets.value_counts())
    print("testing accuracy",accuracy_score(targets, predictions))
    print('')
    print ('Manual Accuracy of logistic regression(percentage of correctly labelled datapoints):')
    print(float((np.dot(targets,predictions) + np.dot(1-targets,1-predictions))/float(targets.size)*100))

    self.F1Score(predictions, targets)

def regression_evaluation(predictions, targets):
  from sklearn.metrics import mean_absolute_error,mean_squared_error
  mae = mean_absolute_error(y_true=targets,y_pred=predictions)
  mse = mean_squared_error(y_true=targets,y_pred=predictions)
  rmse = mean_squared_error(y_true=targets,y_pred=predictions,squared=False)

  print('MAE: ', mae)
  print('MSE: ', mse)
  print('RMSE: ', rmse)