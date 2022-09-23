###This code plot calibration curve
import numpy as np
from sklearn.calibration import calibration_curve
import pandas as pd

Dir="results/performance/data_missing_type_norace_binary_NN/"
Fold=2


File="Fold_"+str(Fold)+"_Y_test.csv"
Y_test=pd.read_csv(Dir+File)
Y_test=np.array(Y_test.iloc[:,1])

File="Fold_"+str(Fold)+"_Pred.csv"
Pred=pd.read_csv(Dir+File)
Pred=np.array(Pred.iloc[:,1])

prob_true, prob_pred = calibration_curve(Y_test, Pred, n_bins=10)

import matplotlib.pyplot as plt

plt.plot(prob_pred,prob_true)
plt.xlabel("Prediction score bin")
plt.ylabel("Prob. of no show")

plt.savefig("Callibration.png")