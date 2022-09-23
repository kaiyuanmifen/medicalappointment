
print("this code explores whether cancellation rate is significantly associated with any factors")

import sys
sys.path.insert(0,'Stage6/')
import numpy as np
import pandas as pd


from DataLoader import DataLoad

ExperimentType="data_missing_type"

NumberOfFolds=10

for Fold in range(NumberOfFolds):
    print("working on fold "+str(Fold))
    ColNames, X_Test, Y_Test, X_Val, Y_Val, X_Train, Y_Train = DataLoad(Type=ExperimentType)

    X_Train=X_Train[Y_Train[:,0]!=1,]
    Y_Train=Y_Train[Y_Train[:,0]!=1]
    Y_Train=np.argmax(np.asarray(Y_Train), axis=1)
    Y_Train[Y_Train==2]=0 #now Y=1 is cancellation


    X_Test=X_Test[Y_Test[:,0]!=1,]
    Y_Test=Y_Test[Y_Test[:,0]!=1]
    Y_Test=np.argmax(np.asarray(Y_Test), axis=1)
    Y_Test[Y_Test==2]=0 #now Y=1 is cancellation



    #model , logistic regression
    import Models
    from Models import LogisticRegression

    neuralnetwork=LogisticRegression(X_Train.shape[1])

    neuralnetwork.compile(optimizer='adam', loss='binary_crossentropy')

    hist=neuralnetwork.fit(X_Train, Y_Train,
                                epochs=20,
                                batch_size=500,
                                shuffle=True)

    pred = neuralnetwork.predict(np.asarray(X_Test))

    from sklearn.metrics import roc_auc_score
    print("AUCROC missing data:" + str(roc_auc_score(Y_Test, pred)))

    #Get weights
    len(neuralnetwork.get_weights()[0])

    Cancel_noshow_wieghts=pd.DataFrame({"Feature":ColNames,"Weights:":neuralnetwork.get_weights()[0].reshape(neuralnetwork.get_weights()[0].shape[0])})

    SaveDir = "results/Cancel_vs_noshow/"+"Fold"+str(Fold)+"/"

    import os
    if not os.path.exists(SaveDir):
        os.makedirs(SaveDir)

    Cancel_noshow_wieghts.to_csv(SaveDir+"FeatureImportance.csv")