

def Get_model_performance(Pred,Y):
    #performance and saving results
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix, cohen_kappa_score

    Confusion_Mat=confusion_matrix(Y, Pred)
    Cohen_kappa_Score=cohen_kappa_score(Y, Pred)
    Metrics=metrics.classification_report(Y, Pred, digits=3)

    return(Confusion_Mat,Cohen_kappa_Score,Metrics)



def RunExperiemnts__Keras_binary(ExperimentType="data_complete_only",
                                 NumberOfFolds=5,NEpochs=50,model="NN",
                                 NameToExclude=None):


    print("This code runs experiment on "+str(ExperimentType)+" setting with "+str(NumberOfFolds)+" folds" )

    import random
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    from sklearn.metrics import f1_score

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from keras.layers import Input, Dense
    from keras.models import Model, load_model


    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve



    for Fold in range(NumberOfFolds):

        print("Fold "+str(Fold))



        #load data

        # load data

        from DataLoader import DataLoad

        ColNames, X_Test, Y_Test, X_Val, Y_Val, X_Train, Y_Train = DataLoad(Type=ExperimentType,NameToExclude=NameToExclude)

        #no show is 1 , cancelled or show up are 0s, change this setting for different analysis
        Y_Train = np.argmax(np.asarray(Y_Train), axis=1)
        Vec = Y_Train.copy()
        Y_Train[Vec == 0] = 0
        Y_Train[Vec == 2] = 1
        Y_Train[Vec == 1] = 0

        Y_Test = np.argmax(np.asarray(Y_Test), axis=1)
        Vec = Y_Test.copy()
        Y_Test[Vec == 0] = 0
        Y_Test[Vec == 2] = 1
        Y_Test[Vec == 1] = 0


        import keras
        from keras.models import Sequential
        from keras.models import load_model
        from keras import models
        from keras import layers
        from keras.models import Model
        from keras import optimizers
        from keras.layers import Dense, Activation
        from keras.layers import Input, Dense, Dropout
        from keras.models import Model, load_model
        from keras import regularizers
        import keras.backend as K
        import tensorflow as tf



        #Model


        def PredictiveModel(X_Train,Y_Train):
            # class weight to tackle imbalanced data
            #class_weight = {0: np.sum(Y_Train == 1) / Y_Train.shape[0],
            #                1: 100*(np.sum(Y_Train == 0) / Y_Train.shape[0])}

            #model
            Inputshape = X_Train.shape[1]
            Outputshape=1
            # neuralnetwork = Sequential([
            #     Dense(1, input_dim=X_Train.shape[1],kernel_regularizer=regularizers.l1(0.01),activation='sigmoid')
            # ])

            import Models
            from Models import PredictiveNeuralNetwork_binary,LogisticRegression
            if model == "NN":
                neuralnetwork = PredictiveNeuralNetwork_binary(Inputshape,Outputshape)
            if model == "Logistic":
                neuralnetwork = LogisticRegression(Inputshape)


            neuralnetwork.compile(optimizer='adam', loss='binary_crossentropy')
            es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

            hist=neuralnetwork.fit(X_Train, Y_Train,
                            epochs=NEpochs,
                            batch_size=500,
                            shuffle=True,
                            validation_split=0.1,
                            callbacks=[es])
            return neuralnetwork,hist


        #Run model
        Keras_model,hist=PredictiveModel(X_Train,Y_Train)
        Keras_model.summary()



        #Save model

        import os
        if not os.path.exists("results"):
            os.makedirs("results")


        SaveDir="results/performance/"+ExperimentType+"_binary_"+model
        if not os.path.exists(SaveDir):
            os.makedirs(SaveDir)

        print("save dir:"+SaveDir)

        Keras_model.save(SaveDir+"/Model_Fold_"+str(Fold)+".h5")


        #performance and saving results
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve
        from sklearn.metrics import confusion_matrix, cohen_kappa_score


        #data with missing

        pred = Keras_model.predict(np.asarray(X_Test))

        print("AUCROC missing data:" + str(roc_auc_score(Y_Test, pred)))

        pred=pd.DataFrame(pred)

        Y_Test=pd.DataFrame(Y_Test)
        X_Test=pd.DataFrame(X_Test)

        pred.to_csv(SaveDir+'/Fold_'+str(Fold)+'_Pred.csv')
        Y_Test.to_csv(SaveDir+'/Fold_'+str(Fold)+'_Y_Test.csv')
        X_Test.to_csv(SaveDir+'/Fold_'+str(Fold)+'_X_Test.csv')
        pd.DataFrame(ColNames).to_csv(SaveDir+'/ColNames.csv')

        print("Experiment finished")


def RunExperiemnts_NN(ExperimentType="data_complete_only",NumberOfFolds=5,NEpochs=50):


    print("This code runs experiment on "+str(ExperimentType)+" setting with "+str(NumberOfFolds)+" folds" )

    import random
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    from sklearn.metrics import f1_score

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from keras.layers import Input, Dense
    from keras.models import Model, load_model


    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve

    from keras.models import Sequential
    from keras.models import load_model
    from keras import models
    from keras import layers
    from keras.models import Model
    from keras import optimizers
    from keras.layers import Dense, Activation
    from keras.layers import Input, Dense, Dropout
    from keras.models import Model, load_model
    from keras import regularizers
    import keras.backend as K
    import tensorflow as tf



    for Fold in range(NumberOfFolds):

        print("Fold "+str(Fold))



        #load data

        from DataLoader import DataLoad

        ColNames,X_Test,Y_Test,X_Val,Y_Val,X_Train,Y_Train=DataLoad(Type=ExperimentType)


        #Model



        def PredictiveModel(X_Train,Y_Train):
            # class weight to tackle imbalanced data
            #class_weight = {0: np.sum(Y_Train == 1) / Y_Train.shape[0],
            #                1: 100*(np.sum(Y_Train == 0) / Y_Train.shape[0])}

            #model
            Inputshape = X_Train.shape[1]
            Outputshape=Y_Train.shape[1]
            # neuralnetwork = Sequential([
            #     Dense(1, input_dim=X_Train.shape[1],kernel_regularizer=regularizers.l1(0.01),activation='sigmoid')
            # ])


            import Models
            from Models import PredictiveNeuralNetwork

            MLModel= Models.PredictiveNeuralNetwork(Inputshape,Outputshape)


            MLModel.compile(optimizer='adam', loss='categorical_crossentropy')

            #set class weight proportional to each type of outcome
            '''
            class_weight = {0: (Y_Train.shape[0]/np.sum(Y_Train,0))[0],
                            1: (Y_Train.shape[0]/np.sum(Y_Train,0))[1],
                            2: (Y_Train.shape[0]/np.sum(Y_Train,0))[2]}
                            
            '''

            #this class weight is hyper parameter, you need to adjust according to data
            class_weight = {0: 1,
                            1: 5,
                            2: 10}

            import keras
            es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

            hist=MLModel.fit(X_Train, Y_Train,
                            epochs=NEpochs,
                            batch_size=500,
                            validation_data=(X_Val,Y_Val),
                            shuffle=True,
                            class_weight=class_weight,
                            callbacks=[es])



            return MLModel,hist








        #Run model
        Keras_model,hist=PredictiveModel(X_Train,Y_Train)
        Keras_model.summary()



        #Save model

        import os
        if not os.path.exists("results"):
            os.makedirs("results")


        SaveDir="results/performance/"+ExperimentType+"_Softmax_"+"NN"
        if not os.path.exists(SaveDir):
            os.makedirs(SaveDir)

        print("save dir:"+SaveDir)

        Keras_model.save(SaveDir+"/Model_Fold_"+str(Fold)+".h5")

        #pred = Imp_model.predict(Autoencoder.predict(X_Test))



        #performance and saving results
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve
        from sklearn.metrics import confusion_matrix, cohen_kappa_score


        #Performance

        pred = Keras_model.predict(np.asarray(X_Test))
        pred=pd.DataFrame(pred)

        Y_Test=pd.DataFrame(Y_Test)
        X_Test=pd.DataFrame(X_Test)

        #pred.to_csv(SaveDir+'/Fold_'+str(Fold)+'_Weight_NN_Pred_missing.csv')
        #Y_Test.to_csv(SaveDir+'/Fold_'+str(Fold)+'_Weight_NN_Y_Test_missing.csv')
        #X_Test.to_csv(SaveDir+'/Fold_'+str(Fold)+'_Weight_NN_X_Test_missing.csv')

        PredictValue = np.argmax(np.asarray(pred), axis=1)
        Y_Test_Value = np.argmax(np.asarray(Y_Test), axis=1)

        Confusion_Mat, Cohen_kappa_Score, Metrics = Get_model_performance(Pred=PredictValue, Y=Y_Test_Value)


        pd.DataFrame(Confusion_Mat).to_csv(SaveDir+'/Fold_'+str(Fold)+'Confusion_missing.csv')

        with open(SaveDir+'/Fold_'+str(Fold)+'CK.txt', 'w') as f:
            f.write(str(Cohen_kappa_Score))

        with open(SaveDir+'/Fold_'+str(Fold)+'metrics.txt', 'w') as f:
            f.write(Metrics)

        print("Performance on data with missing information")
        print(Metrics)



        print("Experiment finished")


def RunExperiemnts_Native_bayes(ExperimentType="data_missing_type", NumberOfFolds=5):
    print("This code runs experiment on " + str(ExperimentType) + " setting with " + str(NumberOfFolds) + " folds")

    import random
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    from sklearn.metrics import f1_score

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from keras.layers import Input, Dense
    from keras.models import Model, load_model

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve

    from keras.models import Sequential
    from keras.models import load_model
    from keras import models
    from keras import layers
    from keras.models import Model
    from keras import optimizers
    from keras.layers import Dense, Activation
    from keras.layers import Input, Dense, Dropout
    from keras.models import Model, load_model
    from keras import regularizers
    import keras.backend as K
    import tensorflow as tf

    for Fold in range(NumberOfFolds):

        print("Fold " + str(Fold))

        # load data

        from DataLoader import DataLoad

        ColNames, X_Test, Y_Test, X_Val, Y_Val, X_Train, Y_Train = DataLoad(Type=ExperimentType)

        Y_Train=np.argmax(Y_Train,1)
        Y_Val= np.argmax(Y_Val, 1)
        Y_Test = np.argmax(Y_Test, 1)

        # Model

        def PredictiveModel(X_Train, Y_Train):

            from sklearn.naive_bayes import MultinomialNB

            MLModel = MultinomialNB()
            MLModel.fit(X_Train, Y_Train)

            return MLModel

        # Run model
        MLModel = PredictiveModel(X_Train, Y_Train)
       # MLModel.summary()

        # Save model

        import os
        if not os.path.exists("results"):
            os.makedirs("results")

        SaveDir = "results/performance/" + ExperimentType + "_Softmax_" + "NB"
        if not os.path.exists(SaveDir):
            os.makedirs(SaveDir)

        print("save dir:" + SaveDir)

        #import cPickle
        import pickle
        with open(SaveDir+"/Model_Fold_"+str(Fold)+".pkl", 'wb') as fid:
            pickle.dump(MLModel, fid)

        # pred = Imp_model.predict(Autoencoder.predict(X_Test))

        # performance and saving results
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import auc
        from sklearn.metrics import roc_curve
        from sklearn.metrics import confusion_matrix, cohen_kappa_score

        # Performance

        pred = MLModel.predict(np.asarray(X_Test))
        pred = pd.DataFrame(pred)

        Y_Test = pd.DataFrame(Y_Test)
        X_Test = pd.DataFrame(X_Test)

        # pred.to_csv(SaveDir+'/Fold_'+str(Fold)+'_Weight_NN_Pred_missing.csv')
        # Y_Test.to_csv(SaveDir+'/Fold_'+str(Fold)+'_Weight_NN_Y_Test_missing.csv')
        # X_Test.to_csv(SaveDir+'/Fold_'+str(Fold)+'_Weight_NN_X_Test_missing.csv')

        #PredictValue = np.argmax(np.asarray(pred), axis=1)
        #Y_Test_Value = np.argmax(np.asarray(Y_Test), axis=1)

        Confusion_Mat, Cohen_kappa_Score, Metrics = Get_model_performance(Pred=pred, Y=Y_Test)

        pd.DataFrame(Confusion_Mat).to_csv(SaveDir + '/Fold_' + str(Fold) + 'Confusion_missing.csv')

        with open(SaveDir+'/Fold_'+str(Fold)+'CK.txt', 'w') as f:
            f.write(str(Cohen_kappa_Score))

        with open(SaveDir+'/Fold_'+str(Fold)+'metrics.txt', 'w') as f:
            f.write(Metrics)

        print("Performance on data with missing information")
        print(Metrics)


        print("Experiment finished")




def RunExperiemnts_PreiousVisitBaseline(NumberOfFolds=5):
    print("This code runs uses previous history to guess appoitment status ")

    import random
    import pandas as pd
    import numpy as np
    import sklearn
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve
    from sklearn.metrics import f1_score

    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from keras.layers import Input, Dense
    from keras.models import Model, load_model

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve

    from keras.models import Sequential
    from keras.models import load_model
    from keras import models
    from keras import layers
    from keras.models import Model
    from keras import optimizers
    from keras.layers import Dense, Activation
    from keras.layers import Input, Dense, Dropout
    from keras.models import Model, load_model
    from keras import regularizers
    import keras.backend as K
    import tensorflow as tf

    from DataLoader import DataLoad

    ColNames, X_Test, Y_Test, X_Val, Y_Val, X_Train, Y_Train = DataLoad(Type="data_missing_type")

    # load data
    X = np.vstack([X_Test,X_Val,X_Train ])
    Y=np.vstack([Y_Test,Y_Val,Y_Train ])


    Pred=np.argmax(X[:,[667,666,665]],1)

    len(ColNames)

    ProVec = np.stack([1 - X[:, 664], np.repeat(np.sum(Y[:, 1] == 1) / Y.shape[0], Y.shape[0]), X[:, 664]], 1)#probability vector for sampling

    for Fold in range(NumberOfFolds):

        print("Fold " + str(Fold))






        Vec=[]
        for i in range(ProVec.shape[0]):

            Vec.append(np.random.choice([0,1,2],p=ProVec[i,]/np.sum(ProVec[i,])))
        Vec=np.asarray(Vec)
        Pred[X[:, 664] == 1]=Vec[X[:, 664] == 1]
        Y_real=np.argmax(Y,1)
        # Performance


        Confusion_Mat, Cohen_kappa_Score, Metrics = Get_model_performance(Pred=Pred, Y=Y_real)

        SaveDir = "results/performance/" +"_Baseline_"
        import os
        if not os.path.exists(SaveDir):
            os.makedirs(SaveDir)

        pd.DataFrame(Confusion_Mat).to_csv(SaveDir + '/Fold_' + str(Fold) + 'Confusion_missing.csv')

        with open(SaveDir+'/Fold_'+str(Fold)+'CK.txt', 'w') as f:
            f.write(str(Cohen_kappa_Score))

        with open(SaveDir+'/Fold_'+str(Fold)+'metrics.txt', 'w') as f:
            f.write(Metrics)

        print("Performance on data with missing information")
        print(Metrics)


        print("Experiment finished")


