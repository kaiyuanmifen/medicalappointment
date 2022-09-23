
print("This code run the experiments of predicting no show vs. show vs. cancel ")

import sys


from ExperimentFunction import RunExperiemnts_NN,RunExperiemnts_Native_bayes,RunExperiemnts_PreiousVisitBaseline,RunExperiemnts__Keras_binary



NumberOfFolds=10
NEpochs=30 #depending on amount of data available, if unsure, use validaiton set to set early stopping



####NN model without mother's education
RunExperiemnts__Keras_binary("data_missing_type_nomotheredu",NumberOfFolds=NumberOfFolds,
                             NEpochs=NEpochs,model="NN",
                             NameToExclude=['Mother'])

####logisitc model without mother's education
RunExperiemnts__Keras_binary("data_missing_type_nomotheredu",NumberOfFolds=NumberOfFolds,
                             NEpochs=NEpochs,model="Logistic",
                             NameToExclude=['Mother'])

####NN model without race
RunExperiemnts__Keras_binary("data_missing_type_norace",NumberOfFolds=NumberOfFolds,
                             NEpochs=NEpochs,model="NN",
                             NameToExclude=['Language',"Race","Hispanic","Ethnicity","Interpreter"])
####logistic model without race
RunExperiemnts__Keras_binary("data_missing_type_norace",NumberOfFolds=NumberOfFolds,
                             NEpochs=NEpochs,model="Logistic",
                             NameToExclude=['Language',"Race","Hispanic","Ethnicity","Interpreter"])




##NN model with all features
RunExperiemnts__Keras_binary("data_missing_type",NumberOfFolds=NumberOfFolds,
                             NEpochs=NEpochs,model="NN",
                             NameToExclude=None)
####logistic model with all features
RunExperiemnts__Keras_binary("data_missing_type",NumberOfFolds=NumberOfFolds,
                             NEpochs=NEpochs,model="Logistic",
                             NameToExclude=None)




##NN model without race nor mother's education
RunExperiemnts__Keras_binary("data_missing_type_noracemotheredu",NumberOfFolds=NumberOfFolds,
                             NEpochs=NEpochs,model="NN",
                             NameToExclude=['Language',"Race","Hispanic","Ethnicity","Interpreter"])
####logistic model without race nor mother's education
RunExperiemnts__Keras_binary("data_missing_type_noracemotheredu",NumberOfFolds=NumberOfFolds,
                             NEpochs=NEpochs,model="Logistic",
                             NameToExclude=['Language',"Race","Hispanic","Ethnicity","Interpreter"])








