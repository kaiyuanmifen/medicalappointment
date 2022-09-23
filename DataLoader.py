



def DataLoad(Type="data_missing_type",Shuffle=True,NameToExclude=['Language',"Race","Hispanic","Ethnicity","Interpreter"]):
    import pandas as pd
    import numpy as np
    from keras.utils import np_utils

    #print("load data for training type: "+Type)
    print("columns excluded",NameToExclude)

    Y = pd.read_csv("../results/Y.csv")
    list(Y.columns.values)
    Missing = Y["Missing"]

    X=pd.read_csv("../results/X.csv")
    X=X.iloc[:,1:(X.shape[1]-1)]
    X.shape
    ColNames=X.columns.values

    ####remove the columns that shouhls be excluded
    if NameToExclude is not None:
        ColToInclude = list()
        for col in list(ColNames):
            if np.sum([i in col for i in NameToExclude]) == 0:
                ColToInclude.append(col)
        ColNames = np.array(ColToInclude)

        X=X[ColNames].copy()

    X=np.asarray(X)

    if Type=="data_complete_only" :
        ColWithMissing=np.asarray(["Mother.Education.Level_",
        "Mother.Education.Level_DECLINED TO ANSWER", "Mother.Education.Level_UNABLE TO COLLECT",
        "Public.Or.Private_Unavailable", "Public.Or.Private_",
        "Plan_", "Payor_", "Payor_PAYOR IS NULL",
        "Language_", "Language_Declined To Answer", "Language_Declined To Answer",
        "Language_Unable to Collect", "Race_Unknown", "Race_#N/A", "Race_Unable to Answer",
        "Race_Other", "Clinical.PCP.Name_"])

        X=X[:,~np.isin(ColNames,ColWithMissing)]

    if Type=="data_imputed":#data impuated
        X=pd.read_csv('../results/X_SImputed_No_Missing_indicator.csv')
        X = X.iloc[:, 1:(X.shape[1] - 1)]
        X.shape
        ColNames = X.columns.values
        X = np.asarray(X)

    if Type == "data_missing_type_extra":  # data impuated
        X = pd.read_csv("../results/X.csv")
        X = X.iloc[:, 1:(X.shape[1] - 1)]
        Weather = pd.read_csv('../results/WeatherInfor.csv')
        list(Weather.columns.values)
        Weather = Weather.iloc[:, 2:Weather.shape[1]]
        np.array(Weather)

        RedSox = pd.read_csv('../results/RedSoxInformation.csv')
        RedSox = RedSox.iloc[:, 1:RedSox.shape[1]]

        Flu = pd.read_csv('../results/FluInfomration.csv')
        Flu = Flu.iloc[:, 2:Flu.shape[1]]

        X = pd.concat([X.reset_index(drop=True), Weather.reset_index(drop=True),RedSox.reset_index(drop=True),Flu.reset_index(drop=True)],
                      axis=1)
        ColNames = list(X.columns.values)



    Y["value"]=0
    Y.loc[Y["Status"] == 'No Show',"value"] = 2
    Y.loc[Y["Status"] == 'Canceled',"value"] = 1
    Y=np_utils.to_categorical(Y["value"])
    np.sum(Y,0)

    Index=np.array(range(Y.shape[0]))
    if Shuffle==True:
        np.random.shuffle(Index)
    X=np.asarray(X)
    Y=np.asarray(Y)

    X_Train=X[Index[range(0,int(0.6*len(Index)))],:]
    Y_Train=Y[Index[range(0,int(0.6*len(Index)))]]

    X_Val = X[Index[range(int(0.6 * len(Index)), int(0.7 * len(Index)))],]
    Y_Val = Y[Index[range(int(0.6 * len(Index)), int(0.7 * len(Index)))]]

    X_Test = X[Index[range(int(0.7 * len(Index)), int(1 * len(Index)))],]
    Y_Test= Y[Index[range(int(0.7 * len(Index)), int(1 * len(Index)))]]



    print("finish data loading")

    return ColNames,X_Test,Y_Test,X_Val,Y_Val,X_Train,Y_Train