import numpy as np
import matplotlib.pyplot as plt

import collections
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve


def loadData(trainingFile, testFile):
    dfTra = pd.read_csv(trainingFile)
    dfTest = pd.read_csv(testFile)

    #Test data has no label, so fill it as -1 temperally.
    dfTest['IsBadBuy'] = -1
    
    testID = dfTest['RefId'].astype(int)
    df = dfTra.append(dfTest)
    
    return df, testID
  
def cleanData(df):

    ###---------------------deal with small amount missing values---------##

    #df['Color'] = df['Color'].replace('NOT AVAIL', 'OTHER')
    #df['Color'] = df['Color'].replace(np.nan, 'OTHER')
    df['Color'] = df['Color'].replace(np.nan, 'UnknowColor')

    df['Transmission'] = df['Transmission'].replace(np.nan, 'AUTO')
    df['WheelType'] = df['WheelType'].replace(np.nan, 'Unknow')
    df['Nationality'] = df['Nationality'].replace(np.nan, 'OTHER')
    df['Size'] = df['Size'].replace(np.nan, 'MEDIUM')
    df['TopThreeAmericanName'] = df['TopThreeAmericanName'].replace(np.nan, 'OTHER')

    ####-------------------convert date to number-----------------------###############
    
    dateList = df['PurchDate'].tolist()
    numDate = np.zeros(len(dateList),dtype = np.float32)
    for index, date in enumerate(dateList):
        septDate = date.split('/')
        numDate[index] = float(septDate[2])+float(septDate[0])/12+float(septDate[1])/30
    
    numDateDF = pd.DataFrame({'numDate':numDate})

    del df['PurchDate']
    df = pd.concat([df, numDateDF], axis=1)

    ###-------------------deal with other numbers-----------------------#####
    types = df.dtypes
    headerList = list(df)
    for i in range(len(headerList)):
        if types[i] == 'float64':
            #print headerList[i]
            df[headerList[i]] = df[headerList[i]].fillna(df[headerList[i]].mean()).astype(int)
    return df

def featureEngineering_makeMode(df):
    """deal with car make, model, trim and sub model"""

    modelList = df['Model'].tolist()
    makeList = df['Make'].tolist()
    trimList = df['Trim'].tolist()
    
    modelList = list(map(lambda x:x.split(' '), modelList))
    makeModeTrimList = []
    for index, mode in enumerate(modelList):
##        if len(val) >= 2:
##            newList.append(makeList[index]+' '+val[0]+val[1])
##        else:
        #makeModeTrimList.append(makeList[index]+' '+mode[0]+' '+str(trimList[index]).upper())
        makeModeTrimList.append(makeList[index]+' '+mode[0])

    makeModeTrimDF = pd.DataFrame({'makeModeTrim':makeModeTrimList})
    df = pd.concat([df, makeModeTrimDF ], axis=1)
    return df
def featureEngineering_MMR(df, keepOriginal = True):

    MMRList = ['MMRAcquisitionAuctionAveragePrice','MMRAcquisitionAuctionCleanPrice','MMRAcquisitionRetailAveragePrice',
                   'MMRAcquisitonRetailCleanPrice', 'MMRCurrentAuctionAveragePrice',
                   'MMRCurrentAuctionCleanPrice', 'MMRCurrentRetailAveragePrice',
                   'MMRCurrentRetailCleanPrice']

    df['diff_AAP'] = df['MMRAcquisitionAuctionAveragePrice']-df['MMRCurrentAuctionAveragePrice']
    df['diff_ACP'] = df['MMRAcquisitionAuctionCleanPrice']-df['MMRCurrentAuctionCleanPrice']
    df['diff_RAP'] = df['MMRAcquisitionRetailAveragePrice']-df['MMRCurrentRetailAveragePrice']
    df['diff_RCP'] = df['MMRAcquisitonRetailCleanPrice']-df['MMRCurrentRetailCleanPrice']


    df['acqu_diff_AAP_ACP'] = df['MMRAcquisitionAuctionAveragePrice']-df['MMRAcquisitionAuctionCleanPrice']
    df['acqu_diff_RAP_RCP'] = df['MMRAcquisitionRetailAveragePrice']-df['MMRAcquisitonRetailCleanPrice']
    df['curr_diff_AAP_ACP'] = df['MMRCurrentAuctionAveragePrice']-df['MMRCurrentAuctionCleanPrice']
    df['curr_diff_RAP_RCP'] = df['MMRCurrentRetailAveragePrice']-df['MMRCurrentRetailCleanPrice']


    df['acqu_diff_AAP_RAP'] = df['MMRAcquisitionAuctionAveragePrice']-df['MMRAcquisitionRetailAveragePrice']
    df['acqu_diff_ACP_RCP'] = df['MMRAcquisitionAuctionCleanPrice']-df['MMRAcquisitonRetailCleanPrice'] 
    df['curr_diff_AAP_RAP'] = df['MMRCurrentAuctionAveragePrice']-df['MMRCurrentRetailAveragePrice']
    df['curr_diff_ACP_RCP'] = df['MMRCurrentAuctionCleanPrice']-df['MMRCurrentRetailCleanPrice'] 

    if not keepOriginal:
        for col in MMRList:
            del df[col]

    return df
def featureEngineering_Cost(df, keepOriginal = True):

    CostList = ['VehBCost', 'WarrantyCost']

    df['costPercentage'] = df['WarrantyCost']/df['VehBCost']  

    if not keepOriginal:
        for col in CostList:
            del df[col]

    return df

def findCorrelation(df, save):
    type1 = 'pearson'
    type2 = 'spearman'
    corr = df.corr(type1)
    if save:
        corr.to_csv("corr.csv")
    return

def evalModel(prediction,TestLabel):
    
    prediction = prediction.astype(np.float)
    TestLabel = TestLabel.astype(np.float)
    
    Accuracy = accuracy_score(TestLabel, prediction)
    Kappa = cohen_kappa_score(TestLabel, prediction)

    print "accuracy : "+str(Accuracy), "Kappa : "+ str(Kappa)

    #APS = average_precision_score(TestLabel, prediction)
    #print "average_precison_score", APS

    #precision = precision_score(TestLabel, prediction)
    #print "precision", precision 
    #recall = recall_score(TestLabel, prediction)
    #print "recall", recall
    AUC = roc_auc_score(TestLabel, prediction)
    print "AUC : ", str(AUC)

    F1 = f1_score(TestLabel, prediction)

    print "F1 score : ", F1

    return Accuracy, F1


def trainRF(X_train, X_test, y_train, y_test):
    print "Start to train the RF model..."

    del X_train['IsBadBuy']
    del X_test['IsBadBuy']

    RF_clf = RandomForestClassifier(n_estimators =80,
                                    random_state=0,
                                    #max_features=None,
                                    class_weight={1:9,0:1},
                                    min_samples_split = 20,
                                    oob_score=True,
                                    n_jobs = -1
                                    )
    RF_clf.fit(X_train,y_train)

    predictionAll = RF_clf.predict_proba(X_test).tolist()

    prediction1 = pd.Series(map(lambda x:x[1], predictionAll))

    #print type(prediction)
    #print type(y_test)

    accuracy, F1 = evalModel(prediction1.round(), y_test)
    print "training score:", RF_clf.oob_score_

    return RF_clf

def trainGBT(X_train, X_test, y_train, y_test):
    print "Start to train the GBT model..."

    GBT_clf = GradientBoostingClassifier(
                                    learning_rate=0.01,
                                    n_estimators =80,
                                    random_state=0,
                                    #max_features=None,
                                    min_samples_split = 20,
                                    )
    GBT_clf.fit(X_train,y_train)

    predictionAll = GBT_clf.predict_proba(X_test).tolist()

    prediction1 = pd.Series(map(lambda x:x[1], predictionAll))


    accuracy, F1 = evalModel(prediction1.round(), y_test)
    print "training score:", GBT_clf.train_score_

    return GBT_clf
    
def trainSVM(X_train, X_test, y_train, y_test):
    print "Start to train the SVM model..."

    del X_train['IsBadBuy']
    del X_test['IsBadBuy']
    
    SVM_clf = SVC(C = 1,
                  kernel = 'rbf',
                  #gamma = 0.1,
                  #probability = True,
                  class_weight ="balanced",
                  random_state = 0)

    SVM_clf.fit(X_train,y_train)

    prediction = SVM_clf.predict(X_test)

    evalModel(prediction, y_test)

def trainKNN(X_train, X_test, y_train, y_test):
    print "Start to train the KNN model..."

    del X_train['IsBadBuy']
    del X_test['IsBadBuy']
    
    KNN_clf = KNeighborsClassifier(n_jobs = -1)
    KNN_clf.fit(X_train,y_train)

    predictionAll = KNN_clf.predict_proba(X_test).tolist()
    prediction1 = pd.Series(map(lambda x:x[1], predictionAll))
    #print type(prediction)
    #print type(y_test)

    accuracy, F1 = evalModel(prediction1.round(), y_test)

def normalization(df):

    ndArray = df.values
    minmax = preprocessing.MinMaxScaler()
    scaled = minmax.fit_transform(ndArray.astype(np.float32))
    return pd.DataFrame(scaled)

def standardization(df):
    ndArray = df.values
    scaler = preprocessing.MinMaxScaler()
    standardized = scaler.fit_transform(ndArray.astype(np.float32))
    return pd.DataFrame(standardized)

def clfOnTest(clf, R_ID, R_test):

    prediction = clf.predict_proba(R_test).tolist()
    prediction = list(map(lambda x:x[1], prediction))

    #pDF = pd.DataFrame({'IsBadBuy':prediction})
    #output = pd.concat([R_ID, pDF],axis=1)
    #output.to_csv("out.csv",index=False)
    #print "the output csv file is saved"
    return prediction


if __name__ == "__main__":
    print "Loading and cleaning data... \n"
    
    rawDF, testID = loadData(trainingFile = 'training.csv',testFile = 'test.csv')
    rawDF = rawDF.reset_index() #introduce a new index, and we will drop it after we clean the data
    #findCorrelation(rawDF, save = True)
    
    cleanDF = cleanData(rawDF)
    del cleanDF['index']
    
    discardList_Base = ['VehYear','WheelTypeID','VNZIP1','PRIMEUNIT']
    for col in discardList_Base:
        del cleanDF[col]
        
    ## start featureEngineering of Make, Mode ##
    cleanDF = featureEngineering_makeMode(cleanDF)
    
    ##After analysis, discard the attributes below##
    discardList_makeMode = ['Nationality','TopThreeAmericanName','SubModel','Model','Make','Trim']

    for col in discardList_makeMode:
        del cleanDF[col]

    ## start featureEngineering of MMR ##
    cleanDF = featureEngineering_MMR(cleanDF, keepOriginal = True)

    ## start featureEngineering of VehBCost, WarrantyCost##
    #cleanDF = featureEngineering_Cost(cleanDF, keepOriginal = True) 
    
    oneHotEncodingList = ['Auction','Color','Transmission',
                          'WheelType','AUCGUART',
                          'VNST','IsOnlineSale','makeModeTrim',
                          'BYRNO','Size']

    discardList = []#'numDate'] #for feature selection later
    if discardList:
        for col in discardList:
            del cleanDF[col]   

    ####---------------------start one-hot-encoding--------------------------
    for e in oneHotEncodingList:
        if e not in discardList:
            cleanDF_OHE = pd.get_dummies(cleanDF[e])
            cleanDF = pd.concat([cleanDF, cleanDF_OHE], axis=1)
   
    #After OHE, we delete them 
    for e in oneHotEncodingList:
        if e not in discardList:
            del cleanDF[e]
          
    ###split merged table of training and test based on id###      
    trainingDF = cleanDF[cleanDF['RefId'] <= 73014]    #cutoffID = 73014
    testDF     = cleanDF[cleanDF['RefId'] > 73014]

    del trainingDF['RefId']
    del testDF['RefId']
    del testDF['IsBadBuy']


    ###check correlation



    #resampling data to ensemble
    devPredictList = []
    keepProbs = [1, 0.4]
    for keepProb in keepProbs:
        trainingDF_1 = trainingDF[trainingDF['IsBadBuy'] == 1].sample(frac = 1, random_state = 1) #keep all 1s
        trainingDF_0 = trainingDF[trainingDF['IsBadBuy'] == 0].sample(frac = keepProb, random_state = 0) #keep some 0s

        X_features = trainingDF_1.append(trainingDF_0)
        X_features = X_features.sample(frac = 1.0, random_state = 0)

        trainLabel = X_features['IsBadBuy'].astype(int)
        
        #X_features = normalization(trainingDF)
        #testDF     = normalization(testDF)

        #X_features = standardization(trainingDF)
        #testDF     = standardization(testDF)
     
        X_train, X_dev, y_train, y_dev = train_test_split(X_features, trainLabel,test_size= 0.1,
                                                            random_state = 0)
        RF_clf = trainRF(X_train, X_dev, y_train, y_dev)
        T_clf = RF_clf
        
        #print feature importance
##        featureScore = zip(T_clf.feature_importances_, list(X_train))
##        featureScore = sorted(featureScore, key = lambda tup: tup[0])
##        print featureScore[-10:]
        pred = clfOnTest(RF_clf, testID, testDF)
        devPredictList.append(pred)
        #SVM_clf = trainSVM(X_train, X_dev, y_train, y_dev)
        #KNN_clf = trainKNN(X_train, X_dev, y_train, y_dev)
        #GBT_clf = trainGBT(X_train, X_dev, y_train, y_dev)

    delta = 0.6 # for emsemble two Random forest classifiers
    ensemblePred = [devPredictList[0][i]*delta+devPredictList[1][i]*(1-delta) for i in xrange(len(devPredictList[0]))]

    predDF = pd.DataFrame({'IsBadBuy':ensemblePred})
    output = pd.concat([testID, predDF],axis=1)
    output.to_csv("submit.csv",index=False)

    print "the submit csv file is saved"


    

    

    
    


    
    
        






