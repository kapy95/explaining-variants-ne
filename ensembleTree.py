#Ensemble tree model
import xgboost as xgb


#Data management:
import pandas as pd
import numpy as np

#Calculate distances:
from scipy.spatial.distance import cdist

#Random number generator
from numpy.random import default_rng


#Performance measures: 
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

#Representations:
from performanceResults import generateDataMap

import warnings
warnings.filterwarnings("ignore")

#save and load xgboost models:
import joblib


def createAndTrainXGB_early_stop(X_train, y_train, X_val, y_val, params):
    """
    Input:
        - X_method: instances to use in training and validation
        - y_method: instance labels to be used in training and validation
        - params: parameters of xgboost in dictionary format
        - size: float to indicate the size of validation dataset
        - stratification: bolean value to indicate whether the training and validation data should be stratified or not
    
    Output:
        - clf: xgboost trained based on the data, parameters. This model uses early stop to avoid overfitting

    """
    evals_result={}
    dtrain=xgb.DMatrix(X_train,y_train)
    dval = xgb.DMatrix(X_val, y_val)
    clf = xgb.train(params=params, dtrain=dtrain,
                    evals=[(dtrain, "train"), (dval, "validation")],#in each round, we measure the performance in the training set and in the validation set. Only the last set is used for early stopping
                    early_stopping_rounds=5,
                    evals_result=evals_result)
     

    return clf, evals_result


def measureClassifierPerformance(specific_clf, label_clf, X_test, y_test, le_name_mapping_dataset, dirResults):
    """ Description: Function to measure the performance of a set of classifiers in different aspects for each class
        Input: 
            classifiers: dictionary whose keys are the code names of the classifiers, and the values are the classifier models
            X_test: data to be used for test
            y_test: class labels of the the test data

        Output:
            classifiers: dataframe whose rows represent the performance of a classifier for a certain class, like this:

            | Precision | Recall | Classifier | f1-score | Class
                0.01       0.87      clf40        0.5        X
                ...        ...        ...         ...       ...
            
            accus_classifiers: dictionary whose keys represent classification models and the values are the accuracies obtained in the test set
    
    """
    precisions_classifiers=[]
    recalls_classifiers=[]
    f1_classifiers=[]

    labels=[]
    classifiers_labels=[]
    values_labels=list(le_name_mapping_dataset.keys())
    d_test = xgb.DMatrix(X_test)


    joblib.dump(specific_clf, dirResults+"/clf_val"+str(label_clf)+".joblib")
    y_pred_probs_test=specific_clf.predict(d_test)
    if len(set(y_test))>2:
        y_pred_test=[np.argmax(probs) for probs in y_pred_probs_test]
    else:
        y_pred_test=[np.argmax([1-probs, probs]) for probs in y_pred_probs_test]
                         
    dirClfMap=dirResults+"/heatmap_clf_val"+str(label_clf)+".pdf"
    generateDataMap(y_real_test_map=y_test,
                    y_pred_map=y_pred_test,
                    le_name_mapping=le_name_mapping_dataset, 
                    labels=list(le_name_mapping_dataset.values()),
                    dirMap=dirClfMap)

    precision_specific_clf=precision_score(y_true=y_test,y_pred=y_pred_test,average=None)
    accu_specific_clf=accuracy_score(y_true=y_test,y_pred=y_pred_test)
    recall_specific_clf=recall_score(y_true=y_test, y_pred=y_pred_test, average=None)
    f1_specific_clf=f1_score(y_true=y_test, y_pred=y_pred_test, average=None)

    print("Accuracy:"+str(round(accu_specific_clf,2)))
    precisions_classifiers.extend(precision_specific_clf)
    recalls_classifiers.extend(recall_specific_clf)
    labels.extend(values_labels)
    f1_classifiers.extend(f1_specific_clf)
        
    clf_labels=[label_clf for i in range(len(values_labels))]
    classifiers_labels.extend(clf_labels)

    df_classifiers=pd.DataFrame.from_dict({"Precision":precisions_classifiers,
                                           "Recall":recalls_classifiers,
                                           "Classifier":classifiers_labels,
                                           "f1-scores":f1_classifiers,
                                           "Class":labels})
    return df_classifiers
    



def findChampionsInSpecificClassProbs(x_train_bin_class, probs_y_train, bin_class, case_sizes):
    #mapping: dictionary that translates bin to coded classes (i.e., new_le_mapping variable).The keys are classes and the values are the coded classes. 
    #The coded classes represent the indices that will be used to obtain the predicted probability of an instance for that class 
    index_prob_pred=bin_class
    
    cases=x_train_bin_class.index.to_list()#we obtain the case ids of these cases
    assert len(cases)==len(probs_y_train)
    
    if type(probs_y_train[0])==np.float32:#if the first element is just number and not a list, we are in a binary classification problem (class 0 or class 1), and just one probability is provided, which is the probability for class 1
        probs_y_train=[[1-probs,probs] for probs in probs_y_train]#we transform the probability obtained for each case so that we have both probabilities

    #we obtain a dictionary that relates each case to the predicted probability of its class, which is in the position equal to its class
    #For instance, if in the encoding the class is called 3, the fourth position of the list of the probabilities list contains the probability of the class
    bin_class_prob_per_case={case: probs[index_prob_pred] for case,probs in zip (cases, probs_y_train)}
    series_prob_case=pd.Series(bin_class_prob_per_case)

    champion=series_prob_case[series_prob_case==max(series_prob_case)]#obtain the champion, that is the instance with the highest probability to be in its correct class
    # #if there is more than one instance in the results with the same maximum probability:
    if len(champion)>1:
        sizes_possible_champions=case_sizes.loc[list(champion.index)]#get the sizes related to the possible champions 

        #find the case between the possible champios with lowest size
        shortest_champion=sizes_possible_champions[sizes_possible_champions==min(sizes_possible_champions)]

        if len(shortest_champion)>1:#again if there is more than one... (e.g. 12)
            rng = default_rng(seed=0)
            random_number=rng.integers(0, len(shortest_champion), size=1)[0]#generate a random number between the first possible champioon and the last one (e.g. between 1 and 12, it could be 5). We set the seed for reproducibility
            champion_class_id=list(shortest_champion.index)[random_number]#select it randomly based on the number, namely its identifier
            champion_class_label=x_train_bin_class.loc[[champion_class_id]]#filter the champion based on its identifier
        else:
            champion_class_label=x_train_bin_class.loc[[list(shortest_champion.index)[0]]]
    else: #if there is one champion directly, just filter it and save it
        champion_class_label=x_train_bin_class.loc[[list(champion.index)[0]]]

    return champion_class_label





def boundaryCasesBetweenTwoClasses(df_probs_class_x, df_probs_class_y, class_labelX, class_labelY, X_test):
    """
    Description: this function looks for the two closest boundary cases between cases. With boundary cases we understand cases whose prediction in the ensemble tree is the corresponding one to its real class (e.g., if the 
    class of a trace t1 is X, its class is predicted as X in the tree), but in terms of prediction probabilities another class Y is close to class X probability (e.g., prediction=[Probs to belong to X: 0.6, Probs to belong to Y: 0.4])
    
    Input: 
        - df_probs_class_x: dataframe whose rows represent cases and the columns represent the predicted probabilities to belong to a class. The class of all cases is predicted as X
        - df_probs_class_y: dataframe whose rows represent cases and the columns represent the predicted probabilities to belong to a class. The class of all cases is predicted as Y
        - class_labelY: class Y label used by the label encoder. It will be used to know the name of the column of class Y probabilities in the dataframes
        - class labelX: class X label used by the label encoder. It will be used to know the name of the column of class X probabilities in the dataframes
        - X_test: dataframe whose rows represent cases, and the columns confidences to declare rules.
    
    Output:
        - Pair of boundary cases between classes X and Y

    """

    #Firstly we calculate for each case of both dataframes the inverse absolute similarity using the predicted probabilities for class X and class Y. Basically consists in this: 1/(1+abs(probsX-probsY)). 
    # Thus, the less difference there is between probs of X and probs of Y, the greater the similarity will be. Therefore, we reward cases where probsX and probsY are close.
    df_probs_class_x["similarity_probs_labels"]=[1/(1+abs(row[class_labelX]-row[class_labelY])) for index, row in df_probs_class_x.iterrows()]
    df_probs_class_y["similarity_probs_labels"]=[1/(1+abs(row[class_labelX]-row[class_labelY])) for index, row in df_probs_class_y.iterrows()]
    
    #We sort the dataframes based on the similarity measure:
    df_sorted_classX=df_probs_class_x.sort_values(by="similarity_probs_labels",ascending=False)
    df_sorted_classY=df_probs_class_y.sort_values(by="similarity_probs_labels",ascending=False)

    #we obtain the 10 first case ids whose probabilities are most similar
    best_boundary_cases_classX=df_sorted_classX.iloc[0:10]
    best_boundary_cases_classY=df_sorted_classY.iloc[0:10]

    #we filter these cases in X_test
    filtered_cases_classX=X_test.filter(items=list(best_boundary_cases_classX.index), axis=0).fillna(-100)#to be able to calculate the distance, we fill nan values for -1 whihc is -100 in the current scale
    filtered_cases_classY=X_test.filter(items=list(best_boundary_cases_classY.index), axis=0).fillna(-100)

    #Finally we calculate the similarity between the cases based on the euclidean distance between the confidences of declare rules
    matrixBoundaryCases=cdist(filtered_cases_classX.values, filtered_cases_classY.values)

    #               case 1 class Y | case 2 class Y | case 3 class Y| ....
    #case 1 class X      0.5             0.3               0.9
    #case 2 class X      0.4             0.25              0.1
    #....               ....             ...               ...

    indexMinDistance=np.argmin(matrixBoundaryCases)#we obtain the index of the minimum distance
                        
    #Transform the index of the minimum distance to real index:
    min_idx = np.unravel_index(indexMinDistance, matrixBoundaryCases.shape)
    #The x component represents a case of classX, and the y component represents a case of classY, so we filter them:
    min_case_class1=filtered_cases_classX.loc[[list(filtered_cases_classX.index)[min_idx[0]]]]
    min_case_class2=filtered_cases_classY.loc[[list(filtered_cases_classY.index)[min_idx[1]]]]

    return min_case_class1, min_case_class2


    
