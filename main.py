from ensembleTree import createAndTrainXGB_early_stop, measureClassifierPerformance, boundaryCasesBetweenTwoClasses, findChampionsInSpecificClassProbs
from performanceResults import representMatrixMeasure
from sklearn.model_selection import train_test_split
import os
import datetime
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import xgboost as xgb
import argparse
import pm4py


#Library to load models:
import joblib

#SHAP functions:
from SHAP_utils import calculateSHAP_values, getMostAbsoluteImportantFeatures, plotGlobalFeatureImportances, plotFeatureImportanceForSpecificInstance, getSHAP_Values_for_instance

import warnings
warnings.filterwarnings("ignore")



def main(X_train_and_validation, y_train_and_validation, X_test, y_test, case_sizes, param_dist, size, strat_bool, le_name_mapping, dirRes, training, classifier_route):
    if training==True:
        print("Training classifiers....")
        if strat_bool==True:
            print("Stratification is going to be performed when training and validation sets are separated")
            strats=y_train_and_validation
            shuff=True
        else:
            strats=None
            shuff=False

        
        X_train, X_val, y_train, y_val = train_test_split(X_train_and_validation,
                                                          y_train_and_validation,
                                                          test_size=size,
                                                          stratify=strats,
                                                          shuffle=shuff,
                                                          random_state=0)

        clf_specific_size_val, eval_results=createAndTrainXGB_early_stop(X_train,
                                                                         y_train,
                                                                         X_val,
                                                                         y_val, 
                                                                         param_dist)

        dict_result={}
        dict_result["mlogloss-training"]=list(eval_results["train"].values())[0]
        dict_result["mlogloss-validation"]=list(eval_results["validation"].values())[0]
        pd.DataFrame.from_dict(dict_result).to_csv(dirRes+"/eval_results_clf"+".csv")
        
        print("Measuring performance...")
        df_performance_classifiers=measureClassifierPerformance(clf_specific_size_val,
                                                                size, 
                                                                X_test, 
                                                                y_test,
                                                                le_name_mapping,
                                                                dirRes)
        df_performance_classifiers.to_csv(dirRes+"/performance_classifiers.csv")

        
    else:
        if strat_bool==True:
            print("Stratification is going to be performed when training and validation sets are separated")
            strats=y_train_and_validation
            shuff=True
        else:
            strats=None
            shuff=False

        
        X_train, X_val, y_train, y_val = train_test_split(X_train_and_validation,
                                                          y_train_and_validation,
                                                          test_size=size,
                                                          stratify=strats,
                                                          shuffle=shuff, 
                                                          random_state=0)
        
        print("Loading trained classifier...")
        clf_specific_size_val=joblib.load(classifier_route)

    print("Calculating shapley values...")
    #Use the best classifier and the train data to obtain Shapley Values that explain how the classifier works:
    shap_values_training, explainer=calculateSHAP_values(clf_specific_size_val, X_train)

    print("Finding most important features of the classifier...")
    #Based on the shap values find the most important features following the classifier criterion in absolute terms (check function if you do not understand the "absolute terms" part):
    most_important_features, dict_absolute_importance_feature_per_class=getMostAbsoluteImportantFeatures(shap_values_training,
                                                                                                         list(X_train.columns))
    #Now we invert the mapping dictionary to generate a plot with the absolute importances  
    inv_mapping = {v: k for k, v in le_name_mapping.items()}
    dirPlotShap=dirRes+"/global_importances_shap.pdf"
    plotGlobalFeatureImportances(X_train, shap_values_training, inv_mapping, dirPlotShap)#we plot the importances at global level using SHAP 


    dict_probs_cases_per_label={}
    classes=le_name_mapping.keys()


    series_mapping=pd.Series(le_name_mapping)

    for encoded_class_label in set(y_train):#for each class
        print("Finding prototype for class "+str(series_mapping[series_mapping==encoded_class_label].index[0])+"...")
        positions_train_instances_class_label=np.where(y_train == encoded_class_label)#find the indices of the instances of that class in the train data
        X_train_class_label=X_train.iloc[positions_train_instances_class_label]#filter the test data to just contain the instances related to that class

         #after finding the champion, we prepare the test instances related to that class so that we can look later for the boundary cases:
        d_train_class_label=xgb.DMatrix(X_train_class_label)#we transform the dataframe to DMatrix so that the model can perform predictions on it, this is necessary, because we use the xgboost api for the model
        predicts_probs_cases_bin_class=clf_specific_size_val.predict(d_train_class_label)#Example: probs_y_train=[ Trace1:[class1: 0.2, class2: 0.5, class3:0.3] , Trace2[class1: 0.9, class2: 0.1, class3:0.0], ...]
        if len(set(y_train))==2:#if it is a binary classification problem, we have just one predicted probability which belongs to class 1, we calculate the other one now:
            predicts_probs_cases_bin_class=[[1-prob, prob] for prob in predicts_probs_cases_bin_class]

        #Find the champion of the class (i.e., the instance with a highest predicted probability to belong to its class).
        champion_class_label=findChampionsInSpecificClassProbs(X_train_class_label,
                                                               predicts_probs_cases_bin_class,
                                                               encoded_class_label,
                                                               case_sizes)

        #save the champions in csv format
        championClass=series_mapping[series_mapping==encoded_class_label].index[0]
        champion_class_label.to_csv(dirRes+"/best_predicted_instance_class"+str(championClass)+".csv")

        #Generate the shap values plot for its prediction:
        #for that we have to find the numerical index of the champion which is based on its case id
        case_id_champion=list(champion_class_label.index)[0]
        numerical_index_champion=X_train.index.get_indexer([case_id_champion])[0]

        #We also need the expected value of that class to perform a representation, these values are contained in a list where each position corresponds to 
        if len(set(y_train))>2:#if there are more than two classes, we are at multiclassification problem, in that case there are different expected values
            expected_value_ch = explainer.expected_value[encoded_class_label]
            classif_type="multi"
        else:#otherwise there is only one
            expected_value_ch=explainer.expected_value
            classif_type="binary"

        #we get its shap values:
        shap_values_champion=getSHAP_Values_for_instance(shap_values_training,
                                                         encoded_class_label,
                                                         numerical_index_champion,
                                                         classification_type=classif_type)
        
        
        #feature values champion:
        feature_values_ch=X_train.loc[case_id_champion]

        #we generate the plot:
        plotFeatureImportanceForSpecificInstance(shap_values_champion,
                                                 expected_value_ch,
                                                 feature_values_ch,
                                                 "prototype_"+championClass,
                                                 dirRes)

        predicts_cases_bin_class=[np.argmax(probs) for probs in predicts_probs_cases_bin_class]#the class (represented by the index) with the highest probability is the one predicted
        correct_predictions=[1 if prediction==encoded_class_label else 0 for prediction in predicts_cases_bin_class]#if it matches its real class, we will filter them
        x_train_filtered = X_train_class_label[[bool(x) for x in correct_predictions]]

        #we obtain the predicted probabilty of each class for each filtered instance:
        d_train_filtered=xgb.DMatrix(x_train_filtered)
        probs_y_train=clf_specific_size_val.predict(d_train_filtered)

        if len(set(y_train))==2:#if it is a binary classification problem, we have just one predicted probability which belongs to class 1, we calculate the other one now:
            probs_y_train=[[1-prob, prob] for prob in probs_y_train]

        cases=x_train_filtered.index.to_list()#we obtain their case ids
        #Transform the probabilities of the filtered instances into a dataframe 
        df_probs_class_label = pd.DataFrame(probs_y_train, columns=classes, index=cases)

        #format of the dataframe:
        #                                           probs to belong to class X | probs to belong to class Y | probs to belong to class Z     |  .....                                 
        #instance 1 correctly classfied in class X              0.7                         0.20                        0.10                     ...
        #instance N correctly classfied in class X              0.6                         0.35                        0.05                     ...
        #                    ...                                ...                         ...                         ...                      ...

        #and save it in a dictionary along with the instances:
        dict_probs_cases_per_label[encoded_class_label]=(x_train_filtered, df_probs_class_label)

    #Once we have the cases correctly classified for each class, and the champions, we find the boundary cases for each possible combination:
    #For instance consider that we have four classes [A,B,C,D].We will obtain all possible combinations with the following for loops:
    dirAllBoundaryCases=dirRes+"/boundaryCases"
    os.mkdir(dirAllBoundaryCases)

    for i in range(0,len(classes)-1):#e.g. [A,B,C]
        df_probs_classI = dict_probs_cases_per_label[i][1]
        for j in range(i+1,len(classes)):#e.g., [B,C,D]
            df_probs_classJ = dict_probs_cases_per_label[j][1]
            classI=series_mapping[series_mapping==i].index[0]
            classJ=series_mapping[series_mapping==j].index[0]
            print("Finding boundary cases for classes "+str(classI)+" and "+str(classJ)+"....")

            #for each possible combination find the boundary cases based on probabilities:
            boundaryCaseClassX, boundaryCaseClassJ=boundaryCasesBetweenTwoClasses(df_probs_class_x=df_probs_classI,
                                                                                  df_probs_class_y=df_probs_classJ,
                                                                                  class_labelX=i,
                                                                                  class_labelY=j,
                                                                                  X_test=X_train)
            #Filter only the columns of the cases that the most important
            boundaryCaseClassX=boundaryCaseClassX[most_important_features]
            boundaryCaseClassJ=boundaryCaseClassJ[most_important_features]

            #concatenate them in one dataframe
            boundaryCases=boundaryCaseClassX.append(boundaryCaseClassJ)

            #save the dataframe in a csv
            boundaryCases.to_csv(dirRes+"/boundaryCases_Classes"+str(classI)+"-"+str(classJ)+".csv")
            
            #we will obtain the features related to each boundary case:
            identifiers=boundaryCases.index.to_list()#firstly we obtain the identifiers
            
            #using the string index we obtain the feature values of the boundary case of class X
            feature_values_instance_x=X_train.loc[identifiers[0]]

            #we will also obtain its shapley values, for that we need the numerical index inside its class (e.g., it is instance 5 of 1000 instances in class X). This is due to, in our case
            #the structure storing the shapley values has N lists corresponding to the n classes of the classifier. Inside each class list, there are more lists, each one correspond to the 
            #contributions of the features for each instance of each class which are used in a multiclassification problem to estimate the contributions (all instances are used independently of
            #their classes)
            #For instance consider the following dataframe:
            #                   Feature AB | Feature AC | .....
            #instance 1 class X    ...          ...
            #instance 1 class Y    ...          ...
            #instance 2 class X    ...          ...
            #instance 1 class Z    ...          ...
            #instance 3 class X    ...          ...
            #       ...            ...          ...
            # Then the shapley values will be stored like this (if there are three classes)
            # shap values=[ [contributions for class X], [contributions for class Y], [contributions for class Z ]]
            # shap values=[ [ [contributions instance 1],[contributions isntance 2],[contributions isntance 3],... ],  ....]    

            #we obtain the numerical identifier related to boundary case of class X (i.e., its row number)
            numerical_index_boundary_case_x=X_train.index.get_indexer([identifiers[0]])[0]

            #We also the expected value of that class to perform a representation, these values are contained in a list where each position corresponds to the 
            if len(set(y_train))>2:#if there are more than two classes, we are at multiclassification problem, in that case there are different expected values
                expected_value_x = explainer.expected_value[i]
                classif_type="multi"
            else:#otherwise there is only one
                expected_value_x=explainer.expected_value
                classif_type="binary"
            
            #with all shap_values, the class in numerical format, and the numerical index inside its class, we can obtain its shapley values.
            #the way to obtain the shap values varies depending on whether we are in a binary o multi classification problem, which specific by classif type    
            shap_values_boundary_case_x=getSHAP_Values_for_instance(shap_values_training,
                                                                    i,
                                                                    numerical_index_boundary_case_x,
                                                                    classification_type=classif_type)
            
            #we do the same for boundary case of class Y
            numerical_index_boundary_case_y=X_train.index.get_indexer([identifiers[1]])[0]
            if len(set(y_train))>2:
                expected_value_y = explainer.expected_value[j]
            else:
                expected_value_y=explainer.expected_value
    
            feature_values_instance_y=X_train.loc[identifiers[1]]
            shap_values_boundary_case_y=getSHAP_Values_for_instance(shap_values_training,
                                                                    j,
                                                                    numerical_index_boundary_case_y,
                                                                    classification_type=classif_type)

            #we create a directory to store plots related to the contributions
            specific_directory=dirAllBoundaryCases+"/boundaryCases"+str(i)+"-"+str(j)
            os.mkdir(specific_directory)

            plotFeatureImportanceForSpecificInstance(shap_values_boundary_case_x, expected_value_x, feature_values_instance_x, identifiers[0], specific_directory)
            plotFeatureImportanceForSpecificInstance(shap_values_boundary_case_y, expected_value_y, feature_values_instance_y, identifiers[1], specific_directory)


print("Reading dataset file...")

parser = argparse.ArgumentParser(description="Pipeline launcher")
parser.add_argument("data", help="Dataset to be used")
args = parser.parse_args()

data=args.data

if data=="rtfm":
    log_route="./Data/road_traffic/Road_Traffic_Fine_Management_Process.xes"
    dataset_route="./Data/road_traffic/mined_rtfm_relabelled_confidences.csv"
    dataset=pd.read_csv(dataset_route, index_col=0)
    resultsFolder="./results/Ours/rtfm/"
else:
    log_route="./Data/sepsis/sepsis.xes"
    dataset_route="./Data/sepsis/mined_sepsis_confidences_SIRS2OrMore.csv"
    #we change how missing values are identified, because one case id is NA but it does not mean missing value.
    missing_values = ['nan', 'null', 'None', '']#we specify some of the general interpretations for nan values, but no NA as a string
    #then we read the dataset specifying that we will not use the default definition of NA of pandas, instead we specificy it with the previous values
    dataset=pd.read_csv(dataset_route, index_col=0, keep_default_na=False, na_values=missing_values)
    resultsFolder="./results/Ours/sepsis/"

log=pm4py.read_xes(log_route)

dataset = dataset.set_index("case:concept:name")

case_sizes=log.groupby(by=["case:concept:name"]).apply(lambda x: len(x)).to_dict()

series_case_sizes=pd.Series(case_sizes)

X=dataset.drop(columns=["Class"])

y=dataset['Class']

print("No. of features:"+str(len(X.columns)))

le = LabelEncoder()
y_transformed = le.fit_transform(y)
le_name_mapping = dict(zip(le.classes_,le.transform(le.classes_)))
cols=X.columns.to_list()
X_train_and_validation, X_test, y_train_and_validation, y_test = train_test_split(X,
                                                                                  y_transformed,
                                                                                  test_size=0.2,#size of the test dataset
                                                                                  stratify=y_transformed,#divide the data maintaining the class proportions in each dataset 
                                                                                  shuffle=True,#disorder the data
                                                                                  random_state=0)#set a seed so that it is possible to replicate it and it does not change between executions

now=datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
dirResults=resultsFolder+now
os.mkdir(dirResults)
size_val=0.2
stratf=True

if len(set(y_transformed))>2:
    param_dist={
        "objective":"multi:softprob",
        "num_class":len(set(y_transformed)),#it is necessary to indicate the number of classes in the data: set-> only obtain an instance of each class, len-> provide the number of classes
        "seed":0
        }
else:
    param_dist={
        "objective":"binary:logistic",
        "seed":0
    }

classifier_route=""
training=True

main(X_train_and_validation, y_train_and_validation, X_test, y_test, series_case_sizes, param_dist, size_val, stratf, le_name_mapping, dirResults, training, classifier_route)

