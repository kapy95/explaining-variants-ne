import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from numpy.random import default_rng

import pm4py
import os
import datetime
import argparse
import warnings
warnings.filterwarnings("ignore")


def calculateAverageDistance(prot_values, other_data):
    distances=cdist([prot_values], other_data, 'euclidean')
    avg_dist=np.mean(distances)
    return avg_dist



def reconstructData(data_mined_route):
    """
    This function reconstructs the training data used. Specifically, it returns the instances used in the training (X_train) and their corresponding classes (y_train) separated. Each class is coded in a numeric format
    and le_name_mapping is a pandas series whose indices are the real names of the classes, and its values are the class numeric formats.
    """
    if "sepsis" in data_mined_route:
        missing_values = ['nan', 'null', 'None', '']#we specify some of the general interpretations for nan values, but no NA as a string
        #then we read the dataset specifying that we will not use the default definition of NA of pandas, instead we specificy it with the previous values
        dataset=pd.read_csv(data_mined_route, index_col=0, keep_default_na=False, na_values=missing_values)
    else:
        dataset=pd.read_csv(data_mined_route, index_col=0)
    
    dataset = dataset.set_index('case:concept:name')
    X=dataset.drop(columns=["Class"])
    X=X.fillna(-100)#replace nan values for -100 so that we can calculate distances later

    y=dataset['Class']
    print("No. of features:"+str(len(X.columns)))

    le = LabelEncoder()
    y_transformed = le.fit_transform(y)
    le_name_mapping = pd.Series(dict(zip(le.classes_,le.transform(le.classes_))))
    X_train_and_validation, X_test, y_train_and_validation, y_test = train_test_split(X,
                                                                                  y_transformed,
                                                                                  test_size=0.2,
                                                                                  stratify=y_transformed,
                                                                                  shuffle=True,
                                                                                  random_state=0)
    
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_and_validation,
                                                               y_train_and_validation,
                                                               test_size=0.2,
                                                               stratify=y_train_and_validation,
                                                               shuffle=True,
                                                               random_state=0)
    return X_train, y_train, X_test, y_test, le_name_mapping




def returnMedoidClassUsingLength(data_train, classes, log, dataClass):
    position_class_instances=np.where(classes == dataClass)#we find which instances of data belong to that class
    data_train_specific_class=data_train.iloc[position_class_instances]#we filter them
    cases_data_class=log[log["case:concept:name"].isin(list(data_train_specific_class.index))]#we filter the events related to each case of the class
    length_per_case_class=cases_data_class.groupby("case:concept:name").apply(lambda x: len(x["concept:name"]) )#we calculate how many events exist in each case
    avg_length_cases_class=np.mean(length_per_case_class)#we calculate the mean
    distances_avg_mean_size_class=abs(length_per_case_class-avg_length_cases_class)#we calculate the absolute difference between the size of each case and the average
    closest_instance_to_avg=distances_avg_mean_size_class[distances_avg_mean_size_class==min(distances_avg_mean_size_class)]#the one with minimum difference is the closest

    if len(closest_instance_to_avg)>1:#if there is more than one, choose one randomly
        rng = default_rng(seed=0)
        random_number=rng.integers(0, len(closest_instance_to_avg), size=1)[0]
        selected_case=list(closest_instance_to_avg.index)[random_number]
    else:
        selected_case=list(closest_instance_to_avg.index)[0]

    case_length_prototype=log[log["case:concept:name"]==selected_case]
    
    return case_length_prototype




def evaluationPrototypes(dataset_route, log_route, results_baseline_route, results_our_approach, results_evaluation):

    specific_log=pm4py.read_xes(log_route)

    X_train, y_train, X_test, y_test, le_name_mapping=reconstructData(data_mined_route=dataset_route)


    separationsBaseline=[]
    separationsOurProts=[]

    cohesionsBaseline=[]
    cohesionOurProts=[]

    separationsTestBaseline=[]
    separationsTestOurProts=[]
    
    cohesionsTestBaseline=[]
    cohesionsTestOurProts=[]

    classes=[]

    now=datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    dirResultsBaseline=results_baseline_route+"/"+now
    os.mkdir(dirResultsBaseline)

    for className, classValue in le_name_mapping.iteritems():
        print("Evaluating prototype for class "+className)
        #firstly we find the prototype of each class using the baseline and we save it in the provided route: 
        log_medoid_class=returnMedoidClassUsingLength(data_train=X_train,
                                                      classes=y_train,
                                                      log=specific_log,
                                                      dataClass=classValue)
        
        log_medoid_class=log_medoid_class.reset_index(drop=True)
        log_medoid_class.to_csv(dirResultsBaseline+"/medoid_class_"+str(className)+".csv")

        classes.append(className)

        case_id_medoid_class=log_medoid_class["case:concept:name"][0]#obtain its case_id
        case_id_best_class_our_prot=pd.read_csv(results_our_approach+"/best_predicted_instance_class"+str(className)+".csv")["case:concept:name"][0]#we read our prototype and select its case id

        #now we are going to compare the prototype returned by the baseline and our approach measuring their distances with instances of its class (intraclass), and instances of other classes (interclass).
        #A good prototype should have a low distance with the instances of its class, and a high separation with instances of other classes.

        #To do that we divide the data in instances of the same class of the prototype and instances whose class is different
        data_class_X=X_train[y_train==classValue]
        data_other_classes=X_train[y_train!=classValue]

        #We firstly do it for the baseline prototype:
        prot_values_baseline=X_train.loc[case_id_medoid_class].values#we obtain the values of the baseline prototype
        class_data_without_medoid_prot=data_class_X.drop(case_id_medoid_class).values#we obtain the data of its class without the prototype selected by baseline
        cohesionMedoid=calculateAverageDistance(prot_values_baseline,class_data_without_medoid_prot)#we calculate the average euclidean distance of the baseline prototype with the instances of its class
        sepMedoid=calculateAverageDistance(prot_values_baseline, data_other_classes)#we calculate the average euclidean distance of the baseline prototype with the instances of other classes
        #we add the values to the lists:
        cohesionsBaseline.append(cohesionMedoid)
        separationsBaseline.append(sepMedoid)

        #We repeat the same steps of the baseline but for our prototype:
        prot_values_our_prot=X_train.loc[case_id_best_class_our_prot].values
        class_data_without_our_prot=data_class_X.drop(case_id_best_class_our_prot).values
        cohesionOurs=calculateAverageDistance(prot_values_our_prot, class_data_without_our_prot)
        sepOurs=calculateAverageDistance(prot_values_our_prot, data_other_classes)
        cohesionOurProts.append(cohesionOurs)
        separationsOurProts.append(sepOurs)

        

        ######################################################################
        #we repeat the process with the test data:
        data_class_X_test=X_test[y_test==classValue]
        data_other_classes_test=X_test[y_test!=classValue]

        #but we do not have to remove the prototype, it does not exist there
        cohesionMedoidTest=calculateAverageDistance(prot_values_baseline,data_class_X_test)#we calculate the average euclidean distance of the baseline prototype with the instances of its class
        sepMedoidTest=calculateAverageDistance(prot_values_baseline, data_other_classes_test)#we calculate the average euclidean distance of the baseline prototype with the instances of other classes
        #we add the values to the lists:
        cohesionsTestBaseline.append(cohesionMedoidTest)
        separationsTestBaseline.append(sepMedoidTest)

        #We repeat the same steps of the baseline prototype but for our prototype:
        cohesionOursTest=calculateAverageDistance(prot_values_our_prot, data_class_X_test)
        sepOursTest=calculateAverageDistance(prot_values_our_prot, data_other_classes_test)
        cohesionsTestOurProts.append(cohesionOursTest)
        separationsTestOurProts.append(sepOursTest)



    #create a directory to save the prototype evaluation results:
    dirResults=results_evaluation+now
    os.mkdir(dirResults)

    #create a dataframe with the interclass distances of baseline and our approach prototypes:
    df_separations = pd.DataFrame({"Class":classes,
                                   "interclass_baseline_train":separationsBaseline,
                                   "interclass_our_prot_train":separationsOurProts,
                                   "interclass_baseline_test": separationsTestBaseline,
                                   "interclass_our_prot_test":separationsTestOurProts})
    
    df_cohesions = pd.DataFrame({"Class":classes,
                                 "intraclass_baseline_train":cohesionsBaseline,
                                 "intraclass_our_prot_train":cohesionOurProts,
                                 "intraclass_baseline_test":cohesionsTestBaseline,
                                 "intraclass_our_prot_test":cohesionsTestOurProts})
    
    
    df_separations.to_excel(dirResults+"/evaluation_interclass.xlsx")
    df_cohesions.to_excel(dirResults+"/evaluation_intraclass.xlsx")


parser = argparse.ArgumentParser(description="Prototype evaluation launcher")
parser.add_argument("data", help="Dataset to be used")
parser.add_argument("folder", help="folder name with the results")
args = parser.parse_args()

dataset=args.data
folder=args.folder

if dataset=="sepsis":
    mined_dataset_route="./Data/sepsis/mined_sepsis_confidences_SIRS2OrMore.csv"
    specific_log_route="./Data/sepsis/sepsis.xes"
else:
    mined_dataset_route="./Data/road_traffic/mined_rtfm_relabelled_confidences.csv"
    specific_log_route="./Data/road_traffic/Road_Traffic_Fine_Management_Process.xes"

evaluationPrototypes(dataset_route=mined_dataset_route,
                     log_route=specific_log_route,
                     results_baseline_route="./results/length_prot/"+dataset,
                     results_our_approach="./results/Ours/"+dataset+"/"+folder,
                     results_evaluation="./results/evaluationPrototypes/"+dataset+"/")



