import pandas as pd
import pm4py
from dataPreparation import main_prepreprocessing


def labelling_SIRScriteria(log):
    """
    Function that labels the cases of the log in the two classes depending on whether two or more SIRS criteria are achieved (True if occurred). This is stored in the SIRSCriteria2OrMore, so the labelling can be resumed to:
        SIRSCriteria2OrMore==True
        SIRSCriteria2OrMore==False
    """
    #firstly we group the events to their corresponding cases. 
    #Then we drop the event that do not have a value for SIRSCriteria2OrMore attribute, 
    #finally we select the first value (which is the only one) and reset index to transform it into a dataframe
    df_SIRScriteria=log.groupby("case:concept:name").apply(lambda x: list(x["SIRSCriteria2OrMore"].dropna())[0]).reset_index()
    df_SIRScriteria=df_SIRScriteria.rename(columns={0:"Classes", "case:concept:name":"case_id"})
    replacement_map = {True:"SIRS-True", False:"SIRS-False"}
    df_SIRScriteria['Classes'] = df_SIRScriteria["Classes"].replace(replacement_map)#we replace the class values so that they are more intuitive
    return df_SIRScriteria



def selectLabelling(log, label_type):
    "Function to label the cases of the sepsis log depending on the desired labelling type. The output is a dataframe "

    if label_type=="SIRS2OrMore":
        df_cases_labelled=labelling_SIRScriteria(log)
    else:
        print("Not implemented!")

    return df_cases_labelled


route_output_slider_sepsis="./Data/sepsis/output_minerful_slider_sepsis.csv"
sepsis_log=pm4py.read_xes("./Data/sepsis/sepsis.xes")

label_class="SIRS2OrMore"
trace_ids_sepsis=sepsis_log["case:concept:name"].unique()

print("Mapping classes to cases...")
df_cases_labelled_sepsis=selectLabelling(log=sepsis_log, label_type=label_class)

cleaned_concatenated_rules=main_prepreprocessing(route_output_slider=route_output_slider_sepsis,
                                                 trace_ids=trace_ids_sepsis,
                                                 n_ranges=3,
                                                 log_data_with_classes=df_cases_labelled_sepsis,
                                                 route_raw_mined_declared_rules="./Data/sepsis/mined_sepsis_fixed_raw.csv")

cleaned_concatenated_rules.to_csv("./Data/sepsis/mined_sepsis_confidences_"+label_class+".csv") 