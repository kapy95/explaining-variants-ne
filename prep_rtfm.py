import pandas as pd
import pm4py
from dataPreparation import main_prepreprocessing

route_output_slider_rtfm="./Data/minerfulSliderOutput_rtfm.csv"
log=pm4py.read_xes("./Data/road_traffic/Road_Traffic_Fine_Management_Process.xes")

output_fresh_approach=pd.read_csv("./Data/road_traffic/A_Fresh_Approach_to_Analyze_Process_Outcomes/out_df.csv")
trace_ids_rtfm=log["case:concept:name"].unique()

print("Mapping classes to cases...")
classes=[]
for index, row in output_fresh_approach.iterrows():
    if row["paid_full"]==True:
        classes.append("paid_full")

    elif row["dismissed"]==True:
        classes.append("dismissed")
    
    elif row["credit_collection"]==True:
        classes.append("credit_collection")
    
    else:
        classes.append("unresolved")
    
output_fresh_approach["Classes"]=classes
rows_paid_and_collected=output_fresh_approach[(output_fresh_approach["credit_collection"]==True) & (output_fresh_approach["paid_full"]==True)]
rows_paid_and_dismissed=output_fresh_approach[(output_fresh_approach["dismissed"]==True) & (output_fresh_approach["paid_full"]==True)]
rows_to_delete=pd.concat([rows_paid_and_dismissed,rows_paid_and_collected])
output_fresh_approach_cleaned=output_fresh_approach[~output_fresh_approach["case_id"].isin(rows_to_delete["case_id"])]
rtfm_case_ids_to_delete=rows_to_delete['case_id']

print(output_fresh_approach_cleaned["Classes"].unique())
cleaned_concatenated_rules=main_prepreprocessing(route_output_slider=route_output_slider_rtfm,
                                                 trace_ids=trace_ids_rtfm,
                                                 n_ranges=15,
                                                 log_data_with_classes=output_fresh_approach_cleaned,
                                                 route_raw_mined_declared_rules="./Data/road_traffic/mined_rtfm_not_clean.csv",
                                                 case_ids_to_delete=rtfm_case_ids_to_delete)
y=cleaned_concatenated_rules["Class"]
y=y.replace(to_replace = ['credit_collection', 'paid_full'], value = ['collected', 'fully_paid'])#replace some class names to adjust them to the ones in the paper
cleaned_concatenated_rules["Class"]=y
cleaned_concatenated_rules.to_csv("./Data/road_traffic/mined_rtfm_relabelled_confidences_test.csv")

