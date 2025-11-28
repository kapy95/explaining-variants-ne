#Data management
import pandas as pd


def addClassesMinedData(df_rules, df_with_classes):
    classes_data_mined_rules=[]
    for index, row in df_rules.iterrows():
        case_id=row['case:concept:name']
        class_case_id_row=df_with_classes[df_with_classes['case_id']==case_id]
        class_case_id=class_case_id_row.reset_index()['Classes'][0]

        classes_data_mined_rules.append(class_case_id)

    df_rules['Class']=classes_data_mined_rules

    return df_rules



def transformMSResultsToDataframeUsingBins(lines, trace_ids, start_ind, end_ind):

    """
    Overview: this function converts the output of the MINERful slider to a dataframe with the following structure:
    Example:
             Declare rule | trace id | Support | Confidence | Coverage | Trace support | Trace confidence | Trace coverage
    row1      Absence(A)      V876       90          78          56           50                 60             100
    ...         ...           ...       ...         ...         ...         .....               ....            .....

    Input:
        - route_rule_names: route to a file where the names of the declare rules of the MINERful slider are stored. It should be like this:
        'From';'To';'Absence(A)';;;;;;'AtLeast1(A)';;;;;;'AtLeast2(A)';;;;;;'AtLeast3(A)';;;;;;'AtMost1(A)';;;;;;'AtMost2(A)';;;;...
        ";;'Support';'Confidence';'Coverage';'Trace support';'Trace confidence';'Trace coverage';'Support';'Confidence';'Coverage';'Trace support';'Trace confidence';'Trace coverage';'Support';'Confidence';'Coverage'...
        0;1;0.000000000;0.000000000;50.000000000;0.000000000;0.000000000;100.000000000;50.000000000;100.000000000;50.000000000;....
        .....
        - trace ids: identifiers of the traces as they appear in the log

    """

    #firstly, the names of the declare rules are obtained, because that file also contains the names of the measures repeated
    names_declare_rules=lines[0].replace("\n","")[12:].split(";;;;;;")#the first part are the declare rule names which are separated by commas (;;;;;;;), we split them from 12 position, 
    #because it also contains the 'from';'to'; characters which are irreleavant

    start=2
    rows_all_traces=[]
    #we split the file with the values by the \n separator, that returns us the values corresponding to a trace.
    #values_each_trace=lines[2:]#we omit the last result because it is empty
    #the result looks like this: [ [0;1;50;60;10;50;10;40...]] where the two first values represent from and to, 
    #and each six of the following values are the measure values of a rule

    values_each_trace_in_slice=lines[start_ind+2:end_ind+2]#we sum +2 to each slice index because the values of the first trace are at position 2, 0 and 1 are headers
    trace_ids_slice=trace_ids[start_ind:end_ind]

    for values_trace, trace_id in zip(values_each_trace_in_slice,trace_ids_slice):#for the values of each trace and its case id
        values=values_trace.split(";")#divide the values by ; separator. 
        #From and to values are still there. We will omit them by starting at position n+2 and finishing at n+6
        for j, declare_rule in zip(range(start,len(values),6),names_declare_rules):#start at the position of the first value of the first rule, and continue with the first value of the following rule
            values_rule=values[j:j+6]#we take the six values related to that rule
            float_values_rule=[float(i) for i in values_rule]#we transform them to floats 
            row=[declare_rule, trace_id]+float_values_rule#we combine them with the declare rule name and the trace id
            rows_all_traces.append(row)#we added it
        
        columns=["Declare rule",
                 "case:concept:name", 
                 "'Support'",
                 "'Confidence'",
                 "'Coverage'", 
                 "'Trace support'", 
                 "'Trace confidence'",
                 "'Trace coverage'"]
    
    data_rules=pd.DataFrame(data=rows_all_traces, columns=columns)
    return data_rules


def transformMSResultsToDataframe(route_output_slider, trace_ids):

    """
    Overview: this function converts the output of the MINERful slider to a dataframe with the following structure:
    Example:
             Declare rule | trace id | Support | Confidence | Coverage | Trace support | Trace confidence | Trace coverage
    row1      Absence(A)      V876       90          78          56           50                 60             100
    ...         ...           ...       ...         ...         ...         .....               ....            .....

    Input:
        - route_rule_names: route to a file where the names of the declare rules of the MINERful slider are stored. It should be like this:
        'From';'To';'Absence(A)';;;;;;'AtLeast1(A)';;;;;;'AtLeast2(A)';;;;;;'AtLeast3(A)';;;;;;'AtMost1(A)';;;;;;'AtMost2(A)';;;;...
        ";;'Support';'Confidence';'Coverage';'Trace support';'Trace confidence';'Trace coverage';'Support';'Confidence';'Coverage';'Trace support';'Trace confidence';'Trace coverage';'Support';'Confidence';'Coverage'...
        0;1;0.000000000;0.000000000;50.000000000;0.000000000;0.000000000;100.000000000;50.000000000;100.000000000;50.000000000;....
        .....
        - trace ids: identifiers of the traces as they appear in the log

    """
    with open(route_output_slider) as file:
        lines=file.readlines()#we read the file line by line

    #firstly, the names of the declare rules are obtained, because that file also contains the names of the measures repeated
    names_declare_rules=lines[0].replace("\n","")[12:].split(";;;;;;")#the first part are the declare rule names which are separated by commas (;;;;;;;), we split them from 12 position, 
    #because it also contains the 'from';'to'; characters which are irreleavant

    start=0
    rows_all_traces=[]
    #we split the file with the values by the \n separator, that returns us the values corresponding to a trace.
    #values_each_trace=lines[2:]#we omit the last result because it is empty
    #the result looks like this: [ [0;1;50;60;10;50;10;40...]] where the two first values represent from and to, 
    #and each six of the following values are the measure values of a rule

    values_each_trace_in_slice=lines[2:]#we start at 2 because the values of the first trace are at position 2, 0 and 1 are headers
    trace_ids_slice=trace_ids
    i=1
    total=len(values_each_trace_in_slice)
    for values_trace, trace_id in zip(values_each_trace_in_slice,trace_ids_slice):#for the values of each trace and its case id
        values=values_trace.split(";")[2:]#divide the values by ; separator. 

        #From and to values are still there. We will omit them by starting at position n+2 and finishing at n+6
        for j, declare_rule in zip(range(start,len(values),6),names_declare_rules):#start at the position of the first value of the first rule, and continue with the first value of the following rule
            values_rule=values[j:j+6]#we take the six values related to that rule
            float_values_rule=[float(i) for i in values_rule]#we transform them to floats 
            row=[declare_rule, trace_id]+float_values_rule#we combine them with the declare rule name and the trace id
            rows_all_traces.append(row)#we added it
        print("Line "+str(i)+"/"+str(total))
        i=i+1
    
    columns=["Declare rule",
             "case:concept:name", 
             "'Support'",
             "'Confidence'",
             "'Coverage'", 
             "'Trace support'", 
             "'Trace confidence'",
             "'Trace coverage'"]
    
    data_rules=pd.DataFrame(data=rows_all_traces, columns=columns)
    return data_rules

#function to generate indices stating the ranges of traces that will be used depending on the number of ranges desired and traces
#e.g., from 0 to 9999, from 10000 to 20000,....
def generate_index_ranges(num_lines, n_ranges):
    size_range = num_lines // n_ranges
    indices = []

    for i in range(n_ranges):
        start = i * size_range
        end = (i + 1) * size_range if i < n_ranges - 1 else num_lines
        indices.append((start, end))

    return indices


def main_prepreprocessing(route_output_slider, trace_ids, n_ranges, log_data_with_classes, route_raw_mined_declared_rules, case_ids_to_delete=None):
    """
    This function transforms the output of the minerful slider into a dataframe where each column is the confidence of a declare rule
    and each row represents a case. Moreover, at the end of the dataframe there is a column to indicate the class of each case. It receives as inputs:

    - route_output_slider: relative route to the file with MINERful slider output file
    - trace ids: case ids in the same order as they appear in the original event log used in MINERful
    - n_ranges: number of ranges to divide the cases (e.g., divide the cases in 3 groups)
    - log_data_with_classes: dataframe with a column named case_id to represent each case and a column called Classes with the corresponding class of each case
    - route_raw_mined_declared_rules: relative route to save the dataframe where each row is a case and each column is the confidence for a particular declare rule. This dataframe
    contains all the cases even the ones to be deleted if it is indicated.
    - case_ids_to_delete: if it is necessary to remove some cases, please provide a list with them here
    """
    num_lines= len(trace_ids)
    ranges = generate_index_ranges(num_lines, n_ranges)
    dataframes_confidences_to_concat=[]

    #we just consider confidence, so the following columns will be deleted 
    columnsToBeDeleted=["'Support'","'Coverage'", "'Trace support'", "'Trace confidence'"]

    with open(route_output_slider) as file:
        lines=file.readlines()#we read the file line by line

    for bin in ranges:
        print("Reading lines from "+str(bin[0])+" to "+ str(bin[1])+".....")
        print("Generating input data...")
        df=transformMSResultsToDataframeUsingBins(lines, trace_ids, bin[0], bin[1])
        df=df.drop(columns=columnsToBeDeleted)

        print("Fixing confidences")
        index_nan=df[(df["'Trace coverage'"]==0) & (df["'Confidence'"]==0)].index
        confidences_fixed=df["'Confidence'"].copy()
        confidences_fixed.iloc[list(index_nan)]=float("NaN")
        df["'Confidence'"]=confidences_fixed
    
        print("Building confidences dataframe...")
        pivot_df = df.pivot(index='case:concept:name', columns='Declare rule', values="'Confidence'")

        #Remove declare rule as index:
        confidences_dataframe = pivot_df.reset_index()
        confidences_dataframe.columns.name = None
        print("---------------------------------------------------------------------")
        dataframes_confidences_to_concat.append(confidences_dataframe)

    all_confidences_df=pd.concat(dataframes_confidences_to_concat)
    all_confidences_df.to_csv(route_raw_mined_declared_rules)

    #This part can run out of memory, in that case, comment the loading of MINERful slider output and the for loop of above, and just load the csv related to all confidences
    #all_confidences_df=pd.read_csv("./Data/road_traffic/mined_rtfm_relabelled_confidences_not_clean.csv", index_col=0)
    
    if type(case_ids_to_delete)!=type(None):
        cleaned_concatenated_rules=all_confidences_df[~all_confidences_df['case:concept:name'].isin(case_ids_to_delete)]
    else:
        cleaned_concatenated_rules=all_confidences_df
    

    
    print("Adding classes...")
    cleaned_concatenated_rules_with_classes=addClassesMinedData(df_rules=cleaned_concatenated_rules, df_with_classes=log_data_with_classes)
    
    return cleaned_concatenated_rules_with_classes


   