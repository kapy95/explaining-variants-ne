import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def representMatrixMeasure(dataframe, class_column, col_num, performance_metric, dirMatrix):
    sns.set_theme(font_scale=1.35) 
    g = sns.FacetGrid(dataframe, col=class_column,height=4, aspect=1.0, palette="muted", col_wrap=col_num)
    g.map(sns.barplot, "Classifier", performance_metric)
    plt.savefig(dirMatrix)
    plt.close()


def generateDataMap(y_real_test_map, y_pred_map, le_name_mapping, labels, dirMap):
    matrix_base=pd.DataFrame.from_dict({"real_labels":y_real_test_map,"predicted_labels":y_pred_map})
    mapping_series=pd.Series(le_name_mapping)
    data_map=list()
    
    for real_label in labels:
        predicted_labels_for_specific_label=matrix_base[matrix_base['real_labels']==real_label]
        vc_predicted_rl=((predicted_labels_for_specific_label['predicted_labels'].value_counts())/len(predicted_labels_for_specific_label))
        vc_predicted_rl=vc_predicted_rl*100

        for label2 in labels:
            predicted_vals=list(vc_predicted_rl.index)
            if label2 in predicted_vals:
                continue
            else:
                vc_predicted_rl[label2]=0
        vc_predicted_rl=vc_predicted_rl.sort_index()
        data_map.append(vc_predicted_rl.values)

    labels_map=[mapping_series[mapping_series==val].index[0] for val in labels]
    # Plot the heatmap
    plt.figure(figsize=(8, 5))
    sns.set(font_scale=1.4)#increases the sizes of all text (axes, y tick label name,...)
    sns.heatmap(data_map, xticklabels=labels_map, yticklabels=labels_map, annot=True, cmap="coolwarm", fmt=".2f", cbar=False, annot_kws={"size": 20})

    # Add labels and title
    plt.xlabel("Predicted")
    plt.ylabel("Real value")
    plt.savefig(dirMap, bbox_inches='tight')
    plt.close()




