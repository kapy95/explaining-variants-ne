# explaining-variants-ne

All the following steps have been verified in a machine with Windows 11. Take into account that some parts may slightly change in other operating systems.

Preparations (this is a video to help with the preparations: https://www.youtube.com/watch?v=oG-0JqnjRsc):
1. Please install anaconda: https://www.anaconda.com/download
2. Download:
   1 The sepsis log from here in xes format and leave it in /data/sepsis directory: https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639
   2.The road fines log from here in xes format and leave it in /data/rtfm: https://data.4tu.nl/articles/dataset/Road_Traffic_Fine_Management_Process/12683249
   3. The datasets with the mined confidences from here: https://zenodo.org/records/17753577
      1. Leave mined_rtfm_relabelled_confidences.csv in /data/rtfm
      2. Leave mined_sepsis_confidences_SIRS2OrMore.csv in /data/sepsis
4. Leave the folders of the one drive link in the data folder
5. Open an anaconda prompt and execute the following command:
    1. Change the current directory to the project folder (cd directory)
    2. Create a python environment with the following command: conda create --name naturalexamples python=3.8.0 (it may take a while)
    3. Activate the environment: conda activate naturalexamples
    4. Install the dependencies of the project by executing the following command: pip install -r requirements.txt 
6. Install any software related to use python jupyter notebooks (such as visual studio code) or use anaconda.
7. Optional (required if you want to mine the rules for all cases): install JRE+7 and download the MINERful repository from here: https://github.com/Process-in-Chains/MINERful

How to run the approach to obtain the prototypes and boundary cases for a given log using our approach:
0. Make sure that the folders of the one drive link are inside the data folder
1. Open an anaconda console and activate the preparation environment (conda activate naturalexamples)
2. Change the current directory to the repository directory
3. Execute one of the following commands depending on the log:
    1. python main.py sepsis
    2. python main.py rtfm

After running the script, a folder with the date related to execution will be created inside the /results/Ours/sepsis or /results/Ours/rtfm directories.When running the main.py file with the rtfm log beware that its execution will require around 10GB of free RAM for its use, or it could run out of RAM otherwsie. This is a video to help with this: https://www.youtube.com/watch?v=y5rGBTeFXKI

Evaluation of the prototypes:
1. Open an anaconda console, change the current directory to the repository directory, and activate the preparation environment (conda activate naturalexamples)
2. Execute the evaluation.py file passing the dataset to be used (sepsis or rtfm), and the folder with the results of our approach (they are contained in the following route /results/ours/sepsis/date or /results/ours/rtfm/date) For example: python evaluation.py rtfm 27-11-2025_17-50-30
3. The prototypes of the baseline will be stored correspondingly in results/length_prot/sepsis or /results/length_prot/rtfm. Also a folder named with the execution date will be created in /results/evaluationPrototypes/rtfm or  /results/evaluationPrototypes/sepsis depending on the log. This folder will contain excel files with intra and inter class distances.
This is a video to help running the prototype evaluation: https://www.youtube.com/watch?v=FNF7gPMIgQ0&t=199s

Evaluation of the boundary cases (this is a video to help with it: https://www.youtube.com/watch?v=ckiotGmEfQg:
1. Open the validationBoundaryCases-rtfm.ipynb or validationBoundaryCases-sepsis.ipynb files (they are jupyter notebooks).
2. Run the cells
3. Open the explorationBoundaryCases_rtfm.ipynb or explorationBoundaryCases_sepsis.ipynb and change the route variable which is in the first cells of tne notebooks to the folder name with the results. For example if I want to run explorationBoundaryCases_rtfm.ipynb, I have to introduce the name of a folder contained in /results/Ours/rtfm. 


Mining of all rules for all cases in a log (optional):
1. Open the minerful_slider.bat file inside MINERful_launcher folder
2. Replace the following absolute paths:
    1. INPUT_LOG: absolute path to the log that you want to mine the rules
    2. RESULTS: absolute path with the file name to save the MINERful results 
    3. JAVA_BIN: absolute path to the java folder 
    4. MINERFUL_JAR: absolute path to the MINERful.jar that is located in the MINERful repository that you downloaded in the preparation step
    5. LIB_FOLDER: absolute path to lib folder that is located in the MINERful repository that you downloaded in the preparation step
3. Execute minerful_slider.bat
Take into account that the mining of the road fines log is quite extensive, since it has to check a lot of rules for many cases

How to generate the datasets from the output of MINERful (optional):
1. Open the prep_rtfm.py or prep_sepsis_file.oy
2. Change the route_output_slider_sepsis or the route_output_slider_rtfm to the directory where you have the ouyput csv of MINERful (this file should be in the same directory of the repository)
3. Execute the prep_rtfm.py or prep_sepsis_file.oy

The following video can be helpful for the mining and the dataset generation: https://youtu.be/yhSZ8_vp0fo. Nevertheless, it is not necessary to perform these steps with the data contained in one drive folder






