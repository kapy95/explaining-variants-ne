REM #Replace for absolute path to xes file with the log, take into account that it should not contain any NaN value, otherwise MINERful can fail
set INPUT_LOG=C:\Users\ccagu\Documents\Trabajo\EstanciaDeUtrecht\CAiSE2026\explaining-variants-ne\Data\sepsis\sepsis.xes
REM # Replace for absolute path to csv file where the output of MINERful should be stored, take into account that will be heavy
set RESULTS=C:\Users\ccagu\Documents\Trabajo\EstanciaDeUtrecht\CAiSE2026\explaining-variants-ne\Data\sepsis\output_minerful_sepsis.csv

REM # PATHs to MINERful and Java
REM #Replace for absolute path to your java version
set JAVA_BIN=C:\Program Files\Java\jdk-21\bin\java
REM #Replace for the absolute path of MINERful jar
set MINERFUL_JAR=C:\Users\ccagu\Documents\Trabajo\EstanciaDeUtrecht\MINERful\MINERful\MINERful.jar
REM #Replace for the absolute path of MINERful lib 
set LIB_FOLDER=C:\Users\ccagu\Documents\Trabajo\EstanciaDeUtrecht\MINERful\MINERful\lib

REM ##################################################################

REM # MINERful main classes
set MINERFUL_DISCOVERY_MAINCLASS=minerful.MinerFulMinerSlider
set MINERFUL_DISCOVERY_SUPPORT=0.0
set MINERFUL_DISCOVERY_CONFIDENCE=0.0
set MINERFUL_DISCOVERY_COVERAGE=0.0
set INPUT_FORMAT=xes

REM #execution of MINERful, which has been tested in windows. For more information please check MINERful wiki:
"%JAVA_BIN%" -cp "%LIB_FOLDER%\*;%MINERFUL_JAR%" %MINERFUL_DISCOVERY_MAINCLASS% -iLSubLen 1 --slide-by 1 -iLE %INPUT_FORMAT% -iLF %INPUT_LOG% -c %MINERFUL_DISCOVERY_CONFIDENCE% -s %MINERFUL_DISCOVERY_SUPPORT% -g %MINERFUL_DISCOVERY_COVERAGE% -sT 0.0 -cT 0.0 -gT 0.0 --sliding-results-out %RESULTS% --stick-tail "false" -prune "none" -vShush "true"