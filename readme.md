## Machine Learning in Software Engineering
The "ZhenjieFinalDemo" branch includes the final report "Computational Study of Data Preprocessing Methods and Machine Learning Classification Algorithms for Software Bug Prediction" in "Jia_Zhenjie_F19.pdf" and the code that can generate some results and figures in the report in the final_code folder. 
Becacuse this report is a joint work with Xinyi He. The code in the final_code folder can only generate the result and figure that showed in the section 4.2, 5.4, 5.5 of the report, including Figure 2, 12-19, Table 4-6.

## python environment required:
1. python 3.5 or above
2. numpy
3. pandas
4. scipy
5. matplotlib
6. sklearn
7. imblearn
8. openpyxl

## How to generate the result and figure 
In the directory "MLSE/final_code/Zimmermann’s data set/" in the "ZhenjieFinalDemo" branch, The file "Classification of files.ipynb" shows how the get the Accuracy, Recall, Precision and F1 score with different algorithms and data preprocessing methods, including LogisticRegression, RandomForest with SMOTEENN and RandomForest with SMOTE using Eclipse 2.0, Eclipse 2.1, Eclipse 3.0 as train set and test set respectively. After that, I write the results to the “sheet2” of file "result_from_paper.xlsx" and comparing my results with the reuslts in Zimmermann's paper.
