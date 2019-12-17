## Machine Learning in Software Engineering
The "ZhenjieFinalDemo" branch includes the final report "Computational Study of Data Preprocessing Methods and Machine Learning Classification Algorithms for Software Bug Prediction" in "Jia_Zhenjie_F19.pdf" and the code that can generate some results and figures in the report in the "final_code" folder. 

Because this report is a joint work with Xinyi He. The code in the final_code folder can only generate the result and figure that showed in the section 4.2, 5.4, 5.5 of the report, including Figure 2, 12-19, Table 4-6.

## Python environment required
1. python 3.5 or above
2. numpy
3. pandas
4. scipy
5. matplotlib
6. sklearn
7. imblearn
8. openpyxl

## Generating the result and figure 
In the directory "MLSE/final_code/Zimmermann’s data set/" in the "ZhenjieFinalDemo" branch, the file "Classification of files.ipynb" shows how the get the Accuracy, Recall, Precision and F1 score with different algorithms and data preprocessing methods, including "LogisticRegression", "RandomForest with SMOTEENN" and "RandomForest with SMOTE" using "Eclipse 2.0", "Eclipse 2.1" and "Eclipse 3.0" as train set and test set respectively. After that, I write the results to the “sheet2” of file "result_from_paper.xlsx" and compare my results with the results in Zimmermann's paper.

In the directory "MLSE/final_code/D'Ambros' dasa set/" in the "ZhenjieFinalDemo" branch, the file "Classification of files.ipynb" shows how the get the Accuracy, Recall, Precision, F1 score and the average of these four scores with different algorithms and data preprocessing methods, including "LogisticRegression", "MLPClassifier with RandomOverSampler", "RandomForest with RandomOverSampler", "RandomForest with SMOTEENN", "RandomForest with SMOTE" and "RandomForest" using " Eclipse_JDT_Core", "Eclipse_PDE_UI", "Equinox_Framework", "Lucene" and "Mylyn" as train set and test set respectively. After running the file "Classification of files.ipynb", these results can also be written into the file "result.xlsx". After that, I change the file name from "result.xlsx" to "result_from_website.xlsx".

Then, in the directory "MLSE/final_code/plot/paper/" in the "ZhenjieFinalDemo" branch, The file "result_from_paper_plot.ipynb" shows how to get the following figures in the final report with the data in "result_from_paper.xlsx".

Figure 17: Accuracy, Recall, Precision and F1 score with different algorithms using Eclipse 2.0 as train set with different versions of Eclipse on Zimmermann’s data set (Zimmermann, 2007) 

Figure 18: Accuracy, Recall, Precision and F1 score with different algorithms using Eclipse 2.1 as train set with different versions of Eclipse on Zimmermann’s data set (Zimmermann, 2007) 

Figure 19: Accuracy, Recall, Precision and F1 score with different algorithms using Eclipse 3.0 as train set with different versions of Eclipse on Zimmermann’s data set (Zimmermann, 2007)

Besides, with the results above, I can also get the Table 5 and Table 6 in the final report.

Moreover, in the directory "MLSE/final_code/plot/website/" in the "ZhenjieFinalDemo" branch, The file "result_from_website_plot.ipynb" shows how to get the following figures with the data in "result_from_website.xlsx"

Figure 12: Accuracy, Recall, Precision and F1 score with different algorithms using Eclipse_JDT_Core as train set with different projects on D'Ambros’s dataset (D'Ambros, 2010)

Figure 13: Accuracy, Recall, Precision and F1 score with different algorithms using Eclipse_PDE_UI as train set with different projects on D'Ambros’s dataset (D'Ambros, 2010)  

Figure 14: Accuracy, Recall, Precision and F1 score with different algorithms using Equinox_Framework as train set with different projects on D'Ambros’s dataset (D'Ambros, 2010) 

Figure 15: Accuracy, Recall, Precision and F1 score with different algorithms using Lucene as train set with different projects on D'Ambros’s dataset (D'Ambros, 2010)  

Figure 16: Accuracy, Recall, Precision and F1 score with different algorithms using Mylyn as train set with different projects on D'Ambros’s dataset (D'Ambros, 2010)  

What is more, with the results above, I can also get the Table 4 in the final report.

Finally, in the directory "MLSE/final_code/plot/" in the "ZhenjieFinalDemo" branch, The file "Logistic Regression Example.ipynb" shows how to get the Figure 2 in the final report.
