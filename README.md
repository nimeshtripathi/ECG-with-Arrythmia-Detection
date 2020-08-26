# ECG-with-Arrythmia-Detection


#Problem Statement
#The ECG or electrocardiogram, measures the electrical activity of the heart. This was invented in 1887 by Waller ( source: Wikipedia) and has since 
#been an optimal tool in clinics for a measurement of the heart function. Multiple Cardiovascular diseases like arrhythmia , atrioventricular dysfunctions 
#and coronary arterial disease etc. can be detected using ECG monitoring devices. The ECG has gone under several upgrades and here I use multiple Machine 
#Learning algorithms like RNN, KNN, OneVsAll to measure their classification accuracy.

Description of Dataset
• The dataset used in the project is collected from the UCI repository , which has a 279 attributes that may cause arrythmia.

Description of Approach
• There were missing datapoints which were replaced by average of the column.
• To predict the output, three classifiers are used, which are 1. OvA 2. KNN 3.RNN.
• Accuracy with OnehotEncoding and Normalization via Feature Scaling: 78.33%
• Accuracy without OnehotEncoding and Normalization via Feature Scaling: 49%



For RNN, the splitting of data in training and test is 1:1. Since the split is random, this affected the accuracy. However, this can be solved with cross 
validation.
SVM has the best classification accuracy of 98.9%.

Conclusion
• The presence/absence of arrythmia was successfully identified in the project, and it was classified into 16 classes.
• Class 1 is normal i.e. no arrythmia.
• Classes 2 to 15, are the different types of arrythmia, which means that the
patient indeed does have arrythmia.
• Class 16 is unclassified.
