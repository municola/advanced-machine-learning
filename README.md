# Advanced Machine Learning (2020)
This respository contains:
- Personal Summary of the whole course [[PDF](https://github.com/municola/advanced-machine-learning/blob/master/AML.pdf)]
- Code for the projects (Counted 30% to our final grade)

## Tasks
All of these tasks were related to health care and biology. <br>
Final Project Grade: 5.63

### Task 1
Outlieder Detection. The features in the given dataset represent volumes of surface areas of various regions of the brain. We then try to solve a regression task, however the dataset has outliers and bad features. <br>
Our Approach: Use a Random Forest to weight feature importance

### Task 2
Class Imbalance. We get a Dataset that is heavely imbalanced and need to regress values (supervised task). <br>
Our Approach: SVM with class-weights

### Task 3
ECG anomaly detection. As input we get a timeseries of ECG values. OUr task it to classify if the patient of this ECG has an anomaly (Decide between 3 possible anomalies) or has a normal heartbeat. <br>
Our Approach: Feature Extraction with libraries and personal Algorithms to extract Q,R,S point etc. + Gradient Boosting

### Task 4
Sleep detection for mice. As input we get EEG and EMG signals of a mouse for 24 hours. We then need to decide when the mouse slept and when it was awake.<br>
Our Approach: Fourier Transformation weith Hamming windows + CNN with Softmax + Postprocessing with a Dense Neural Net
