# Supervised ML for anomaly detection in IOT to enahnce network security
This project aims to evaluate the effectiveness of supervised machine-learning-based anomaly detection techniques in distinguishing between benign and malicious network traffic in the IoT-23 dataset and to assess the robustness and scalability of different supervised anomaly detection methods in handling the dynamic and heterogeneous nature of IoT network environments.


## Data Set (Aposemat IoT-23)
The dataset used in this project is: [iot_23_datasets_small](https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/iot_23_datasets_small.tar.gz).<br/>
- The lighter version containing only the labeled flows without the pcaps files (8.8 GB) [Aposemat IoT-23 dataset](https://www.stratosphereips.org/datasets-iot23).
- A labeled dataset with malicious and benign IoT network traffic.
- This dataset was created as part of the Avast AIC laboratory with the funding of Avast Software. 

## Data Classification Details
The project is implemented in four distinct steps simulating the essential data processing and analysis phases. <br/>
- Each step is represented in a corresponding Jupyter Notebook inside [IOT-23-dataset-extraction](IOT-23-dataset-extraction).
- Data files (raw, interim, processed) are stored inside the [CSV-data](CSV-data) path.
- Trained data models are stored inside [applied-ML-methods](applied-ML-methods).

### PHASE 1 - Data Cleaning
> Corresponding Jupyter Notebook:  [iot-23-data-cleaning.ipynb](https://github.com/AmazingCoder107856/supervised-ML-anomaly-detection-using-IoT-23-dataset-/blob/main/IOT-23-dataset-extraction/iot-23-data-cleaning.ipynb)

Implemented data exploration and cleaning tasks:
1. Loading the raw dataset file into pandas DataFrame.
2. Exploring dataset summary and statistics.
3. Fixing combined columns.
4. Dropping irrelevant columns.
5. Fixing unset values and validating data types.
6. Checking the cleaned version of the dataset.
7. Storing the cleaned dataset to a csv file.

### PHASE 2 - Data Processing
> Corresponding Jupyter Notebook:  [iot-23-data-preprocessed.ipynb](https://github.com/AmazingCoder107856/supervised-ML-anomaly-detection-using-IoT-23-dataset-/blob/main/IOT-23-dataset-extraction/iot-23-data-preprocessed.ipynb)

Implemented data processing and transformation tasks:
1. Loading dataset file into pandas DataFrame.
2. Exploring dataset summary and statistics.
3. Analyzing the target attribute.
4. Encoding the target attribute using [LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html).
5. Handling outliers using [IQR (Inter-quartile Range)](https://en.wikipedia.org/wiki/Interquartile_range).
6. Encoding IP addresses.
7. Handling missing values:
    1. Impute missing categorical features using [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html).
    2. Impute missing numerical features using [KNNImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html).
8. Scaling numerical attributes using [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
9. Encoding categorical features: handling rare values and applying [One-Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
10. Checking the processed dataset and storing it to a csv file.

### PHASE 3 - Data Training
> Corresponding Jupyter Notebook:  [iot-23-data-training.ipynb](https://github.com/AmazingCoder107856/supervised-ML-anomaly-detection-using-IoT-23-dataset-/blob/main/IOT-23-dataset-extraction/iot-23-data-training.ipynb)

Trained and analyzed classification models:
1. Naive Bayes: [ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html)
2. Decision Tree: [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
3. Logistic Regression: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)    
4. Random Forest: [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
5. Support Vector Classifier: [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
6. K-Nearest Neighbors: [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
7. XGBoost: [XGBClassifier](https://xgboost.readthedocs.io/en/stable/index.html#)

Evaluation method: 
- Cross-Validation Technique: [Stratified K-Folds Cross-Validator](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- Folds number: 5
- Shuffled: Enabled

Results were analyzed and compared for each considered model.<br/>

### PHASE 4 - Data Tuning
> Corresponding Jupyter Notebook:  [iot-23-data-tuning.ipynb](https://github.com/AmazingCoder107856/supervised-ML-anomaly-detection-using-IoT-23-dataset-/blob/main/IOT-23-dataset-extraction/iot-23-data-tuning.ipynb)

Model tuning details:
- Tuned model: Support Vector Classifier - [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)
- Tuning method: [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- Results were analyzed before/after tuning.
