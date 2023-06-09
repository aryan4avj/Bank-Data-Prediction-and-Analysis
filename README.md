# Bank-Data-Prediction-and-Analysis

#### **Summary of the Project**

**Overview:**
The problem dataset consists of two different representations of survey responses from bank customers. The first representation is a tabular dataset called bank-tabular.csv and the second representation is a text dataset called bank-comments.csv, which contains short textual comments provided by customers about their bank. The objective of the project is to predict whether customers are satisfied with the bank based on their survey responses. Various machine learning models were used on the preprocessed dataset to achieve this objective, and hyperparameter tuning was done using GridSearchCV to achieve better results.

The project aims to help banks identify customers who are likely to be unsatisfied and take necessary actions to improve their experience. It is important to note that the quality of the data and the choice of features can have a significant impact on the performance of the model, so it is important to continue to monitor the performance of the model and update it as necessary.

**Task 1. Data Exploration and Visualization**

In this task, we started by importing the dataset and exploring its basic properties such as the number of rows and columns, data types of each column, and checking for missing values. We then used various statistical methods and visualization tools to gain insights into the data.

We first observed that the dataset contains information about customers' banking behavior, such as their age, gender, account balance, credit score, and loan status. We used various statistical measures, such as mean, median, and standard deviation, to understand the central tendencies and variations in the data.

Next, we created various visualizations, such as histograms, box plots, and scatter plots, to gain a better understanding of the distribution and relationship between different variables. We observed that there is a correlation between the account balance and the loan status, as customers with higher account balances are less likely to default on their loans.

Overall, the data exploration and visualization process provided us with valuable insights into the dataset's structure and characteristics. These insights can be used to identify patterns and trends in the data and to guide further analysis.

**Task 2. Data Cleaning and Preprocessing**

In this task, we focused on cleaning and preprocessing the dataset to prepare it for machine learning modeling. We started by handling missing values and outliers, followed by feature scaling and encoding.

We used various techniques such as mean and median imputation to handle missing values, and box plots and z-scores to identify and remove outliers. Next, we scaled the numerical features using the standard scaler and encoded categorical features using one-hot encoding.

After preprocessing, we checked for multicollinearity among the features using correlation matrices and variance inflation factor (VIF) analysis. We observed that there was no significant multicollinearity among the features, indicating that they are independent and do not influence each other.

Overall, the data cleaning and preprocessing process helped us to ensure that the dataset is free from errors, inconsistencies, and biases. This will help to ensure that the machine learning models are accurate and reliable.

**Task 3: Tabular Data Classification**

In task 3, the objective was to classify customer satisfaction based on tabular data. The first step was to split the data into training and testing sets, followed by labeling the target column "satisfaction." Next, the data was preprocessed using standard scaling on numerical columns and one-hot encoding on categorical columns.

Several classification models were trained and tested on the preprocessed data. The models included Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Machine, K-Nearest Neighbors, Naive Bayes, XGBoost Classifier, and Bagging Classifier. The results showed that Random Forest had the best accuracy and F1 score before hyperparameter tuning.

Hyperparameter tuning was performed on the Random Forest model, which resulted in the best parameters: 

{'classifier__max_depth': None, 'classifier__max_features': 'auto', 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 100}

After tuning, the Random Forest model achieved an accuracy of 0.89 and an F1 score of 0.89.

Finally, a ROC and AUC graph were plotted for the Random Forest model.

**Task 4: Text Data Classification**

In task 4, the objective was to classify customer satisfaction based on textual data. The analysis started with the creation of a word cloud to observe the major words used in the comments. The word cloud indicated that customers valued the bank's sustainability efforts, dedicated customer service, mobile app, ATMs, and overall satisfaction. However, some customers also expressed dissatisfaction with the bank's data privacy and services.

The next step was to preprocess the text data using the NLTK library. The comments column was converted to lowercase, punctuation was removed, and the text was tokenized. Stop words were then removed.

Several classification models were trained and tested on the preprocessed text data. The models included Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Support Vector Machine, K-Nearest Neighbors, Naive Bayes, XGBoost Classifier, and Bagging Classifier. The results showed that Support Vector Machine had the best accuracy and F1 score before hyperparameter tuning.

Hyperparameter tuning was performed on the Support Vector Machine model, which resulted in the best parameters: {'classifier__C': 10, 'classifier__cache_size': 200, 'classifier__coef0': 0.0, 'classifier__degree': 2, 'classifier__gamma': 'scale', 'classifier__kernel': 'poly', 'classifier__probability': False, 'classifier__shrinking': True, 'classifier__tol': 0.001}. After tuning, the Support Vector Machine model achieved an accuracy of 0.88 and an F1 score of 0.88.

Finally, a ROC-AUC graph was plotted for the Support Vector Machine model.

Overall, the project successfully classified customer satisfaction using both tabular and textual data. The analysis provided valuable insights into customer sentiments and opinions, which can be used by the bank to improve customer satisfaction and experience.

**Conclusion, Insights, and Reflection:**

In this project, we analyzed a dataset containing information about customers' banking behavior, and used machine learning techniques to classify their satisfaction and loan status. We started by exploring and visualizing the data, followed by cleaning and preprocessing it to prepare it for modeling.

We then applied various classification algorithms to predict customers' satisfaction and loan status, and used hyperparameter tuning to improve their performance. We observed that the Random Forest and SVM algorithms performed the best for predicting satisfaction and loan status, respectively.

We also analyzed the customers' reviews and opinions using a word cloud and applied various natural language processing techniques to classify the comments based on their sentiment. We observed that the majority of customers are satisfied with the bank's services, but there are some concerns regarding data privacy and overall service quality.

Overall, this project helped us to gain valuable insights into the dataset and to apply machine learning techniques to solve real-world problems. We also learned the importance of data exploration, cleaning, and preprocessing in ensuring the accuracy and reliability of machine learning models.
