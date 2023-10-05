# Titanic_Classification_Project
# The Titanic Dataset
Without a doubt, one of the most well-known shipwrecks in history is the sinking of the Titanic.

The presumably "unsinkable" RMS Titanic sank on the 15th of April, 1912, during her first voyage after hitting an iceberg. Unfortunately, there weren't enough lifeboats on board to accommodate everyone and 1502 out of 2224 passengers and staff died.


Even while survival required a certain amount of luck, it appears that some groups of people had a higher chance of living than others.


We're creating a predictive model that addresses the following question: "What kinds of people were more likely to survive?" utilizing traveller information, such as name, age, gender, socioeconomic status, etc.

Link to download the dataset - https://www.kaggle.com/datasets/yasserh/titanic-dataset 

The Titanic dataset contains 891 rows and 12 columns. The columns are
1. PassengerId - PassengerId.
2. Survival - Survival in Titanic.
3. pclass - Passenger class.
4. Name - Name of the passenger.
5. Sex - Sex.
6. Age - Age in year.
7. SibSp - Siblings/spouses aboard the Titanic.
8. Parch - Parents/children aboard the Titanic.
9. Ticket - ticket number.
10. Fare - Passenger fare.
11. Cabin - cabin number.
12. Embarked - Port of embarkation (C-Cherbourg, Q- Queenstown, S- Southampton, O- others)

# Objective
- Understand the Dataset & cleanup (if required).
- Build a strong classification model to predict whether the passenger survives or not.
- Also fine-tune the hyperparameters & compare the evaluation metrics of various classification algorithms.


# Section
Data Preprocessing and Cleaning

1. Deals with NaN values using heatmap.
2. Drop the Cabin column.
3. Impute NaN Age based on the mean age.
4. Impute NaN Embarked based on the mode embarked.

Data Visualization

1. Plotting histogram of the dataset.
2. Illustrate the trends in survival rates and demographic characteristics.
3. Plotting passenger class survives.
4. Showing Distribution of Pclass Sex wise.

Building Model

1. Label encoding.
2. Selecting independent and dependent features.
3. Split data for training and testing.
4. Train, Predict, and evaluate the Logistic Regression, Random Forest Classifier, DecisionTreeClassifier, KNeighborsClassifier, SVC, and AdaBoostClassifier.
