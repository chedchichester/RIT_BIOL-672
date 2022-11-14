# Name: Ched Chichester
# Operating System: Windows 11
# Context: Unit 2 Assignment Part 4
# Date Created: November 13, 2022
# Date of Last Revision: November 13, 2022
# Packages: pandas, kaggle, sklearn
# Data files: heart.csv (Downloaded from https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

import pandas as pd  # Import pandas for reading csv
import zipfile  # Import zipfile to deal with dataset that downloads as a zip file
from kaggle.api.kaggle_api_extended import KaggleApi  # Import kaggle API
from sklearn.model_selection import train_test_split  # Import train_test_split for creating training and testing sets
from sklearn.ensemble import RandomForestClassifier  # Import random forest classifier
from sklearn.ensemble import AdaBoostClassifier  # Import adaboost classifier
from sklearn.metrics import confusion_matrix  # Import for creating confusion matrix

api = KaggleApi()  # Return Kaggle API object
api.authenticate()  # Authenticate API object
api.dataset_download_file("fedesoriano/heart-failure-prediction", file_name="heart.csv")  # Download dataset
heart_dataframe = pd.read_csv("heart.csv", header=0, delimiter=",")  # Read job postings csv file
heart_disease = heart_dataframe["HeartDisease"]  # Separate classification into separate vector
heart_feature = heart_dataframe.drop(["HeartDisease", "Sex", "ChestPainType", "RestingECG", "ExerciseAngina",
                                      "ST_Slope"], axis="columns")  # Drop classification from feature dataframe along
# with other strings
heart_feature_train, heart_feature_test, heart_disease_train, heart_disease_test = train_test_split(heart_feature,
                                                                                                    heart_disease,
                                                                                                    test_size=0.5)
# Split data into even training/test sets


def execute_random_forest():  # Execute random forest classifier
    random_forest = RandomForestClassifier()  # Assign random forest classifier
    random_forest.fit(heart_feature_train, heart_disease_train)  # Fit random forest classifier
    random_forest_prediction = random_forest.predict(heart_feature_test)  # Extract random forest predictions
    cm_random_forest = confusion_matrix(heart_disease_test, random_forest_prediction)  # Create confusion matrix for
    # random forest
    print("\nRandom Forest Confusion Matrix: \n" + str(cm_random_forest))  # Print confusion matrix for random forest
    print("Random Forest Accuracy: " + str(random_forest.score(heart_feature_test, heart_disease_test) * 100) + "%")
    # Print random forest accuracy


def execute_adaboost():  # Execute random forest classifier
    adaboost = AdaBoostClassifier()  # Assign adaboost classifier
    adaboost.fit(heart_feature_train, heart_disease_train)  # Fit adaboost classifier
    adaboost_prediction = adaboost.predict(heart_feature_test)  # Extract adaboost predictions
    cm_adaboost = confusion_matrix(heart_disease_test, adaboost_prediction)  # Create confusion matrix for adaboost
    print("\nAdaboost Confusion Matrix: \n" + str(cm_adaboost))  # Print confusion matrix for adaboost
    print("Adaboost Accuracy: " + str(adaboost.score(heart_feature_test, heart_disease_test) * 100) + "%")
    # Print adaboost accuracy
    print("The random forest performs slightly better than the adaboost algorithm in this case by about 4%. Looking"
          "at the confusion matrix, the adaboost algorithm wrongly has a higher false negative rate for predicting "
          "heart disease, \nbut a similar false positive rate compared to the random forest algorithm.")


def main():  # Execute defined functions
    execute_random_forest()
    execute_adaboost()


if __name__ == "__main__":  # Execute Python functions
    main()  # Execute main function above
