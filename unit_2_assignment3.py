# Name: Ched Chichester
# Operating System: Windows 11
# Context: Unit 2 Assignment Part 3
# Date Created: November 13, 2022
# Date of Last Revision: November 13, 2022
# Packages: pandas, kaggle, sklearn
# Data files: diabetes.csv (Downloaded from https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)
# nflcombine.csv (Downloaded from https://www.kaggle.com/datasets/thedevastator/nfl-team-stats-and-outcomes)

import pandas as pd  # Import pandas for reading csv
from kaggle.api.kaggle_api_extended import KaggleApi  # Import kaggle API
from sklearn.model_selection import train_test_split  # Import train_test_split for creating training and testing sets
from sklearn.neural_network import MLPClassifier  # Import neural network package
from sklearn.metrics import confusion_matrix  # Import for creating confusion matrix

api = KaggleApi()  # Return Kaggle API object
api.authenticate()  # Authenticate API object
api.dataset_download_file("akshaydattatraykhare/diabetes-dataset", file_name="diabetes.csv")  # Download dataset
api.dataset_download_file("thedevastator/nfl-team-stats-and-outcomes", file_name="nflcombine.csv")  # Download dataset
diabetes_dataframe = pd.read_csv("diabetes.csv", header=0, delimiter=",")  # Read diabetes csv file
diabetes_outcome = diabetes_dataframe["Outcome"]  # Separate classification into separate vector
diabetes_feature = diabetes_dataframe.drop("Outcome", axis="columns")  # Drop classification from feature dataframe
feature_training_set, feature_test_set, outcome_training_set, outcome_test_set = train_test_split(diabetes_feature,
                                                                                                  diabetes_outcome,
                                                                                                  test_size=0.5)
# Split data into even training/test sets

nfl_dataframe = pd.read_csv("nflcombine.csv", header=0, delimiter=",")  # Read nfl combine csv file
nfl_turnover = nfl_dataframe["TO"]  # Separate classification into separate vector
nfl_feature = nfl_dataframe.drop(["TO", "Opp", "Home_team", "Winner"],
                                 axis="columns")  # Drop classification and strings
nfl_feature_train, nfl_feature_test, nfl_turnover_train, nfl_turnover_test = train_test_split(nfl_feature,
                                                                                              nfl_turnover,
                                                                                              test_size=0.5)
# Split data into even training/test sets


def execute_ann():  # Execute artificial neural network
    ann = MLPClassifier()  # Assigns the ANN as a multi-layer perceptron classifier
    ann.fit(feature_training_set, outcome_training_set)  # Train ANN
    ann_prediction = ann.predict(feature_test_set)  # Test ANN predictions
    cm_ann = confusion_matrix(outcome_test_set, ann_prediction)  # Create confusion matrix for ANN
    print("\nANN Confusion Matrix: \n" + str(cm_ann))  # Print confusion matrix for ANN
    print("ANN Accuracy: " + str(ann.score(feature_test_set, outcome_test_set) * 100) + "%")  # Print ANN accuracy
    print("The artifical neural network performs similarly to the other machine learning methods with an accuracy of"
          "around 70%. Thus, the other methods are better suited for this classification because they are simpler to "
          "interpret.")

    deep_ann = MLPClassifier(hidden_layer_sizes=(10, 10, 10, 10, 10))  # Assigns teh deep ANN as a multi-layer
    # perceptron classifier with 5 layers of 10 nodes each
    deep_ann.fit(nfl_feature_train, nfl_turnover_train)  # Train ANN
    deep_ann_prediction = deep_ann.predict(nfl_feature_test)  # Test ANN predictions
    cm_ann = confusion_matrix(nfl_turnover_test, deep_ann_prediction)  # Create confusion matrix for QDA
    print("\nDeep ANN Confusion Matrix: \n" + str(cm_ann))  # Print confusion matrix for ANN
    print("Deep ANN Accuracy: " + str(deep_ann.score(nfl_feature_test, nfl_turnover_test) * 100) + "%")  # Print deep
    # ANN accuracy


def main():  # Execute defined functions
    execute_ann()


if __name__ == "__main__":  # Execute Python functions
    main()  # Execute main function above
