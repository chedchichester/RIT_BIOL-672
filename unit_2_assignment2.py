# Name: Ched Chichester
# Operating System: Windows 11
# Context: Unit 2 Assignment Part 2
# Date Created: November 13, 2022
# Date of Last Revision: November 13, 2022
# Packages: pandas, kaggle, sklearn
# Data files: diabetes.csv (Downloaded from https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)

import pandas as pd  # Import pandas for reading csv
from kaggle.api.kaggle_api_extended import KaggleApi  # Import kaggle API
from sklearn.model_selection import train_test_split  # Import train_test_split for creating training and testing sets
from sklearn import svm  # Import svm

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


def execute_svm():  # Execute support vector machine
    support_vector_linear = svm.SVC(kernel="linear")  # Assigns linear SVM model
    support_vector_linear.fit(feature_training_set, outcome_training_set)  # Fit linear SVM model
    print("\nLinear SVM Accuracy: " + str(support_vector_linear.score(feature_test_set, outcome_test_set) * 100) + "%")
    # Print linear SVM accuracy
    support_vector_polynomial = svm.SVC(kernel="poly")  # Assigns polynomial SVM model
    support_vector_polynomial.fit(feature_training_set, outcome_training_set)  # Fit polynomial SVM model
    print("Polynomial SVM Accuracy: " + str(support_vector_polynomial.score(feature_test_set, outcome_test_set) *
                                            100) + "%")  # Print polynomial SVM accuracy
    support_vector_rbf = svm.SVC(kernel="rbf")  # Assigns radial basis function SVM model
    support_vector_rbf.fit(feature_training_set, outcome_training_set)  # Fit radial basis function SVM model
    print("Radial Basis Function SVM Accuracy: " + str(support_vector_rbf.score(feature_test_set, outcome_test_set) *
                                                       100) + "%")  # Print radial basis function SVM accuracy
    support_vector_sigmoid = svm.SVC(kernel="sigmoid")  # Assigns sigmoid SVM model
    support_vector_sigmoid.fit(feature_training_set, outcome_training_set)  # Fit sigmoid SVM model
    print("Sigmoid SVM Accuracy: " + str(support_vector_sigmoid.score(feature_test_set, outcome_test_set) * 100) + "%")
    # Print sigmoid SVM accuracy
    print("Besides the sigmoid kernel, all of the other kernel functions produced similar accuracy results to one "
          "another with results hovering between 70% and 75%. The sigmoid kernel function, however, only produced"
          " an accuracy of \nabout 55%. The linear, polynomial, and radial basis function based SVM methods also "
          "produced similar accuracy results when compared to the simpler machine learning methods, which all were"
          " between 70% and 80% accurate. SVM \nis not as simple as the other methods, so it is not very useful in this"
          "scenario because it does not provide a significant boost in prediction accuracy.")


def main():  # Execute defined functions
    execute_svm()


if __name__ == "__main__":  # Execute Python functions
    main()  # Execute main function above
