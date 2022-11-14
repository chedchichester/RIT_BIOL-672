# Name: Ched Chichester
# Operating System: Windows 11
# Context: Unit 2 Assignment Part 1
# Date Created: November 1, 2022
# Date of Last Revision: November 13, 2022
# Packages: pandas, kaggle, sklearn
# Data files: diabetes.csv (Downloaded from https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)

import pandas as pd  # Import pandas for reading csv
from kaggle.api.kaggle_api_extended import KaggleApi  # Import kaggle API
from sklearn.model_selection import train_test_split  # Import train_test_split for creating training and testing sets
from sklearn.neighbors import KNeighborsClassifier  # Import for utilizing KNN model
from sklearn.naive_bayes import GaussianNB  # Import Gaussian Naive Bayes classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Import LDA from Sci-kit learn
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  # Import QDA from Sci-kit learn
from sklearn import mixture  # Import mixture for GMM model with EM algorithm
from sklearn.model_selection import cross_val_score  # Import for k-fold cross validation
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


def execute_knn():  # Execute k-nearest neighbors method
    knn = KNeighborsClassifier(n_neighbors=5)  # Use 3 nearest neighbors for fitting KNN model
    knn.fit(feature_training_set, outcome_training_set)  # Fit KNN training model
    knn_5fold_scores = cross_val_score(knn, feature_training_set, outcome_training_set, cv=5)  # KNN 5-fold
    # cross validation
    outcome_prediction_knn = knn.predict(feature_test_set)  # Extract KNN predictions
    cm_knn = confusion_matrix(outcome_test_set, outcome_prediction_knn)  # Create confusion matrix for KNN
    print("\nKNN Confusion Matrix: \n" + str(cm_knn))  # Print confusion matrix for KNN
    print("KNN Accuracy: " + str(knn.score(feature_test_set, outcome_test_set) * 100) + "%")  # Print KNN accuracy
    print("KNN 5-Fold Cross Validation Accuracy Mean: " + str(knn_5fold_scores.mean() * 100) + "%")  # Print mean
    print("KNN 5-FOld Cross Validation Standard Deviation: " + str(knn_5fold_scores.std() * 100) + "%")  # Print std


def execute_naive_bayes():  # Execute naive bayes method
    nb_model = GaussianNB()  # Assign model as Gaussian Naive Bayes
    nb_model.fit(feature_training_set, outcome_training_set)  # Fit NB training model
    nb_5fold_scores = cross_val_score(nb_model, feature_training_set,
                                      outcome_training_set, cv=5)  # NB 5-fold cross validation
    outcome_prediction_nb = nb_model.predict(feature_test_set)  # Extract NB predictions
    cm_nb = confusion_matrix(outcome_test_set, outcome_prediction_nb)  # Create confusion matrix for NB
    print("\nNB Confusion Matrix: \n" + str(cm_nb))  # Print confusion matrix for NB
    print("NB Accuracy: " + str(nb_model.score(feature_test_set, outcome_test_set) * 100) + "%")  # Print NB accuracy
    print("NB 5-Fold Cross Validation Accuracy Mean: " + str(nb_5fold_scores.mean() * 100) + "%")  # Print mean
    print("NB 5-FOld Cross Validation Standard Deviation: " + str(nb_5fold_scores.std() * 100) + "%")  # Print std


def execute_lda():  # Execute linear discriminant analysis
    lda = LinearDiscriminantAnalysis()  # Assign model as linear discriminant analysis
    lda.fit(feature_training_set, outcome_training_set)  # Fit LDA training model
    lda_5fold_scores = cross_val_score(lda, feature_training_set, outcome_training_set, cv=5)  # LDA 5-fold cross
    # validation
    outcome_prediction_lda = lda.predict(feature_test_set)  # Extract LDA predictions
    cm_lda = confusion_matrix(outcome_test_set, outcome_prediction_lda)  # Create confusion matrix for LDA
    print("\nLDA Confusion Matrix: \n" + str(cm_lda))  # Print confusion matrix for LDA
    print("LDA Accuracy: " + str(lda.score(feature_test_set, outcome_test_set) * 100) + "%")  # Print LDA accuracy
    print("LDA 5-Fold Cross Validation Accuracy Mean: " + str(lda_5fold_scores.mean() * 100) + "%")  # Print mean
    print("LDA 5-FOld Cross Validation Standard Deviation: " + str(lda_5fold_scores.std() * 100) + "%")  # Print std


def execute_qda():  # Execute quadratic discriminant analysis
    qda = QuadraticDiscriminantAnalysis()  # Assign model as quadratic discriminant analysis
    qda.fit(feature_training_set, outcome_training_set)  # Fit QDA training model
    qda_5fold_scores = cross_val_score(qda, feature_training_set, outcome_training_set, cv=5)  # QDA 5-fold cross
    # validation
    outcome_prediction_qda = qda.predict(feature_test_set)  # Extract QDA predictions
    cm_qda = confusion_matrix(outcome_test_set, outcome_prediction_qda)  # Create confusion matrix for QDA
    print("\nQDA Confusion Matrix: \n" + str(cm_qda))  # Print confusion matrix for QDA
    print("QDA Accuracy: " + str(qda.score(feature_test_set, outcome_test_set) * 100) + "%")  # Print QDA accuracy
    print("QDA 5-Fold Cross Validation Accuracy Mean: " + str(qda_5fold_scores.mean() * 100) + "%")  # Print mean
    print("QDA 5-FOld Cross Validation Standard Deviation: " + str(qda_5fold_scores.std() * 100) + "%")  # Print std


def execute_em():  # Execute expectation maximization
    em = mixture.GaussianMixture(n_components=2)  # Creates GMM object with 2 classes/latent variables
    em.fit(feature_training_set)  # Fit GMM to model without classifications (unsupervised)
    outcome_prediction_em = em.predict(feature_test_set)  # Extract EM predictions
    cm_em = confusion_matrix(outcome_test_set, outcome_prediction_em)  # Create confusion matrix for EM
    print("\nEM Confusion Matrix: \n" + str(cm_em))  # Print confusion matrix for EM
    em_accuracy = ((cm_em[0][0] + cm_em[1][1]) / (cm_em[0][0] + cm_em[0][1] + cm_em[1][0] + cm_em[1][1])) * 100
    # Calculate accuracy for EM predictions
    print("EM Accuracy: " + str(em_accuracy) + "%")  # Print EM accuracy


def main():  # Execute defined functions
    execute_knn()
    execute_naive_bayes()
    execute_lda()
    execute_qda()
    execute_em()


if __name__ == "__main__":  # Execute Python functions
    main()  # Execute main function above
