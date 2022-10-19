# ChedChichester_unit1_BIOL672.py
# Name: Ched Chichester
# Operating System: Windows 11
# Context: Unit 1 Assignment Part 2
# Date Created: September 28, 2022
# Date of Last Revision: October 19, 2022
# Packages: math, matplotlib, subprocess, pandas, sklearn, distfit, factor_analyzer
# Data files: iris_tab.txt, iris_size_tab.txt

import subprocess as sub  # Import subprocess to call R function
import pandas as pd  # Imports pandas for creating dataframes
import seaborn as sea  # Import seaborn for plotting sca
from math import log  # Import log to calculate BIC
from distfit import distfit  # Import distfit for distribution fitting
from matplotlib import pyplot as plt  # Imports pyplot for plotting
from sklearn.datasets import load_wine  # Import datasets for analysis
from sklearn.preprocessing import StandardScaler  # Import StandardScaler to standardize values for PCA
from sklearn.decomposition import PCA  # Import PCA to return principal components of data
from sklearn.cluster import KMeans  # Import KMeans to perform k-means clustering
from sklearn import mixture  # Import mixture for GMM model
from factor_analyzer import FactorAnalyzer  # Import FactorAnalyzer to perform factor analysis

# I wanted to execute my R script from this Python script. The following code seems to technically execute my R script,
# but it seems to have an issue with the portion of my R code that writes to .txt files and prints to pdf files (at
# least when I run it on my Windows 11 PC). It does print anything that I would print to my R console to my Python
# console, but I printed all of my R data to output files so nothing will be output to my Python window from this
# except for the "Running R Script..." and "Done running R script!"
sub.call(["C:\\Program Files\\R\\R-4.2.1\\bin\\Rscript", "--vanilla",
          "C:\\Users\\ched1\\RStudio Projects\\unit_1_assignment.r"])  # Run R code via Rscript and denote filepath

# Initialize variables for analysis in downstream-defined functions
iris_size_data = pd.read_csv("iris_size_tab.txt", delimiter="\t")  # Read in iris_size_tab.txt data
iris_size_data = iris_size_data.drop(columns=["species", "avg_size"])  # Drop the species column
sepal_length = iris_size_data.sepal_length  # Assign sepal_length factor
sepal_width = iris_size_data.sepal_width  # Assign sepal_width factor
petal_length = iris_size_data.petal_length  # Assign petal_length factor
petal_width = iris_size_data.petal_width  # Assign petal_width factor
iris_features = [sepal_length, sepal_width, petal_length, petal_width]  # Create array of features for analysis
iris_size_data_standardized = StandardScaler().fit_transform(iris_features)  # Standardize data before PCA and FA


def execute_pca():  # Define function for PCA

    # Create PCA object and execute PCA to obtain loadings and dataframe
    pca = PCA(n_components=4)  # Initialized PCA object with four dimensions
    pca_data = pca.fit_transform(iris_size_data_standardized)  # Execute PCA on data with 4 dimensions
    pca_df = pd.DataFrame(data=pca_data,
                          columns=["PC 1", "PC 2", "PC 3", "PC 4"])  # Create dataframe with principal components
    print("\nPCA Loadings: \n", pca_df)  # Print PCA loadings for analysis
    explained_variance_ratio = pca.explained_variance_ratio_  # Assign the explained variance ratio
    print("\nExplained variance ratios:" + "\nPC 1: " + str(explained_variance_ratio[0]) + "\nPC 2: " +
          str(explained_variance_ratio[1]) + "\nPC 3: " +
          str(explained_variance_ratio[2]) + "\nPC 4: " +
          str(explained_variance_ratio[3]))  # Print explained variance ratio for analysis

    # Plot Explained Variance Ratio figure
    plt.figure(figsize=(6, 6))  # Creates figure with specified size
    plt.bar(range(4), explained_variance_ratio, align="center", label=' Individual Explained Variance', color="blue")
    # Plots bar graph to show how much explained variance is attributed to each principal component
    plt.ylabel('Explained Variance Ratio')  # Label y axis
    plt.xlabel('Principal Components')  # Label x axis
    plt.title("PCA Scree Plot")  # Title plot
    plt.legend(loc='best')  # Create legend in ideal location
    plt.tight_layout()  # Fits labels within figure window
    plt.show()  # Show plot
    print(
        "\nThe data reduction was successful. As shown by the scree plot, only three principal components are required"
        "to explain the variance in the data. This means that the data can be adequately analyzed in three dimensions"
        " without \nlosing any variability. Furthermore, as can be seen visually from the scree plot and "
        "quantitatively from the explained variance ratios, PC 1 and PC 2 account for about 99.7% of the total "
        "variance. Therefore, one could \narguably reduce the dimensions down from four to two and only sacrifice a"
        "very small amount of variability in the data. With regards to the principal component loadings, none of them"
        "had all positive or negative values \n(with the exception of PC 4, which technically had all positive "
        "values, which were effectively 0) and therefore the principal components were all constructed with some "
        "combination of positive and negative combinations of \nsepal_length, sepal_width, petal_length, and "
        "petal_width. It is difficult to say any of the principal components captures much more size variation than "
        "the others because the average loading value for each principal \ncomponent is 0. So, each component is "
        "taking equal positive and negative weights of the features. However, it is worth noting that PC 1 has the "
        "largest weighting values, and therefore it could be said PC 1 is taking the \nhighest percentage of the "
        "features into account. PC 1 is particularly strongly driven by sepal_length and petal_width while PC 2 is "
        "strongly driven by sepal_width and petal_length.")  # Verbal interpretation of PCA results


def execute_factor_analysis():  # Define function for factor analysis
    factor_analysis = FactorAnalyzer(n_factors=3, rotation="varimax")  # Prepare factor analysis model with 3 factors
    factor_analysis.fit(iris_size_data)  # Fit factor analysis model to data
    ev, v = factor_analysis.get_eigenvalues()  # Return eigenvalues and eigenvectors
    print("\nFactor Analysis Eigenvalues: " + str(ev))  # Print eigenvalues
    print("\nFactor Analysis Loadings: \n" + str(factor_analysis.loadings_))  # Print FA loadings for analysis
    print("\nLooking at the factor analysis loadings, factor 1 is strongly associated with the sepal_length,"
          " petal_length, and petal_width. Factor 2 is strongly associated with the sepal_width."
          "By analyzing the eigenvalues from the factor \nanalysis of the covariance matrix, only one eigenvalue is "
          "greater than 1, with a second eigenvalue being nearly 1. Thus, factor 1 is the most significant factor"
          "with factor 2 potentially being somewhat significant as well. \nFactor 3 is not very significant. The main"
          "characteristics that cluster together are sepal_length, petal_length, and petal_width as can be seen in "
          "factor 1. Also, petal_length and petal_width cluster together as both factor 2 \nand factor 3 have values "
          "for these variables that are similar to one another. If two traits are far apart along the axis of a "
          "significant factor, this signifies the traits are not correlated or even inversely correlated, and \nvice "
          "versa. Yes, this factor analysis was successful in identifying two latent factors that signify relationships"
          " between the variables within the model.")  # Verbal interpretation of factor analysis


def execute_kmeans():  # Define function for k-means clustering
    iris_data = pd.read_csv("iris_tab.txt", delimiter="\t")  # Read in iris_tab.txt data
    sea.lmplot(x="petal_length", y="petal_width", data=iris_data, hue="species", fit_reg=False)  # Plot scatter plot of
    # petal_length vs petal_width without regression fit lines and select point color by species
    plt.title("Species-Categorized Petal Length vs Petal Width")  # Title plot
    plt.xlabel("Petal Length")  # Labels x axis
    plt.ylabel("Petal Width")  # Labels y axis
    plt.tight_layout()  # Fits labels within figure window
    plt.show()  # Show plot

    k_means = KMeans(n_clusters=2)  # Specify the number of clusters to predict via k-means clustering algorithm
    k_means_cluster = k_means.fit_predict(iris_data[["petal_length", "petal_width"]])  # Predict k-means clusters for
    # petal_length and petal_width
    iris_data["cluster"] = k_means_cluster  # Create new column to match each row with its assigned cluster
    iris_df_1 = iris_data[iris_data.cluster == 0]  # Create new dataframe for the first cluster
    iris_df_2 = iris_data[iris_data.cluster == 1]  # Create new dataframe for teh second cluster
    plt.scatter(iris_df_1.petal_length, iris_df_1.petal_width, color="red")  # Plot scatter of first cluster red
    plt.scatter(iris_df_2.petal_length, iris_df_2.petal_width, color="blue")  # Plot scatter of second cluster blue
    plt.xlabel("Petal Length")  # Label x axis
    plt.ylabel("Petal Width")  # Label y axis
    plt.title("K-Means-Clustered Petal Length vs Petal Width")  # Title plot
    plt.tight_layout()  # Fits labels within figure window
    plt.show()  # Show plot


def multi_model_inference():  # Define function for multi-model inference
    wine_data = load_wine()  # Load wine dataset
    wine_dataframe = pd.DataFrame(wine_data.data)  # Creates dataframe of the wine dataset
    wine_alcohol_content = wine_dataframe.iloc[:, 0]  # Extracts first column of dataframe, which represents alcohol
    # content
    n = len(wine_alcohol_content)  # Calculate n for BIC calculation
    k = 1  # Assign k for BIC calculation

    dist_norm = distfit(distr="norm")  # Creates normal distribution fit object
    dist_norm.fit_transform(wine_alcohol_content)  # Fits normal distribution to data
    dist_norm.plot(title="Normal Distribution Fitting", figsize=(8, 8))  # Plots normal distribution fit
    plt.show()  # Show plot
    norm_rss = 1.21089  # Assign normal distribution RSS to calculate BIC
    norm_bic = n * log(norm_rss/n) + k * log(n)  # Calculate normal distribution BIC

    dist_lognorm = distfit(distr="lognorm")  # Creates log-normal distribution fit object
    dist_lognorm.fit_transform(wine_alcohol_content)  # Fits log-normal distribution to data
    dist_lognorm.plot(title="Log-Normal Distribution Fitting", figsize=(8, 8))  # Plots log-normal distribution fit
    plt.show()  # Show plot
    lognorm_rss = 1.21219  # Assign normal distribution RSS to calculate BIC
    lognorm_bic = n * log(lognorm_rss/n) + k * log(n)  # Calculate log-normal distribution BIC

    dist_expon = distfit(distr="expon")  # Creates exponential distribution fit object
    dist_expon.fit_transform(wine_alcohol_content)  # Fits exponential distribution to data
    dist_expon.plot(title="Exponential Distribution Fitting", figsize=(8, 8))  # Plots exponential distribution fit
    plt.show()  # Show plot
    expon_rss = 3.32596  # Assign normal distribution RSS to calculate BIC
    expon_bic = n * log(expon_rss/n) + k * log(n)  # Calculate exponential distribution BIC

    dist_all = distfit()  # Creates fit object without defining distribution to compare each method
    dist_all.fit_transform(wine_alcohol_content)  # Fits data to several distributions for comparison
    dist_all.plot_summary()  # Return summary of distribution performances
    plt.title("Distribution Fit Summary")  # Titles plot
    plt.show()  # Show plot

    wine_alcohol_content = wine_alcohol_content.values
    wine_alcohol_content = wine_alcohol_content.reshape(-1, 1)
    gmm = mixture.GaussianMixture(n_components=3)  # Creates GMM object with 3 classes/latent variables
    gmm.fit(wine_alcohol_content)  # Fit GMM to model
    gmm_bic = gmm.bic(wine_alcohol_content)  # Calculate BIC on GMM fit

    print("\nThe normal distribution BIC is :" + str(norm_bic))  # Print normal distribution BIC
    print("\nThe log-normal distribution BIC is :" + str(lognorm_bic))  # Print log-normal distribution BIC
    print("\nThe exponential distribution BIC is :" + str(expon_bic))  # Print exponential distribution BIC
    print("\nThe GMM BIC is: " + str(gmm_bic))  # Print GMM BIC
    print("THe result of my model fitting and BIC analysis shows that the log-normal distribution is the best fit for"
          " the data. Thus, this model testing did not indicate the presence of latency. Interestingly, the BIC values"
          "\nI calculated for the normal, log-normal, and exponential distributions are all negative while the GMM BIC "
          "is positive. I calculated the BIC for the non-GMM distributions utilizing the RSS from their fits, but "
          "perhaps there\nis an issue or discrepancy between this method and the method from the GMM module, which had"
          " a built-in method to return the BIC. It's also worth mentioning that the BIC values for the normal, "
          "log-normal, and exponential\ndistributions align with the RSS values in that the ones with lower RSS values"
          "have lower BIC values, indicated a better fit. Unfortunately, I had some trouble producing a plot of the"
          " GMM distribution, so simply assessing\nthe BIC values is all I am able to do for now.")
    # Print verbal interpretation


def main():  # Execute defined functions
    execute_pca()  # Execute PCA
    execute_factor_analysis()  # Execute factor analysis
    execute_kmeans()  # Execute k-means clustering
    multi_model_inference()  # Execute multi-model inference


if __name__ == "__main__":  # Execute Python functions
    main()  # Execute main function above
