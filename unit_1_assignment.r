# ChedChichester_unit1_BIOL672.r
# Name: Ched Chichester
# Operating System: Windows 11
# Context: Unit 1 Assignment Part 1
# Date Created: September 21, 2022
# Date of Last Revision: October 19, 2022
# Libraries: ggplot2, dplyr
# Data files: mtcars, diamonds, iris_tab, iris_purchase, iris_tab_bgnoise, 
# iris_tab_missing, iris_tab_randclass

print("Running R script...") # Let the user know the R script is now running

library(ggplot2) # Loads the ggplot2 library for ggplot plotting features
library(dplyr) # Loads the dplyr library to group dataframe elements

# The following code will execute #1, #2, and #3 of Unit Assignment 1
sink("desc.txt") # Opens link to desc.txt
data_set <- rnorm(5000) # Creates a random data set of 
# 5000 normally distributed numbers
sample_mean <- mean(data_set) # Calculates the mean of the data set
standard_deviation <- sd(data_set) # Calculates the standard deviation of the 
# data set
normal_data_sequence <- seq(min(data_set), max(data_set), length = 50) # Creates
# a sequence of values
normal_data <- dnorm(normal_data_sequence, mean = mean(data_set),
                     sd = sd(data_set)) # Creates a normally distributed 
# distribution
sample_mean_string <- sprintf("The sample mean is %s", sample_mean) # Creates 
# string of sample mean
standard_deviation_string<- sprintf("The sample standard deviation is %s",
                                    standard_deviation) # Creates string of 
# standard deviation
print(sample_mean_string) # Print Sample Mean
print(standard_deviation_string) # Print standard deviation
sink() # Close sink connection

pdf("histo.pdf")
plot_1 <- hist(data_set, prob = TRUE, main = 
                 "Histogram With Density Curve and Normal Curve") # Titles plot
lines(density(data_set), col = 4, lwd = 2) # Plots distribution curve
lines(normal_data_sequence, normal_data, col = 2, lwd = 2) # Plots normal curve
dev.off() # Closes link to PDF file

# The following code will execute #4 of Unit Assignment 1
write.csv(mtcars,"ANOVA_Test_Data") # Writes mtcars built-in R data to a CSV 
# file (not necessary to do, but this is just to practice writing/ reading data)
anova_data <- read.csv("ANOVA_Test_Data", sep = ",") # Read in ANOVA data
cyl <- anova_data$cyl # Extracts cyl column from data
hp <- anova_data$hp # Extracts hp column from data
anova_test <- oneway.test(hp~cyl) # Conducts one-way ANOVA
anova_dataframe <- data.frame(cyl, hp) %>% # Creates dataframe with cyl and hp
  group_by(cyl) %>% # Groups dataframe by cyl
  summarise(mean_hp = mean(hp), sd_hp = sd(hp), count = n(), se_hp = 
              (sd_hp/sqrt(count))) # Summarizes statistics for each group

plot_2 <- ggplot(anova_dataframe,
                 aes(x = cyl, y = mean_hp,fill = factor(cyl))) +
  geom_bar(stat = "identity", position = position_dodge()) + 
  labs(x = "Number of Cylinders", y = "Horesepower", 
       title = "mtcars Cylinders vs Horsepower") + 
  theme(plot.title = element_text(hjust =0.5)) + 
  geom_errorbar(aes(ymin = mean_hp-se_hp, ymax = mean_hp+se_hp)) # Plots bar
# chart with error bars for each cyl group and adds labels/titles accordingly

pdf("Error_Bar_Chart.pdf")
print(plot_2) # Prints the plot
dev.off() # Close link to PDF file

anova_data <- data.frame(cyl, hp)
split_anova_data <- split(anova_data, anova_data$cyl) # Splits data by cyl value
split_anova_data_4 <- unlist(split_anova_data$`4`) # Changes data type from list
split_anova_data_6 <- unlist(split_anova_data$`6`) # Changes data type from list
split_anova_data_8 <- unlist(split_anova_data$`8`) # Changes data type from list

# Execute t tests with Benjamini-Hochberg correction
t_test_1 <- t.test(split_anova_data_4, split_anova_data_6, mu = 0, 
                   alt = "two.sided", paired = FALSE, var.equal = FALSE, 
                   conf.level = 0.95, p.adjust.method = "hochberg")
t_test_2 <- t.test(split_anova_data_4, split_anova_data_8, mu = 0, 
                   alt = "two.sided", paired = FALSE, var.equal = FALSE, 
                   conf.level = 0.95, p.adjust.method = "hochberg")
t_test_3 <- t.test(split_anova_data_6, split_anova_data_8, mu = 0, 
                   alt = "two.sided", paired = FALSE, var.equal = FALSE, 
                   conf.level = 0.95, p.adjust.method = "hochberg") 

# Execute t tests with Bonferroni correction
t_test_4 <- t.test(split_anova_data_4, split_anova_data_6, mu = 0, 
                   alt = "two.sided", paired = FALSE, var.equal = FALSE, 
                   conf.level = 0.95, p.adjust.method = "bonferroni")
t_test_5 <- t.test(split_anova_data_4, split_anova_data_8, mu = 0, 
                   alt = "two.sided", paired = FALSE, var.equal = FALSE, 
                   conf.level = 0.95, p.adjust.method = "bonferroni")
t_test_6 <- t.test(split_anova_data_6, split_anova_data_8, mu = 0, 
                   alt = "two.sided", paired = FALSE, var.equal = FALSE, 
                   conf.level = 0.95, p.adjust.method = "bonferroni") 

sink("ANOVA_test_results.txt") # Opens sink to print to .txt file
print("ANOVA test results for mtcars of cyl and hp:")
print(anova_test) # Prints ANOVA results
print("t_test results of 4 cyl and 6 cyl with Benjamini-Hochberg correction:")
# Prints description of results
print(t_test_1) # Prints t test 1 results
print("t_test results of 4 cyl and 8 cyl with Benjamini-Hochberg correction:")
# Prints description of results
print(t_test_2) # Prints t test 2 results
print("t_test results of 6 cyl and 8 cyl with Benjamini-Hochberg correction:")
# Prints description of results
print(t_test_3) # Prints t test 3 results
print("t_test results of 4 cyl and 6 cyl with Bonferroni correction:")
# Prints description of results
print(t_test_4) # Prints t test 4 results
print("t_test results of 4 cyl and 8 cyl with Bonferroni correction:")
# Prints description of results
print(t_test_5) # Prints t test 5 results
print("t_test results of 6 cyl and 8 cyl with Bonferroni correction:")
# Prints description of results
print(t_test_6) # Prints t test 6 results
print("From the data, we can see that the t test results were exactly the same
      for both theh Bonferroni and Benjamini-Hochberg corrections. Although, 
      this is not typically the case, it may have resulted here due to the 
      rather small sample size.")
sink() # Close sink connection

# The following code will execute #5 of Unit Assignment 1
sink("Non-parametric_Tests.txt")
kw_test <- kruskal.test(hp~cyl) # Performs Kruskal-Wallis test
print(kw_test)
pearson_correlation <- cor(x = anova_data,use = "complete.obs", 
                           method = "pearson") # Calculates Pearson correlation
spearman_correlation <- cor(x = anova_data,use = "complete.obs", 
                           method = "spearman") # Calculates Spearman correlation
print("Pearson Correlation Test:")
print(pearson_correlation) # Print Pearson correlation results
print("Spearman Correlation Test:")
print(spearman_correlation) # Print Spearman correlation results

ks_test <- ks.test(cyl, hp) # Performs Kolmogorov-Smirnov test
print(ks_test) # Prints KS results
plot_3 <- ggplot(anova_data, aes(x = cyl, y = hp, color = factor(cyl))) + 
  geom_point() + labs(x = "Number of Cylinders", y = "Horsepower",  
                      title = "Scatterplot of Cylinders vs Horsepower") + 
theme(plot.title = element_text(hjust =0.5)) # Plot cyl vs hp scatterplot)
pdf("Scatterplot_of_Cylinders_vs_Horsepower.pdf")
print(plot_3) # Print scatterplot
dev.off()

# The following code will execute #6 of Unit Assignment 1
linear_regression <- lm(anova_data) # Returns linear regression slope and 
# coefficient
print("Linear regression results:")
print(linear_regression) # Print linear regression results
print("The assumption of normality did not hold as the p value for the Kolmogorov-Smirnov test was 2.2e-16. This was expected as the data consists of horespower generated from the number of cylinders in an engine, which would be expected to increase as cylinders increase. However, in this case, the results form both the parametric and non-parametric tests found similar results of significant difference between the three groups. This is likely due to the large disparity in the horespower for each cylinder group. If the disparity was not as large, the tests may have differed more in their findings.")
sink() # Close connection to .txt file

pdf("Cylinder vs_Horsepower_Scatterplot.pdf") # Opens pdf to print plot
plot_4 <- ggplot(anova_data, aes(x = cyl, y = hp, color = factor(cyl))) + 
  geom_point() + labs(x = "Number of Cylinders", y = "Horsepower",  
       title = "Scatterplot of Cylinders vs Horsepower") + 
  geom_smooth(aes(x = cyl, y = hp), method = "lm", formula = y~x)
  theme(plot.title = element_text(hjust =0.5)) # Plot cyl vs hp scatterplot with linear regression line
print(plot_4) # Print plot
plot.new() # Open new plot for text command
text(0.5,0.5,"The results are consistent with what was discovered in the 
correlation analysis earlier. The data in this case follows a linear 
relationship where the horsepower variable is very highly positively correlated 
with the cylinders variable. Correlation is better suited for quickly 
determining how strong of a relationship exists between the variables while 
regression is better for representing the model and interpolating or 
     extrapolating data points.")
dev.off() # Close link to scatterplot pdf

# The following code will execute #7 and #8 of Unit Assignment 1
write.csv(diamonds,"Multivariate_Data.txt") # Writes diamonds built-in R data to a CSV 
# file (not necessary to do, but this is just to practice writing/ reading data)
multivariate_data <- read.csv("Multivariate_Data.txt", sep = ",") # Read in data
cut <- multivariate_data$cut
cut_category <- as.numeric(factor(cut, levels = c("Ideal", "Premium","Very Good", "Good", "Fair"))) # Converts strings of cut into numerical values
depth <- multivariate_data$depth
table <- multivariate_data$table
x <- multivariate_data$x
y <- multivariate_data$y
z <- multivariate_data$z
multivariate_dataframe <- data.frame(cut_category, depth, table, x, y, z) # Create dataframe with 5 variables and 1 category (cut)
dependent_vars <- cbind(depth, table, x, y, z) # Combine dependent variables into a dataframe
manova_model <- manova(formula = dependent_vars~cut_category, data = multivariate_data)
sink("MANOVA_results")
print(summary(manova_model, test = "Pillai")) # Prints a summary of a Pillai MANOVA test
print("As can be seen in the MANOVA summary, the p value is extremely small at only 2.2e-16, which I believe is the minimum value that R can report here. This suggests that the cut of the diamond is significantly influenced by the depth, table, x, y, and z variables. This intuitively makes sense as the dimensions of the diamond are what are assessed in determining the quality of the cut.")
sink() # Close connection

# The following code will execute #9 of Unit Assignment 1
sink("Multiple_Regression_Results.txt") # Opens a new file to print multiple regression results
multiple_regression_data <- lm(formula = cut_category ~ depth + table + x + y + z, data = multivariate_dataframe) # Fits
# model of cut_category to other independent variables
within_category_df <- multivariate_dataframe[multivariate_dataframe$cut_category == 1, ]
multiple_regression_data_within_category <- lm(formula = cut_category ~ depth + table + x + y + z, data = within_category_df) # Fits model within cut category
print(summary(multiple_regression_data)) # Prints summary of lm results
print(summary(multiple_regression_data_within_category)) # Prints summary of lm results
print("Per the results of the multiple regression, all of the independent variables in this scenario of depth, table, x, y, and z together have a significant impact on the cut category, but they do not have much of a different impact than one another. This aligns with my expectation that the cut category should be directly related to each of these measures. Thus, similar data may be achieved even if a few of the variables were removed. It is interesting to note that the p values for depth and table are much lower than the p values for x, y, and z. This implies that the depth and table are more influential in determining the cut category than the other independent variables. Therefore, if I were to pursue dimensionality reduction on this data, I might consider neglecting some of the variables, especially the z variable, which contributes the least to cut category.")
sink() # Close connection

# The following code will execute #10 of Unit Assignment 1
sink("ANCOVA_Results.txt") # Open .txt file for ANCOVA results
clarity <- multivariate_data$clarity
clarity_category <- as.numeric(factor(clarity, levels = c("FL", "IF","VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1", "I2", "I3"))) # Converts strings of clarity into numerical values
price <- multivariate_data$price
ancova_dataframe <- data.frame(price, cut_category, clarity_category)
ancova_data <- aov(formula = price ~ clarity_category*cut_category, data = ancova_dataframe) # Run ANCOVA test
print(summary(ancova_data)) # Print ANCOVA results
print("As expected, the clarity and cut both significantly impact the price of the diamond in their own right. However, when controlling for the clarity covariate, the cut by itself has a somewhat less significant impact on the price. In addition, the interaction of the two independent variables is quite strong with a very low p value found for this interaction." )
sink() # Close connections

# The following code will execute #11 and #12 of Unit Assignment 1
sink("iris_original_and_corrupted_results.txt")
normal_iris <- read.delim("iris_tab.txt", sep = "\t")
sepal_length <- normal_iris$sepal_length
sepal_width <- normal_iris$sepal_width
petal_length <- normal_iris$petal_length
petal_width <- normal_iris$petal_width
species <- as.numeric(factor(normal_iris$species, levels = c("setosa", "versicolor","virginica")))
normal_iris_df <- data.frame(sepal_length, sepal_width, petal_length, petal_width, species)
normal_multiple_regression <- lm(formula = species ~ sepal_length + sepal_width + petal_length + petal_width, data = normal_iris_df)
# Fits multiple regression to normal iris data
print(summary(normal_multiple_regression))

# Repeat for iris_tab_bgnoise:
bgnoise_iris <- read.delim("iris_tab_bgnoise.txt", sep = "\t")
sepal_length <- normal_iris$sepal_length
sepal_width <- normal_iris$sepal_width
petal_length <- normal_iris$petal_length
petal_width <- normal_iris$petal_width
species <- as.numeric(factor(bgnoise_iris$species, levels = c("setosa", "versicolor","virginica")))
bgnoise_iris_df <- data.frame(sepal_length, sepal_width, petal_length, petal_width, species)
bgnoise_multiple_regression <- lm(formula = species ~ sepal_length + sepal_width + petal_length + petal_width, data = bgnoise_iris_df)
# Fits multiple regression to normal iris data
print(summary(bgnoise_multiple_regression))

# Repeat for iris_tab_randclass:
randclass_iris <- read.delim("iris_tab_randclass.txt", sep = "\t")
sepal_length <- normal_iris$sepal_length
sepal_width <- normal_iris$sepal_width
petal_length <- normal_iris$petal_length
petal_width <- normal_iris$petal_width
species <- as.numeric(factor(randclass_iris$species, levels = c("setosa", "versicolor","virginica")))
randclass_iris_df <- data.frame(sepal_length, sepal_width, petal_length, petal_width, species)
randclass_multiple_regression <- lm(formula = species ~ sepal_length + sepal_width + petal_length + petal_width, data = randclass_iris_df)
# Fits multiple regression to normal iris data
print(summary(randclass_multiple_regression))

# Repeat for iris_tab_missing:
missing_iris <- read.delim("iris_tab_missing.txt", sep = "\t")
sepal_length <- normal_iris$sepal_length
sepal_width <- normal_iris$sepal_width
petal_length <- normal_iris$petal_length
petal_width <- normal_iris$petal_width
species <- as.numeric(factor(missing_iris$species, levels = c("setosa", "versicolor","virginica")))
missing_iris_df <- data.frame(sepal_length, sepal_width, petal_length, petal_width, species)
missing_multiple_regression <- lm(formula = species ~ sepal_length + sepal_width + petal_length + petal_width, data = missing_iris_df)
# Fits multiple regression to normal iris data
print(summary(missing_multiple_regression))

# Print interpretation of normal_iris vs corrupted iris data
print("It's interesting to note that the multiple regression results are exactly the same for the original iris dataset as the corrupted data sets with added noise or with missing values. The only corrupted data set I analyzed that returned different results was the random classified data. It appears the bgnoise and missing data sets have no impact on the influence of the factors on the dependent variable. However, the randomly classified data set greatly reduces the ability of any factor to predict the classification. Though, even in the randomly classified data set, the combination of all four factors is stil able to predict the classification fairly well with a p value of ~0.0003. Although, we cannot be certain the classification being predicted is correct (due to it being randomly classified) without further analysis of the data.")
sink() # Close connection

# The following code will execute #12 of Unit Assignment 1
sink("iris_purchase_results.txt")
iris_purchase <- read.csv("iris_purchase.txt", sep = "\t")
color <- as.numeric(factor(iris_purchase$color, levels = c("yellow", "orange", "red", "blue")))
attractiveness <- iris_purchase$attractiveness
review <- iris_purchase$review
likelytobuy <- iris_purchase$likelytobuy
sold <- as.numeric(factor(iris_purchase$sold, levels = c("TRUE", "FALSE")))
iris_purchase_df <- data.frame(color, attractiveness, review, likelytobuy, sold)
iris_purchase_multiple_regression <- lm(sold ~ color + attractiveness + review + likelytobuy)
print(summary(iris_purchase_multiple_regression))
print("Based on the results of the multiple regression comparing the influence of color, attractiveness, review, and likelytobuy on if a flower is sold, it can be seen that no individual factor has a significant impact on selling the flower. Although, all factors together have a strong predictability on if the flower is sold. The most significant factor on if a flower sells is the likelytobuy factor, which intuitively makes sense because this factor should aim to capture the likelihood that a flower is sold to a certain customer.")
sink() # Close connection

print("Done running R script") # Let the user know the R script is done running
