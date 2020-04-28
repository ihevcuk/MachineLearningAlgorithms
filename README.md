# MachineLearningAlgorithms

## Logistic regression
Implementation is based on Coursera course "Machine learning". Algorithm tries to predict tennis match winner. Training data is csv file where each row is player's statistics in some match. Ending column denotes winner (0 for loser, 1 for winner). The accuracy obtained from test data is 87%.

### Details
1. Separate .csv file to training and test set
2. Group training set in N groups of training set and cross validation set
3. Use optimization functions to get optimal parameters
4. Use parameters on test set to predict if player won the match

##
This code implies that all features are decimal number and number of features is not limited by this code.

Code could be easily used for any use case where logistic regression is suitable.
