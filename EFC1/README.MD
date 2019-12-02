## Linear Regression for MNIST Database (Q1)
  Here we have a linear classification model. The inputs are MNIST 28x28 matrices which are stacked into vector with size 784. All the 60000 samples then form the collumns of the data matrix X, where the first collumn is a fixed polarizing input collumn of ones. The output is calculated multiplyiing X by W, where W is the parameter matrix (785x10) which can be obtained through least squares regression. What we want is:
  
  <p align="center">
  <img src=https://github.com/andre91998/EA072/blob/master/EFC1/min.PNG />
</p>

Where y is the output matrix from the training data. We then obtain that, since we cannot guarantee that X has a complete rank: 

  <p align="center">
  <img src=https://github.com/andre91998/EA072/blob/master/EFC1/W.PNG />
</p>

Where lambda we call the "regularization coefficient", which has only one value for all ten classes. A good value for lambda should be set. The search for this value is done by seperating the data set into training and validation (two thirds, one third respectively, but this can be easily altered). We then train the model for the following values: {2<sup>-10</sup>,2<sup>-8</sup>,...,2<sup>10</sup>}. With this search, we analyze the classification error and select a interval for a "refined" search so as to obtain the optimal lambda. Finally, having obtained the optimal value for the "regularization coefficient", the final classification model can be obtained using the entire dataset (60000 samples).


## Extreme Learning Machine for MNIST Database (Q2)

## MLP for MNIST Database (Q3)

## CNN for MNIST Database (Q4)
