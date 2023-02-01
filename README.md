# Predicting NFL Scores using Sportsbook Odds
Nick Colvin
CS 6375 Fall 2022

_Introduction_

The dataset used for this prediction problem comprises 4219 NFL games from the beginning of the 2007-2008 season to the most recently completed week (Week 13) of the 2022-2023 season. It includes regular season and playoff games. According to the source website, data is sourced from various offshore and Nevada based sportsbooks. Results are updated weekly.

_Methods_

Two models were used for this problem: MLPRegressor from scikit-learn and my own implementation. The data was reorganized into a more interpretable format for the models. Team names were encoded into integer labels, and every sample was normalized according to the test and training sets. The training set size was 75% of the overall dataset size, and was selected arbitrarily for brevity by choosing a split that performed well with both models. Hyperparameters were also chosen in an ad hoc manner: different configurations of 1-3 hidden layers consisting of 10-200 neurons using either tanh, sigmoid, or ReLU activation functions and MSE or MAE loss functions were tested. Ultimately, a single hidden layer of 100 neurons using ReLU activation was used along with a learning rate of 0.003 over 100 training iterations for the following analysis of features.

In order to judge the relative importance of each feature to the model predictions, the models were trained on the same dataset each time with a single feature removed. This process was repeated 10 times in order to calculate averages and standard deviations for the scores, records, and profits using the modified datasets.

Additionally, bar charts were generated in order to gain insight as to which features were weighed more heavily by my implementation of the model. Each bar represents a sample, and the 25 best predictions are shown. The size of each section represents the norm of the product of feature weights and sample value. These charts were generated after the performance table, and do not reflect the weights for those models. However, they provide similar examples of what a single training session might generate. Features were removed in the same order as in the table.

_Results_

Scikit-learn MLPRegressor

<img width="867" alt="mlpregressor performance table" src="https://user-images.githubusercontent.com/7663086/216105150-11c20d3a-cde0-4125-a2c5-c4e1af4f2cd2.png">

Own Implementation

<img width="876" alt="own implementation performance table" src="https://user-images.githubusercontent.com/7663086/216105331-1cfe89ca-383b-4582-bab0-a76c2f1cc5b7.png">

<img width="694" alt="all, no advantage" src="https://user-images.githubusercontent.com/7663086/216105514-7e8f2502-65bd-43a3-9630-558763d7e7a6.png">

<img width="690" alt="no name, no moneyline" src="https://user-images.githubusercontent.com/7663086/216105627-89a38c6d-5730-4f2e-9335-47fc5105e409.png">

<img width="685" alt="no spread open, no spread close" src="https://user-images.githubusercontent.com/7663086/216105733-571da31b-808c-4090-ab4c-a214145b9cb5.png">

<img width="684" alt="no ou open, no ou close" src="https://user-images.githubusercontent.com/7663086/216106358-6b3eee4e-d2a3-420e-b64b-2cbaace03dff.png">

_Analysis & Conclusions_

The most important features for my model on average were the moneyline, team IDs, closing total, and closing spread. This coincides with lower R2 values when those features were removed from the dataset. However, while scores were worse, returns were better without moneyline in the dataset. It was also a bit surprising that MAE was lower without team names, but an explanation is that the model can afford greater degrees of freedom with respect to the other features. The biggest surprise from my model was an average profit of over 20 units betting the total when the closing total was removed. This was the only positive return made by any of the models, and a reasonable standard deviation suggests it was not entirely by chance.

The MLPRegressor from Scikit-learn also appeared to place the greatest weight on the moneyline, and removing it from the dataset was the only time the R2 score was lowered. Removing the opening total seemed to have a positive effect on R2, MAE, and MSE. Moreover, removing either opening or closing total from both models made a positive impact on returns. This is peculiar behavior that could be explained by the models having greater freedom to predict different totals. Overall, both models performed reasonably similar with respect to scores and returns, which is not surprising given their similar implementations. The only notable difference was standard deviation of returns was lower for my model than Scikit-learn’s. This is likely due to Scikit-learn being more sophisticated, and using other processes like shuffling, minibatches, and regularization.

One drawback of my model was training time. This was a bottleneck in terms of getting the best performance. Ideally, these models would have been averaged over 10 sessions using 1000 iterations, however it simply took too long. There did not appear to be any evidence of overfitting even with such a large number of iterations. Another possible improvement would be some sort of loss function to factor in the returns on the training set. Lower error did not necessarily imply higher returns, so it seems like a reasonable consideration.
