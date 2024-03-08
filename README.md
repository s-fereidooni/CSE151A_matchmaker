# CSE 151A Matchmaker

Link to our Jupyter notebooks: 
* MS2: [https://colab.research.google.com/drive/17Tuyk_vncUdV1RXNU-dlD5jrgjd07v-H?usp=sharing](https://colab.research.google.com/drive/17Tuyk_vncUdV1RXNU-dlD5jrgjd07v-H?usp=sharing)
* MS3: [https://colab.research.google.com/drive/1YiTdME0R0TYR0eGEjBOzBj24NQI2w02b?usp=sharing](https://colab.research.google.com/drive/1z7sOpXUmxRID6vVpQlkCrXNFihgwi_LU?usp=sharing) 
* MS4: [https://colab.research.google.com/drive/1Wfwq86W_myw6D7NHexcekC2eitLdB0yb?usp=sharing](https://colab.research.google.com/drive/1Wfwq86W_myw6D7NHexcekC2eitLdB0yb?usp=sharing) 

Link to Dataset: https://www.kaggle.com/datasets/ulrikthygepedersen/speed-dating/data \
The csv to the out dataset is also on the repository. 

# Milestone 2
### Introduction to our Dataset
Our dataset consists of questionnaire responses from a speed dating event in 2004. The questionnaire collected basic information about participants, their hobbies/interests, how attractive they found the other person, how they rate themselves on several factors etc. 

Besides the demographic information collected from participants (e.g. age, race, etc.), the data for questionnaire responses exists mostly in the form of rating scales with different bounds (e.g. sports interest has scale of 1-10, importance of attractiveness has scale of 0-100). 

The last field is whether or not they ended up being a match, which is the target attribute. We wanted to use some of the data in here as metrics to generate a predictive model that, when given similar data about 2 different people, can predict whether or not they will be a match. 

### Limitations
Perceptions of attractiveness, interests, and other subjective measures are inherently personal and can vary widely among individuals. What one person finds attractive or interesting, another might not. This subjectivity can introduce variability into the data that may not be easily captured or generalized by a predictive model. Moreover, the data was collected from a speed dating event in 2004, which means it reflects the cultural norms, dating preferences, and societal attitudes of that specific time and place. Over time, societal norms and individual preferences evolve, potentially making the dataset less applicable to current or future scenarios, especially in different cultural contexts. Also, the findings from this dataset may not be generalizable to other forms of dating or relationship formation, such as online dating or meeting through mutual interests, where the dynamics and factors influencing a match might differ significantly.

### Implications & Applications
The implications of developing this predictive model based on the dataset could be transformed into a significant accessible dating application that replicates the speed-dating questionnaire process without program components that require a physical meet up. By analyzing questionnaire responses that include demographic information, personal interests, self-assessments, and perceptions of others, this model aims to uncover the underlying factors that contribute to successful matches. Such a model could be instrumental in enhancing the algorithms behind dating apps, enabling them to offer more personalized and accurate match recommendations.

Potential applications of this model in a dating app context are vast. For instance, the app could use the model to refine its matching algorithms, taking into account not just the superficial preferences but also the nuanced aspects of attraction and compatibility revealed by the dataset. This could lead to more meaningful connections by matching individuals based on deeper levels of compatibility, such as shared values, interests, and mutual perceptions of attractiveness. Furthermore, insights gained from the model could assist in the development of features that encourage users to explore potential matches they might not have considered otherwise, thereby expanding their horizons and increasing the chances of forming successful relationships.

Moreover, this approach opens up possibilities for dynamic feedback mechanisms where the model adjusts and learns from the outcomes of its predictions, thereby improving its accuracy over time. As the app collects more data on matches and user interactions, it could continuously refine its understanding of what makes a successful match, leading to ever-improving recommendations for its users. This not only enhances user satisfaction but also positions the app as a leader in leveraging advanced machine learning techniques to foster human connections.

---
### Initial Findings in Exploratory Data Analysis 
The first thing we noticed was that there are a lot of features (123 columns in total). We first decided to narrow down the features to those that started with a "d", i.e. those that represented the difference in answers between the two people. These datapoints reflect a direct relationship between the two participants, and are hence relevant to our project. Because we are doing more of a blind-dating matchmaker, we will not be using the features that require the participants to meet (eg. how attractive do you think your partner is, how likely do you think they will like you etc).

We can then further split the "difference" data into 2 subsections: difference in hobby ratings and difference in importance of [trait] in partner. We plotted heatmaps and pairplots, and the correlations of individual features and the match class seem relatively weak. Difference in interest in art seem to have the highest correlation, but that is still a meagre 0.038. Similarly for difference in importance of [trait], "funny" has a correlation of 0.035 with match, and that is the highest value.

It is too soon to make any concrete conclusions, but this informed us that our model will need to be more complex than a simple logistic regression model, which has drawbacks when dealing with complex patterns and non-linear relationships in data. At this stage, we are still considering random forest classifier models and multi-layer perceptron neural networks, with the latter as our more likely choice due to the sheer volume of the data we have in this data set. 

---
### Preprocessing: What is the next step?
Some preprocessing was already done at the time of submission, in that we had to drop a lot of columns for an interpretable exploratory data analysis. As the difference data and match class were both provided as objects, we converted them to numerical values. The scale for difference in importance of [trait] is much larger at 1 - 100 compared to the difference data for hobby ratings (1-10). We probably have to scale the data such that they are all normalized. Our target values are already binary, so only conversion to numerics is needed for our target class.

In the future preprocessing stages, we plan to first normalize our ratings data into scales of 0-1 proportionate to the current rating scales. Furthermore, we can consider normalizing "object" type data (e.g. strings) into discrete numerical values to incorporate demographics data as well as other columns into our model for a more multi-faceted approach. We can consider a larger 80-20 split between our training and testing data as we have enough entries to facilitate a bigger testing split.

---
# Milestone 3
### Preprocessing
It was decided that the relevant data is the difference between preferences of two people. This means that we need to drop all irrelevant columns. After dropping, the values were string values representing a range of values, so these values need to be encoded to represent actual integer values. Then, the output values in the matching column also needs to be encoded, so we first converted the string values to 0's and 1's, then used one-hot-encoding to create two columns: match and no match.

---
### Our Model + Future Models
For our first model, we used a neural network to predict matches based on differences in preferences.  We are considering using Random Forest Classifier and a Decision Tree as our next models. A Random Forest Classifier is less prone to overfitting and can give useful insights into the relevance/importance of the features in our data. Random Forest Classifiers also excel with categorical data. While our dataset has numerical values, it is more along the lines of categorical data since the numerical values represent categories of differences rather than actual continuous values. On the other hand, a Decision Tree is a bit more likely to overfit if it is deep, but it is easily interpretable and does not make assumptions about the data distribution. Decision Trees can help with understanding how different data values affect the output, and are also less computaionally heavy, which should result in faster training and classifying times.

---
### Evaluating the Model
It seems that our model is fitting well--perhaps even somewhat of a best-fit! The trained model boasts a train MSE of 0.13509749 and a test MSE of 0.1443914. When observing the MSE of the model as it trains, it decreases from 0.169 to 0.101 and the val_mse decreases concurrently from 0.160 to 0.106, and the two eventually converge. It is expected that the test and val have a slightly higher MSE as the train MSE. Additionally, the accuracy starts at 0.82 and increases to end at 0.87, as the validation accuracy also follows a similar increasing trend starting at 0.85 to 0.86. 

![image](https://github.com/s-fereidooni/CSE151A_matchmaker/assets/107325918/4219dbe7-27bb-4cb3-aad2-a3eaff4aca26)

On the fitting graph, our model falls near the line of best fit since the test MSE and train MSE both decrease over time and somewhat converge, but the test MSE is still higher than that of the train. 

It is important to note that the model is still not perfect, and requires further exploration/exploration of different models. Currently, as seen in our classification report, our model is relatively good at predicting non-matches in both precision and recall, but does not do as well in predicting matches. Our future models, the Random Forest Classifier and the Decision Tree, will hopefully resolve this match-predicting issue. 

---
### Conclusion

---

# Milestone 4
### Preprocessing
#### 1. Evaluate your data, labels and loss function. Were they sufficient or did you have have to change them.
In milestone 2 we cleaned our data of 8378 observations with 123 features to 56 features. We found this to work great in milestone 3 as well as the current milestone so we kept it as is with 56 features. Furthermore we value encoded the data as the original data was composed of strings of numerical ranges. We ended up averaging the ranges which would be the encoding for that feature, for example ‘[3-5]’ would turn into 4. This made the values a lot easier to work with for our model.

### Evaluating the Model
#### 3. Evaluate your model compare training vs test error
For this milestone we implemented a decision tree for our classification task. In order to find the best hyperparameters used grid search which yielded the following hyperparameters:
 - criterion: entropy
 - max_depth: 5
 - min_samples_leaf: 2
 - min_samples_split: 4

Which resulted in the following training and test accuracies:
 - Training set accuracy: 0.8433 
 - Test set accuracy: 0.8455

We can see that the training accuracy and test accuracy were very similar suggesting that the model was a good fit.


#### 4. Where does your model fit in the fitting graph, how does it compare to your first model?
As seen by the accuracies above, our model was a good fit for the data as our training and test accuracies were very similar. Furthermore upon inspection of the tree diagram we saw that there were 31 leaf nodes and 5 layers which we believed to be a good amount in relation to the number of observations. We also saw that each leaf node had at a minimum 9 observations which is not low enough to suggest overfitting. Grid search assisted in ensuring that our model would be a good fit for our data as it searched for the best hyperparameters and did 5 fold cross validation. In the last milestone we were able to achieve 0.87 accuracy on the training set and 0.86 accuracy on the test set by training a neural network. Although the decision tree did worse than the neural network, we are happy to find that it came within 2-3% accuracy of the neural network.

### Our Model + Future Models
#### 5. Did you perform hyper parameter tuning? K-fold Cross validation? Feature expansion? What were the results?
We performed hyper parameter tuning and K-fold Cross validation using Grid Search, but the model’s accuracy was not increasing even with the best found model. Our original accuracy was at around 0.85 and regardless of what was changed, even the best possible model we found was not able to exceed that. 

#### 6. What is the plan for the next model you are thinking of and why?
The next model we are thinking of is an SVM. This is because it’s effective in even nonlinear classification and where there's a lot of dimensions of data without much data. In this case, we have various dimensions to our data, as there are a lot of criteria for choosing/simulating the choosing of finding a partner. 

#### 7. Conclusion section: What is the conclusion of your 2nd model? What can be done to possibly improve it? How did it perform to your first and why?
The second model worked well enough, with around an 85% accuracy, which is similar to our previous model's accuracy as well. However, we don't think it's very possible to improve the decision tree classifier we used more than we already have. We used hyper parameter tuning to improve the model's accuracy, but it stayed around the same number. Pruning the tree in different ways could possibly improve the model's accuracy as well. We've tried to do Cost-complexity pruning, however, it made no effect on our model, but we could potentially try other pruning methods like weakest link pruning to better the model. 


