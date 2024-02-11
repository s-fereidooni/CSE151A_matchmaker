# CSE 151A Matchmaker

Link to our Jupyter notebook: https://colab.research.google.com/drive/17Tuyk_vncUdV1RXNU-dlD5jrgjd07v-H?usp=sharing
Link to Dataset: https://www.kaggle.com/datasets/ulrikthygepedersen/speed-dating/data
The csv to the out dataset is also on the repository. 

## Introduction to our Dataset
Our dataset consists of questionnaire responses from a speed dating event in 2004. The questionnaire collected basic information about participants, their hobbies/interests, how attractive they found the other person, how they rate themselves on several factors etc. The last field is whether or not they ended up being a match. We wanted to use some of the data in here to generate a predictive model that, when given certain data about 2 different people, can predict whether or not they will be a match.

## Initial Findings in Data Exploration Step
The first thing we noticed was that there are a lot of features (123 columns in total). We first decided to narrow down the features to those that started with a "d", i.e. those that represented the difference in answers between the two people. These datapoints reflect a direct relationship between the two participants, and are hence relevant to our project. Because we are doing more of a blind-dating matchmaker, we will not be using the features that require the participants to meet (eg. how attractive do you think your partner is, how likely do you think they will like you etc).\
We can then further split the "difference" data into 2 subsections: difference in hobby ratings and difference in importance of [trait] in partner. We plotted heatmaps and pairplots, and the correlations of individual features and the match class seem relatively weak. Difference in interest in art seem to have the highest correlation, but that is still a meagre 0.038. Similarly for difference in importance of [trait], "funny" has a correlation of 0.035 with match, and that is the highest value.\
It is too soon to make any concrete conclusions, but this may mean that our model will need to be more complex than a linear regression model.

## Preprocessing: What is the next step
A bit of preprocessing was already done here, in that we had to drop a lot of columns. As the difference data and match class were both provided as objects, we converted them to numerical values. The scale for difference in importance of [trait] is much larger at 1 - 100 compared to the difference data for hobby ratings (1-10). We probably have to scale the data such that they are all normalized. Luckily, our target values are already converted to binary, so no encoding is needed for our class.
