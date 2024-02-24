# CSE 151A Matchmaker

Link to our Jupyter notebooks: 
* MS2: https://colab.research.google.com/drive/17Tuyk_vncUdV1RXNU-dlD5jrgjd07v-H?usp=sharing
* MS3: https://colab.research.google.com/drive/1kMx4RTVwOA_aT0HvRO7E_Wl7IKOk4dXm?usp=sharing \
Link to Dataset: https://www.kaggle.com/datasets/ulrikthygepedersen/speed-dating/data \
The csv to the out dataset is also on the repository. 

## Introduction to our Dataset
Our dataset consists of questionnaire responses from a speed dating event in 2004. The questionnaire collected basic information about participants, their hobbies/interests, how attractive they found the other person, how they rate themselves on several factors etc. 

Besides the demographic information collected from participants (e.g. age, race, etc.), the data for questionnaire responses exists mostly in the form of rating scales with different bounds (e.g. sports interest has scale of 1-10, importance of attractiveness has scale of 0-100). 

The last field is whether or not they ended up being a match, which is the target attribute. We wanted to use some of the data in here as metrics to generate a predictive model that, when given similar data about 2 different people, can predict whether or not they will be a match. 

## Limitations
Perceptions of attractiveness, interests, and other subjective measures are inherently personal and can vary widely among individuals. What one person finds attractive or interesting, another might not. This subjectivity can introduce variability into the data that may not be easily captured or generalized by a predictive model. Moreover, the data was collected from a speed dating event in 2004, which means it reflects the cultural norms, dating preferences, and societal attitudes of that specific time and place. Over time, societal norms and individual preferences evolve, potentially making the dataset less applicable to current or future scenarios, especially in different cultural contexts. Also, the findings from this dataset may not be generalizable to other forms of dating or relationship formation, such as online dating or meeting through mutual interests, where the dynamics and factors influencing a match might differ significantly.

## Implications & Applications
The implications of developing this predictive model based on the dataset could be transformed into a significant accessible dating application that replicates the speed-dating questionnaire process without program components that require a physical meet up. By analyzing questionnaire responses that include demographic information, personal interests, self-assessments, and perceptions of others, this model aims to uncover the underlying factors that contribute to successful matches. Such a model could be instrumental in enhancing the algorithms behind dating apps, enabling them to offer more personalized and accurate match recommendations.

Potential applications of this model in a dating app context are vast. For instance, the app could use the model to refine its matching algorithms, taking into account not just the superficial preferences but also the nuanced aspects of attraction and compatibility revealed by the dataset. This could lead to more meaningful connections by matching individuals based on deeper levels of compatibility, such as shared values, interests, and mutual perceptions of attractiveness. Furthermore, insights gained from the model could assist in the development of features that encourage users to explore potential matches they might not have considered otherwise, thereby expanding their horizons and increasing the chances of forming successful relationships.

Moreover, this approach opens up possibilities for dynamic feedback mechanisms where the model adjusts and learns from the outcomes of its predictions, thereby improving its accuracy over time. As the app collects more data on matches and user interactions, it could continuously refine its understanding of what makes a successful match, leading to ever-improving recommendations for its users. This not only enhances user satisfaction but also positions the app as a leader in leveraging advanced machine learning techniques to foster human connections.

---
# Initial Findings in Exploratory Data Analysis 
The first thing we noticed was that there are a lot of features (123 columns in total). We first decided to narrow down the features to those that started with a "d", i.e. those that represented the difference in answers between the two people. These datapoints reflect a direct relationship between the two participants, and are hence relevant to our project. Because we are doing more of a blind-dating matchmaker, we will not be using the features that require the participants to meet (eg. how attractive do you think your partner is, how likely do you think they will like you etc).

We can then further split the "difference" data into 2 subsections: difference in hobby ratings and difference in importance of [trait] in partner. We plotted heatmaps and pairplots, and the correlations of individual features and the match class seem relatively weak. Difference in interest in art seem to have the highest correlation, but that is still a meagre 0.038. Similarly for difference in importance of [trait], "funny" has a correlation of 0.035 with match, and that is the highest value.

It is too soon to make any concrete conclusions, but this informed us that our model will need to be more complex than a simple logistic regression model, which has drawbacks when dealing with complex patterns and non-linear relationships in data. At this stage, we are still considering random forest classifier models and multi-layer perceptron neural networks, with the latter as our more likely choice due to the sheer volume of the data we have in this data set. 

---
# Preprocessing: What is the next step?
Some preprocessing was already done at the time of submission, in that we had to drop a lot of columns for an interpretable exploratory data analysis. As the difference data and match class were both provided as objects, we converted them to numerical values. The scale for difference in importance of [trait] is much larger at 1 - 100 compared to the difference data for hobby ratings (1-10). We probably have to scale the data such that they are all normalized. Our target values are already binary, so only conversion to numerics is needed for our target class.

In the future preprocessing stages, we plan to first normalize our ratings data into scales of 0-1 proportionate to the current rating scales. Furthermore, we can consider normalizing "object" type data (e.g. strings) into discrete numerical values to incorporate demographics data as well as other columns into our model for a more multi-faceted approach. We can consider a larger 80-20 split between our training and testing data as we have enough entries to facilitate a bigger testing split.

---
# Preprocessing
It was decided that the relevant data is the difference between preferences of two people. This means that we need to drop all irrelevant columns. After dropping, the values were string values representing a range of values, so these values need to be encoded to represent actual integer values. Then, the output values in the matching column also needs to be encoded, so we used one-hot-encoding to create two columns: match and no match.

---
# Training the Model + Future Models
For our first model, we used a simple neural network to predict matches based on differences in preferences.

---
# Evaluating the Model

---
# Conclusion

---
# Future Possible Models
