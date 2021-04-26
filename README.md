**Covid-Tweet-Sentiment-Classifier**
 - uses Covid-related Tweets Sentiment Analysis dataset from Kaggle.
 - classifies the tweets according to the given sentiment analysis rated from Extremely Positive to Extremely Negative or 1 to 5.
 - pandas, nltk and sklearn libraries for creating models, reading from csv files and preprocessing of data.

<pre>
->Using sklearn to create Naive-Bayes, Logistic Regression and Stochastic Gradient Descent classifiers
->Training the models to find the training and test accuracies as well as precision, recall, f1-score and support
->The program's purpose is to find the most efficient among the three models.
<pre>

The output obtained was as follows:

- **Stochastic Gradient Descent Classifier**
Training accuracy Score:  0.9164905119420754
Test accuracy Score:  0.5650342285413376
                    precision    recall  f1-score   support

Extremely Negative       0.63      0.58      0.61       636
Extremely Positive       0.70      0.65      0.67       642
          Negative       0.42      0.52      0.47       849
           Neutral       0.73      0.60      0.66       747
          Positive       0.49      0.50      0.50       924

          accuracy                           0.57      3798
         macro avg       0.59      0.57      0.58      3798
      weighted avg       0.58      0.57      0.57      3798
<pre>

- **Naive-Bayes Classifier**
<pre>
Training accuracy score:  0.770682994387346
Test accuracy score:  0.42759347024749866
                    precision    recall  f1-score   support


Extremely Negative       0.21      0.64      0.32       198
Extremely Positive       0.24      0.74      0.36       194
          Negative       0.54      0.41      0.47      1365
           Neutral       0.20      0.68      0.31       182
          Positive       0.70      0.36      0.47      1859

          accuracy                           0.43      3798
         macro avg       0.38      0.57      0.39      3798
      weighted avg       0.57      0.43      0.45      3798
<pre>

- **Logistic Regression Classifier**
<pre>
Training accuracy Score:  0.9553660373690989
Validation accuracy Score:  0.6024223275408109
                    precision    recall  f1-score   support

Extremely Negative       0.53      0.65      0.59       487
Extremely Positive       0.60      0.71      0.65       503
          Negative       0.56      0.54      0.55      1067
           Neutral       0.73      0.65      0.68       696
          Positive       0.62      0.56      0.59      1045

          accuracy                           0.60      3798
         macro avg       0.61      0.62      0.61      3798
      weighted avg       0.61      0.60      0.60      3798
<pre>
