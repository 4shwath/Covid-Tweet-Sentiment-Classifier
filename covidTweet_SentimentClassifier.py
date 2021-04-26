import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

stop = list(stopwords.words('english'))
vectorizer = CountVectorizer(decode_error = 'replace',stop_words = stop)

#loading the datasets
def load_dataset(filename, cols):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset.columns = cols
    return dataset

#removing unwanted columns
def remove_unwanted_cols(dataset, cols):
    for col in cols:
        del dataset[col]
    return dataset

#assigning file names and columns of dataset
train_file = 'C:\\Users\\ashwa\\Downloads\\Corona_NLP_train.csv'
test_file = 'C:\\Users\\ashwa\\Downloads\\Corona_NLP_test.csv'
columns = ['UserName', 'ScreenName', 'Location', 'TweetAt', 'OriginalTweet', 'Sentiment']
unreq_columns = ['UserName', 'ScreenName', 'Location', 'TweetAt']

#loading dataset and removing unwanted columns
train_data = load_dataset(train_file, columns)
train_data = remove_unwanted_cols(train_data, unreq_columns)
test_data = load_dataset(test_file, columns)
test_data = remove_unwanted_cols(test_data, unreq_columns)

#vectorization
x_train = vectorizer.fit_transform(train_data.OriginalTweet.values)
x_test = vectorizer.transform(test_data.OriginalTweet.values)

y_train = train_data.Sentiment.values
y_test = test_data.Sentiment.values

#Stochastic Gradient Descent Classifier
def sgdClassifier():
    sgd_clf = SGDClassifier(loss = 'hinge', penalty = 'l2', random_state=0)

    sgd_clf.fit(x_train,y_train)
    
    sgd_prediction = sgd_clf.predict(x_test)
    sgd_accuracy = accuracy_score(y_test, sgd_prediction)
    print("Training accuracy Score: ",sgd_clf.score(x_train, y_train))
    print("Test accuracy Score: ",sgd_accuracy )
    print(metrics.classification_report(sgd_prediction, y_test))

#Naive-Bayes Classifier
def naiveBayesModel():
    NB_model = MultinomialNB()
    NB_model.fit(x_train, y_train)
    pred = NB_model.predict(x_test)
    
    #printing accuracy, precision, f score and recall
    NB_accuracy = accuracy_score(y_test, pred)
    print("Training accuracy score: ", NB_model.score(x_train, y_train))
    print("Test accuracy score: ", NB_accuracy)
    print(metrics.classification_report(pred, y_test))

#Logistic Regression Model
def logisticRegression():
    logreg = LogisticRegression()

    logreg.fit(x_train, y_train)
    
    logreg_prediction = logreg.predict(x_test)
    logreg_accuracy = accuracy_score(y_test,logreg_prediction)
    print("Training accuracy Score: ",logreg.score(x_train,y_train))
    print("Validation accuracy Score: ",logreg_accuracy )
    print(metrics.classification_report(logreg_prediction,y_test))

sgdClassifier()
naiveBayesModel()
logisticRegression()