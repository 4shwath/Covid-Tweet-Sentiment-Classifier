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
def load_df(filename, cols):
    df = pd.read_csv(filename, encoding='latin-1')
    df.columns = cols
    return df

#removing unwanted columns
def remove_unwanted_cols(df, cols):
    for col in cols:
        del df[col]
    return df

train_file = 'C:\\Users\\ashwa\\Downloads\\Corona_NLP_train.csv'
test_file = 'C:\\Users\\ashwa\\Downloads\\Corona_NLP_test.csv'
columns = ['UserName', 'ScreenName', 'Location', 'TweetAt', 'OriginalTweet', 'Sentiment']
unreq_columns = ['UserName', 'ScreenName', 'Location', 'TweetAt']

#loading dataset and removing unwanted columns
train_data = load_df(train_file, columns)
train_data = remove_unwanted_cols(train_data, unreq_columns)
test_data = load_df(test_file, columns)
test_data = remove_unwanted_cols(test_data, unreq_columns)

#vectorization
x_train = vectorizer.fit_transform(train_data.OriginalTweet.values)
x_test = vectorizer.transform(test_data.OriginalTweet.values)

y_train = train_data.Sentiment.values
y_test = test_data.Sentiment.values

#Stochastic Gradient Descent Classifier
def sgdClassifier():
    clf = SGDClassifier(loss = 'hinge', penalty = 'l2', random_state=0)
    clf.fit(x_train,y_train)
    clf_pred = clf.predict(x_test)
    clf_accuracy = accuracy_score(y_test, clf_pred)

    print("Training accuracy Score: ",clf.score(x_train, y_train))
    print("Test accuracy Score: ",accuracy )
    print(metrics.classification_report(clf_pred, y_test))

#Naive-Bayes Classifier
def naiveBayesModel():
    clf = MultinomialNB()
    clf.fit(x_train, y_train)
    clf_pred = clf.predict(x_test)
    clf_accuracy = accuracy_score(y_test, clfpred)

    #accuracy, precision, f-score and recall
    print("Training accuracy score: ", clf.score(x_train, y_train))
    print("Test accuracy score: ", clf_accuracy)
    print(metrics.classification_report(clf_pred, y_test))

#Logistic Regression Model
def logisticRegression():
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    clf_pred = clf.predict(x_test)
    clf_accuracy = accuracy_score(y_test,clf_pred)
    print("Training accuracy Score: ",clf.score(x_train,y_train))
    print("Validation accuracy Score: ",clf_accuracy )
    print(metrics.classification_report(clf_pred,y_test))

sgdClassifier()
naiveBayesModel()
logisticRegression()