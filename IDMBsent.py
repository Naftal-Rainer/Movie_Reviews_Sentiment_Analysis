import nltk
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
import string 
from collections import Counter
from nltk.classify import NaiveBayesClassifier



# Import the movie reviews data

doc = movie_reviews.words()
print('The document has ' +str(len(doc)) + ' characters')

# Declare words with no weighty meanings as stopwords 

stop_words = stopwords.words('english') + list(string.punctuation)

# Filter meaningful words

filtered_words = [word for word in doc if word not in stop_words]
print('The new document has '+str(len(filtered_words))+ ' characters')

# Check for the frequency of occurance. First 10 

word_counter = Counter(filtered_words)
most_common = word_counter.most_common()[:10]
print(most_common)

# Sentiment Analysis

def bow_feaures(words):
    return { word:1 for word in words if word not in stop_words}

positive_reviews = movie_reviews.fileids('pos')
negative_reviews = movie_reviews.fileids('neg')

negative_features = [(bow_feaures(movie_reviews.words(fileids = [f])), 'neg')
                     for f in negative_reviews]

positive_features = [(bow_feaures(movie_reviews.words(fileids = [f])), 'pos')
                     for f in positive_reviews]

print(len(negative_features))
print(len(positive_features))

# Training the algorithm

split = 800

sentiment_classifier = NaiveBayesClassifier.train(positive_features[:split] + negative_features[:split])

acc_training = nltk.classify.util.accuracy(sentiment_classifier, positive_features[:split] + negative_features[:split] )
print(acc_training)

acc_testing = nltk.classify.util.accuracy(sentiment_classifier, positive_features[split:] + negative_features[split:] )
print(acc_testing)

# Most Informative features

print(sentiment_classifier.show_most_informative_features()
)
