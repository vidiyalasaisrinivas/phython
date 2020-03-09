from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
#from sklearn.neighbors import KNeighborsClassifier


twenty_train = fetch_20newsgroups(categories=['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware','comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles'], shuffle=True)
stop_words = set(stopwords.words('english'))


tfidf_Vect = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf=SVC()
#clf = svm.SVC(kernel='linear', C=1,gamma=0).fit(X_train_tfidf, twenty_train.target)
#clf = svm.SVC()

clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)


predicted = clf.predict(X_test_tfidf)

score = metrics.accuracy_score(twenty_test.target, predicted)
print(score)