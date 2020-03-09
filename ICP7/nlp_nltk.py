import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from nltk.stem import LancasterStemmer, SnowballStemmer, WordNetLemmatizer, PorterStemmer
from nltk import pos_tag, ne_chunk, ngrams

# Reading Text from the URL
wikiURL = "https://en.wikipedia.org/wiki/Google"
openURL = requests.get(wikiURL).text

# Assigning Parsed Web Page into a Variable
soup = BeautifulSoup(openURL, "html.parser")

# Kill all script and style elements
for script in soup(["script", "style"]):
    script.extract()

# get text
text = soup.body.get_text()

# Saving to a Text File
with open('Input.txt', 'w') as text_file:
    text_file.write(str(text.encode("utf-8")))


# Tokenization
######Sentence Tokenization######
sentTok = sent_tokenize(text)
print("Sentence Tokenization : \n", sentTok)

####Word Tokenization#######
tokens = [word_tokenize(t) for t in sentTok]
print ("Word Tokenization : \n", tokens)


# Stemming
#LancasterStemmer
lStem = LancasterStemmer()
print("Lancaster Stemming : \n")
for i in tokens[0:50]:
    print(lStem.stem(str(i)))

#SnowBallStemmer
sStem = SnowballStemmer('english')
print("SnowBall Stemming : \n")
for i in tokens[0:50]:
    print(sStem.stem(str(i)))

#PorterStemmer
pStem = PorterStemmer()
print("Porter Stemming : \n")
for i in tokens[0:50]:
    print(pStem.stem(str(i)))

# POS-tagging
print("Part of Speech Tagging :\n", pos_tag(word_tokenize(text)))

# Lemmatization
lemmatizer = WordNetLemmatizer()
print("Lemmatization :\n")
for tok in tokens[0:50]:
    print(lemmatizer.lemmatize(str(tok)))

# Trigram
print("Trigrams :\n")
trigram = []
for x in tokens[0:20]:
    trigram.append(list(ngrams(x, 3)))
print(trigram)

# Named Entity Recognition
print("NER : \n", ne_chunk(pos_tag(wordpunct_tokenize(str(tokens)))))
