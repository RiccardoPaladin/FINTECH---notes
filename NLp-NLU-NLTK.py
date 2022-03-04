#The code below allow to open a tab and download the package needed
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()




import nltk
nltk.download('punkt')
text = 'this is an sample.'
tokens = nltk.word_tokenize(text)   #split also the dot at the end and present a list for each element
tokens                              #if the document is long can be done sentence tonkenization adn then do word tokenizarion for each sentence with a loop
text2 = "this isn't a good test"
tokens1 = nltk.word_tokenize(text2)   #split is and n't
tokens1
tag = nltk.pos_tag(tokens)       #it gives for each element of the list which kind of word is, like noun berb or abjective  POS= part of speech
tag   #JJ= objective   VB=verb VBD= past verb  (look at pos labels list)


from nltk.util import ngrams

n_grams = ngrams(tokens, 3 )
n_grams       #group the sentence slitted in list in groups of 3 words

from nltk.stem.porter import *
stemmer = PorterStemmer()
stemmer.stem("jumps")     #it gives the stem of the word (jump) so the base form

from nltk.stem import WordNetLemmatizer
nltk.download('WordNet')
lemmatizer= WordNetLemmatizer()    #decognugate vrebs, from past to infinite
lemmatizer.lemmatize('was')       #it should give be but it give wa
#the lemmatizer assumes that a single word is a noun and not a verb, so it needs a phrase or a part of speech
lemmatizer.lemmatize('jumped', 'v') #this gives jumb since i specified that is a verb
lemmatizer.lemmatize('was', 'v')  #now it give be
#to match two sentences but are using different time-verb we can pass all them in the lemmatizer and eliminate the past sentences to have all the presen t
#get the sentence, make them a world token, bring them in the part of the speech and then pass through the lemmatizer
#to make the process simpler we can use textblob with pip install then


# -m textblob.download_corporate lite
from textblob import TextBlob   #with the capital letter to use the class
blob = TextBlob(""" To understand why the Arab language is written from right to left, a review of the history of language is in order. The first evidence of the existence of a writing system dates back to the 4th millennium BCE, around -3500. Cuneiform writing was used in Mesopotamia""")
blob[0:20]   #to bring the first 20 characters
blob.tags  #all the object separed in tuples, it is a list
blob.tags[5]  #to take the 5 elements
bolb.noun_phrases  #it groups worlds if together have ameaning
bolb.noun_phrases.count('world')    #count the number the world is repeted but only if is characterized as a single world

blob = TextBlob('This is a great string')
blob.sentiment    #polarity: compunds score from -1 to 1 (negative to positive)
#subjectivity : from 0 to 1 how much is subjective thoughts, 0 pure fact, 1 very subjective
blob.sentences  #each sentences is a textblob
for s in blob.sentences:
    print(s.entiment)
#polarity and subjectivity for each sentences
#useful for speach for investors, it takes all the negativity
blob.words
blob.words[0] #is the word is plular
blob.words[0].singularize()  #it gives the singular
blob.words[0].pluralize()  #from singular to plular
blob.detect_lenguage
blob.translate(from_lang= 'en' ,to = 'it')


import spacy
#spacy is useful for word vectorization
nlp = spacy.load("file")
token1 = nlp('red')
token1.vector  #series of number useful only for red, they are associated with red 300X300 vector
token2 = nlp('yellow')
similarity = token1.similarity(token2)  #rank based on the cosine of the angol amon the two vectors
#we can use math from word, it can be operations among word since it refers to vectors of the same size so can be multiplied or subtracted
#if we add al the vectors together we can loose the meaning of the sentence
#we cannot can go from vector to word



#pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


sia = SentimentIntensityAnalizer
sia.polarity_scores("this is a string.")   #it gives a dictionary with positive, negative compound words based on the sense of the phrase. it gives the percentage of positivity and negativity
#it recognize the meaning or the words and also things like esclamation points
#when we change the sententence look at the compound
#it recognize also smile :)
#it didn't work thta much because cannot get the sarcasm and intonations
#it can be used on influence people and create a bot to invest on stocks that are mentioned by very influence people


