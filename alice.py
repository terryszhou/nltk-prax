import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

text = open("alice.txt", "r").read()

sentences = nltk.sent_tokenize(text)

# print(sentences[10])

sentence = "The rain in Spain stays mainly in the plains."

def sentiment_scores(sentence):
  sid = SentimentIntensityAnalyzer()
  sentiment_dict = sid.polarity_scores(sentence)
  print(f"Overall sentiment dictionary is: {sentiment_dict}.")
  print(f"Sentence was rated as: {sentiment_dict['neg']*100}% negative")
  print(f"Sentence was rated as: {sentiment_dict['neu']*100}% neutral")
  print(f"Sentence was rated as: {sentiment_dict['pos']*100}% positive")
  print("Sentence overall rated as", end=" ")
  if sentiment_dict["compound"] >= 0.05:
    print("positive")
  elif sentiment_dict["compound"] <= -0.05:
    print("negative")
  else:
    print("neutral")

sentiment_scores(sentences[5])