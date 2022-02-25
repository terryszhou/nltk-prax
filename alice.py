import nltk
import pandas
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

alice_text = open("alice.txt", "r").read()

alice_sentences = nltk.sent_tokenize(alice_text)

# Takes sentence as string and parses using Vader Sentiment Intensity Analyzer.
def sentiment_scores(sentence):
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

compound_scores = []

for sentence in alice_sentences:
  compound_scores.append((sentence.replace("\n", " "), sid.polarity_scores(sentence)["compound"],
  sid.polarity_scores(sentence)["pos"],
  sid.polarity_scores(sentence)["neg"],
  sid.polarity_scores(sentence)["neu"]))

# Skips Table of Contents, etc.
filtered_scores = compound_scores[16:]

# Sample set of positive sentences.
def pos_sentences():
  i = 0
  for sent in filtered_scores:
    if sent[1] < 0:
      print(sent, "\n")
      i += 1
    if i > 5:
      break

# Sample set of negative sentences.
def neg_sentences():
  i = 0
  for sent in filtered_scores:
    if sent[1] < 0:
      print(sent, "\n")
      i += 1
    if i > 5:
      break

all_scores = []

def most_polar_sentences():
  for sent in filtered_scores:
    all_scores.append(sent[1])
  for sent in filtered_scores:
    if sent[1] == max(all_scores):
      print(f"The most positive compound score was assigned to:\n{sent}.")
    elif sent[1] == min(all_scores):
      print(f"The most negative compound score was assigned to:\n{sent}.")

# most_polar_sentences()

# Generates .xlsx of full text, plus SID values.
def create_alice_sid():
  d = {"Sentence": [], "Compound": [], "Positive": [], "Negative": [], "Neutral": []}
  for sent in filtered_scores:
    d["Sentence"].append(sent[0])
    d["Compound"].append(sent[1])
    d["Positive"].append(sent[2])
    d["Negative"].append(sent[3])
    d["Neutral"].append(sent[4])
  df = pandas.DataFrame(data = d)
  df.to_excel("alice.xlsx")

# create_alice_sid()
