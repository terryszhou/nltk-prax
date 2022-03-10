import nltk
import pandas
import numpy
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

# Sample set of positive sentences.
def pos_sentences():
  i = 0
  for sent in compound_scores:
    if sent[1] < 0:
      print(sent, "\n")
      i += 1
    if i > 5:
      break

# Sample set of negative sentences.
def neg_sentences():
  i = 0
  for sent in compound_scores:
    if sent[1] < 0:
      print(sent, "\n")
      i += 1
    if i > 5:
      break

all_scores = []

def most_polar_sentences():
  for sent in compound_scores:
    all_scores.append(sent[1])
  for sent in compound_scores:
    if sent[1] == max(all_scores):
      print(f"The most positive compound score was assigned to:\n{sent}.\n")
    elif sent[1] == min(all_scores):
      print(f"The most negative compound score was assigned to:\n{sent}.\n")

# most_polar_sentences()

# Generates .xlsx of full text, plus SID values.
def create_alice_sid():
  # Create Pandas DataFrame
  df = pandas.DataFrame(compound_scores)
  # Create 'chapter' column
  df['chapter'] = numpy.where(df[0].str.find('CHAPTER') != -1,
                              "CHAPTER" + df[0].str.split('CHAPTER').str[1],
                              numpy.nan)
  # Fixes odd string bug in Chapter V
  df.loc[df['chapter'].str.contains("Advice from a Caterpillar", na=False, case=False), 'chapter'] = 'CHAPTER V.'
  # Removes periods from end of chapter strings
  df['chapter'] = df['chapter'].str.replace(".", "", regex=True)
  # Removes NA values
  df.fillna(method="ffill", inplace=True)
  # Drops all rows that simply read "Chapter..." etc, etc
  df.drop(df.loc[df[0].str.find('CHAPTER') != -1].index, inplace=True)
  # Defines column names
  df.columns = ["sentences", "compound_score", "pos_score", "neg_score", "neu_score", "chapter"]
  # Writes dataframe to Excel file
  df.to_excel("alice.xlsx")

create_alice_sid()
