import nltk
import pandas
import numpy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
import matplotlib.pyplot as plt

alice_text = open("alice.txt", "r").read()

alice_sentences = nltk.sent_tokenize(alice_text)

compound_scores = []

for sentence in alice_sentences:
  compound_scores.append((sentence.replace("\n", " "),
  sid.polarity_scores(sentence)["compound"],
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

df = pandas.DataFrame(compound_scores)

# Generates .xlsx of full text, plus SID values.
def create_alice_sid():
  # Creates 'chapter' column. Fills with all strings that contain 'CHAPTER', else NA
  df['chapter'] = numpy.where(df[0].str.find('CHAPTER') != -1,
                              "CHAPTER" + df[0].str.split('CHAPTER').str[1],
                              numpy.nan)
  # Fixes odd string bug in Chapter V
  df.loc[df['chapter'].str.contains("Advice from a Caterpillar", na=False, case=False), 'chapter'] = 'CHAPTER V.'
  # Removes periods from end of chapter strings
  df['chapter'] = df['chapter'].str.replace(".", "", regex=True)
  # Removes NA values; ffill (forward fill) method propagates last valid value forward
  df.fillna(method="ffill", inplace=True)
  # Drops all rows that simply read "Chapter..." etc, etc
  df.drop(df.loc[df[0].str.find('CHAPTER') != -1].index, inplace=True)
  # Defines column names
  df.columns = ["sentences", "compound_score", "pos_score", "neg_score", "neu_score", "chapter"]
  # Writes dataframe to Excel file
  # df.to_excel("alice.xlsx")

create_alice_sid()

def create_alice_graph():
  fig, ax = plt.subplots(figsize=(15,5))
  ax.plot(df.groupby('chapter', sort=False).mean()['compound_score'].index,
          df.groupby('chapter', sort=False).mean()['compound_score'].values, linewidth=3, color="#a53363")
  ax.set_xticklabels(df.groupby('chapter', sort=False).mean()['compound_score'].index,
                    rotation=30)
  ax.set_ylim(-0.2, 0.2)
  ax.axhline(y=0, linestyle=':', color='grey')
  ax.set_title('Mean compound sentiment score of each chapter - Alice in Wonderland', fontsize=16)
  ax.spines['top'].set_visible(False)
  ax.fill_between(x=ax.get_xticks(), y1=-0.05, y2=0.05, color='grey', alpha=0.1)
  ax.spines['right'].set_visible(False)
  ax.text(x=0, y=-0.03, s='neutral')
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.yaxis.grid(alpha=0.2)
  plt.show()

create_alice_graph()
