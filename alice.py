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
  df.to_excel("alice.xlsx")

create_alice_sid()

def alice_compound_graph():
  # Separates plt tuple into figure and axis
  fig, ax = plt.subplots(figsize=(15,7))
  # Plots mean compound scores by chapter
  ax.plot(df.groupby('chapter', sort=False).mean()['compound_score'].index,
          df.groupby('chapter', sort=False).mean()['compound_score'].values,
          linewidth=3, color="#a53363")
  # Labels and skews x-axis
  ax.set_xticklabels(df.groupby('chapter', sort=False).mean()['compound_score'].index, rotation=30)
  # Sets y-axis max/min values
  ax.set_ylim(-0.15, 0.15)
  # Sets central x-axis line
  ax.axhline(y=0, linestyle=':', color='grey')
  # Sets title
  ax.set_title('Mean compound sentiment score of each chapter - Alice in Wonderland', fontsize=16)
  # Sets top/right borders invisible
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  # Sets bottom/left border 20% opaque
  ax.spines['bottom'].set_alpha(0.2)
  ax.spines['left'].set_alpha(0.2)
  # Creates central neutral area
  ax.fill_between(x=ax.get_xticks(), y1=-0.05, y2=0.05, color='grey', alpha=0.1)
  # Writes text for central neutral area
  ax.text(x=0, y=0.025, s='neutral')
  # Sets opacity of x/y guide lines
  ax.xaxis.grid(alpha=0.2)
  ax.yaxis.grid(alpha=0.2)
  # Saves figure
  fig.savefig("public/images/alice_compound_graph.png")
  # Shows graph in terminal
  plt.show()

# alice_compound_graph()

def alice_overall_sent_totals():
  df['pos_score'] = numpy.where(df['compound_score'] >= 0.05, 1, 0)
  df['neg_score'] = numpy.where(df['compound_score'] <= -0.05, 1, 0)
  df['neu_score'] = numpy.where((df['compound_score'] > -0.05) &
                            (df['compound_score'] < 0.05), 1, 0)
  print(f"Number of overall positive sentences is: {len(df[df['pos_score'] == 1])}")
  print(f"Number of overall neutral sentences is: {len(df[df['neg_score'] == 1])}")
  print(f"Number of overall negative sentences is: {len(df[df['neu_score'] == 1])}")

alice_overall_sent_totals()