import nltk
import pandas
import numpy
import string
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

def alice_chapter_sentiment_graph():
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

# alice_chapter_sentiment_graph()

def alice_overall_sent_totals():
  df['pos_score'] = numpy.where(df['compound_score'] >= 0.05, 1, 0)
  df['neg_score'] = numpy.where(df['compound_score'] <= -0.05, 1, 0)
  df['neu_score'] = numpy.where((df['compound_score'] > -0.05) &
                            (df['compound_score'] < 0.05), 1, 0)
  print(f"Number of overall positive sentences is: {len(df[df['pos_score'] == 1])}")
  print(f"Number of overall neutral sentences is: {len(df[df['neg_score'] == 1])}")
  print(f"Number of overall negative sentences is: {len(df[df['neu_score'] == 1])}")

# alice_overall_sent_totals()

def alice_chap_sent_count_graph():
  fig, ax = plt.subplots(figsize=(15,7))
  ax.bar(x=df.groupby('chapter').nunique()['sentences'].index, 
          height=df.groupby('chapter').nunique()['sentences'].values,
          color="#a53363")
  ax.set_xticklabels(df['chapter'].unique(), rotation=30)
  ax.set_title('Sentence count per chapter - Alice in Wonderland', fontsize=16)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_alpha(0.2)
  ax.spines['left'].set_alpha(0.2)
  ax.yaxis.grid(alpha=0.2)
  fig.savefig("public/images/alice_chap_sent_count_graph.png")
  plt.show()

# alice_chap_sent_count_graph()

def alice_chap_vader_sent_graph():
  alice_overall_sent_totals()
  fig, ax = plt.subplots(figsize=(15,7))
  ax.bar(x=df[df['pos_score'] == 1].groupby('chapter').nunique()['sentences'].index, 
        height=df[df['pos_score'] == 1].groupby('chapter').nunique()['sentences'].values / df.groupby('chapter').nunique()['sentences'] * 100, alpha=0.8,
        label='positive', color='purple')
  ax.bar(x=df[df['neu_score'] == 1].groupby('chapter').nunique()['sentences'].index, 
        height=df[df['neu_score'] == 1].groupby('chapter').nunique()['sentences'].values / df.groupby('chapter').nunique()['sentences'] * 100, alpha=0.3,
        bottom=df[df['pos_score'] == 1].groupby('chapter').nunique()['sentences'].values / df.groupby('chapter').nunique()['sentences'] * 100,
        label='neutral', color='purple')
  ax.bar(x=df[df['neg_score'] == 1].groupby('chapter').nunique()['sentences'].index, 
        height=df[df['neg_score'] == 1].groupby('chapter').nunique()['sentences'].values / df.groupby('chapter').nunique()['sentences'] * 100,
        bottom=df[df['neu_score'] == 1].groupby('chapter').nunique()['sentences'].values / df.groupby('chapter').nunique()['sentences'] * 100 +
        df[df['pos_score'] == 1].groupby('chapter').nunique()['sentences'].values / df.groupby('chapter').nunique()['sentences'] * 100,
        label='negative', color='red')
  ax.legend(frameon=False, bbox_to_anchor=(1, 0.5))
  ax.set_xticklabels(df['chapter'].unique(), rotation=30)
  ax.set_title('Sentence sentiment by chapter - Alice in Wonderland', fontsize=16)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_alpha(0.2)
  ax.spines['left'].set_alpha(0.2)
  ax.set_ylabel('[%]')
  ax.yaxis.grid(alpha=0.2)
  fig.savefig("public/images/alice_chap_vader_sent_graph.png")
  plt.show()

# alice_chap_vader_sent_graph()

def alice_avg_sent_length_graph():
  df['sentences_length'] = df['sentences'].apply(lambda x: len(x.split(' ')))
  fig, ax = plt.subplots(figsize=(15,7))
  ax.bar(x=df.groupby('chapter').nunique()['sentences_length'].index, 
        height=df.groupby('chapter').nunique()['sentences_length'].values,
        alpha=0.7, color='#a53363')
  ax.set_xticklabels(df['chapter'].unique(), rotation=30)
  ax.set_title('Avg. sentence length - Alice in Wonderland', fontsize=16)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_alpha(0.2)
  ax.spines['left'].set_alpha(0.2)
  ax.yaxis.grid(alpha=0.2)
  ax.set_ylabel('Word Count')
  fig.savefig("public/images/alice_avg_sent_length_graph.png")
  plt.show()

# alice_avg_sent_length_graph()

def get_chunks(text):
  entities = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
  entity_list = []
  current_entity = []
  for entity in entities:
    if type(entity) == nltk.tree.Tree:
      current_entity.append(" ".join([token for token, pos in entity.leaves()]))
    if current_entity:
      named_entity = " ".join(current_entity)
      if named_entity not in entity_list:
        entity_list.append(named_entity)
        current_entity = []
    else:
      continue
  return entity_list

def clean_sentences():
  string.punctuation += '“”‘—'
  string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~“”‘—'
  translator = str.maketrans('','',string.punctuation)
  df['cleaned_sentences'] = df['sentences'].apply(lambda x: x.translate(translator))
  df['cleaned_sentences'] = df['cleaned_sentences'].str.strip()
  print(df['cleaned_sentences'].loc[50])

  df['tagged_sent'] = df['cleaned_sentences'].str.split(' ').apply(lambda x: nltk.pos_tag(x))
  print(df.loc[50]['tagged_sent'])

  df['ne'] = df['tagged_sent'].apply(lambda x: nltk.ne_chunk(x, binary=True))
  print(df.loc[50]['ne'])

  df['named_entities'] = df['cleaned_sentences'].apply(lambda x: get_chunks(x))
  print(df.loc[50]['named_entities'])

clean_sentences()