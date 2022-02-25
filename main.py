from nltk.corpus import twitter_samples
from nltk.tag import pos_tag_sents, pos_tag
from nltk.tokenize import word_tokenize
from nltk.help import upenn_tagset
import nltk.data
import pandas

# # TWITTER SAMPLE ANALYSIS - - - - - - - -

# Renders each tweet as a string
tweets = twitter_samples.strings("positive_tweets.json")

# Renders each string into a list of parseable tokens
tweets_tokens = twitter_samples.tokenized("positive_tweets.json")

# Renders a list of tuples containing each token and its POS (Part-of-Speech)
# Adjectives are marked JJ. Nouns are NN.
tweets_tagged = pos_tag_sents(tweets_tokens)

# Iterates through POS-tagged tweets and tots up nouns and adjectives.
def count_POS():
  JJ_COUNT = 0
  NN_COUNT = 0
  for tweet in tweets_tagged:
    for pair in tweet:
      tag = pair[1]
      if tag == "JJ":
        JJ_COUNT += 1
      elif tag == "NN":
        NN_COUNT += 1
  print(f"Total number of adjectives is {JJ_COUNT}.")
  print(f"Total number of nouns is {NN_COUNT}.")

# count_POS()

# # CREATED POS GLOSSARY .CSV FILE - - - - - - - -

def create_POS_glossary():
  tagdict = nltk.data.load('help/tagsets/upenn_tagset.pickle')
  pos_list = []
  def_list = []
  ex_list = []
  for item in list(tagdict.items()):
    pos_list.append(item[0])
    def_list.append(item[1][0])
    ex_list.append(item[1][1])
  d = {"POS": pos_list, "Definition": def_list, "Examples": ex_list}
  df = pandas.DataFrame(data = d)
  df.to_excel("pos_glossary.xlsx")

# create_POS_glossary()

# # SAMPLE SENTENCE ANALYSIS - - - - - - - -

sentence = """At eight o'clock on Thursday morning Arthur didn't feel very good."""

# word_tokenize: converts string into tokens.
tokens = word_tokenize(sentence)

# pos_tag: converts tokens into POS-tagged tuples
tagged = pos_tag(tokens)
