import nltk
import pandas

# # TWITTER SAMPLE ANALYSIS - - - - - - - -

# Renders each tweet as a string
tweets = nltk.corpus.twitter_samples.strings("positive_tweets.json")

# Renders each string into a list of parseable tokens
tweets_tokens = nltk.corpus.twitter_samples.tokenized("positive_tweets.json")

# Renders a list of tuples containing each token and its POS (Part-of-Speech)
# Adjectives are marked JJ. Nouns are NN.
tweets_tagged = nltk.tag.pos_tag_sents(tweets_tokens)

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
tokens = nltk.word_tokenize(sentence)

# pos_tag: converts tokens into POS-tagged tuples.
tagged = nltk.pos_tag(tokens)

# ne_chunk: returns a tree of named entities.
entities = nltk.ne_chunk(tagged)

# get_chunks: parses chunk tree and extracts names as a list of strings.
def get_chunks(text):
  chunked =  nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
  continuous_chunk = []
  current_chunk = []
  for chunk in chunked:
    if type(chunk) == nltk.tree.Tree:
      current_chunk.append(" ".join([token for token, pos in chunk.leaves()]))
    if current_chunk:
      named_entity = " ".join(current_chunk)
      if named_entity not in continuous_chunk:
        continuous_chunk.append(named_entity)
        current_chunk = []
    else:
      continue
  return continuous_chunk

print(get_chunks(sentence))

# returns true if word exists in standard English corpora.
def check_word_exists(str):
  wnl = nltk.stem.WordNetLemmatizer()
  words = nltk.corpus.words
  print(wnl.lemmatize(str) in words.words())

# check_word_exists("dog")