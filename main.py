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

sentence = "The rain in Spain stays mainly in the plains."

# word_tokenize: converts string into tokens.
tokens = nltk.word_tokenize(sentence)

# pos_tag: converts tokens into POS-tagged tuples.
tagged = nltk.pos_tag(tokens)

# ne_chunk: returns a tree of named entities.
entities = nltk.ne_chunk(tagged)

# get_chunks: parses chunk tree and extracts names as a list of strings.
# Note: preferentially targets capitalized words.
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

# print(get_chunks(sentence))

# returns true if word exists in standard English corpora.
def check_word_exists(str):
  wnl = nltk.stem.WordNetLemmatizer()
  words = nltk.corpus.words
  print(wnl.lemmatize(str) in words.words())

# check_word_exists("dog")

def basic_stats(text):
  gutenberg = nltk.corpus.gutenberg
  # returns total number of characters
  num_chars = len(gutenberg.raw(text))
  # returns total word count
  num_words = len(gutenberg.words(text))
  # returns total number of sentences
  num_sents = len(gutenberg.sents(text))
  # returns total number of unique vocabulary occurrences
  num_vocab = len(set(w.lower() for w in gutenberg.words(text)))
  # returns average word length (Note: generally returns 4 - 5 regardless of text)
  avg_words = round(num_chars/num_words)
  # returns average sentence length.
  avg_sents = round(num_words/num_sents)
  # returns average unique vocabulary frequency
  avg_vocab = round(num_words/num_vocab)

  # print(num_chars, num_words, num_sents, num_vocab)
  print(avg_words, avg_sents, avg_vocab)

# basic_stats("carroll-alice.txt")

# Sentences of a Gutenberg text.
alice_sentences = nltk.corpus.gutenberg.sents("carroll-alice.txt")

# Length of longest sentence.
longest_len = max([len(s) for s in alice_sentences])

# Text of longest sentence (split into list of strings).
[s for s in alice_sentences if len(s) == longest_len]
