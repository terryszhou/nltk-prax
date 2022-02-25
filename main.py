from nltk.corpus import twitter_samples
from nltk.tag import pos_tag_sents

# Renders each tweet as a string
tweets = twitter_samples.strings("positive_tweets.json")

# Renders each string into a list of parseable tokens
tweets_tokens = twitter_samples.tokenized("positive_tweets.json")

# Renders a list of tuples containing each token and its POS (Part-of-Speech)
# Adjectives are marked JJ. Nouns are NN or NNS based on plurality.
tweets_tagged = pos_tag_sents(tweets_tokens)

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
  return f"Total number of adjectives is {JJ_COUNT}.", f"Total number of nouns is {NN_COUNT}."

print(count_POS())