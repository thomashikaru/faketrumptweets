import numpy as np


class MarkovModel:
    """Represents a Markov Model for a given text"""

    def __init__(self, n, text):
        """Constructor takes n-gram length and training text
        and builds dictionary mapping n-grams to
        character-probability mappings."""
        self.n = n
        self.d = {}
        for i in range(len(text)-n-1):
            ngram = text[i:i+n]
            nextchar = text[i+n:i+n+1]
            if ngram in self.d:
                if nextchar in self.d[ngram]:
                    self.d[ngram][nextchar] += 1
                else:
                    self.d[ngram][nextchar] = 1
            else:
                self.d[ngram] = {nextchar: 1}

    def test_init(self):
        for x in (list(self.d.items())[:10]):
            print(x)

    def get_next_char(self, ngram):
        """Generates a single next character based to come after the provided n-gram,
        based on the probability distribution learned from the text."""
        if ngram in self.d:
            dist = self.d[ngram]
            distlist = list(dist.items())
            keys = [k for k, _ in distlist]
            vals = [v for _, v in distlist]
            valsum = sum(vals)
            vals = list(map(lambda x: x/valsum, vals))
            return np.random.choice(keys, 1, p=vals)[0]
        else:
            # this should never happen if start string n-gram exists in train text
            return np.random.choice([x for x in "abcdefghijklmnopqrstuvwxyz"])

    def get_n_chars(self, length, ngram):
        """Returns a generated sequence of specified length,
        using the given n-gram as a starting seed."""
        s = []
        for i in range(length):
            nextchar = self.get_next_char(ngram)
            ngram = ngram[1:]+nextchar
            s.append(nextchar)
        return ''.join(s)


def main():
    """Load the data, build the Markov Model, and generate an example."""
    f = open("trump_tweets_all.txt")
    text = " ".join(f.readlines())
    text = " ".join(text.split())
    text = text.encode("ascii", errors="ignore").decode()
    text.replace("&amp;", "&")
    f.close()
    ngram_length = 4
    tweet_length = 280
    model = MarkovModel(ngram_length, text)
    initial_ngram = "Hill"[:ngram_length]
    print(initial_ngram + model.get_n_chars(tweet_length, initial_ngram))


if __name__ == "__main__":
    main()

