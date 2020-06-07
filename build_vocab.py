import pickle
import argparse

def main(args):
    with open(args.text_path, 'r') as f:
    w = f.readlines()
    w_dict = {}
    for l in w:
    if l.split('#')[0] in list(w_dict.keys()):
        w_dict[l.split('#')[0]].append(l.split('#')[1][3:-4])
    else:
        w_dict[l.split('#')[0]] = [l.split('#')[1][3:-4]]

    vocab = {}
    for li in w_dict.values():
    for sen in li:
        for word in sen.split(' '):
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
    words = [word for word, count in vocab.items() if count >= args.word_freq]

    class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.id2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if word not in self.word2idx:
        self.word2idx[word] = self.idx
        self.id2word[self.idx] = word
        self.idx += 1
    
    def __call__(self, word):
        if not word in self.word2idx:
        return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    for word in words:
    vocab.add_word(word)
    
    vocab_path = args.vocab_path    # path to store the vocabulary file

    # dump the vocabulary object into a pickle file
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_path', type=str,
                        default='data/annotations/captions_train2014.json',
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--word_freq', type=int, default=4,
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)