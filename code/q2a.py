import rnn
import csv
from utils import *
from rnnmath import *

data_folder = "data/"
train_size = 1000
dev_size = 1000
vocab_size = 2000

with open('data/q2a_exp.csv', 'r') as f:
    reader = csv.reader(f)
    experiments = list(reader)

# get the data set vocabulary
vocab = pd.read_table(
    data_folder + "/vocab.wiki.txt",
    header=None,
    sep="\s+",
    index_col=0,
    names=['count', 'freq'],
)
num_to_word = dict(enumerate(vocab.index[:vocab_size]))
word_to_num = invert_dict(num_to_word)

# calculate loss vocabulary words due to vocab_size
fraction_lost = fraq_loss(vocab, word_to_num, vocab_size)
print("Retained %d words from %d (%.02f%% of all tokens)\n" %
        (vocab_size, len(vocab), 100 * (1 - fraction_lost)))

docs = load_lm_dataset(data_folder + '/wiki-train.txt')
S_train = docs_to_indices(docs, word_to_num, 1, 1)
X_train, D_train = seqs_to_lmXY(S_train)

# Load the dev set (for tuning hyperparameters)
docs = load_lm_dataset(data_folder + '/wiki-dev.txt')
S_dev = docs_to_indices(docs, word_to_num, 1, 1)
X_dev, D_dev = seqs_to_lmXY(S_dev)

X_train = X_train[:train_size]
D_train = D_train[:train_size]
X_dev = X_dev[:dev_size]
D_dev = D_dev[:dev_size]

# q = best unigram frequency from omitted vocab
# this is the best expected loss out of that set
q = vocab.freq[vocab_size] / sum(vocab.freq[vocab_size:])


for e in experiments:
    no = e[0]
    hdim = int(e[1])
    lookback = int(e[2])
    lr = float(e[3])

    ### Implemented model training ----------------------------------------------

    model = rnn.RNN(vocab_size, hdim, vocab_size)
    model.train(X_train, D_train, X_dev, D_dev, epochs = 10, back_steps = lookback, learning_rate= lr)

    np.save("../results/rnn_{}.U.npy".format(no), model.U)
    np.save("../results/rnn_{}.V.npy".format(no), model.V)
    np.save("../results/rnn_{}.W.npy".format(no), model.W)

    ### -------------------------------------------------------------------------
    run_loss = -1
    adjusted_loss = -1

    print("Unadjusted: %.03f" % np.exp(run_loss))
    print("Adjusted for missing vocab: %.03f" % np.exp(adjusted_loss))