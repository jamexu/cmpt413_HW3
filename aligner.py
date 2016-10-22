#!/usr/bin/env python
import optparse, sys, os, logging
from collections import defaultdict

def get_args():
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--iterations", dest="iterations", default=5, help="number of training iterations")
    optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
    optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
    optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
    optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
    optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
    (opts, _) = optparser.parse_args()

    if opts.logfile:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    return opts

def get_condprob_dict(fe_data, num_iterations):

    e_count = defaultdict(float)
    fe_count = defaultdict(float)

    cond_probs = {}

    # initialization for conditional probabilities
    # this initializes them uniformly
    sys.stderr.write("initializing data")
    f_vocab = set()
    for (n, (f, e)) in enumerate(fe_data):
        f_vocab.update(set(f))

    #f_vocab_set = set(f_vocab)
    default_prob = 1.0 / len(f_vocab)


    for iteration in range(num_iterations):
        sys.stderr.write("\niteration %d-" % iteration)
        e_count.clear()
        fe_count.clear()
        for (n, (f, e)) in enumerate(fe_data):
            for f_i in set(f):

                # set up a normailization value
                normalize = 0.0
                for e_j in set(e):
                    key = (f_i, e_j)
                    normalize += cond_probs.setdefault(key, default_prob)

                # add the normalized amount to the counts
                for e_j in set(e):
                    key = (f_i, e_j)
                    added_count = cond_probs[key] / normalize
                    fe_count[key] += added_count
                    e_count[key] += added_count
            if n % 500 == 0:
                sys.stderr.write(".")

        # update the conditional probabilities for this iteration
        for key in fe_count.iterkeys():
            cond_probs[key] = fe_count[key] / e_count[key]

    return cond_probs


def main(opts):
    f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
    e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

    prob_f_given_e = get_condprob_dict(bitext, opts.iterations)



if __name__ == '__main__':
    main(get_args())

