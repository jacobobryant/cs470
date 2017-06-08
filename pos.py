from argparse import ArgumentParser
from collections import namedtuple
from fractions import Fraction as frac

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st].get(obs[0], 0), "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = max(V[t-1][prev_st]["prob"]*trans_p[prev_st].get(st, 0) for prev_st in states)
            for prev_st in states:
                if V[t-1][prev_st]["prob"] * trans_p[prev_st].get(st, 0) == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st].get(obs[t], 0)
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break
    #for line in dptable(V):
    #    print(line)
    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    return opt
    #print('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)

def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

def read_words(filename):
    Word = namedtuple('Word', 'word pos')

    with open(filename, 'r') as f:
        return [Word(*word.rsplit('_', 1)) for word in f.read().split()]

def main(training_file, testing_file):
    word_list = read_words(training_file)

    transitions = {}
    emissions = {}

    # count the transitions
    for previous, current in zip(word_list[:-1], word_list[1:]):
        if previous.pos not in transitions:
            transitions[previous.pos] = {}
        if current.pos not in transitions[previous.pos]:
            transitions[previous.pos][current.pos] = 0
        transitions[previous.pos][current.pos] += 1

    # count the emissions
    for word in word_list:
        if word.pos not in emissions:
            emissions[word.pos] = {}
        if word.word not in emissions[word.pos]:
            emissions[word.pos][word.word] = 0
        emissions[word.pos][word.word] += 1

    # normalize the counts
    for structure in [transitions, emissions]:
        for key1 in structure:
            total = sum(structure[key1][key2] for key2 in structure[key1])
            for key2 in structure[key1]:
                structure[key1][key2] = frac(structure[key1][key2], total)
    
    states = set(emissions.keys())
    start_p = {state: frac(1, len(states)) for state in states}
    
    hits = 0
    misses = 0
    testing_words = read_words(testing_file)

    observations = [word.word for word in testing_words]
    predictions = viterbi(observations, states, start_p, transitions, emissions)
    for observation, prediction in zip(testing_words, predictions):
        if observation.pos == prediction:
            hits += 1
        else:
            misses += 1

    print(hits / (hits + misses))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--training", '-f', default="training_dataset.txt")
    parser.add_argument("--testing", '-t', default="testing_dataset.txt")
    args = parser.parse_args()
    main(args.training, args.testing)
