from argparse import ArgumentParser
from collections import namedtuple
from fractions import Fraction as frac
from heapq import nlargest
import random

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

def read_sentences(filename):
    Word = namedtuple('Word', 'word pos')

    with open(filename, 'r') as f:
        return [[Word(*word.rsplit('_', 1)) for word in sentence.split()]
                for sentence in f.read().lower().splitlines()]

def print_result(r):
    print(' '.join(word.word for word in r.test_data))
    print("    actual: " + ' '.join(word.pos for word in r.test_data))
    print("prediction: " + ' '.join(pos for pos in r.predictions))
    print("     score: {}".format(r.score))
    print()

def print_analysis(transitions, emissions):
    print("\nTRANSITIONS")
    for pos in transitions:
        most_likely  = max(transitions[pos], key=lambda key: transitions[pos][key])
        least_likely = min(transitions[pos], key=lambda key: transitions[pos][key])
        print("{}: {}, {}".format(pos, most_likely, least_likely))

    print("\nEMISSIONS")
    for pos in emissions:
        most_likely =  max(emissions[pos], key=lambda key: emissions[pos][key])
        least_likely = min(emissions[pos], key=lambda key: emissions[pos][key])
        print("{}: {}, {}".format(pos, most_likely, least_likely))
    print()

def main(training_file, testing_file):
    sentence_list = read_sentences(training_file)

    transitions = {}
    emissions = {}

    # count the transitions
    for sentence in sentence_list:
        for previous, current in zip(sentence[:-1], sentence[1:]):
            if previous.pos not in transitions:
                transitions[previous.pos] = {}
            if current.pos not in transitions[previous.pos]:
                transitions[previous.pos][current.pos] = 0
            transitions[previous.pos][current.pos] += 1

        # count the emissions
        for word in sentence:
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

    print_analysis(transitions, emissions)
    
    states = set(emissions.keys())
    start_p = {state: frac(1, len(states)) for state in states}
    
    total_hits = 0
    total_misses = 0
    test_sentences = read_sentences(testing_file)
    results = []
    Result = namedtuple("Result", "test_data predictions score")

    # get results from viterbi
    for i, sentence in enumerate(test_sentences):
        print("\r{}%".format((i*100) // len(test_sentences)), end="")

        hits = 0
        misses = 0
        observations = [word.word for word in sentence]
        predictions = viterbi(observations, states, start_p, transitions, emissions)

        for observation, prediction in zip(sentence, predictions):
            if observation.pos == prediction:
                hits += 1
            else:
                misses += 1

        score = hits / (hits + misses)
        results.append(Result(sentence, predictions, score))
        total_hits += hits
        total_misses += misses

    print("\n\nBEST:")
    best = nlargest(10, results, key=lambda r: r.score)
    for r in best:
        print_result(r)

    print("\nWORST:")
    worst = [r for r in results if r.score < 0.90]
    random.shuffle(worst)
    for r in worst[:20]:
        print_result(r)

    print("total accuracy:", total_hits / (total_hits + total_misses))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--training", '-f', default="training_dataset.txt")
    parser.add_argument("--testing", '-t', default="testing_dataset.txt")
    args = parser.parse_args()
    main(args.training, args.testing)
