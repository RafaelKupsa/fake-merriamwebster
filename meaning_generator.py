import os.path

import nltk
from nltk.tree import Tree
from nltk.corpus import ptb
from gensim.models import KeyedVectors
import spacy
import pyinflect
import numpy as np
from scipy.spatial.distance import cosine
import random
import re
from word_generator import WordGenerator, Suffix


# Download spacy for checking if a word already exists
NLP = spacy.load('en_core_web_sm')

# Load a list of mass nouns
with open(os.path.join("data", "mass_nouns.txt")) as f:
    MASS_NOUNS = [w.strip() for w in f.readlines()]


def load_possible_phrases(pos):
    """
    Load possible phrases for the given part of speech
    Params:
        pos: the part of speech, can be "noun", "verb", "adj", "adv"
    Returns:
        a list of phrases as nltk.tree.Tree objects
    """
    fp = {"noun": os.path.join("data", "nps.txt"),
          "verb": os.path.join("data", "vps.txt"),
          "adj": os.path.join("data", "adjps.txt"),
          "adv": os.path.join("data", "advps.txt")}[pos]
    with open(fp) as f:
        lines = [tuple(line.strip().split("\t")) for line in f.readlines()]
    phrases = [(Tree.fromstring(sentence_str), Tree.fromstring(phrase_str)) for sentence_str, phrase_str in lines]
    return phrases


def find_possible_phrases(treebank_fp, pos):
    """
    Finds possible phrases in the PENN Treebank for the given part of speech
    Params:
        treebank_fp: the filepath of the treebank
        pos: the part of speech, can be "noun", "verb", "adj", "adv"
    Returns:
        a list of phrases as nltk.tree.Tree objects
    """

    nltk.download("ptb")
    ptb_stream = ptb.parsed_sents(treebank_fp)
    phrase = {"noun": "NP", "verb": "VP", "adj": "ADJP", "adv": "ADVP"}[pos]
    phrases = []
    for tree in ptb_stream:
        phrases += [(tree, subtree) for subtree in
                    tree.subtrees(lambda x: x.label() == phrase and len(x.leaves()) < 15)]
    return filter_phrases(pos, phrases)


def filter_phrases(pos, phrases):
    """
    Filters the given phrases to useful ones.
    Excludes:
        Phrases with names or specific numbers
        Phrases with a single word only
        Noun phrases with a determiner/certain adjectives + a single word
        Noun phrase with a single word + 's
        Verb phrases starting with to or a modal/auxiliary verb (since those have dependent verb phrases with the verb as a head)
        Verb phrases starting with forms of "to be"
    Params:
        pos: the part of speech of the phrases
        phrases: list of phrases as nltk.tree.Tree objects
    Returns:
        a list of filtered phrases
    """

    filtered_phrases = []

    # List of adjectives that act as determiners
    no_article_jjs = ["many", "much", "more", "other", "own", "several", "all"]

    for tree, phrase in phrases:
        p = phrase.copy(deep=True)
        
        # Filter words with names or specific numbers
        if any([x.label().startswith("NNP") or x.label() == "CD" for x in p.subtrees()]):
            continue

        p = remove_brackets(p)

        # Filter verbs
        if pos == "verb":
            # Filter words starting with to or modal verbs
            if p[0].label() in ["TO", "MD"]:
                continue
            
            # Filter words with auxiliaries or forms of "to be"
            if p[0].label().startswith("VB"):
                dependent_verb_phrases = [x for x in p[1:] if x.label() == "VP"]
                if len(dependent_verb_phrases) > 0:
                    dependent_verbs = list(dependent_verb_phrases[0].subtrees(lambda x: x.label().startswith("VB")))
                    if len(dependent_verbs) > 0:
                        dependent_verb = dependent_verbs[0]
                        if inflect(p[0][0], "VB") == "have" and dependent_verb.label() == "VBN":
                            continue
                        if inflect(p[0][0], "VB") == "be": 
                            continue
                        if inflect(p[0][0], "VB") == "will" and dependent_verb.label() == "VB":
                            continue

        # Remove determiners/certain adjectives and possessive phrases from noun phrases
        if pos == "noun":
            if len(p) > 0 and p[-1].label() == "POS":
                p.remove(p[-1])
            if len(p) > 0 and p[0].leaves()[0] in no_article_jjs:
                p.remove(p[0])
            if len(p) > 0 and p[0].label() in ["QP", "DT", "PRP$", "CD"]:
                p.remove(p[0])

        # Only keep phrases with more than one word
        if len(p.leaves()) > 1:
            filtered_phrases.append((tree, phrase))

    return filtered_phrases



def concat(words):
    """
    Concatenates and a list of tokens to a single string and cleans it up
    Params:
        words: list of tokens
    Returns:
        words: a string of words
    """

    words = " ".join(words)
    words = words.replace("-LRB- ", "(")
    words = words.replace("-LCB- ", "(")
    words = words.replace(" -RRB-", ")")
    words = words.replace(" -RCB-", ")")
    words = words.replace("--", "–")
    words = words.replace("`` ", "\"")
    words = words.replace(" ''", "\"")
    words = words.replace("\\/", "/")

    words = words.replace(" '", "'")
    words = words.replace(" n't", "n't")
    words = re.sub(r" ([,.:;!?%\]}])", "\g<1>", words)
    words = re.sub(r"([\[{$§]) ", "\g<1>", words)

    return words


def inflect(word, inflection):
    """
    Inflects a given word according to the given inflection tag
    Params:
        word: the word to inflect as a string
        inflection: the inflection tag, can be any tag from the pyinflect module
    Returns:
        the inflected word
    """

    # Try to inflect it using the pyinflect module
    token = NLP(word)[0]
    inflected = token._.inflect(inflection)
    if inflected is not None:
        return inflected

    # If the word is unknown to pyinflect

    # Inflects nouns to plural and verbs to 3rd person present form
    if inflection in ["NNS", "VBZ"]:
        s_endings = ["s", "sh", "ch", "x", "z"]
        for ending in s_endings:
            if word.endswith(ending):
                return word + "es"
        if word[-1:] == "y" and word[-2:] not in ["ay", "ey", "oy", "uy"]:
            return word[:-1] + "ies"
        return word + "s"

    # Inflects verbs to past tense / past participle
    elif inflection in ["VBD", "VBN"]:
        ed = Suffix("ed", "english", assimilation=[("", ["e"]), ("i", ["y"]), ("er", ["er"]), ("en", ["en"])] + [(v + c, [v + c]) for c in "bdgmnprt" for v in ["ee", "ai", "oi", "oo", "au", "ou", "ea", "oa"]] + [(v + c * 2, [v + c]) for c in "bdgmnprt" for v in "aeiou"]),
        return ed[0].attach(word)

    # Inflects verbs to gerund (-ing form)
    elif inflection == "VBG":
        ing = Suffix("ing", "english", assimilation=[("", ["e"]), ("er", ["er"]), ("en", ["en"])] + [(v + c, [v + c]) for c in "bdgmnprt" for v in ["ee", "ai", "oi", "oo", "au", "ou", "ea", "oa"]] + [(v + c * 2, [v + c]) for c in "bdgmnprt" for v in "aeiou"]),
        return ing[0].attach(word)

    # otherwise return word unchanged
    return word


def remove_brackets(phrase):
    """
    Removes brackets and quotation mark subtrees from a given phrase
    Params:
        phrase: a nltk.tree.Tree object
    Returns
        the phrase without brackets and quotation mark subtrees
    """
    for sub in phrase.subtrees():
        for p in sub:
            if not isinstance(p, str) and p.label() in ["``", "''", "-LCB-", "-RCB-", "-LRB-", "-RRB-"]:
                sub.remove(p)
    return phrase



class MeaningGenerator:
    """
    Class for generating random meanings
    """
    def __init__(self, fasttext_vector_limit=1000000, similarity_threshold=0.5):
        """
        Initializes the meaning generator
        Params:
            fasttext_vector_limit: the number of fasttext vectors to be loaded
            similarity threshold: how similar should the phrases be to the meanings given to the generator
        """

        print("Loading fasttext...")
        self.FASTTEXT = KeyedVectors.load_word2vec_format(os.path.join("data", "wiki-news-300d-1M-subword.vec"), limit=fasttext_vector_limit)
        print("Fasttext loaded.")

        print("Loading phrases...")
        self.PHRASES = {pos: load_possible_phrases(pos) for pos in ["noun", "verb", "adj", "adv"]}
        print("Phrases loaded.")

        self.SIMILARITY_THRESHOLD = similarity_threshold

    def generate(self, word, pos, meanings):
        """
        Generates a random meaning for the given word with the given part of speech and some associated meanings
        Params:
            word: the word as a string
            pos: the part of speech of the word, can be "noun", "verb", "adj" or "adv"
            meanings: a list of associated meanings to the word
        Returns:
            meaning: a string representing the generated meaning
            example: a string representing an example sentence using the word
        """

        if len(meanings) == 0:
            sentence, phrase = self._choose_random_phrase(pos)
        else:
            sentence, phrase = self._choose_phrase_from_meanings(pos, meanings)

        # Generalizes the selected phrase (noun to singular form, verb to infinitive, ...)
        meaning, changes = self._generalize(pos, phrase)

        # Inserts the word into the sentence in place of the selected phrase
        example = self._insert(word, pos, changes, sentence, phrase)

        return concat(meaning.leaves()), concat(example.leaves())

    def _choose_random_phrase(self, pos):
        """
        Chooses a random phrase with the given part of speech
        Params:
            pos: part of speech, can be "noun", "verb", "adj", "adv"
        """
        return self.PHRASES[pos][random.randrange(len(self.PHRASES[pos]))]

    def _choose_phrase_from_meanings(self, pos, meanings):
        """
        Chooses a phrase with the given part of speech and a word sense similar to the given meanings
        Params:
            pos: a part of speech, can be "noun", "verb", "adj", "adv"
        Returns:
            meanings: a list of strings
        """
        
        # Embed the meanings with fasttext
        embedded_meanings = [self.FASTTEXT[m] for m in meanings if m in self.FASTTEXT.key_to_index.keys()]
        if len(embedded_meanings) == 0:
            return self._choose_random_phrase(pos)

        # Take the mean
        embedded_meanings = np.mean(embedded_meanings, 0)

        # Embed the sentences
        embedded_sents = [
            (i, [
                self.FASTTEXT[word]
                for word in phrase.leaves()
                if word in self.FASTTEXT.key_to_index.keys()
            ])
            for i, (_, phrase) in enumerate(self.PHRASES[pos])
        ]

        # Take the mean
        embedded_sents = [(i, np.mean(words, 0)) if len(words) > 0 else (i, 0) for i, words in embedded_sents]

        # Sort the embedded sentences by their similarity to the embedded meanings and keep only the most similar
        cosine_differences = [(i, cosine(embedded_meanings, emb)) for i, emb in embedded_sents]
        sorted_cosine_differences = sorted(cosine_differences, key=lambda x: x[1])
        sorted_cosine_differences = [diff for diff in sorted_cosine_differences if 0 < diff[1] <= self.SIMILARITY_THRESHOLD]

        # If there are only a few similar phrases, return a random one
        if len(sorted_cosine_differences) < 10:
            return self._choose_random_phrase(pos)

        # Return a random one of the 30 most similar phrases
        idx, _ = sorted_cosine_differences[random.randrange(min(len(sorted_cosine_differences), 30))]

        return self.PHRASES[pos][idx]

    def _generalize(self, pos, phrase):
        """
        Generalizes a phrase (nouns to singular, verbs to infinitive, ...)
        Params:
            pos: part of speech of the phrase head, can be "noun", "verb", "adj" or "adv"
            phrase: the phrase as a nltk.tree.Tree
        Returns:
            the generalized phrase
            the changes made during generalization
        """

        changes = []
        phrase = phrase.copy(deep=True)
        phrase = remove_brackets(phrase)

        # If the first word is capitalized, lower it
        first_leaf_tp = phrase.leaf_treeposition(0)
        if phrase[first_leaf_tp][0].isupper():
            phrase[first_leaf_tp] = phrase[first_leaf_tp].lower()
            changes.append("CAP")

        # Generalize the syntactic head
        if pos == "noun":
            phrase = generalize_np_head(phrase, changes)
        elif pos == "verb":
            phrase = generalize_vp_head(phrase, changes)

        # Remove anaphora
        phrase = remove_anaphora(phrase, changes)

        # Remove deixis
        phrase = remove_deixis(phrase, changes)

        return phrase, changes

    def _insert(self, word, pos, changes, sentence, phrase_to_replace):
        """
        Inserts the given word into the given sentence in place of the given phrase_to_replace
        Params:
            word: the word to insert as a string
            pos: the part of speech of the word, can be "noun", "verb", "adj" or "adv"
            changes: a list of changes to apply to the word before inserting
            sentence: the sentence as a nltk.tree.Tree
            phrase_to_replace: has to occur as a subtree in the sentence
        Returns:
            the sentence as a nltk.tree.Tree with the word inserted
        """

        # Determine the tree labels
        word_label = {"noun": "NN", "verb": "VB", "adj": "JJ", "adv": "RB"}[pos]
        phrase_label = {"noun": "NP", "verb": "VP", "adj": "ADJP", "adv": "ADVP"}[pos]
        phrases_before, phrases_after = [], []

        # Apply changes to the word
        cap = False
        for change in changes:
            if isinstance(change, Tree):
                phrases_before.append(change)
            elif change == "POS":
                poss = "'" if word.endswith("s") else "'s"
                phrases_after.append(Tree("POS", [poss]))
            elif change == "CAP":
                cap = True
            else:
                word = inflect(word, change)
                word_label = change

        # Build the tree for the word
        subphrases = phrases_before + [Tree(word_label, [word])] + phrases_after
        if pos in ["adj", "adv"] and len(subphrases) == 1:
            phrase = subphrases[0]
        else:
            phrase = Tree(phrase_label, subphrases)

        # capitalize the first word if necessary
        if cap:
            first_leaf_tp = phrase.leaf_treeposition(0)
            phrase[first_leaf_tp] = phrase[first_leaf_tp].capitalize()

        # Insert the word
        for x in sentence.subtrees():
            if pos == "adj" and phrase_to_replace in x:
                idx = x.index(phrase_to_replace)
                # Adapt indefinite article to new phrase
                if idx > 1 and x[idx - 1].label() == "DT":
                    if x[idx - 1][0] in ["a", "an"]:
                        x[idx - 1][0] = "an" if phrase.leaves()[0][0] in "aeiou" else "a"
                # Trailing adjective phrases are replaced by preceding adjectives e.g. "the thing ready to do x" > "the {new word} thing"
                if x.label() == "NP":
                    adjp_idx = x.index(phrase_to_replace)
                    np_idxs = [i for i, p in enumerate(x) if p.label() in ["NN", "NNS", "NP"]]
                    if len(np_idxs) > 0:
                        np_idx = np_idxs[-1]
                        if np_idx < adjp_idx:
                            np_with_nn = x
                            if x[np_idx].label() != "NN":
                                np_with_nn, nn_idx = get_np_with_nn(x[np_idx])
                            if np_with_nn is not None:
                                x.pop(adjp_idx)
                                np_with_nn.insert(np_idx, phrase_to_replace)
            if x == phrase_to_replace:
                x.set_label(phrase.label())
                x.clear()
                x.extend(phrase)

        return sentence


def generalize_np_head(phrase, changes):
    """
    Generalizes the head of a noun phrase
    Params:
        phrase: the noun phrase to generalize as a nltk.tree.Tree
        changes: a list where changes made are appended
    Returns:
        the generalized phrase as a nltk.tree.Tree
    """

    # List of adjectives that act as determiners
    no_article_jjs = ["many", "much", "more", "other", "own", "several", "all"]

    # Change noun to singular except in coordinated noun phrases
    parent_phrase = None
    head_phrase = phrase
    if any([child.label() == "CC" for child in head_phrase]):
        changes.append("NNS")
        return phrase
    while head_phrase[0].label() == "NP" and head_phrase[0][-1].label() != "POS":
        parent_phrase = phrase
        head_phrase = head_phrase[0]
        if any([child.label() == "CC" for child in head_phrase]):
            return phrase

    # In possessive phrases remove the 's
    if head_phrase[-1].label() == "POS":
        head_phrase.remove(head_phrase[-1])
        changes.append("POS")

    if head_phrase[-1].label().startswith("NN"):
        if parent_phrase is not None and parent_phrase[1].label() == "PP" and parent_phrase[1][0].leaves()[0] == "of":
            return phrase
        # Remove determiners
        if head_phrase[0].label() in ["DT", "QP", "PRP$", "CD"] or head_phrase[0].leaves()[0] in no_article_jjs:
            changes.append(head_phrase[0])
            head_phrase.remove(head_phrase[0])
        # Insert indefinite article except if there is a possessive phrase or the noun is a mass noun
        if head_phrase[0].label() != "NP" and head_phrase[-1].leaves() not in MASS_NOUNS:
            a = "an" if head_phrase.leaves()[0][0] in "aeiou" else "a"
            head_phrase.insert(0, Tree("DT", [a]))
        # Change the verb in depending verb phrases to singular present
        if head_phrase[-1].label() == "NNS":
            head_phrase[-1].set_label("NN")
            head_phrase[-1][0] = inflect(head_phrase[-1][0], "NN")
            changes.append("NNS")
            if parent_phrase is not None and parent_phrase[1].label() == "SBAR":
                dependent_verb = list(parent_phrase[1].subtrees(lambda x: x.label().startswith("VB")))[0]
                if dependent_verb.label() in ["VBP", "VBD"]:
                    dependent_verb.set_label("VBZ")
                    verb = dependent_verb.pop(0)
                    dependent_verb.insert(0, inflect(verb, "VBZ"))
    return phrase


def generalize_vp_head(phrase, changes):
    """
    Generalizes the head of a verb phrase
    Params:
        phrase: the verb phrase to generalize as a nltk.tree.Tree
        changes: a list where changes made are appended
    Returns:
        the generalized phrase as a nltk.tree.Tree
    """

    head_phrase = phrase
    for p in head_phrase:
        if len(p) <= 1:
            continue
        for x in p:
            if x.label() == "RB":
                # Remove "so"
                if x[0] == "so":
                    p.remove(x)
                    changes.append(x)
                # Verb + n't > not Verb
                elif x[0] in ["n't", "not"]:
                    not_phrase = x
                    p.remove(x)
                    head_phrase.insert(0, not_phrase)

    # Inflect head verb to infinitive
    for p in head_phrase:
        if p.label().startswith("VB"):
            changes.append(p.label())
            p.set_label("VB")
            verb = p.pop(0)
            p.insert(0, inflect(verb, "VB"))

    # Append "to"
    phrase = Tree("VP", [Tree("TO", ["to"]), head_phrase])

    return phrase


def generalize_adjp_head(phrase, changes):
    """
    Generalizes the head of an adjective phrase
    Params:
        phrase: the adjective phrase to generalize as a nltk.tree.Tree
        changes: a list where changes made are appended
    Returns:
        the generalized phrase as a nltk.tree.Tree
    """


    head_phrase = phrase
    for p in head_phrase:
        if len(p) <= 1:
            continue
        for x in p:
            # Remove "so"
            if x.label() == "RB" and x[0] == "so":
                p.remove(x)
                changes.append(x)

    return phrase


def remove_anaphora(phrase, changes):
    """
    Generalizes anaphoric subphrases
    Params:
        phrase: the phrase to generalize as a nltk.tree.Tree
        changes: a list where changes made are appended
    Returns:
        the generalized phrase as a nltk.tree.Tree
    """

    poss_prons = ["my", "your", "his", "her", "its", "our", "their"]
    pers_prons = ["I", "me", "you", "she", "her", "he", "him", "it", "we", "us", "they", "them"]
    refl_prons = ["myself", "yourself", "himself", "herself", "itself", "themself", "ourselves", "yourselves", "themselves"]

    for p in phrase.subtrees():
        # Replace possessive pronouns with one's
        if p.label() == "PRP$":
            for pron in poss_prons:
                p[0] = p[0].replace(pron, "one's")
        if p.label() == "PRP":
            # Replace reflexive pronouns with oneself
            for pron in refl_prons:
                p[0] = p[0].replace(pron, "oneself")
            # Replace personal pronouns with someone
            for pron in pers_prons:
                p[0] = p[0].replace(pron, "someone")

    return phrase


def remove_deixis(phrase, changes):
    """
    Generalizes deictic subphrases
    Params:
        phrase: the phrase to generalize as a nltk.tree.Tree
        changes: a list where changes made are appended
    Returns:
        the generalized phrase as a nltk.tree.Tree
    """

    # Replace temporal phrases with a generalized verion
    replacement_rules = [
        ("NN", "today", "the current day"),
        ("NN", "yesterday", "the day before"),
        ("NN", "tomorrow", "the next day"),
        ("NP", "this week", "the current week"),
        ("NP", "last week", "the week before"),
        ("NP", "next week", "the next week"),
        ("NP", "this month", "the current month"),
        ("NP", "last month", "the month before"),
        ("NP", "next month", "the next month"),
        ("NP", "this year", "the current year"),
        ("NP", "last year", "the year before"),
        ("NP", "next year", "the next year")
    ]
    for p in phrase.subtrees():
        for label, match, replacement in replacement_rules:
            if p.label() == label and " ".join(p.leaves()) == match:
                p.clear()
                p.append(replacement)

    # Replace this/that/these/those with the indefinite article
    for p in phrase.subtrees():
        if p.label() == "NP" and len(p) > 0 and not isinstance(p[0], str) and p[0].label() == "DT" and p[-1].label().startswith("NN"):
            if p[0][0] in ["this", "that"]:
                p[0][0] = "an" if p[1].leaves()[0][0] in "aeiou" else "a"
            if p[0][0] in ["these", "those"]:
                p[0][0] = ""

    return phrase


def get_np_with_nn(np):
    """
    Helper function to get the noun phrase containing the noun within a noun phrase
    Params:
        a noun phrase as a nltk.tree.Tree
    Returns:
        the noun phrase
        the index of the noun in the noun phrase
    """
    while True:
        np_idxs = [i for i, p in enumerate(np) if p.label() in ["NP"]]
        if len(np_idxs) == 0:
            return None, None
        else:
            np = np[np_idxs[-1]]
            nn_idxs = [i for i, p in enumerate(np) if p.label() in ["NN", "NNS"]]
            if len(nn_idxs) > 0:
                return np, nn_idxs[-1]


if __name__ == "__main__":
    # Generates 100 random words and meanigns
    wg = WordGenerator()
    mg = MeaningGenerator()
    for i in range(100):
        word, pos, _, meanings = wg.generate()
        meaning, example = mg.generate(word, pos, meanings)
        print(f"{word}, {pos}: {meaning}")
        print(f"Example sentence: {example}")
        print()


