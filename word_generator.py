import numpy as np
import spacy
import os

# Reference:
# https://en.wikipedia.org/wiki/List_of_Greek_and_Latin_roots_in_English/A%E2%80%93G

# Download spacy for checking if a word already exists
NLP = spacy.load('en_core_web_sm')

# bad word list adapted from https://www.kaggle.com/nicapotato/bad-bad-words
with open(os.path.join("data", "bad_words.txt")) as f:
    BAD_WORDS = [w.strip() for w in f.readlines()]

class WordGenerator:
    """
    class for generating random non-existent words adhering to English phonology
    """
    def __init__(self):
        """
        Initializes a WordGenerator
        """

        # List of possible prefixes
        self.PREFIXES = [
            Prefix("a", "english", meanings=["away", "on", "more"], assimilation=[("an", ["a", "e", "i", "o", "u"])]),
            Prefix("a", "greek", meanings=["not", "without"], assimilation=[("an", ["a", "e", "i", "o", "u", "y"])]),
            Prefix("ab", "latin", meanings=["of", "off", "from"], assimilation=[("a", ["m", "p", "v", "gn", "f"]), ("abs", ["t", "c"])]),
            Prefix("ad", "latin", meanings=["toward", "near", "at", "add"], assimilation=[("a", ["sc", "sp", "st", "sq", "j", "gn"]), ("ac", ["c", "q"]), ("af", ["f"]), ("ag", ["g"]), ("al", ["l"]), ("ap", ["p"]), ("ar", ["r"]), ("as", ["s"]), ("at", ["t"])]),
            Prefix("ambi", "latin", meanings=["both", "two"], assimilation=[("amb", ["i"])]),
            Prefix("ante", "latin", meanings=["before", "front"], assimilation=[("ant", ["e"])]),
            Prefix("anti", "greek", meanings=["against", "opposing"], assimilation=[("ant", ["i"])]),
            Prefix("auto", "greek", meanings=["self"], assimilation=[("aut", ["o"])]),
            Prefix("be", "english", meanings=["near", "around", "about", "on", "off", "much"], assimilation=[("ben", ["e"])]),
            Prefix("bi", "latin", meanings=["two", "twice"], assimilation=[("bin", ["i"])]),
            Prefix("centi", "latin", meanings=["hundred", "hundredth"], assimilation=[("cent", ["i"])]),
            Prefix("circum", "latin", meanings=["around", "circle"]),
            Prefix("cis", "latin", meanings=["here", "side"], assimilation=[("ci", ["sc", "sp", "st", "sq"])]),
            Prefix("co", "latin", meanings=["with", "together", "combined"]),
            Prefix("con", "latin", meanings=["with", "together", "combined"], assimilation=[("co", ["gn"]), ("com", ["m", "p", "b"]), ("col", ["l"]), ("cor", ["r"])]),
            Prefix("contra", "latin", meanings=["against", "contra"], assimilation=[("contr", ["a"])]),
            Prefix("crypto", "greek", meanings=["hidden", "secret"], assimilation=[("crypt", ["o"])]),
            Prefix("de", "latin", meanings=["off, down, away"], assimilation=[("den", ["e"])]),
            Prefix("deca", "latin", meanings=["ten"], assimilation=[("dec", ["a", "o", "u"])]),
            Prefix("deca", "greek", meanings=["ten", "tenth"], assimilation=[("dec", ["a", "o", "u", "y"])]),
            Prefix("di", "latin", meanings=["two", "twice"], assimilation=[("du", ["e", "i"])]),
            Prefix("di", "greek", meanings=["two", "twice"], assimilation=[("du", ["e", "i"])]),
            Prefix("dia", "greek", meanings=["through", "complete"], assimilation=[("di", ["a"])]),
            Prefix("dis", "latin", meanings=["opposite", "not"], assimilation=[("di", ["sc", "sp", "st", "sq"])]),
            Prefix("down", "english", meanings=["down", "decrease"]),
            Prefix("dys", "greek", meanings=["bad", "difficult"], assimilation=[("dy", ["z", "sc", "sm", "sp", "st", "ct", "chth", "pt", "phth", "x", "ps", "cn", "gn", "mn", "pn"])]),
            Prefix("en", "english", meanings=["cause", "put into"], assimilation=[("em", ["m", "p", "b"])]),
            Prefix("endo", "greek", meanings=["in", "within"], assimilation=[("end", ["o"])]),
            Prefix("ennea", "greek", meanings=["nine"], assimilation=[("enn", ["e"]), ("enne", ["a", "i", "o", "u", "y"])]),
            Prefix("epi", "greek", meanings=["upon", "close", "after"], assimilation=[("ep", ["i"])]),
            Prefix("eu", "greek", meanings=["good", "well"], assimilation=[("eun", ["u"])]),
            Prefix("ex", "latin", meanings=["out of", "away"], assimilation=[("e", ["b", "d", "g", "j", "l", "m", "n", "r", "v"]), ("ec", ["c", "q"]), ("ef", ["f"])]),
            Prefix("extra", "latin", meanings=["beyond", "more"], assimilation=[("extr", ["a"])]),
            Prefix("for", "english", meanings=["away", "out", "complete"]),
            Prefix("fore", "english", meanings=["before"], assimilation=[("for", ["a", "e", "i", "o", "u"])]),
            Prefix("giga", "greek", meanings=["billion", "large", "huge"], assimilation=[("gig", ["a"])]),
            Prefix("hecto", "greek", meanings=["hundred"], assimilation=[("hect", ["o"])]),
            Prefix("hemi", "greek", meanings=["half", "middle"], assimilation=[("hem", ["i"])]),
            Prefix("hepta", "greek", meanings=["seven"], assimilation=[("hept", ["a"])]),
            Prefix("hetero", "greek", meanings=["different", "other"], assimilation=[("heter", ["o"])]),
            Prefix("hexa", "greek", meanings=["six"], assimilation=[("hex", ["a"])]),
            Prefix("homo", "greek", meanings=["same"], assimilation=[("hom", ["o"])]),
            Prefix("hyper", "greek", meanings=["over", "above", "exaggerated"]),
            Prefix("hypo", "greek", meanings=["under", "below"], assimilation=[("hyp", ["o"])]),
            Prefix("icosa", "greek", meanings=["twenty"], assimilation=[("icos", ["a"])]),
            Prefix("in", "latin", meanings=["not", "in", "into"], assimilation=[("i", ["gn"]), ("im", ["m", "p", "b"]), ("il", ["l"]), ("ir", ["r"])]),
            Prefix("infra", "latin", meanings=["beneath", "below", "under"], assimilation=[("infr", ["a"])]),
            Prefix("inter", "latin", meanings=["between"]),
            Prefix("intra", "latin", meanings=["among", "within"], assimilation=[("intr", ["a"])]),
            Prefix("iso", "greek", meanings=["equal", "same"], assimilation=[("is", ["o"])]),
            Prefix("kilo", "greek", meanings=["thousand"], assimilation=[("kil", ["o"])]),
            Prefix("macro", "greek", meanings=["large", "big"], assimilation=[("macr", ["o"])]),
            Prefix("mega", "greek", meanings=["million", "large", "huge"], assimilation=[("meg", ["a"])]),
            Prefix("micro", "greek", meanings=["millionth", "small"], assimilation=[("micr", ["o"])]),
            Prefix("mid", "english", meanings=["middle"]),
            Prefix("milli", "latin", meanings=["small", "tiny", "thousand", "thousandth"], assimilation=[("mill", ["i"])]),
            Prefix("mis", "english", meanings=["wrong", "false"]),
            Prefix("mono", "greek", meanings=["one", "single", "alone"], assimilation=[("mon", ["o"])]),
            Prefix("multi", "latin", meanings=["many", "much", "amount"], assimilation=[("mult", ["i"])]),
            Prefix("nano", "greek", meanings=["billionth", "small", "tiny"], assimilation=[("nan", ["o"])]),
            Prefix("neo", "greek", meanings=["new", "fresh"], assimilation=[("ne", ["o"])]),
            Prefix("non", "latin", meanings=["not", "without"]),
            Prefix("novem", "latin", meanings=["nine"], assimilation=[("noven", ["n", "t", "s", "d", "g", "c", "q"])]),
            Prefix("ob", "latin", meanings=["against"], assimilation=[("o", ["sc", "st", "sp"]), ("obs", ["t"]), ("oc", ["c"]), ("of", ["f"]), ("og", ["g"]), ("op", ["p"])]),
            Prefix("octo", "greek", meanings=["eight"], assimilation=[("oct", ["o"])]),
            Prefix("octo", "latin", meanings=["eight"], assimilation=[("oct", ["o"])]),
            Prefix("oligo", "greek", meanings=["few", "less"], assimilation=[("olig", ["o"])]),
            Prefix("omni", "latin", meanings=["all", "every"], assimilation=[("omn", ["i"])]),
            Prefix("out", "english", meanings=["out", "outside", "more"]),
            Prefix("over", "english", meanings=["over", "above", "more", "cover"]),
            Prefix("pan", "greek", meanings=["all", "every"]),
            Prefix("para", "greek", meanings=["beside", "side"], assimilation=[("par", ["a"])]),
            Prefix("penta", "greek", meanings=["five"], assimilation=[("pent", ["a"])]),
            Prefix("per", "latin", meanings=["through", "thorough", "very"]),
            Prefix("peri", "greek", meanings=["around", "about"], assimilation=[("per", ["i"])]),
            Prefix("pluri", "latin", meanings=["many", "much", "amount"], assimilation=[("plur", ["i"])]),
            Prefix("poly", "greek", meanings=["many", "much", "amount"]),
            Prefix("post", "latin", meanings=["after", "later"], assimilation=[("pos", ["t"])]),
            Prefix("pre", "latin", meanings=["before", "early"]),
            Prefix("pro", "latin", meanings=["forward", "for"]),
            Prefix("pro", "greek", meanings=["before"], assimilation=[("pr", ["o"])]),
            Prefix("pseudo", "greek", meanings=["fake", "false"], assimilation=[("pseud", ["o"])]),
            Prefix("quadri", "latin", meanings=["four"], assimilation=[("quadr", ["i"])]),
            Prefix("quasi", "greek", meanings=["almost", "approximately"], assimilation=[("quas", ["i"])]),
            Prefix("quin", "latin", meanings=["five"], assimilation=[("quim", ["m", "b", "p"])]),
            Prefix("re", "latin", meanings=["again", "back", "repeat"]),
            Prefix("retro", "latin", meanings=["backwards", "behind"], assimilation=[("retr", ["o"])]),
            Prefix("semi", "latin", meanings=["half", "middle"], assimilation=[("sem", ["i"])]),
            Prefix("septem", "latin", meanings=["seven"], assimilation=[("septen", ["n", "t", "s", "d", "g", "c", "q"])]),
            Prefix("sexa", "latin", meanings=["six"], assimilation=[("sex", ["a"])]),
            Prefix("sub", "latin", meanings=["under", "below"]),
            Prefix("super", "latin", meanings=["above", "over"]),
            Prefix("syn", "greek", meanings=["with", "together", "combined"], assimilation=[("sym", ["m", "p", "b"]), ("syl", ["l"])]),
            Prefix("tele", "greek", meanings=["distance", "long"], assimilation=[("tel", ["e"])]),
            Prefix("tera", "greek", meanings=["trillion"], assimilation=[("ter", ["a"])]),
            Prefix("tetra", "greek", meanings=["four"], assimilation=[("tetr", ["a"])]),
            Prefix("thermo", "greek", meanings=["heat", "hot", "warm"], assimilation=[("therm", ["o"])]),
            Prefix("trans", "latin", meanings=["across", "beyond", "change"], assimilation=[("tran", ["s"])]),
            Prefix("tri", "latin", meanings=["three"], assimilation=[("trin", ["i"])]),
            Prefix("tri", "greek", meanings=["three"], assimilation=[("trin", ["a"])]),
            Prefix("ultra", "latin", meanings=["more", "beyond"], assimilation=[("ultr", ["a"])]),
            Prefix("un", "english", meanings=["not", "without"]),
            Prefix("under", "english", meanings=["under", "insufficient", "below"]),
            Prefix("uni", "latin", meanings=["one", "single", "alone"], assimilation=[("un", ["i"])]),
            Prefix("up", "english", meanings=["up", "more", "high"]),
            Prefix("with", "english", meanings=["against", "back", "off", "with"])
        ]

        # Dictionary countaining a list for all possible suffixes per part of speech
        self.SUFFIXES = {}

        self.SUFFIXES["noun"] = [
            Suffix("ability", "latin", meanings=["able", "capable", "can"], assimilation=[("", ["a"])]),
            Suffix("acity", "latin", meanings=["quality", "have"], assimilation=[("", ["a"])]),
            Suffix("acy", "latin", meanings=["quality", "state", "condition"], assimilation=[("", ["a"])]),
            Suffix("ade", "latin", meanings=["action", "do", "happen"], assimilation=[("", ["a"])]),
            Suffix("age", "latin", meanings=["many", "action", "state", "process"], assimilation=[("", ["a"])]),
            Suffix("al", "latin", meanings=["action", "do", "happen"], assimilation=[("", ["a"]), ("r", ["l"])]),
            Suffix("an", "latin", meanings=["person", "individual", "human", "guy", "man", "woman"], assimilation=[("", ["a"])]),
            Suffix("ation", "latin", meanings=["action", "process", "result", "do", "happen"], assimilation=[("", ["a"])]),
            Suffix("ator", "latin", meanings=["person", "individual", "human", "guy", "man", "woman"], assimilation=[("", ["a"])]),
            Suffix("ance", "latin", meanings=["process", "action", "state", "condition"], assimilation=[("", ["a"])]),
            Suffix("dom", "english", meanings=["condition", "state", "realm", "domain"], assimilation=[("i", ["y"])]),
            Suffix("ence", "latin", meanings=["process", "action", "state", "condition"], assimilation=[("", ["e", "a"])]),
            Suffix("er", "english", meanings=["person", "individual", "human", "guy", "man", "woman", "act", "do"], assimilation=[("", ["e"]), ("i", ["y"]), ("er", ["er"]), ("en", ["en"])] + [(v + c, [v + c]) for c in "bdgmnprt" for v in ["ee", "ai", "oi", "oo", "au", "ou", "ea", "oa"]] + [(v + c * 2, [v + c]) for c in "bdgmnprt" for v in "aeiou"]),
            Suffix("ery", "english", meanings=["place", "location", "site", "spot", "house", "work"], assimilation=[("", ["e"]), ("i", ["y"]), ("er", ["er"]), ("en", ["en"])] + [(v + c, [v + c]) for c in "bdgmnprt" for v in ["ee", "ai", "oi", "oo", "au", "ou", "ea", "oa"]] + [(v + c * 2, [v + c]) for c in "bdgmnprt" for v in "aeiou"]),
            Suffix("escence", "latin", meanings=["change", "state"], assimilation=[("", ["e", "a"])]),
            Suffix("ess", "english", meanings=["woman", "she", "female", "feminin"], assimilation=[("", ["e"]), ("i", ["y"]), ("er", ["er"]), ("en", ["en"])] + [(v + c, [v + c]) for c in "bdgmnprt" for v in ["ee", "ai", "oi", "oo", "au", "ou", "ea", "oa"]] + [(v + c * 2, [v + c]) for c in "bdgmnprt" for v in "aeiou"]),
            Suffix("ibility", "latin", meanings=["able", "capable", "can"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("ifier", "latin", meanings=["person", "individual", "act", "do", "change", "affect"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("ification", "latin", meanings=["change", "effect"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("ition", "latin", meanings=["action", "process", "result", "do", "happen"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("ism", "greek", meanings=["doctrine", "belief", "faith", "think"], assimilation=[("", ["i"])]),
            Suffix("ism", "latin", meanings=["doctrine", "belief", "faith", "think"], assimilation=[("", ["e", "i"])]),
            Suffix("ism", "english", meanings=["doctrine", "belief", "faith", "think"], assimilation=[("", ["e"])]),
            Suffix("ist", "latin", meanings=["person", "individual", "belief", "work", "occupation"], assimilation=[("", ["e", "i"])]),
            Suffix("ist", "greek", meanings=["person", "individual", "belief", "work", "occupation"], assimilation=[("", ["i"])]),
            Suffix("itis", "greek", meanings=["sickness", "infection", "ill"], assimilation=[("", ["i"])]),
            Suffix("itude", "latin", meanings=["quality", "be", "have"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("ity", "latin", meanings=["quality", "have"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("iveness", "latin", meanings=["quality", "tend", "have"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("ivity", "latin", meanings=["quality", "tend", "have"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("ization", "latin", meanings=["change", "effect", "result", "action"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("izer", "latin", meanings=["person", "change", "effect", "result", "act", "do"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("lysis", "greek", meanings=["loss", "chemistry", "science"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ment", "latin", meanings=["action", "result"]),
            Suffix("ness", "english", meanings=["quality", "be", "have"]),
            Suffix("edness", "english", meanings=["quality", "be", "have"], assimilation=[("", ["e"]), ("i", ["y"]), ("er", ["er"]), ("en", ["en"])] + [(v + c, [v + c]) for c in "bdgmnprt" for v in ["ee", "ai", "oi", "oo", "au", "ou", "ea", "oa"]] + [(v + c * 2, [v + c]) for c in "bdgmnprt" for v in "aeiou"]),
            Suffix("liness", "english", meanings=["quality", "be", "have"]),
            Suffix("fulness", "english", meanings=["quality", "be", "have", "full"]),
            Suffix("ling", "english", meanings=["person", "small", "guy", "man", "woman", "child"]),
            Suffix("metry", "greek", meanings=["measure", "science", "length", "size", "weight"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ometry", "greek", meanings=["measure", "science", "length", "size", "weight"], assimilation=[("", ["o", "a"])]),
            Suffix("logy", "greek", meanings=["science", "chemistry", "biology", "religion", "physics", "math", "philosophy"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ology", "greek", meanings=["science", "chemistry", "biology", "religion", "physics", "math", "philosophy"], assimilation=[("", ["o", "a"])]),
            Suffix("or", "latin", meanings=["person", "individual", "human", "guy", "man", "woman", "act", "do"], assimilation=[("", ["a", "o"])]),
            Suffix("ory", "latin", meanings=["place", "location", "site", "spot", "room"], assimilation=[("", ["a", "o"])]),
            Suffix("osity", "latin", meanings=["quality", "be", "have"], assimilation=[("", ["a", "o"])]),
            Suffix("sis", "greek", meanings=["condition", "infection", "science"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("osis", "greek", meanings=["condition", "infection", "science"], assimilation=[("", ["o", "a"])]),
            Suffix("pathy", "greek", meanings=["condition", "sickness"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("opathy", "greek", meanings=["condition", "sickness"], assimilation=[("", ["o", "a"])]),
            Suffix("philia", "greek", meanings=["love", "condition"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ophilia", "greek", meanings=["love", "condition"], assimilation=[("", ["o", "a"])]),
            Suffix("phobia", "greek", meanings=["hate", "fear", "bad", "condition"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ophobia", "greek", meanings=["hate", "fear", "bad", "condition"], assimilation=[("", ["o", "a"])]),
            Suffix("ship", "english", meanings=["state", "fellowship", "role"], assimilation=[("", ["s"])]),
            Suffix("ion", "latin", meanings=["quality", "action"], assimilation=[("at", ["a"]), ("et", ["e"]), ("it", ["i"]), ("ot", ["o"]), ("ut", ["u"])]),
            Suffix("ulation", "latin", meanings=["quality", "action"], assimilation=[("c", ["qu"]), ("", ["u", "a", "o"])]),
            Suffix("ure", "latin", meanings=["process", "condition", "entity"], assimilation=[("c", ["qu"]), ("", ["u", "a", "o"])]),
            Suffix("ia", "greek", meanings=["science", "land", "taxonomy", "disease", "flower", "collection"], assimilation=[("ar", ["a"]), ("er", ["e"]), ("ir", ["i"]), ("or", ["o"]), ("ur", ["u"]), ("yr", ["y"])]),
            Suffix("ria", "greek", meanings=["science", "land", "taxonomy", "disease", "flower", "collection"], assimilation=[(c + "e", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ics", "greek", meanings=["doctrine", "science", "chemistry", "biology", "religion", "physics", "math", "philosophy"], assimilation=[("", ["i", "y"])])
        ]

        self.SUFFIXES["verb"] = [
            Suffix("ate", "latin", assimilation=[("", ["a"])]),
            Suffix("ationize", "latin", assimilation=[("", ["a"])]),
            Suffix("ationalize", "latin", assimilation=[("", ["a"])]),
            Suffix("itate", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("itionize", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("itionalize", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("utate", "latin", assimilation=[("c", ["qu"]), ("", ["u", "a", "o"])]),
            Suffix("esce", "latin", meanings=["change"], assimilation=[("", ["e", "a"])]),
            Suffix("ule", "latin", assimilation=[("c", ["qu"]), ("", ["u", "a", "o"])]),
            Suffix("ulate", "latin", assimilation=[("c", ["qu"]), ("", ["u", "a", "o"])]),
            Suffix("urate", "latin", assimilation=[("c", ["qu"]), ("", ["u", "a", "o"])]),
            Suffix("ify", "latin", meanings=["change", "effect"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("ionize", "latin", assimilation=[("at", ["a"]), ("et", ["e"]), ("it", ["i"]), ("ot", ["o"]), ("ut", ["u"])]),
            Suffix("ivize", "latin", meanings=["quality", "tend", "have"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("ize", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("logize", "greek", meanings=["doctrine", "science", "chemistry", "biology", "religion", "physics", "math", "philosophy"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ologize", "greek", meanings=["doctrine", "science", "chemistry", "biology", "religion", "physics", "math", "philosophy"], assimilation=[("", ["o", "a"])]),
            Suffix("icalize", "greek", assimilation=[("", ["i", "y"])])
        ]

        self.SUFFIXES["adj"] = [
            Suffix("able", "latin", meanings=["able", "capable", "can"], assimilation=[("", ["a"])]),
            Suffix("aceous", "latin", meanings=["similar", "containing", "belonging"], assimilation=[("", ["a"])]),
            Suffix("al", "latin", assimilation=[("", ["a"])]),
            Suffix("arious", "latin", assimilation=[("", ["a"])]),
            Suffix("ant", "latin", assimilation=[("", ["a"])]),
            Suffix("ative", "latin", meanings=["quality", "tend", "have"], assimilation=[("", ["a"])]),
            Suffix("ating", "latin", assimilation=[("", ["a"])]),
            Suffix("ational", "latin", assimilation=[("", ["a"])]),
            Suffix("ent", "latin", assimilation=[("", ["e", "a"])]),
            Suffix("eous", "latin", meanings=["similar", "containing", "belonging"], assimilation=[("or", ["o"]), ("ur", ["u"]), ("ir", ["i"]), ("", ["e", "a"])]),
            Suffix("escent", "latin", meanings=["change", "similar"], assimilation=[("", ["e", "a"])]),
            Suffix("ful", "english", meanings=["quality", "containing", "belonging"], assimilation=[("i", ["y"])]),
            Suffix("ible", "latin", meanings=["able", "capable", "can"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("inous", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("id", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("itional", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("itive", "latin", meanings=["quality", "tend", "have"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("ic", "greek", assimilation=[("", ["i"])]),
            Suffix("ical", "greek", assimilation=[("", ["i"])]),
            Suffix("ing", "english", assimilation=[("", ["e"]), ("er", ["er"]), ("en", ["en"])] + [(v + c, [v + c]) for c in "bdgmnprt" for v in ["ee", "ai", "oi", "oo", "au", "ou", "ea", "oa"]] + [(v + c * 2, [v + c]) for c in "bdgmnprt" for v in "aeiou"]),
            Suffix("ed", "english", assimilation=[("", ["e"]), ("i", ["y"]), ("er", ["er"]), ("en", ["en"])] + [(v + c, [v + c]) for c in "bdgmnprt" for v in ["ee", "ai", "oi", "oo", "au", "ou", "ea", "oa"]] + [(v + c * 2, [v + c]) for c in "bdgmnprt" for v in "aeiou"]),
            Suffix("ish", "english", assimilation=[("", ["e", "i", "a", "y"]), ("er", ["er"]), ("en", ["en"])] + [(v + c, [v + c]) for c in "bdgmnprt" for v in ["ee", "ai", "oi", "oo", "au", "ou", "ea", "oa"]] + [(v + c * 2, [v + c]) for c in "bdgmnprt" for v in "aeiou"]),
            Suffix("ile", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("ine", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("ifying", "latin", meanings=["change", "effect"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("itant", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("itating", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("iting", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("ive", "latin", meanings=["quality", "tend", "have"], assimilation=[("", ["e", "i", "a"])]),
            Suffix("izing", "latin", assimilation=[("", ["e", "i", "a"])]),
            Suffix("less", "english", meanings=["without"]),
            Suffix("like", "english", meanings=["resembling", "similar"]),
            Suffix("oid", "greek", meanings=["resembling", "similar"], assimilation=[("", ["o"])]),
            Suffix("one", "latin", assimilation=[("", ["a", "o"])]),
            Suffix("ous", "latin", meanings=["similar", "containing", "belonging"], assimilation=[("", ["a", "o"])]),
            Suffix("ose", "latin", assimilation=[("", ["a", "o"])]),
            Suffix("ular", "latin", assimilation=[("c", ["qu"]), ("", ["u", "a", "o"])]),
            Suffix("ural", "latin", assimilation=[("c", ["qu"]), ("", ["u", "a", "o"])]),
            Suffix("ulating", "latin", assimilation=[("c", ["qu"]), ("", ["u", "a", "o"])]),
            Suffix("urating", "latin", assimilation=[("c", ["qu"]), ("", ["u", "a", "o"])]),
            Suffix("uline", "latin", assimilation=[("c", ["qu"]), ("", ["u", "a", "o"])]),
            Suffix("urine", "latin", assimilation=[("c", ["qu"]), ("", ["u", "a", "o"])]),
            Suffix("y", "english", assimilation=[("", ["e", "y"])]),
            Suffix("ite", "greek", assimilation=[("", ["i", "y"])]),
            Suffix("logic", "greek", meanings=["logical", "scientific", "science", "chemistry", "biology", "religion", "physics", "math", "philosophy"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ologic", "greek", meanings=["logical", "scientific", "science", "chemistry", "biology", "religion", "physics", "math", "philosophy"], assimilation=[("", ["o", "a"])]),
            Suffix("logical", "greek", meanings=["logical", "scientific", "science", "chemistry", "biology", "religion", "physics", "math", "philosophy"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ological", "greek", meanings=["logical", "scientific", "science", "chemistry", "biology", "religion", "physics", "math", "philosophy"], assimilation=[("", ["o", "a"])]),
            Suffix("pathic", "greek", meanings=["sick", "ill", "condition"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("opathic", "greek", meanings=["sick", "ill", "condition"], assimilation=[("", ["o", "a"])]),
            Suffix("pathical", "greek", meanings=["sick", "ill", "condition"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("opathical", "greek", meanings=["sick", "ill", "condition"], assimilation=[("", ["o", "a"])]),
            Suffix("lytic", "greek", meanings=["analytic", "logical", "loss", "chemistry", "science"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("olytic", "greek", meanings=["analytic", "logical", "loss", "chemistry", "science"], assimilation=[("", ["o", "a"])]),
            Suffix("lytical", "greek", meanings=["analytic", "logical", "loss", "chemistry", "science"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("olytical", "greek", meanings=["analytic", "logical", "loss", "chemistry", "science"], assimilation=[("", ["o", "a"])]),
            Suffix("philic", "greek", meanings=["loving", "feeling", "love", "condition"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ophilic", "greek", meanings=["loving", "feeling", "love", "condition"], assimilation=[("", ["o", "a"])]),
            Suffix("phobic", "greek", meanings=["afraid", "hate", "fear", "bad", "condition"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ophobic", "greek", meanings=["afraid", "hate", "fear", "bad", "condition"], assimilation=[("", ["o", "a"])]),
            Suffix("metric", "greek", meanings=["measure", "science", "length", "size", "weight"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ometric", "greek", meanings=["measure", "science", "length", "size", "weight"], assimilation=[("", ["o", "a"])]),
            Suffix("metrical", "greek", meanings=["measure", "science", "length", "size", "weight"], assimilation=[(c + "o", [c]) for c in "bcdghlmnprstxz"]),
            Suffix("ometrical", "greek", meanings=["measure", "science", "length", "size", "weight"], assimilation=[("", ["o", "a"])]),
        ]

        self.SUFFIXES["adv"] = [
            Suffix("ly", "english", assimilation=[("i", ["y"])]),
            Suffix("wise", "english", meanings=["similar", "direction", "towards"]),
            Suffix("ally", "latin", assimilation=[("", ["a"])]),
            Suffix("fully", "english", meanings=["quality", "have"], assimilation=[("i", ["y"])]),
            Suffix("edly", "english", assimilation=[("", ["e"]), ("i", ["y"]), ("er", ["er"]), ("en", ["en"])] + [(v + c, [v + c]) for c in "bdgmnprt" for v in ["ee", "ai", "oi", "oo", "au", "ou", "ea", "oa"]] + [(v + c * 2, [v + c]) for c in "bdgmnprt" for v in "aeiou"]),
            Suffix("edly", "latin", assimilation=[("", ["e"]), ("i", ["y"]), ("er", ["er"]), ("en", ["en"])] + [(v + c, [v + c]) for c in "bdgmnprt" for v in ["ee", "ai", "oi", "oo", "au", "ou", "ea", "oa"]] + [(v + c * 2, [v + c]) for c in "bdgmnprt" for v in "aeiou"]),
            Suffix("ically", "greek", assimilation=[("", ["i"])]),
            Suffix("ingly", "english", assimilation=[("", ["e"]), ("er", ["er"]), ("en", ["en"])] + [(v + c, [v + c]) for c in "bdgmnprt" for v in ["ee", "ai", "oi", "oo", "au", "ou", "ea", "oa"]] + [(v + c * 2, [v + c]) for c in "bdgmnprt" for v in "aeiou"]),
            Suffix("ily", "english", assimilation=[("", ["e", "y"])]),
            Suffix("itely", "greek", assimilation=[("", ["i", "y"])])
        ]

    def generate(self, pos=None, origin=None):
        """
        Generate a new random non-existent word
        Params:
            pos: part of speech of the new word, can be "noun", "verb", "adj" or "adv"
            origin: origin of the word, can be "english", "latin" or "greek"
        Returns:
            word: the generated word as a string
            pos: the part of speech of the word
            origin: the origin of the word
            meanings: list of meanings of the prefixes and suffixes
        """

        if pos is None:
            pos = np.random.choice(["noun", "verb", "adj", "adv"], p=[0.4, 0.3, 0.2, 0.1])
        if origin is None:
            origin = np.random.choice(["english", "latin", "greek"], p=[0.35, 0.4, 0.25])

        if origin == "greek":
            word = str(GreekRoot())
            meanings = []
            if np.random.random() < 0.7:
                word, pf_meanings = self._append_prefix(word, origin)
                meanings += pf_meanings
            word, sf_meanings = self._append_suffix(word, origin, pos)
            meanings += sf_meanings

        elif origin == "latin":
            root = LatinRoot()
            word = str(root)
            meanings = []
            has_suffix = False
            if np.random.random() < 0.8:
                word, sf_meanings = self._append_suffix(word, origin, pos)
                meanings += sf_meanings
                has_suffix = True
            else:
                root.no_suffix()
                word = str(root)
            if not has_suffix or np.random.random() < 0.7:
                word, pf_meanings = self._append_prefix(word, origin)
                meanings += pf_meanings

        else:
            word = str(EnglishRoot())
            meanings = []
            if np.random.random() < 0.6:
                word, pf_meanings = self._append_prefix(word, origin)
                meanings += pf_meanings
            if np.random.random() < 0.6:
                word, sf_meanings = self._append_suffix(word, origin, pos)
                meanings += sf_meanings

        # check if word already exists or is vulgar
        if word in list(NLP.vocab.strings) or self._is_vulgar(word):
            return self.generate(pos, origin)

        return word, pos, origin, meanings

    def _append_prefix(self, root, origin):
        """
        Appends a random prefix of the given origin to the word root
        Params:
            root: the root as a string
            origin: the origin of the root
        Returns:
            root: the root with the prefix attached as a string
            meanings: List of meanings of the prefix
        """
        origin_prefixes = [p for p in self.PREFIXES if p.origin == origin]
        if len(origin_prefixes) > 0:
            prefix = np.random.choice(origin_prefixes) if len(origin_prefixes) > 1 else origin_prefixes[0]
            return prefix.attach(root), prefix.meanings
        else:
            return root, []

    def _append_suffix(self, root, origin, pos):
        """
        Appends a random suffix of the given origin and part of speech to the word root
        Params:
            root: the root as a string
            origin: the origin of the root
            pos: the part of speech of the root
        Returns:
            root: the root with the suffix attached as a string
            meanings: List of meanings of the suffix
        """
        origin_suffixes = [p for p in self.SUFFIXES[pos] if p.origin == origin]
        if len(origin_suffixes) > 0:
            suffix = np.random.choice(origin_suffixes) if len(origin_suffixes) > 1 else origin_suffixes[0]
            return suffix.attach(root), suffix.meanings
        else:
            return root, []

    def _is_vulgar(self, word):
        """
        Checks if a given word contains vulgar/inappropriate language
        Params:
            word: the word as a string
        Returns:
            True or False
        """
        for bw in BAD_WORDS:
            if bw in word:
                return True
        return False


class Prefix:
    """
    class for a prefix
    """
    def __init__(self, prefix, origin, meanings=None, assimilation=None):
        """
        Initializes a prefix
        Params:
            prefix: the prefix as a string
            origin: the origin of the prefix, can be "english", "latin", "greek"
            meanings: a list of meanings that are usually associated with the prefix
            assimilation: a list of 2-tuples specifying assimilation patterns of the prefix
                the first element of the tuple denotes the alternative spelling of the prefix
                the second element is a list of strings that trigger an assimilation if the words begin with it
        """
        self.prefix = prefix
        self.origin = origin
        self.meanings = meanings if meanings is not None else []
        self.assimilation = assimilation if assimilation is not None else []

    def attach(self, root):
        """
        Attaches the prefix to the given root
        Params:
            root: the root that the prefix should be attached to as a string
        Returns:
            the root with the prefix attached
        """
        for a in self.assimilation:
            for pat in a[1]:
                if root.startswith(pat):
                    if self.origin == "latin" and root.startswith("gn") and a[0][-1] not in "aeiou":
                        return a[0] + root[1:]
                    return a[0] + root
        return self.prefix + root


class Suffix:
    """
    class for a suffix
    """
    def __init__(self, suffix, origin, meanings=None, assimilation=None):
        """
        Initializes a suffix
        Params:
            prefix: the suffix as a string
            origin: the origin of the suffix, can be "english", "latin", "greek"
            meanings: a list of meanings that are usually associated with the suffix
            assimilation: a list of 2-tuples specifying assimilation patterns of the suffix
                the first element of the tuple denotes the alternative spelling of the word's end
                the second element is a list of strings that trigger an assimilation if a word ends with it
        """
        self.suffix = suffix
        self.origin = origin
        self.meanings = meanings if meanings is not None else []
        self.assimilation = assimilation if assimilation is not None else []

    def attach(self, root):
        """
        Attaches the suffix to the given root
        Params:
            root: the root that the suffix should be attached to as a string
        Returns:
            the root with the suffix attached
        """
        for a in self.assimilation:
            for pat in a[1]:
                if root.endswith(pat):
                    return root[:-len(pat)] + a[0] + self.suffix
        return root + self.suffix


def get_probs(choices, length_penalty):
    """
    Get probability distribution of the elements in choices, adjusted by the values in length_penalty
    Params:
        choices: a list of strings
        length_penalty: a list of floats, the index of a value corresponds to the associated length
    Returns:
        probs: a list of floats that add up to one, specifying a probability distribution over the elements in choices
    """
    probs = [length_penalty[len(c)] for c in choices]
    probs = [p / sum(probs) for p in probs]
    return probs


class EnglishRoot:
    """
    class for generating a random English word root that looks like it has English (Germanic) origin
    """
    def __init__(self):
        """
        Generates a random English looking word root
        """
        self.onset = self._gen_onset()
        self.nucleus = self._gen_nucleus()
        self.coda = self._gen_coda()
        self.final = self._gen_final()

    def _gen_onset(self):
        """
        Generates a random syllable onset
        Returns:
            the syllable onset as a string
        """
        choices = [
            "", "b", "c", "ch", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p",
            "r", "s", "sc", "sk", "sh", "sp", "st", "t", "th", "w", "y", "z",
            "bl", "cl", "fl", "gl", "pl", "sl", "spl",
            "br", "cr", "dr", "fr", "gr", "pr", "scr", "shr", "spr", "str", "tr", "thr", "wr",
            "qu", "dw", "sw", "squ", "tw",
            "gn", "kn", "sn",
            "sm", "wh"
        ]
        return np.random.choice(choices, p=get_probs(choices, [3, 1, 0.4, 0.3]))

    def _gen_nucleus(self):
        """
        Generates a random syllable nucleus
        Returns:
            the nucleus as a tuple of strings
                the first element is the vowel letter after the coda consonant
                the second element is a possible vowel letter after the coda consonant
        """
        choices = [("a", ""), ("e", ""), ("i", ""), ("o", ""), ("u", ""),
                   ("a", "e"), ("i", "e"), ("o", "e"),
                   ("ai", ""), ("ee", ""), ("oa", ""), ("ew", ""),
                   ("ea", ""), ("oo", ""), ("ou", ""), ("ow", ""), ("oi", ""),
                   ("au", ""), ("aw", "")]

        # Excludes ce, ci, sce, sci
        if self.onset[-1:] == "c":
            choices = [c for c in choices if c[0][0] not in "ei"]

        # Excludes ka, ko, ku, ska, sko, sku
        if self.onset[-1:] == "k":
            choices = [c for c in choices if c[0][0] not in "aou"]

        # Excludes wew, wow, wrew, wrow, quew, quow, dwew, dwow, swew, swow, twew, twow
        if "w" in self.onset or "q" in self.onset:
            choices.remove(("ew", ""))
            choices.remove(("ow", ""))
            choices.remove(("aw", ""))

        # Excludes quu
        if "qu" in self.onset:
            choices.remove(("u", ""))

        index = np.random.choice(len(choices))
        return choices[index]

    def _gen_coda(self):
        """
        Generates a random syllable coda
        Returns:
            the syllable coda as a string
        """
        
        choices_short = [
            "b", "ck", "d", "dge", "ff", "g", "gh", "ll", "m", "n", "p", "r", "ss", "t", "th", "tch", "x", "zz",
            "lb", "lch", "ld", "lge", "lf", "lk", "lm", "ln", "lp", "lse", "lsh", "lt", "lth", "lve",
            "rb", "rch", "rd", "rge", "rf", "rk", "rl", "rm", "rn", "rp", "rse", "rsh", "rt", "rth", "rve",
            "mb", "mp", "nd", "nt", "ng", "nk", "nch", "nge", "nce", "nth",
            "ft", "sk", "sp", "st", "ght"
        ]
        choices_long = ["", "b", "c", "d", "f", "g", "gh", "ght", "k", "l", "m", "n", "p", "r", "s", "t", "th", "v",
                        "z"]
        
        # Sets the choices only two codas occurring with short vowels
        if len(self.nucleus[0]) == 1 and len(self.nucleus[1]) == 0:
            choices = choices_short
            
            # if the vowel is not i, a coda with gh is not possible
            if self.nucleus[0] != "i":
                choices.remove("gh")
                choices.remove("ght")
        
        # Sets the choices only two codas occurring with long vowels
        else:
            choices = choices_long
            
            # if the vowel is not ou, ei or au, a coda with gh is not possible
            if self.nucleus[0] not in ["ou", "ei", "au"]:
                choices.remove("gh")
                choices.remove("ght")
            
            # ou, ei, oa, and au cannot have an empty coda
            # Excludes also word with neither onset nor coda which are rare and not very interesting
            if self.nucleus[0] in ["ou", "ei", "oa", "au"] or self.onset == "":
                choices.remove("")
            
            # long nuclei orthographically cannot end in c, s, v or z without an e attached
            if len(self.nucleus[0]) > 1:
                choices.remove("c")
                choices.append("ce")
                choices.remove("s")
                choices.append("se")
                choices.remove("v")
                choices.append("ve")
                choices.remove("z")
                choices.append("ze")
            
            # oi is a rare nucleus and occurs only with a few coda consonants, e.g. toil, noise, coin, voice, boi (> boy)
            if self.nucleus[0] == "oi":
                choices = ["", "l", "se", "n", "ce"]


        # Excludes words that have l in both the onset and the coda which occurs rarely in English
        if "l" in self.onset:
            choices = [c for c in choices if "l" not in c]

        # Excludes words that have r in both the onset and the coda which occurs rarely in English
        if "r" in self.onset:
            choices = [c for c in choices if "r" not in c]


        coda = np.random.choice(choices, p=get_probs(choices, [0.5, 1, 0.4, 0.3]))

        # if an empty coda was generated, the orthography of some nuclei changes: ai/ae > ay, oi > oy
        if coda == "":
            if self.nucleus[0] == "ai" or self.nucleus == ("a", "e"):
                self.nucleus = ("ay", "")
            if self.nucleus[0] == "oi":
                self.nucleus = ("oy", "")

        return coda

    def _gen_final(self):
        """
        Generates a random word ending such as "", "en", "le", "er", "ow", "y"
        Returns:
            the word ending as a string
        """
        choices = ["", "en", "le", "er", "ow", "y"]
        probs = {"": 1, "en": 0.05, "le": 0.2, "er": 0.3, "ow": 0.1, "y": 0.3}

        # Excludes -ghle, -lle and -lele
        if self.coda == "gh" or self.coda.endswith("l"):
            choices.remove("le")

        # ow only occurs with some double consonants
        if self.coda not in ["dd", "rr", "ll", "nn", "nd"]:
            choices.remove("ow")

        # if there is no coda, only no ending or "er" are possible
        if self.coda == "":
            choices = ["", "er"]

        probs = [probs[c] for c in choices]
        probs = [p / sum(probs) for p in probs]
        final = np.random.choice(choices, p=probs)


        if final != "":

            # A trailing -e is removed, e.g. crazey > crazy, nobele > noble
            if len(self.nucleus[0]) == 1 and len(self.nucleus[1]) == 1:
                self.nucleus = (self.nucleus[0], "")

            # Single coda consonants are doubled for short vowels
            elif len(self.nucleus[0]) == 1 and len(self.nucleus[1]) == 0 and self.coda in "bdgmnprt":
                self.coda = self.coda * 2

            # -le changes to -el after certain consonants
            if final == "le" and self.coda in ["mm", "nn", "rr", "m", "n", "r", "dge", "ce"]:
                final = "el"
            # -ceel > -cel, -dgeel > dgel
            if self.coda.endswith("e"):
                self.coda = self.coda[:-1]


        return final

    def __str__(self):
        """
        Returns:
            the root as a combined string
        """
        return self.onset + self.nucleus[0] + self.coda + self.nucleus[1] + self.final


class LatinRoot:
    """
    class for generating a random English word root that looks like it has Latin origin
    """
    def __init__(self):
        """
        Generates a random English looking word root
        """
        self.word_parts = [self._gen_onset()]
        self.word_parts.append(self._gen_nucleus())
        if np.random.random() < 0.9:
            self.word_parts.append(self._gen_coda_onset())
            if np.random.random() < 0.1:
                self.word_parts.append(self._gen_onset_glide())
            if np.random.random() < 0.1:
                self.word_parts.append(self._gen_nucleus())
                self.word_parts.append(self._gen_coda_onset())
                if np.random.random() < 0.2:
                    self.word_parts.append(self._gen_onset_glide())
            if (self.word_parts[-1].endswith("c") or self.word_parts[-1].endswith("g")) and np.random.random() < 0.5:
                self.word_parts[-1] += "e"

    def _gen_onset(self):
        """
        Generates a random syllable onset
        Returns:
            the syllable onset as a string
        """
        choices = [
            "", "b", "c", "d", "f", "g", "h", "j", "l", "m",
            "n", "p", "r", "s", "sc", "sp", "st", "t", "v",
            "bl", "cl", "fl", "gl", "pl", "spl",
            "br", "cr", "dr", "fr", "gr", "pr", "scr", "spr", "str", "tr",
            "qu", "squ", "gn"
        ]
        return np.random.choice(choices, p=get_probs(choices, [5, 1, 0.4, 0.3]))

    def _gen_onset_glide(self):
        """
        Generates an onset glide (either i or u)
        Returns:
            the glide as a string
        """
        return "i" if self.word_parts[-1].endswith("qu") else np.random.choice(["i", "u"])

    def _gen_nucleus(self):
        """
        Generates a random syllable nucleus
        Returns:
            the nucleus as a string
        """
        choices = ["a", "e", "i", "o", "u", "au"]

        # Excludes vowels if an onset glide of the same quality is preceding
        if self.word_parts[-1].endswith("i"):
            choices.remove("i")
        if self.word_parts[-1].endswith("u"):
            choices.remove("u")

        return np.random.choice(choices, p=get_probs(choices, [0, 1, 0.3]))

    def _gen_coda_onset(self):
        """
        Generates a random syllable coda plus a following syllable onset
        Returns:
            the coda and onset as a string
        """
        choices = [
            "", "b", "c", "d", "g", "h",
            "l", "m", "n", "p", "r", "s", "t", "v",
            "bl", "cl", "fl", "gl", "pl",
            "br", "cr", "dr", "fr", "gr", "pr", "tr"
                                                "ll", "nn", "rr", "ss", "tt",
            "lb", "lc", "ld", "lg", "lm", "ln", "lp", "ls", "lt", "lv",
            "lbr", "lcr", "ldr", "lgr", "lpr", "ltr",
            "rb", "rc", "rd", "rg", "rm", "rn", "rp", "rs", "rt", "rv",
            "rbl", "rcl", "rfl", "rgl", "rpl",
            "mb", "nc", "nd", "ng", "mp", "ns", "nt",
            "mbl", "ncl", "ngl", "mpl",
            "mbr", "ncr", "ndr", "ngr", "mpr", "ntr",
            "ct", "ps", "pt", "x",
            "ctr", "ptr",
            "sc", "sp", "st",
            "scr", "spr", "str", "scl", "spl",
            "qu", "gn", "nqu", "squ",
            "nct", "mpt", "nst", "xt", "pst",
            "nctr", "mptr", "nstr", "xtr", "pstr"
        ]

        # If the preceding nucleus is au, codas starting with l, r, m or n or double consonants (except for ss) are not allowed
        if self.word_parts[-1] == "au":
            choices = [c for c in choices if len(c) > 0 and c[0] not in "lrmnt" and c != "ss"]
        
        # Excludes words that have r in both the onset and the coda
        if "r" in self.word_parts[-2]:
            choices = [c for c in choices if "r" not in c]

        # Excludes words that have l in both the onset and the coda
        if "l" in self.word_parts[-2]:
            choices = [c for c in choices if "l" not in c]

        return np.random.choice(choices, p=get_probs(choices, [0.4, 1, 0.4, 0.2, 0.1]))

    def no_suffix(self):
        """
        Adapts the word if there is no suffix
        """
        
        # Removes final empty onsets or h-onsets
        if self.word_parts[-1] == "h" or self.word_parts[-1] == "":
            self.word_parts = self.word_parts[:-1]
        
        # Words that end in ce or ge are ok
        if self.word_parts[-1].endswith("ce") or self.word_parts[-1].endswith("ge"):
            return

        # -a > -ay
        if self.word_parts[-1] in "a":
            self.word_parts[-1] += "y"
            return
        # -e > -ey
        if self.word_parts[-1] in "e":
            self.word_parts[-1] += "y"
            return
        # -o > -oe, -u > -ue
        if self.word_parts[-1] in "ou":
            self.word_parts[-1] += "e"
            return
        #  -i > -y
        if self.word_parts[-1] == "i":
            self.word_parts[-1] = "y"
            return

        # -Cr > -Cer
        if self.word_parts[-1].endswith("r") and len(self.word_parts[-1]) > 1 and self.word_parts[-1][-2] in "bcdfgjlmnpstv":
            self.word_parts[-1] = self.word_parts[-1][:-1] + "er"
            return
        # -Cl > -Cle
        if self.word_parts[-1].endswith("l") and len(self.word_parts[-1]) > 1 and self.word_parts[-1][-2] in "bcdfgjmnprstv":
            self.word_parts[-1] += "e"
            return

        # -qu > -cue
        if self.word_parts[-1].endswith("qu"):
            self.word_parts[-1] = self.word_parts[-1][:-2] + "cue"
            return
        # -gn > -gne
        if self.word_parts[-1].endswith("gn"):
            self.word_parts[-1] = self.word_parts[-1][:-2] + "ne"
            return
        # -nc > -nk
        if self.word_parts[-1].endswith("nc"):
            self.word_parts[-1] = self.word_parts[-1][:-2] + "nk"
            return

        # -v > ve, -ps > pse, -ns > nse, -rs > -rse, -ls > lse
        if any([self.word_parts[-1].endswith(c) for c in ["v", "ps", "ns", "rs", "ls"]]):
            self.word_parts += "e"
            return

        # -VC > -VCe
        if self.word_parts[-1] in "bcdfgjlmnprst" and self.word_parts[-2] != "au":
            self.word_parts[-1] += "e"
            return

    def __str__(self):
        """
        Returns:
            the root as a combined string
        """
        return "".join(self.word_parts)


class GreekRoot:
    """
    class for generating a word root with Greek origin
    """
    def __init__(self):
        """
        Generates a random English word root that looks as if it is a Greek loanword
        """
        self.word_parts = [self._gen_onset()]
        self.word_parts.append(self._gen_nucleus())
        if np.random.random() < 0.9:
            self.word_parts.append(self._gen_coda_onset())
            if np.random.random() < 0.3:
                self.word_parts.append(self._gen_nucleus())

    def _gen_onset(self):
        """
        Generates a random syllable onset
        Returns:
            the syllable onset as a string
        """
        choices = [
            "", "b", "c", "ch", "d", "g", "h", "l", "m",
            "n", "p", "ph", "rh", "s", "t", "th", "z",
            "bl", "cl", "chl", "gl", "pl", "phl",
            "br", "cr", "chr", "dr", "gr", "pr", "phr", "tr", "thr",
            "sc", "sch", "sm", "sp", "sph", "st", "sth",
            "scl", "spr", "sphr", "str",
            "ct", "chth", "pt", "phth", "x", "ps",
            "cn", "gn", "mn", "pn"
        ]

        return np.random.choice(choices, p=get_probs(choices, [5, 1, 0.4, 0.2, 0.2]))

    def _gen_nucleus(self):
        """
        Generates a random syllable nucleus
        Returns:
            the nucleus as a string
        """
        choices = [
            "a", "e", "i", "o", "u", "y",
            "au", "eu", "ei"
        ]

        # Excludes words starting with y
        if len(self.word_parts[-1]) == 0:
            choices.remove("y")

        # The second nucleus in a word is always short
        if len(self.word_parts) > 1:
            choices = [c for c in choices if len(c) == 1]

        return np.random.choice(choices, p=get_probs(choices, [0, 1, 0.8]))

    def _gen_coda_onset(self):
        """
        Generates a random syllable coda plus a following syllable onset
        Returns:
            the coda and onset as a string
        """

        choices_long = [
            "", "b", "c", "ch", "d", "g", "l", "m",
            "n", "p", "ph", "r", "s", "t", "th", "z",
            "bl", "cl", "chl", "gl", "pl", "phl", "tl", "thl",
            "br", "cr", "chr", "dr", "gr", "pr", "phr", "tr", "thr",
        ]
        choices_short = choices_long + [
            "sc", "sch", "sm", "sp", "sph", "st", "sth", "sb", "sd",
            "scl", "spr", "sphr", "str",
            "ct", "chth", "pt", "phth", "x", "ps",
            "cn", "gn", "mn", "pn", "chn", "dn", "phn", "tn", "thn",
            "cm", "chm", "dm", "gm", "lm", "rm", "sm", "tm", "thm",
            "bd", "gd",
            "mb", "nc", "nch", "nd", "ng", "mp", "mph", "ns", "nt", "nth",
            "mbr", "ncr", "nchr", "ndr", "ngr", "mpr", "mphr", "ntr", "nthr",
            "mbl", "ncl", "nchl", "ngl", "mpl", "mphl",
            "rb", "rc", "rch", "rd", "rg", "rp", "rph", "rs", "rt", "rth",
            "rbr", "rcr", "rchr", "rdr", "rgr", "rpr", "rphr", "rtr", "rthr",
            "rbl", "rcl", "rchl", "rgl", "rpl", "rphl",
            "lb", "lc", "lch", "ld", "lg", "lp", "lph", "ls", "lt", "lth",
            "cc", "cch", "ll", "mm", "nn", "pp", "pph", "rrh", "ss", "tt"
        ]

        # The choices differ depending on the length of the preceding nucleus
        choices = choices_long if len(self.word_parts[-1]) > 1 else choices_short

        # Excludes words that have r in both the onset and the coda
        if "r" in self.word_parts[-2]:
            choices = [c for c in choices if not c.startswith("r")]

        # Excludes words that have l in both the onset and the coda
        if "l" in self.word_parts[-2]:
            choices = [c for c in choices if not c.startswith("l")]

        return np.random.choice(choices, p=get_probs(choices, [0.4, 1, 0.4, 0.2, 0.1]))

    def __str__(self):
        """
        Returns:
            the root as a combined string
        """
        return "".join(self.word_parts)


if __name__ == "__main__":
    # Print 100 randomly generated words
    g = WordGenerator()
    for i in range(100):
        word, pos, root, affix_meanings = g.generate()
        print(word, pos, root, affix_meanings)



