"""Segment judge: which of the six segments does a generated answer describe?

Free and deterministic — no API. The testbed's segments come with their
LibriSpeech transcripts, so attribution is lexical: content-word overlap
between the answer and each segment's transcript, weighting words by how
few segments contain them (a word unique to one segment is strong evidence;
a word in all six is none). Chance = 1/6.

    seg, scores = attribute(answer, transcripts)   # seg in 0..5, or -1 if no evidence

Used for (a) E2's steering metric (does the answer describe the biased-to
segment?), (b) the unsteered per-arm accuracy on "what does segment k talk
about", and (c) aligning free descriptions to segments for the
generation-conditioned E1 re-score.

Known limits: paraphrase-heavy answers ("a man talks about his feelings")
may score 0 → -1 (reported separately, never counted as correct); a spelled
name ("Mister Marshall" vs "Mr. Marshall") is handled by a few normalisations.
"""

import math
import re
from collections import Counter

_STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being
am do does did doing have has had having it its this that these those he she they them his her their
there here what which who whom whose why how when where i you we me my your our us not no nor so than
too very can will just about into over under again further then once all any both each few more most
other some such only own same s t don should now up down out off very also than then them very
one two three four five six seven eight nine ten first second third fourth fifth sixth last speaker
speakers segment segments talk talks talking about says said say saying speaks speaking mentions mentioned
describes description audio recording person people man woman someone something thing things topic
part parts number sixth voice voices tells telling told discuss discusses discussing""".split())

_NORM = {"mr": "mister", "mrs": "missus", "dr": "doctor", "st": "saint", "&": "and"}


def _tokens(text):
    text = text.lower()
    text = re.sub(r"[’']", "", text)
    words = re.findall(r"[a-z0-9]+", text)
    out = []
    for w in words:
        w = _NORM.get(w, w)
        if w in _STOP or len(w) < 3:
            continue
        # crude stemming so "embraced"/"embrace", "kings"/"king" meet
        for suf in ("ingly", "edly", "ing", "ed", "es", "s", "ly"):
            if len(w) > 4 and w.endswith(suf):
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


def attribute(answer, transcripts, min_evidence=1.0):
    """Returns (best_segment_index or -1, per-segment scores)."""
    a = set(_tokens(answer))
    segs = [set(_tokens(t)) for t in transcripts]
    n = len(segs)
    df = Counter(w for s in segs for w in s)
    scores = []
    for s in segs:
        common = a & s
        # idf over segments: unique-to-one-segment words get log(6/1)=1.79, shared-by-all get 0
        scores.append(sum(math.log(n / df[w]) for w in common))
    best = max(range(n), key=lambda i: scores[i])
    if scores[best] < min_evidence:
        return -1, scores
    # tie -> no decision
    if sorted(scores)[-2] == scores[best]:
        return -1, scores
    return best, scores


if __name__ == "__main__":
    # smoke test on the two probes seen in the first runs
    T = ["IT IS BLACK IN MISFORTUNE IT IS BLACKER STILL IN CRIME THESE TWO BLACKNESSES AMALGAMATED COMPOSE SLANG",
         "SO NO TALES GOT OUT TO THE NEIGHBORS BESIDES IT WAS A LONELY PLACE AND BY GOOD LUCK NO ONE CAME THAT WAY",
         "HE SMILED GUILTILY AS HE ADDED BUT I MUST ADMIT I WAS MORE THAN A LITTLE CONCERNED MYSELF",
         "BUT SOFTLY AS THE NAME WAS BREATHED MARY GRANT ALREADY AWAKENED BY THE SOUNDS IN THE HUT SLIPPED OVER",
         "I'M GOING TO SEE MISTER MARSHALL SAID KENNETH AND DISCOVER WHAT I CAN DO TO ASSIST YOU THANK YOU SIR",
         "SLOWLY BUT STEADILY THE SLENDER LINE WAS PAID OUT AMID A TENSE SILENCE ON THE PART OF THE LITTLE GROUP"]
    for ans in ["The speaker talks about going to see Mr. Marshall and discovering what they can do to assist.",
                "It is blacker still in crime, these two blacknesses.",
                "A man talks about his feelings.",
                "The last speaker, Kenneth, talks about going to see Mister Marshall."]:
        print(attribute(ans, T)[0], [round(x, 2) for x in attribute(ans, T)[1]], "|", ans)
