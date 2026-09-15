"""V15 judge: which of six ACOUSTICALLY-distinguished segments does an answer describe?

`judge.attribute` is useless on the prosody strips by design — all six segments
speak the identical sentence, so content-word overlap is uniform and carries no
information. That is the whole point of the corpus: if targeting still works
here, something must be routing acoustic information rather than transcript
content.

Two endpoints, one per axis, both deterministic and API-free:

  emotion axis (`testbed/prosody_emotion`): the six segments carry six DISTINCT
    emotions, so an answer naming an emotion identifies exactly one segment.
    Chance = 1/6, directly comparable to every other steering number in this
    project.

  speaker axis (`testbed/prosody_speaker`): the six segments are six different
    actors reading the same line with the same emotion. RAVDESS gives us gender
    and nothing finer, and strips are gender-balanced 3/3 on purpose, so gender
    narrows to three segments rather than one. The endpoint is therefore whether
    the described gender MATCHES the targeted segment's, and **chance is 1/2,
    not 1/6**. Weaker, and labelled as such wherever it is reported — but a
    steering effect on it is still causal routing of a purely acoustic property,
    since content and emotion are held constant.

Both report an unattributed rate; an unattributed answer is never counted correct.

    seg, scores = attribute_emotion(answer, emotions)      # seg in 0..5 or -1
    ok, pred    = attribute_gender(answer, target_gender)  # ok in {True, False, None}
"""

import re

# RAVDESS labels -> words a caption model actually uses for them. Deliberately
# generous on synonyms and deliberately EMPTY of overlap between classes: a word
# that would fit two emotions (e.g. "upset" for both sad and angry) is omitted
# rather than assigned, so a vague answer goes unattributed instead of guessed.
_EMOTION_WORDS = {
    "neutral":   ["neutral", "flat", "monotone", "matter of fact", "unemotional",
                  "even toned", "plain", "deadpan"],
    "calm":      ["calm", "relaxed", "soothing", "gentle", "serene", "soft spoken",
                  "tranquil", "measured"],
    "happy":     ["happy", "happily", "cheerful", "joyful", "joyous", "delighted",
                  "upbeat", "excited", "enthusiastic", "pleased", "glad", "elated"],
    "sad":       ["sad", "sadly", "sorrowful", "unhappy", "melancholy", "mournful",
                  "gloomy", "depressed", "downcast", "dejected", "grief"],
    "angry":     ["angry", "angrily", "anger", "furious", "irritated", "annoyed",
                  "enraged", "irate", "hostile", "aggressive", "mad", "outraged"],
    "fearful":   ["fearful", "afraid", "scared", "frightened", "terrified",
                  "anxious", "nervous", "panicked", "alarmed", "fear"],
    "disgust":   ["disgust", "disgusted", "revolted", "repulsed", "contempt",
                  "contemptuous", "sickened", "distaste"],
    "surprised": ["surprised", "surprise", "astonished", "shocked", "amazed",
                  "startled", "stunned", "incredulous"],
}

_MALE = ["man", "male", "men", "he", "him", "his", "gentleman", "guy", "boy",
         "masculine", "deep voiced", "baritone"]
_FEMALE = ["woman", "female", "women", "she", "her", "hers", "lady", "girl",
           "feminine", "high pitched female", "soprano"]


def _norm(text):
    return " " + re.sub(r"[^a-z ]+", " ", text.lower()) + " "


def _count(text, words):
    t = _norm(text)
    return sum(1 for w in words if f" {w} " in t)


def attribute_emotion(answer, emotions):
    """emotions: the six segments' emotion labels, in seg_idx order.
    Returns (best segment index or -1, per-segment scores)."""
    scores = [_count(answer, _EMOTION_WORDS.get(e, [])) for e in emotions]
    best = max(scores)
    if best == 0:
        return -1, scores
    # a tie means the answer named two emotions; refuse rather than pick one
    if sum(1 for s in scores if s == best) > 1:
        return -1, scores
    return scores.index(best), scores


def attribute_gender(answer, target_gender):
    """Returns (correct or None if unattributed, predicted gender or None).
    Chance is 1/2 — this endpoint identifies a gender, not a segment."""
    m, f = _count(answer, _MALE), _count(answer, _FEMALE)
    if m == f:
        return None, None
    pred = "male" if m > f else "female"
    return pred == target_gender, pred


def attribute(answer, seg_labels, axis):
    """Uniform entry point mirroring judge.attribute's contract.

    emotion axis -> (segment index or -1, scores), chance 1/6.
    speaker axis -> (0/1 coded as a 'segment' is NOT meaningful), so this
    returns (-1, []) and callers must use attribute_gender directly. Raising
    would be worse: it would tempt a caller to fall back on the lexical judge,
    which is exactly the thing that does not work on this corpus.
    """
    if axis == "emotion":
        return attribute_emotion(answer, seg_labels)
    return -1, []
