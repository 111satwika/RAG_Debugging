"""BM25 keyword search: scores chunks by the query words they contain, so exact terms (IDs, names, error codes) match.

Dense search matches meaning and can miss an exact string such as 'QL-1003'; BM25 matches the string itself and
cannot match a paraphrase. Hybrid search (see hybrid.py) uses both.
"""
import math
import re
from collections import Counter

K1 = 1.5     # how fast repeating a word stops helping
B = 0.75     # how much a long chunk is penalised
TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")      # keeps QL-1003, d_model and 0.831 whole
# Filler words. With a few dozen chunks BM25 cannot learn that they are uninformative, and a question made of them
# ("what is the ...") would otherwise match unrelated chunks and pull them into the hybrid ranking.
STOPWORDS = frozenset("""a about after all also am an and any are as at be been being but by can could did do does for from
had has have how i if in into is it its me my no not of on or our s so than that the their them then there these they this
to was we were what when where which who whom why will with would you your""".split())


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    """Lowercase words. A joined token such as 'ql-1003' is kept and also split ('ql', '1003'), so a query for
    either the whole code or a part of it matches."""
    tokens = []
    for match in TOKEN.finditer(text.lower()):
        token = match.group()
        tokens.append(token)
        parts = re.split(r"[-_.]", token)
        if len(parts) > 1:
            tokens.extend(part for part in parts if part)
    return [t for t in tokens if t not in STOPWORDS] if drop_stopwords else tokens


class BM25Index:
    def __init__(self, chunks: list[dict], k1: float = K1, b: float = B):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.counts = [Counter(tokenize(chunk["text"])) for chunk in chunks]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.avg_length = sum(self.lengths) / max(len(chunks), 1)
        document_frequency = Counter(term for count in self.counts for term in count)
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in document_frequency.items()}

    def score(self, query_terms: list[str], index: int) -> float:
        count, length = self.counts[index], self.lengths[index]
        total = 0.0
        for term in query_terms:
            frequency = count.get(term, 0)
            if frequency:
                norm = frequency + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                total += self.idf[term] * frequency * (self.k1 + 1) / norm
        return total

    def search(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[dict]:
        """The k best chunks with a score above zero (a chunk with no query word is never returned)."""
        terms = list(dict.fromkeys(tokenize(query)))
        scored = []
        for index, chunk in enumerate(self.chunks):
            if doc_ids and chunk["doc_id"] not in doc_ids:
                continue
            score = self.score(terms, index)
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [{**self.chunks[index], "rank": rank, "score": round(score, 4)}
                for rank, (score, index) in enumerate(scored[:k], start=1)]
