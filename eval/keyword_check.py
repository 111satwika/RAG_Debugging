"""Keyword and verdict checks for the golden set.

must_include: each item is a string, or a list of alternatives (any one must match).
expected_verdict: "yes" / "no" -> the answer must start with Yes / No.
Unanswerable questions (no source_docs) must have found == false; others found == true.
"""
import re

CITATION = re.compile(r"\[S\d+(?:\s*,\s*S?\d+)*\]")


def normalize(text: str) -> str:
    t = CITATION.sub("", text).lower()
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)      # 37,000 -> 37000
    t = t.replace("^2", "²").replace("*", "·").replace("⋅", "·")
    t = re.sub(r"[\s\-]+", "", t)                 # drop spaces and hyphens
    return t


def _match(item, norm_answer):
    options = item if isinstance(item, list) else [item]
    return any(normalize(o) in norm_answer for o in options)


def matches(item, text: str) -> bool:
    """Does a key fact (a string, or a list of alternatives) appear in `text`, by the same rules the answer check uses."""
    return _match(item, normalize(text))


def keyword_check(q: dict, answer: str, found: bool) -> dict:
    norm = normalize(answer)
    missing = [item for item in q.get("must_include", []) if not _match(item, norm)]
    wrong = [item for item in q.get("must_not_include", []) if _match(item, norm)]

    verdict_ok = True
    if q.get("expected_verdict"):
        first = re.sub(r"^[^a-z]+", "", CITATION.sub("", answer).lower())
        verdict_ok = first.startswith(q["expected_verdict"])

    found_ok = found == bool(q["source_docs"])

    return {
        "pass": not missing and not wrong and verdict_ok and found_ok,
        "missing": missing,
        "wrong": wrong,
        "verdict_ok": verdict_ok,
        "found_ok": found_ok,
    }


if __name__ == "__main__":
    import json, sys
    golden = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "golden_set.json", encoding="utf-8"))
    bad = 0
    for q in golden:
        if q["type"] == "unanswerable":
            continue
        ans = q.get("required_answer", q["expected_answer"]) + " " + q.get("optional_details", "")
        r = keyword_check(q, ans, found=True)
        if not r["pass"]:
            bad += 1
            print(q["id"], r)
    print("self-test:", "all reference answers pass" if not bad else f"{bad} failed")
