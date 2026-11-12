"""LLM-as-judge for the generation metrics. JUDGE_VOTES calls per answer (the median counts), JSON in and out."""
import json
import re

from src.config import JUDGE_PROMPT_VERSION, JUDGE_VOTES
from src.generate.answer import NOT_FOUND
from src.generate.llm import LLM

SCORES = (0.0, 0.5, 1.0)

# v2: the reference split into required_answer and optional_details.
# v3: v2 plus explicit grading rules for the mistakes the gpt-4o-mini judge made with v2 (missing optional details
# marked down (q19), an accepted alternative value marked wrong (h08)) and the reasoning written before the scores.
JUDGE_PROMPTS = {
    "v2": """You are grading the answer of a retrieval-augmented question answering system.
You are given the question, the reference answer, the context the system retrieved, and the system's answer.

Score each criterion with 0, 0.5 or 1:
- correctness: 1 = the facts in the answer agree with the reference answer and nothing contradicts it; 0.5 = partly correct or has a minor error; 0 = wrong, contradicts the reference, or gives a value the reference marks as outdated.
- relevance: 1 = the answer addresses the question that was asked; 0.5 = partly on topic; 0 = off topic. An appropriate refusal ("the documents do not say") to an unanswerable question counts as relevant. A refusal to an answerable question does not.
- completeness: 1 = covers every key fact in the reference answer; 0.5 = covers some of them; 0 = covers none.
- faithfulness: 1 = every claim in the answer is supported by the context; 0.5 = mostly supported but some detail is not; 0 = makes claims that are not in the context (for example facts taken from the model's own knowledge). Judge this against the context only, NOT against the reference answer. A refusal that states nothing unsupported scores 1.
- abstained: true if the answer declines to answer or says the information is not available; otherwise false.

If the reference answer starts with "ABSTAIN", the question cannot be answered from the documents and the correct behaviour is to abstain.

Question type: {qtype}
Question: {question}
Reference answer: {expected}{optional}

Context retrieved by the system:
{context}

System answer:
{answer}

Reply with JSON only, in exactly this shape:
{{"correctness": 0, "relevance": 0, "completeness": 0, "faithfulness": 0, "abstained": false, "unsupported_claims": [], "reasoning": "one or two sentences"}}
""",
}

# v3 splits the judge in two calls, because with one prompt gpt-4o-mini let its own (wrong) reading of a badly parsed table
# overrule the reference (q19: "41.8 ... is correct according to the reference answer. However, the context indicates ...
# 41.0 ... Therefore, the answer is incorrect"). The reference call never sees the context and the faithfulness call never
# sees the reference, so neither can overrule the other.
REFERENCE_PROMPT = """You are grading the answer of a question answering system against a reference answer.

Score each criterion with 0, 0.5 or 1:
- correctness: 1 = the facts in the answer agree with the reference answer and nothing contradicts it; 0.5 = partly correct or has a minor error; 0 = wrong, contradicts the reference, or gives a value the reference marks as outdated.
- relevance: 1 = the answer addresses the question that was asked; 0.5 = partly on topic; 0 = off topic. An appropriate refusal ("the documents do not say") to an unanswerable question counts as relevant. A refusal to an answerable question does not.
- completeness: 1 = covers every key fact in the reference answer; 0.5 = covers some of them; 0 = covers none.
- abstained: true if the answer declines to answer or says the information is not available; otherwise false.

Grading rules:
1. The reference answer is the truth. Wording, length, order and citations such as [S1] do not matter; only the facts do.
2. Where the reference accepts more than one value ("41.8 or 41.0", "either value is correct"), an answer that gives any one accepted value is fully correct and fully complete for that fact: it does not need to give the others.
3. Do not lower any score for a missing detail that the reference does not ask for: where a value was reported, which section it is in, or extra background.
4. Extra facts are not errors unless they contradict the reference.
5. If the reference answer starts with "ABSTAIN", the question cannot be answered from the documents and the correct behaviour is to abstain.
6. A "No, <the document> does not mention <it>" answer is an answer, not an abstention, when the reference says the document does not mention it.

Question type: {qtype}
Question: {question}
Reference answer: {expected}{optional}

System answer:
{answer}

First write your reasoning: compare each fact in the answer with the reference. Then give the scores.
Reply with JSON only, in exactly this shape:
{{"reasoning": "one or two sentences", "correctness": 0, "relevance": 0, "completeness": 0, "abstained": false}}
"""

FAITHFULNESS_PROMPT = """You are checking whether the answer of a retrieval-augmented question answering system is supported by the context it retrieved. Do not judge whether the answer is right, only whether the context supports it.

- faithfulness: 1 = every claim in the answer is supported by the context; 0.5 = mostly supported but some detail is not; 0 = makes claims that are not in the context (for example facts taken from the model's own knowledge). A refusal ("the documents do not say") states nothing unsupported and scores 1. A statement that a document does not mention something is supported when the context from that document does not mention it.
- A number, name or statement that appears in the context is supported. Do not mark it unsupported because you are unsure which table column or row it belongs to (tables in the context can be badly formatted).
- Documents sometimes contradict themselves (for example a value in the text and a different value in a table). When the context gives different values for the same thing, a claim that matches any one of them is supported.
- unsupported_claims: the claims of the answer that the context does not support (an empty list when there are none).

Question: {question}

Context retrieved by the system:
{context}

System answer:
{answer}

First write your reasoning, then give the score.
Reply with JSON only, in exactly this shape:
{{"reasoning": "one or two sentences", "faithfulness": 0, "unsupported_claims": []}}
"""

SPLIT_PROMPTS = {"v3": (REFERENCE_PROMPT, FAITHFULNESS_PROMPT)}

# v4: the reference call no longer gives scores. It lists the key facts of the reference and marks each one, and the scores
# are computed from that list in code (score_facts). With v3, gpt-4o-mini's own scores were erratic on wording: it gave 0 to
# a correct answer because it added a true detail the reference does not ask for (n02: "from an average of 2 days to under
# 4 hours"). A yes/no answer whose verdict is the opposite of the question's expected_verdict is checked in code as well.
FACTS_PROMPT = """You are checking the answer of a question answering system against a reference answer.

1. List the key facts of the reference answer: each value, name, item or verdict (Yes / No) that the question asks for. Only the reference answer, not the optional details. Do not list a fact that the question itself already states: repeating the question earns nothing (for "Which project found X?", the key fact is the project, not X).
2. For each key fact, give its status in the system answer:
   - "present": the answer states it, in any wording. Where the reference accepts several values ("41.8 or 41.0", "either value is correct", "naming either project"), any one accepted value is present.
   - "missing": the answer does not state it.
   - "contradicted": the answer states a different value or the opposite verdict for the same thing.
3. List the wrong claims of the answer: claims that contradict the reference answer. A fact the reference does not mention is NOT a wrong claim (for example extra background or where a value comes from). When the question asks which items there are, the reference list is complete: an extra item in the answer is a wrong claim.
4. relevance: 1 = the answer addresses the question that was asked; 0.5 = partly on topic; 0 = off topic. A refusal to an answerable question is not relevant.
5. abstained: true if the answer declines to answer or says the information is not available. A "No, <the document> does not mention <it>" answer is an answer, not an abstention.

If the reference answer starts with "ABSTAIN", the question cannot be answered from the documents: list no facts, and only give relevance and abstained (a refusal is relevant).

Question: {question}
Reference answer: {expected}{optional}

System answer:
{answer}

Reply with JSON only, in exactly this shape:
{{"reasoning": "one or two sentences", "facts": [{{"fact": "<a key fact of the reference>", "status": "present"}}], "wrong_claims": [], "relevance": 1, "abstained": false}}
"""

# v4 faithfulness: the judge lists the answer's claims, each with an exact quote from the context, and code checks that the
# quote is really in the context (quote_in_context) and scores the share of supported claims (score_claims). Asked for a
# score directly, gpt-4o-mini mixed faithfulness up with correctness: it gave 0 to "41.8" and to "41.0" although the paper
# states both (q19, h08), and 0 to a sentence copied from the leave table because its Yes/No was wrong (q40).
CLAIMS_PROMPT = """You are checking whether each claim in the answer of a question answering system appears in the context it retrieved. Do not judge whether the answer is right, only whether the context states each claim.

1. List the factual claims of the answer (values, names, statements). A refusal ("the documents do not say", "not found") makes no claim. A claim that a document does not mention something is supported when none of the context from that document mentions it: give the quote "" and supported_by_absence true.
2. For each claim, copy the exact words from the context that state it: a short, contiguous piece of the context, copied character for character (a table row may be copied as it appears). If the context does not state the claim, give the quote "".
3. A claim that is a calculation or comparison made from values in the context (a difference, a ratio, "larger", "at least", "all of them") is derived: give the quote "" and, in derived_from, the exact quotes of the values it is made from. Do not check whether the calculation is right.
4. Documents sometimes contradict themselves (a value in the text and a different value in a table). A claim that matches any one of the values the context gives is stated by the context: quote that place.

Question: {question}

Context retrieved by the system:
{context}

System answer:
{answer}

Reply with JSON only, in exactly this shape:
{{"claims": [{{"claim": "<a claim of the answer>", "quote": "<exact words from the context, or empty>", "derived_from": [], "supported_by_absence": false}}]}}
"""

FACT_PROMPTS = {"v4": (FACTS_PROMPT, CLAIMS_PROMPT)}
WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
# Absence supports only a negative claim ("the resume does not mention X"); the judge also set supported_by_absence on a plain
# positive claim it could not find ("ReleaseForge AI uses Ollama", n16).
COMPARISON = re.compile(r"\b(larger|smaller|higher|lower|more|less|fewer|greater|bigger|faster|slower|longer|shorter|exceeds?|"
                        r"at least|at most|both|all|every|each|same|equal|twice|times|difference|outperforms?|better|worse)\b", re.I)
NEGATION = re.compile(r"\b(not|no|never|none|nothing|without|doesn't|isn't|aren't|didn't)\b", re.I)


def _flat(text: str) -> str:
    """Lower case, table pipes and markup removed, spaces collapsed: a quote and the context compare in this form."""
    return re.sub(r"\s+", " ", re.sub(r"[|*#`>]", " ", text.lower())).strip()


def quote_in_context(quote: str, context: str) -> bool:
    """True when the quote is in the context: verbatim (after _flat), or, for a quote the judge retyped slightly, when every
    number in it and at least 80% of its words occur in the context."""
    quote, flat = _flat(quote), _flat(context)
    if len(quote) < 3:
        return False
    if quote in flat:
        return True
    words, context_words = WORD.findall(quote), set(WORD.findall(flat))
    numbers = [w for w in words if any(ch.isdigit() for ch in w)]
    return bool(words) and all(n in context_words for n in numbers) and sum(w in context_words for w in words) >= 0.8 * len(words)


NUMBER = re.compile(r"\d+(?:\.\d+)?")


def calculated_from_answer(claim: str, context: str, answer: str) -> bool:
    """True when the claim states numbers and each one is in the context or is the difference, sum or ratio of two numbers
    the answer states (q34: "larger by approximately 31.5 million" = 36 - 4.5). The judge splits such a sentence into a
    fragment with no subject, which neither the quote check nor the entailment check can tie to its source values."""
    numbers = [float(n) for n in NUMBER.findall(claim.replace(",", ""))]
    if not numbers:
        return False
    in_context = {float(n) for n in NUMBER.findall(context.replace(",", ""))}
    stated = sorted({float(n) for n in NUMBER.findall(answer.replace(",", ""))} - set(numbers))

    def derived(x: float) -> bool:
        close = lambda a, b: abs(a - b) <= 0.01 * max(abs(b), 1e-9)
        return any(close(x, a - b) or close(x, a + b) or (b and close(x, a / b)) for a in stated for b in stated if a != b)
    return all(n in in_context or derived(n) for n in numbers)


def claim_supported(claim: dict, context: str) -> bool:
    """A claim is supported by its own quote, by the quotes of the values it is calculated from (q34: "larger by about 31.5
    million" from 36M and 4.5 million), or, for a negative claim only, by the absence of what it says is not mentioned.
    gpt-4o-mini rarely fills derived_from, so a comparison claim with no quote at all counts as derived from the answer's
    other claims; whether the comparison is right is the correctness judge's job (a flipped Yes/No scores 0 there)."""
    sources = claim.get("derived_from") if isinstance(claim.get("derived_from"), list) else []
    text, quote = claim.get("claim", ""), claim.get("quote") or ""
    return bool(quote_in_context(quote, context)
                or sources and all(quote_in_context(q or "", context) for q in sources)
                or not quote and not sources and COMPARISON.search(text)
                or claim.get("supported_by_absence") and NEGATION.search(text))


ENTAILMENT_PROMPT = """For each numbered claim, decide whether the context states it or directly implies it. Reading a table counts: a row and its column headers state the row's values (a "No" under "Carry Forward" states that the leave cannot be carried forward). A comparison of two values that the context gives counts, and so does a calculation from them (a difference, a ratio, a sum: "larger by about 31.5 million" from 36M and 4.5 million). A claim that needs knowledge from outside the context, or names something the context does not connect to it, is not supported.

Context:
{context}

Claims:
{claims}

Reply with JSON only: one entry per claim, keyed by its number, in exactly this shape:
{{"verdicts": {{"1": "supported", "2": "not supported"}}}}
"""


def entailed(judge: LLM, claims: list[str], context: str, question: dict, retries: int) -> tuple[list[bool], dict]:
    """Second chance for claims whose quote could not be verified: one call asks, claim by claim, whether the context states
    or directly implies it. With quotes only, correct answers read from a table or rephrased failed (q08, q16, q38, q44)."""
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, start=1))
    try:
        data, tokens = _ask(judge, ENTAILMENT_PROMPT.format(context=context, claims=numbered), ("verdicts",), question,
                            retries, valid=lambda d: isinstance(d["verdicts"], (dict, list)))
    except ValueError:                      # no usable reply: the claims stay unsupported, the verdict is not lost
        return [False] * len(claims), {"in": 0, "out": 0}
    verdicts = data["verdicts"]
    if isinstance(verdicts, list):          # a plain list in claim order is accepted too
        verdicts = {str(i): v for i, v in enumerate(verdicts, start=1)}
    # a claim the reply leaves out counts as not supported
    return [str(verdicts.get(str(i), "")).strip().lower() == "supported" for i in range(1, len(claims) + 1)], tokens


def score_claims(claims: list[dict], context: str, rechecked: dict | None = None) -> tuple[float, list[str]]:
    """Faithfulness from the judge's claim list: 1 = every claim supported (or no claim at all, as in a refusal), 0.5 = at
    least half of them, 0 = fewer. A claim is supported when its quote is verified in the context, when it says a document
    does not mention something (the judge's supported_by_absence, accepted only for a negative claim), or when the
    entailment check (`rechecked`: claim -> supported) found it stated or implied."""
    rechecked = rechecked or {}
    unsupported = [c.get("claim", "") for c in claims
                   if not claim_supported(c, context) and not rechecked.get(c.get("claim", ""))]
    if not claims or not unsupported:
        return 1.0, []
    supported = len(claims) - len(unsupported)
    return (0.5 if supported * 2 >= len(claims) else 0.0), unsupported
VERDICT = re.compile(r"^\W*(yes|no)\b", re.I)


def verdict_flipped(question: dict, answer: str) -> bool:
    """True when a yes/no answer starts with the opposite of the question's expected_verdict."""
    expected = question.get("expected_verdict")
    match = VERDICT.match(answer)
    return bool(expected and match and match.group(1).lower() != expected.lower())


def score_facts(question: dict, answer: str, facts: list[dict], wrong_claims: list) -> dict:
    """Correctness and completeness from the judge's fact list.

    completeness  1 = every key fact present, 0.5 = some, 0 = none: how much of the reference the answer covers
    correctness   whether what the answer says is right: 0 when the yes/no verdict is the opposite of the expected one;
                  with a contradicted fact or a wrong claim, 0.5 when some key fact is present and 0 when none is; with
                  nothing wrong, 1 when at least one key fact is present (an incomplete answer is incomplete, not wrong:
                  q44 left out the exact latencies) and 0 when none is (the answer does not answer)
    """
    present = sum(f.get("status") == "present" for f in facts)
    contradicted = sum(f.get("status") == "contradicted" for f in facts)
    completeness = 1.0 if facts and present == len(facts) else 0.5 if present else 0.0
    if verdict_flipped(question, answer):
        correctness = 0.0
    elif contradicted or wrong_claims:
        correctness = 0.5 if present else 0.0
    else:
        correctness = 1.0 if present else 0.0
    return {"correctness": correctness, "completeness": completeness}
JUDGE_PROMPT = JUDGE_PROMPTS.get(JUDGE_PROMPT_VERSION)

REQUIRED = ("correctness", "relevance", "completeness", "faithfulness", "abstained")

OPTIONAL_BLOCK = """
Optional details (in the source documents, but the question does not ask for them): {details}
Do not require these for completeness and do not lower any score because they are missing. Do not mark them as wrong if the answer includes them and the context supports them."""


def reference_parts(question: dict) -> tuple[str, str]:
    """The reference the judge grades against, and the optional-details block ('' when the question has none)."""
    required = question.get("required_answer") or question["expected_answer"]
    details = question.get("optional_details")
    return required, OPTIONAL_BLOCK.format(details=details) if details else ""


def _snap(value) -> float:
    return min(SCORES, key=lambda s: abs(s - float(value)))


def parse_verdict(text: str, question: dict) -> dict:
    data = json.loads(text)
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise ValueError(f"judge reply is missing {missing}")
    abstained = data["abstained"]
    if isinstance(abstained, str):
        abstained = abstained.strip().lower() == "true"
    verdict = {
        "correctness": _snap(data["correctness"]),
        "relevance": _snap(data["relevance"]),
        "completeness": _snap(data["completeness"]),
        "faithfulness": _snap(data["faithfulness"]),
        "abstained": bool(abstained),
        "unsupported_claims": data.get("unsupported_claims") or [],
        "reasoning": data.get("reasoning", ""),
    }
    if question["type"] == "unanswerable":
        # Nothing to be complete about; refusing is the only correct behaviour.
        verdict["correctness"] = 1.0 if verdict["abstained"] else 0.0
        verdict["completeness"] = None
        if verdict["abstained"] and not verdict["unsupported_claims"]:
            # The judge prompt already says a refusal here is relevant and states nothing unsupported, but the judge
            # still scores it 0 now and then (h12): a correct refusal is fixed to 1 instead of left to judge noise.
            verdict["relevance"] = 1.0
            verdict["faithfulness"] = 1.0
    return verdict


def combine_votes(verdicts: list[dict]) -> dict:
    """One verdict from several judge calls on the same answer: the median of each score, the majority for abstained.
    A single call is noisy: gpt-4o scored the same, word-for-word identical h08 answer 1.0 once and 0.5 four times."""
    if len(verdicts) == 1:
        return verdicts[0]

    def median(values):
        values = sorted(v for v in values if v is not None)
        return values[len(values) // 2] if values else None

    out = dict(verdicts[0])
    for m in ("correctness", "relevance", "completeness", "faithfulness"):
        out[m] = median(v[m] for v in verdicts)
    out["abstained"] = sum(v["abstained"] for v in verdicts) * 2 > len(verdicts)
    out["votes"] = [{m: v[m] for m in ("correctness", "relevance", "completeness", "faithfulness")} for v in verdicts]
    return out


def judge_answer(judge: LLM, question: dict, answer: str, chunks: list[dict], retries: int = 1,
                 votes: int = JUDGE_VOTES, found: bool | None = None) -> dict:
    """The judge's verdict on one answer: the median of `votes` calls (see combine_votes). `found` is the answer model's own
    flag (False: it said the documents do not hold the answer), which counts as a refusal whatever the judge reads."""
    verdicts = [_judge_once(judge, question, answer, chunks, retries, found=found) for _ in range(votes)]
    verdict = combine_votes(verdicts)
    verdict["judge_tokens"] = {k: sum(v["judge_tokens"][k] or 0 for v in verdicts) for k in ("in", "out")}
    return verdict


def _judge_once(judge: LLM, question: dict, answer: str, chunks: list[dict], retries: int,
                version: str = JUDGE_PROMPT_VERSION, found: bool | None = None) -> dict:
    context = "\n\n".join(f"[{c.get('source_id') or c['chunk_id']}] {c['text']}" for c in chunks)
    expected, optional = reference_parts(question)
    fields = dict(qtype=question["type"], question=question["question"], expected=expected, optional=optional,
                  context=context, answer=answer)
    refused = found is False or answer.strip() == NOT_FOUND
    if version in FACT_PROMPTS:
        facts_prompt, faithfulness_prompt = FACT_PROMPTS[version]
        unanswerable = question["type"] == "unanswerable"
        keys = ("relevance", "abstained") if unanswerable else ("facts", "relevance", "abstained")
        graded, tokens_a = _ask(judge, facts_prompt.format(**fields), keys, question, retries,
                                valid=lambda d: unanswerable or (isinstance(d["facts"], list) and len(d["facts"]) > 0))
        listed, tokens_b = _ask(judge, faithfulness_prompt.format(**fields), ("claims",), question, retries,
                                valid=lambda d: isinstance(d["claims"], list))
        claims = [] if refused else [c for c in listed["claims"] if isinstance(c, dict)]
        unquoted = [c.get("claim", "") for c in claims if not claim_supported(c, context)]
        calculated = [c for c in unquoted if calculated_from_answer(c, context, answer)]
        doubtful = [c for c in unquoted if c not in calculated]
        rechecked, tokens_c = {c: True for c in calculated}, {"in": 0, "out": 0}
        if doubtful:
            verdicts, tokens_c = entailed(judge, doubtful, context, question, retries)
            rechecked.update(zip(doubtful, verdicts))
        faithfulness, unsupported = score_claims(claims, context, rechecked)
        tokens_b = {k: (tokens_b[k] or 0) + (tokens_c[k] or 0) for k in ("in", "out")}
        checked = {"faithfulness": faithfulness, "unsupported_claims": unsupported,
                   "reasoning": f"{len(claims)} claims checked, {len(doubtful)} by entailment"}
        facts = [] if unanswerable else [f for f in graded["facts"] if isinstance(f, dict)]
        wrong = graded.get("wrong_claims") or []
        if unanswerable:
            scores = {"correctness": 0, "completeness": 0}          # parse_verdict scores the refusal
        elif refused:
            # A refusal states none of the reference's facts. The judge counted "Information not found" as matching a
            # reference of "the resume does not mention X" (h17); the right reply there is "No, it is not mentioned".
            scores = {"correctness": 0.0, "completeness": 0.0}
        else:
            scores = score_facts(question, answer, facts, wrong)
        reasoning = f"{graded.get('reasoning', '')} | faithfulness: {checked.get('reasoning', '')}"
        verdict = parse_verdict(json.dumps({**graded, **checked, **scores, "abstained": refused or graded["abstained"],
                                            "reasoning": reasoning}), question)
        verdict.update(facts=facts, wrong_claims=wrong, verdict_flipped=verdict_flipped(question, answer), claims=claims)
        tokens = {k: (tokens_a[k] or 0) + (tokens_b[k] or 0) for k in ("in", "out")}
    elif version in SPLIT_PROMPTS:
        reference_prompt, faithfulness_prompt = SPLIT_PROMPTS[version]
        graded, tokens_a = _ask(judge, reference_prompt.format(**fields),
                                ("correctness", "relevance", "completeness", "abstained"), question, retries)
        checked, tokens_b = _ask(judge, faithfulness_prompt.format(**fields), ("faithfulness",), question, retries)
        if refused:
            # The answer model's found=false (or the pipeline's own refusal text) is a refusal by definition; the reference
            # call marked it as not abstained on every unanswerable question of the fresh held-out set (n23-n25).
            graded["abstained"] = True
        reasoning = f"{graded.get('reasoning', '')} | faithfulness: {checked.get('reasoning', '')}"
        verdict = parse_verdict(json.dumps({**graded, **checked, "reasoning": reasoning}), question)
        tokens = {k: (tokens_a[k] or 0) + (tokens_b[k] or 0) for k in ("in", "out")}
    else:
        data, tokens = _ask(judge, JUDGE_PROMPTS[version].format(**fields), REQUIRED, question, retries)
        verdict = parse_verdict(json.dumps(data), question)
    verdict.update(judge_model=judge.name, judge_prompt=version, judge_tokens=tokens)
    return verdict


def _ask(judge: LLM, prompt: str, keys: tuple, question: dict, retries: int, valid=None) -> tuple[dict, dict]:
    """One judge call, retried when the reply is not JSON with `keys` (or fails `valid`). Returns the reply and its tokens."""
    last_error = None
    for _ in range(retries + 1):
        reply = judge.complete(prompt, json_mode=True)
        try:
            data = json.loads(reply["text"])
            missing = [k for k in keys if k not in data]
            if missing:
                raise ValueError(f"judge reply is missing {missing}")
            if valid and not valid(data):
                raise ValueError("judge reply does not have the expected content")
        except (ValueError, TypeError) as exc:      # json errors are ValueErrors
            last_error = exc
            continue
        return data, {"in": reply["input_tokens"], "out": reply["output_tokens"]}
    raise ValueError(f"judge gave an unusable reply for {question['id']}: {last_error}")
