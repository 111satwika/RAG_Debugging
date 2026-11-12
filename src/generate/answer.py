"""The structured answer of the cited prompts (v2 and later): the answer text with [S#] citations, the sources it cites and
whether the documents contained the answer. The model's JSON is validated here, against the sources that were in the prompt."""
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

CITATION = re.compile(r"\[(S\d+)\]")
NOT_FOUND = "Information not found in the provided documents."


class AnswerStream:
    """Pull the text of the "answer" field out of a JSON reply while it is still being written, so the user sees the answer
    stream instead of the JSON around it (and the "checks" written before it). feed() takes the next raw piece and returns
    the new answer characters, decoded; escapes split across pieces are held back until complete.

    A reply that is not JSON at all (plain prompts) is passed through as it is.
    """
    KEY = re.compile(r'"answer"\s*:\s*"')
    ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}

    def __init__(self, json_reply: bool = True):
        self.json_reply = json_reply
        self.raw = ""
        self.pos = None          # index in raw of the next answer character to decode; None until the key is seen
        self.done = False

    def feed(self, piece: str) -> str:
        if not self.json_reply:
            return piece
        self.raw += piece
        if self.done:
            return ""
        if self.pos is None:
            match = self.KEY.search(self.raw)
            if not match:
                return ""
            self.pos = match.end()
        out = []
        while self.pos < len(self.raw):
            ch = self.raw[self.pos]
            if ch == '"':
                self.done = True
                break
            if ch != "\\":
                out.append(ch)
                self.pos += 1
                continue
            if self.pos + 1 >= len(self.raw):
                break                                      # the escape is not complete yet
            nxt = self.raw[self.pos + 1]
            if nxt == "u":
                if self.pos + 6 > len(self.raw):
                    break
                out.append(chr(int(self.raw[self.pos + 2:self.pos + 6], 16)))
                self.pos += 6
            else:
                out.append(self.ESCAPES.get(nxt, nxt))
                self.pos += 2
        return "".join(out)


class Answer(BaseModel):
    # Scratch work the model writes before the answer; never shown, so any shape is accepted: with list[str], a reply whose
    # checks came as a list of lists failed validation and the raw JSON was shown as the answer (q34 in the v9 run).
    evidence: list[Any] = Field(default_factory=list)     # v2.6 to v2.8: the facts copied out before the answer is written
    checks: list[Any] = Field(default_factory=list)       # v2.9 and later: item-by-item checks and attribution checks
    answer: str
    sources: list[str] = Field(default_factory=list)      # source ids such as "S1"
    found: bool = True


class ParsedAnswer(BaseModel):
    """What the pipeline keeps of a reply: the text to show, the cited source ids and how well the reply followed the format."""
    text: str
    found: bool | None                  # None when the reply was not valid JSON
    sources: list[str]                  # cited ids that exist in the prompt, in the order they were listed
    status: Literal["ok", "flawed", "invalid"]
    issues: list[str] = Field(default_factory=list)


def parse_answer(reply: str, valid_ids: list[str]) -> ParsedAnswer:
    """Turn the model's reply into a ParsedAnswer.

    ok       valid JSON, every listed source exists, the text cites only sources that exist and every listed source is cited
    flawed   valid JSON with a problem (a source that does not exist, a claim without a source, a source that is listed but
             not cited, "not found" with sources): the unusable ids are dropped and the problems are recorded in `issues`
    invalid  not JSON of the right shape: the raw reply is kept as the answer and nothing is cited
    """
    try:
        data = Answer.model_validate(json.loads(reply))
    except (ValueError, ValidationError) as exc:
        return ParsedAnswer(text=reply, found=None, sources=[], status="invalid", issues=[f"not a valid answer: {type(exc).__name__}"])

    issues = []
    if data.answer.strip().startswith(("{", "[")) and not CITATION.match(data.answer.strip()):
        issues.append("the answer is JSON or code, not text")      # q05: '{"DevPilot AI": {"tool_servers": 6}, ...}'
    listed = list(dict.fromkeys(s.strip().strip("[]") for s in data.sources))
    unknown = [s for s in listed if s not in valid_ids]
    if unknown:
        issues.append(f"cites sources that were not provided: {', '.join(unknown)}")
    cited_in_text = [s for s in dict.fromkeys(CITATION.findall(data.answer)) if s in valid_ids]
    bad_in_text = [s for s in dict.fromkeys(CITATION.findall(data.answer)) if s not in valid_ids]
    if bad_in_text:
        issues.append(f"the text cites sources that were not provided: {', '.join(bad_in_text)}")
    sources = [s for s in listed if s in valid_ids]
    if data.found and not sources:
        issues.append("an answer was given without a source")
    if not data.found and sources:
        issues.append("the answer says nothing was found but lists sources")
        sources = []
    missing = [s for s in sources if s not in cited_in_text]
    if missing and data.found:
        issues.append(f"sources listed but not cited in the text: {', '.join(missing)}")
    return ParsedAnswer(text=data.answer, found=data.found, sources=sources, status="flawed" if issues else "ok", issues=issues)
