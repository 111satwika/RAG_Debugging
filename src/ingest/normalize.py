"""Step 1b: normalize parsed text before chunking (cleanup only, structure is kept)."""
import re
import unicodedata
from collections import Counter

PICTURE_TEXT = re.compile(r"<!-- Start of picture text -->.*?<!-- End of picture text -->", re.S)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
BOLD = re.compile(r"\*\*(.+?)\*\*")
SPLIT_IDENTIFIER = re.compile(r"_(\w+)_\s+(\w+)_")  # parser turns _warmup_steps_ into "_warmup_  steps_"
ITALIC = re.compile(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)")  # _x_ but not warmup_steps
UNDERLINE = re.compile(r"</?u>")
SPLIT_DECIMAL = re.compile(r"(?<=\d) \. (?=\d)")  # italic decimal point extracts as "0 . 98"
SUP = re.compile(r"<sup>(.*?)</sup>")
SUB = re.compile(r"<sub>(.*?)</sub>")
BR = re.compile(r"\s*<br\s*/?>\s*")
PAGE_NUMBER = re.compile(r"^(page\s*)?\d+(\s*(of|/)\s*\d+)?$", re.I)
EXPONENT = re.compile(r"^[+-]?\w+$")
# Table cleanup. The parser leaves italic underscores inside numbers ("1_.0·_10^20", "params _×_10^6", "|_d_ff|") and splits a
# value that spans two columns over both cells ("|3_.3·_|10^18|", Transformer rows of the paper's Table 2), which shifts it
# into the next column.
NUMBER_UNDERSCORE = re.compile(r"(?<=[\d·])_(?=[.·\d])|(?<=[.·])_(?=\d)|(?<=·)_|_(?=×)|(?<=×)_")
CELL_UNDERSCORE = re.compile(r"(?<=\|)_(?=[A-Za-z])")
SPLIT_SPAN_CELL = re.compile(r"(\d·)\|(10\^-?\d+)\|$", re.M)
# Headers the parser cannot rebuild: a multi-row header whose spanning cells it cuts into fragments ("BL|EU", "Training C|ost")
# and whose sub-header rows ("|Model|||||", "||EN-DE|...") it emits as data rows. Without the fix nothing says which column is
# English-to-French BLEU, and both the answer model and the judge misread the table (n12, q19). Exact text after cleanup.
TABLE_FIXES = {
    "||BL|EU|Training C|ost (FLOPs)|\n|---|---|---|---|---|\n|Model|||||\n||EN-DE|EN-FR|EN-DE|EN-FR|":
        "|Model|BLEU EN-DE|BLEU EN-FR|Training Cost (FLOPs) EN-DE|Training Cost (FLOPs) EN-FR|\n|---|---|---|---|---|",
}

CHAR_MAP = str.maketrans({
    "\u2212": "-",                  # unicode minus -> hyphen
    "\u2018": "'", "\u2019": "'",   # curly single quotes
    "\u201c": '"', "\u201d": '"',   # curly double quotes
    "\u00a0": " ", "\u2009": " ", "\u202f": " ", "\u200b": "",
    "\u00ad": "",                   # soft hyphen
})


def _key(line: str) -> str:
    """Line identity ignoring digits, so 'Page 1' and 'Page 2' count as the same line."""
    return re.sub(r"\d+", "#", line.strip().lower())


def find_boilerplate(pages: list[dict], edge: int = 3) -> set[str]:
    """Lines near the top/bottom of a page that repeat on 2+ pages (headers, footers)."""
    if len(pages) < 2:
        return set()
    seen = Counter()
    for page in pages:
        lines = [l for l in page["text"].split("\n") if l.strip()]
        edge_lines = {_key(l) for l in lines[:edge] + lines[-edge:] if not l.lstrip().startswith("|")}
        seen.update(edge_lines)
    return {k for k, n in seen.items() if n >= 2}


def _superscript(match: re.Match) -> str:
    content = re.sub(r"[_*]", "", match.group(1)).strip()
    return f"^{content}" if EXPONENT.match(content) else content


def normalize_text(text: str, boilerplate: set[str] = frozenset()) -> str:
    text = PICTURE_TEXT.sub("", text)
    text = HTML_COMMENT.sub("", text)
    text = unicodedata.normalize("NFC", text).translate(CHAR_MAP)
    text = UNDERLINE.sub("", text)
    text = BOLD.sub(r"\1", text)
    text = SPLIT_IDENTIFIER.sub(r"\1_\2", text)
    text = ITALIC.sub(r"\1", text)
    text = SUP.sub(_superscript, text)
    text = SUB.sub(r"_\1", text)
    text = BR.sub(" ", text)
    text = SPLIT_DECIMAL.sub(".", text)

    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        is_boilerplate = (
            (line and _key(line) in boilerplate and not line.startswith("|"))
            or PAGE_NUMBER.match(line)
        )
        if not is_boilerplate:
            if line.startswith("|"):
                line = CELL_UNDERSCORE.sub("", NUMBER_UNDERSCORE.sub("", line))
                line = SPLIT_SPAN_CELL.sub(r"\1\2||", line)      # one value spanning two columns: keep it in the first
            lines.append(line)
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    for broken, fixed in TABLE_FIXES.items():
        text = text.replace(broken, fixed)
    return text


def normalize_document(doc: dict) -> dict:
    boilerplate = find_boilerplate(doc["pages"])
    pages = [{"page": p["page"], "text": normalize_text(p["text"], boilerplate)} for p in doc["pages"]]
    return {**doc, "pages": pages}
