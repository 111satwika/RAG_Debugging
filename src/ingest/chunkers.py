"""Chunking strategies. Each one turns a normalized document into text pieces with a page number.

fixed      consecutive windows of `size` tokens, optionally overlapping (the baseline)
recursive  cut at paragraph, then line, then sentence boundaries, and pack pieces up to `size` tokens
structure  one chunk per heading section (resume sections and projects, paper headings, spec numbered
           sections), so chunks vary in size. Options:
             overlap      a section over `size` is cut at paragraph or sentence boundaries, and each piece starts
                          with the last `overlap` tokens of the one before (never across sections)
             headers      prepend "doc title > section" to every piece
             keep_tables  never cut a table, however long, up to the embedding model's input limit (a table beyond
                          that is cut at row boundaries and every part repeats the header row)
             min_tokens   a piece smaller than this is merged with a neighbour from the same section, and a tiny
                          sub-section is merged with its parent's lead-in text or a sibling sub-section

`child_size` turns any of these into parent-child retrieval: see make_children.
`size` is a budget for the whole chunk, header included. Only a whole table (keep_tables) or a merge of a tiny piece
(up to `min_tokens` more) may go over it.
"""
import re
from dataclasses import dataclass

from src.config import SPECIAL_TOKENS
from src.ingest.tokenization import Tokenizer

STRATEGIES = ("fixed", "recursive", "structure")
SEPARATORS = (r"\n\n", r"\n", r"(?<=[.!?])\s+")        # paragraph, line, sentence
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
SECTION_LEVELS = {"resume": 3, "spec": 2, "paper": 3}   # cut at headings up to this level
DEFAULT_SECTION_LEVEL = 2
HEADER_SEPARATOR = "\n\n"
TABLE_SEPARATOR_ROW = re.compile(r"^\|\s*:?-{2,}")
TABLE_CAPTION = re.compile(r"^(Table \d+[:.].*)$", re.M)


@dataclass(frozen=True)
class ChunkSpec:
    strategy: str = "fixed"
    size: int = 500
    overlap: int = 0
    headers: bool = False
    keep_tables: bool = False
    min_tokens: int = 0      # > 0: merge pieces smaller than this with a neighbour from the same section
    child_size: int = 0      # > 0: parent-child retrieval; the chunks above are parents, each cut into children
    table_captions: bool = False   # a chunk holding a captioned table is headed by the caption, not the section before it

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown chunking strategy '{self.strategy}'. Choose from {list(STRATEGIES)}")
        if self.size < 1 or not 0 <= self.overlap < self.size:
            raise ValueError("overlap must be smaller than the chunk size")
        if (self.headers or self.keep_tables or self.min_tokens) and self.strategy != "structure":
            raise ValueError("headers, keep_tables and min_tokens need the structure strategy")
        if self.table_captions and not self.headers:
            raise ValueError("table_captions changes the chunk header, so it needs headers")
        if self.min_tokens < 0 or self.min_tokens >= self.size:
            raise ValueError("min_tokens must be smaller than the chunk size")
        if self.child_size and not 20 <= self.child_size < self.size:
            raise ValueError("child_size must be between 20 and the chunk size")

    @property
    def label(self) -> str:
        parts = [self.strategy, str(self.size)]
        if self.overlap:
            parts.append(f"overlap{self.overlap}")
        if self.headers:
            parts.append("headers")
        if self.keep_tables:
            parts.append("tables")
        if self.min_tokens:
            parts.append(f"min{self.min_tokens}")
        if self.child_size:
            parts.append(f"child{self.child_size}")
        if self.table_captions:
            parts.append("captions")
        return "-".join(parts)


@dataclass
class Piece:
    text: str          # the whole chunk text, header included
    page: int
    header: str = ""   # the contextual header at the start of `text`, if any


# ---------------------------------------------------------------- fixed

def split_fixed(text: str, tokenizer: Tokenizer, size: int, overlap: int = 0) -> list[str]:
    """Consecutive windows of `size` tokens; each window starts `size - overlap` tokens after the last."""
    offsets = tokenizer.offsets(text)
    step = size - overlap
    pieces, start = [], 0
    while start < len(offsets):
        window = offsets[start:start + size]
        piece = text[window[0][0]:window[-1][1]].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(offsets):
            break
        start += step
    return pieces


# ---------------------------------------------------------------- recursive

def _cut(text: str, level: int) -> list[str]:
    """Cut at one separator level, keeping the separator on the piece before it."""
    parts, last = [], 0
    for match in re.finditer(SEPARATORS[level], text):
        parts.append(text[last:match.end()])
        last = match.end()
    parts.append(text[last:])
    return [p for p in parts if p.strip()]


def _atomize(text: str, tokenizer: Tokenizer, size: int, level: int = 0) -> list[str]:
    """Pieces of at most `size` tokens, cut at the coarsest boundary that works."""
    if tokenizer.count(text) <= size:
        return [text]
    if level >= len(SEPARATORS):
        return [piece + " " for piece in split_fixed(text, tokenizer, size)]     # a space, so merged pieces stay separate words
    parts = _cut(text, level)
    if len(parts) <= 1:
        return _atomize(text, tokenizer, size, level + 1)
    return [piece for part in parts for piece in _atomize(part, tokenizer, size, level + 1)]


def _pack(units: list[str], tokenizer: Tokenizer, size: int, joiner: str = "", overlap: int = 0,
          min_tokens: int = 0) -> list[str]:
    """Greedily join consecutive units while they fit in `size` tokens.

    With `overlap`, a new chunk starts with the last few units of the previous one (up to `overlap` tokens).
    With `min_tokens`, a last chunk whose new units are fewer tokens than that joins the chunk before it.
    """
    chunks, current, carried_count = [], [], 0
    for unit in units:
        if current and tokenizer.count(joiner.join(current + [unit])) > size:
            chunks.append((current, carried_count))
            carried, total = [], 0
            for previous in reversed(current):
                total += tokenizer.count(previous)
                if total > overlap:
                    break
                carried.insert(0, previous)
            current = carried if tokenizer.count(joiner.join(carried + [unit])) <= size else []
            carried_count = len(current)
        current.append(unit)
    chunks.append((current, carried_count))

    if min_tokens and len(chunks) > 1:
        last, carried_count = chunks[-1]
        fresh = last[carried_count:]
        if tokenizer.count(joiner.join(fresh)) < min_tokens and \
                tokenizer.count(joiner.join(chunks[-2][0] + fresh)) <= size + min_tokens:
            chunks[-2] = (chunks[-2][0] + fresh, chunks[-2][1])
            chunks.pop()
    return [joiner.join(units_).strip() for units_, _ in chunks if joiner.join(units_).strip()]


def split_recursive(text: str, tokenizer: Tokenizer, size: int, overlap: int = 0, min_tokens: int = 0) -> list[str]:
    return _pack(_atomize(text, tokenizer, size), tokenizer, size, overlap=overlap, min_tokens=min_tokens)


# ---------------------------------------------------------------- tables

def _blocks(text: str) -> list[tuple[bool, str]]:
    """Alternating (is_table, text) blocks; a table is a run of lines that start with '|'."""
    blocks, run, in_table = [], [], False
    for line in text.split("\n"):
        is_row = line.lstrip().startswith("|")
        if run and is_row != in_table:
            blocks.append((in_table, "\n".join(run)))
            run = []
        in_table = is_row
        run.append(line)
    if run:
        blocks.append((in_table, "\n".join(run)))
    return blocks


def split_table(table: str, tokenizer: Tokenizer, size: int) -> list[str]:
    """Cut a long table at row boundaries; every part repeats the header row."""
    lines = table.split("\n")
    head = lines[:2] if len(lines) > 1 and TABLE_SEPARATOR_ROW.match(lines[1].lstrip()) else lines[:1]
    parts, current = [], list(head)
    for row in lines[len(head):]:
        if len(current) > len(head) and tokenizer.count("\n".join(current + [row])) > size:
            parts.append("\n".join(current))
            current = list(head)
        current.append(row)
    parts.append("\n".join(current))
    return parts


def table_limit(tokenizer: Tokenizer, header_tokens: int = 0) -> int:
    """The most a whole table may hold: what the embedding model accepts, less its special tokens and the header."""
    return tokenizer.max_tokens - SPECIAL_TOKENS - header_tokens if tokenizer.max_tokens else 10 ** 9


def _pieces_keeping_tables(text: str, tokenizer: Tokenizer, size: int, overlap: int = 0, min_tokens: int = 0,
                           whole_table_limit: int | None = None) -> list[str]:
    """Pack text blocks and tables. A table stays whole up to `whole_table_limit`, which may exceed `size`.
    Overlap applies only inside a long run of text, never to a table."""
    limit = whole_table_limit or size
    units = []
    for is_table, block in _blocks(text):
        if not block.strip():
            continue
        tokens = tokenizer.count(block)
        if tokens <= size or (is_table and tokens <= limit):
            units.append(block)
        elif is_table:
            units.extend(split_table(block, tokenizer, limit))
        else:
            units.extend(split_recursive(block, tokenizer, size, overlap, min_tokens))
    return _pack(units, tokenizer, size, joiner="\n")


def _merge_small(pieces: list[str], tokenizer: Tokenizer, size: int, min_tokens: int) -> list[str]:
    """Merge a piece smaller than `min_tokens` into a neighbour from the same section (previous first, then next)."""
    def fits(a: str, b: str) -> bool:
        ta, tb = tokenizer.count(a), tokenizer.count(b)
        return ta + tb <= size + min_tokens or max(ta, tb) > size       # a whole table may already be over `size`

    pieces, merged, index = list(pieces), [], 0
    while index < len(pieces):
        piece = pieces[index]
        if tokenizer.count(piece) < min_tokens:
            if merged and fits(merged[-1], piece):
                merged[-1] = merged[-1] + "\n\n" + piece
                index += 1
                continue
            if index + 1 < len(pieces) and fits(piece, pieces[index + 1]):
                pieces[index + 1] = piece + "\n\n" + pieces[index + 1]
                index += 1
                continue
        merged.append(piece)
        index += 1
    return merged


# ---------------------------------------------------------------- structure

def doc_title(doc: dict) -> str:
    """The first top-level heading, or the file name made readable."""
    for page in doc["pages"]:
        for line in page["text"].split("\n"):
            match = HEADING.match(line)
            if match and len(match.group(1)) == 1:
                return match.group(2)
    return re.sub(r"[_\s]+", " ", doc["doc_id"]).strip()


def build_sections(doc: dict, max_level: int) -> list[dict]:
    """Cut a document at its headings. A heading with no text under it joins the next section."""
    raw, stack = [], []
    current = {"path": [], "page": doc["pages"][0]["page"] if doc["pages"] else 1, "lines": [], "has_body": False}
    for page in doc["pages"]:
        for line in page["text"].split("\n"):
            match = HEADING.match(line)
            if match and len(match.group(1)) <= max_level:
                level, title = len(match.group(1)), match.group(2)
                stack = [(lv, t) for lv, t in stack if lv < level] + [(level, title)]
                if current["lines"]:
                    raw.append(current)
                current = {"path": [t for _, t in stack], "page": page["page"], "lines": [line], "has_body": False}
            else:
                current["lines"].append(line)
                current["has_body"] = current["has_body"] or bool(line.strip())
    raw.append(current)

    sections, carried = [], None
    for index, section in enumerate(raw):
        if not section["has_body"] and index < len(raw) - 1:
            carried = {"lines": (carried["lines"] if carried else []) + section["lines"],
                       "page": carried["page"] if carried else section["page"]}
            continue
        if carried:
            section = {**section, "lines": carried["lines"] + section["lines"], "page": carried["page"]}
            carried = None
        sections.append(section)
    return sections


def merge_small_sections(sections: list[dict], tokenizer: Tokenizer, title: str, size: int, min_tokens: int) -> list[dict]:
    """Merge a tiny section with its parent's lead-in text or a sibling sub-section (never across top-level sections).

    The merged piece takes the heading path the two have in common, so its header names the parent section.
    """
    def path(section):
        return section["path"][1:] if section["path"][:1] == [title] else section["path"]

    def tokens(section):
        return tokenizer.count("\n".join(section["lines"]).strip())

    def related(a, b):
        pa, pb = path(a), path(b)
        return bool(pb) and (pb[:-1] == pa or (len(pa) > 1 and pa[:-1] == pb[:-1]))

    merged = []
    for section in sections:
        previous = merged[-1] if merged else None
        if previous and related(previous, section) and (tokens(previous) < min_tokens or tokens(section) < min_tokens) \
                and tokens(previous) + tokens(section) <= size - 25:
            common = []
            for a, b in zip(previous["path"], section["path"]):
                if a != b:
                    break
                common.append(a)
            merged[-1] = {"path": common, "page": previous["page"], "lines": previous["lines"] + [""] + section["lines"],
                          "has_body": True}
        else:
            merged.append(section)
    return merged


def _structure_pieces(doc: dict, tokenizer: Tokenizer, spec: ChunkSpec) -> list[Piece]:
    title = doc_title(doc)
    sections = build_sections(doc, SECTION_LEVELS.get(doc["doc_type"], DEFAULT_SECTION_LEVEL))
    if spec.min_tokens:
        sections = merge_small_sections(sections, tokenizer, title, spec.size, spec.min_tokens)
    pieces = []
    for section in sections:
        text = "\n".join(section["lines"]).strip()
        if not text:
            continue
        path = section["path"][1:] if section["path"][:1] == [title] else section["path"]
        header = " > ".join([title, *path]) if spec.headers else ""
        header_tokens = tokenizer.count(header + HEADER_SEPARATOR) if header else 0
        budget = spec.size - header_tokens
        if budget < 20:
            raise ValueError(f"the header '{header}' leaves no room for text in a {spec.size}-token chunk")
        whole_table = table_limit(tokenizer, header_tokens) if spec.keep_tables else budget
        if tokenizer.count(text) <= budget:
            bodies = [text]
        elif spec.keep_tables:
            bodies = _pieces_keeping_tables(text, tokenizer, budget, spec.overlap, spec.min_tokens, whole_table)
        else:
            bodies = split_recursive(text, tokenizer, budget, spec.overlap, spec.min_tokens)
        if spec.min_tokens:
            bodies = _merge_small(bodies, tokenizer, budget, spec.min_tokens)
        pending = None                      # a caption that ended the previous piece, for the table that follows it
        for body in bodies:
            for part in ([body] if tokenizer.count(body) <= max(budget + spec.min_tokens, whole_table)
                         else split_fixed(body, tokenizer, budget)):
                head = header
                if spec.table_captions:
                    caption, pending = table_caption(part, pending)
                    if caption:
                        head = f"{title} > {caption}"
                pieces.append(Piece(head + HEADER_SEPARATOR + part if head else part, section["page"], head))
    return pieces


def table_caption(part: str, pending: str | None) -> tuple[str | None, str | None]:
    """The caption to head this piece with, and the caption left over for the next piece.

    The parser puts a table under the section that precedes it on the page: the paper's Table 2 (BLEU scores) was headed
    "5 Training > 5.4 Regularization" and Table 1 "3.4 Embeddings and Softmax", so the reranker ranked the text of the
    Results section above them (q18-q20). A piece with table rows is headed by the caption before its first row, in the
    piece itself (Tables 1, 2) or at the end of the previous piece (Tables 3, 4: the caption closes the text before them).
    """
    lines = part.split("\n")
    first_row = next((i for i, line in enumerate(lines) if line.startswith("|")), None)
    captions = [(i, m.group(1).strip()) for i, line in enumerate(lines) if (m := TABLE_CAPTION.match(line))]
    caption = None
    if first_row is not None and sum(line.startswith("|") for line in lines) >= 3:
        before = [c for i, c in captions if i < first_row]
        caption = before[-1] if before else pending
    after_rows = [c for i, c in captions if first_row is None or i > first_row]
    short = lambda c: c if c is None else re.split(r"(?<=[.!?])\s", c, maxsplit=1)[0][:160]
    return short(caption), (after_rows[-1] if after_rows else None)


# ---------------------------------------------------------------- entry point

def make_pieces(doc: dict, tokenizer: Tokenizer, spec: ChunkSpec) -> list[Piece]:
    if spec.strategy == "structure":
        return _structure_pieces(doc, tokenizer, spec)
    pieces = []
    for page in doc["pages"]:
        if spec.strategy == "fixed":
            texts = split_fixed(page["text"], tokenizer, spec.size, spec.overlap)
        else:
            texts = split_recursive(page["text"], tokenizer, spec.size, spec.overlap)
        pieces.extend(Piece(text, page["page"]) for text in texts)
    return pieces


def make_children(parents: list[dict], tokenizer: Tokenizer, spec: ChunkSpec) -> list[dict]:
    """Parent-child retrieval: cut every parent chunk into small children. The children are what gets embedded
    and searched, and the parent is what the model reads. A child repeats its parent's header for context."""
    children = []
    for parent in parents:
        header = parent.get("header", "")
        body = parent["text"][len(header):].strip() if header else parent["text"]
        budget = max(spec.child_size - (tokenizer.count(header + HEADER_SEPARATOR) if header else 0), 5)
        parts = [body] if tokenizer.count(body) <= budget else split_recursive(body, tokenizer, budget, spec.overlap // 2)
        for index, part in enumerate(parts):
            text = header + HEADER_SEPARATOR + part if header else part
            children.append({**parent, "chunk_id": f"{parent['chunk_id']}_k{index}", "parent_id": parent["chunk_id"],
                             "n_tokens": tokenizer.count(text), "text": text})
    return children
