"""Names for atomic openings, in the grammar standard chess already uses.

Atomic has no ECO, no published index, and — outside a handful of community
coinages — no names at all. `book.py` answers "what does everyone play here",
but a repertoire of forty positions identified by 4-field FEN is a repertoire
nobody can memorise. A name is not decoration: it is the handle that lets a
line be remembered, discussed and looked for.

**Three layers, checked in this order.**

1. **Yours** (`opening_names`). Whatever you type in the UI wins over
   everything below, forever. You are the authority on your own repertoire.
2. **Curated** (`data/openings.json`). A small seed file, and every entry
   carries a `source` saying where the name came from — `standard` for a name
   borrowed from standard chess for the identical move order, `community` for
   one atomic players actually use, `descriptive` for one coined here. That
   column exists so a coined name is never mistaken for received theory.
3. **Derived** (this module). Everything else is named by rule, from the
   deepest named ancestor plus a label for the move that branches off it.

**The grammar.** Standard chess does not name variations arbitrarily: the
suffix says what the move *is*. A Defence is Black's answer to the first move;
a Gambit offers material; an Attack is an aggressive plan; a Counterattack is
Black seizing the initiative back rather than merely equalising. Reproducing
that grammar means classifying the move, and each of those words is a testable
claim about the position:

| Suffix | Test |
|---|---|
| `Defence` | Black's reply at ply 1, assertive by none of the tests below |
| `Gambit` | after the move, a capture wins material and survives the blast |
| `Attack` | a sortie into their half, a check, or a profitable hit on a piece |
| `Counterattack` | the same or a central thrust, by Black, **and** it gains ground |
| `Counter-Gambit` | a Gambit that also gains ground, played by Black |
| `Variation` | none of the above fired |

"Gains ground" is the part that keeps `Counterattack` honest, and it is
measured rather than asserted: Black's score in the field *after* this move,
against Black's score in the position it was played in. 1.Nf3 f6 2.e3 d5 lifts
Black from 38.6% to 47.0% — a real counterattack, and the community's own view
of that move — while 1.Nf3 d5 is the same kind of move and drops Black from
41.0% to 12.0%, so it is named a Defence and nothing more. Without book
statistics the test cannot run, and then nothing is called a Counterattack at
all: a missing measurement produces a weaker claim, never a guessed one.

The atomic-specific part is `_offers_material`. A gambit in standard chess is a
piece left en prise; here the capturer dies with its victim, so half of what
looks hanging is not, and the test has to be run through the explosion rules —
which is why 1.Nf3 e5 is not a gambit even though the pawn can be taken.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import chess
import chess.variant

from lab.methods import ATOMIC_PIECE_VALUES
from lab.rules import material_after

from . import book

#: The curated seed. Shipped in the repo, not in the database, so it is
#: reviewable in a diff and cannot be lost with a rebuilt store.
NAMES_PATH = Path(__file__).resolve().parent.parent / "data" / "openings.json"

#: How deep a name is worth deriving. Past this the book is thin and a name
#: would be a label on noise rather than a line anyone plays.
DEFAULT_DEPTH = 12

#: A pawn, in the lab's own piece values. The unit `_offers_material` works in.
PAWN = ATOMIC_PIECE_VALUES[chess.PAWN]

#: Black must not *lose* ground for an aggressive move to be a Counterattack.
#: Zero rather than a positive margin: the claim being tested is "took the
#: initiative back", and holding level after White's best try already is that.
COUNTER_GAIN = 0.0

#: Suffixes, in the order the classifier considers them.
KINDS = ("Counter-Gambit", "Counterattack", "Gambit", "Attack", "Defence", "Variation")


# -- the curated file ------------------------------------------------------


@dataclass(frozen=True)
class Curated:
    """One hand-written name, and where the name came from."""

    line: tuple[str, ...]
    name: str
    source: str            # standard / community / descriptive
    note: str = ""
    aliases: tuple[str, ...] = ()


@lru_cache(maxsize=1)
def curated(path: str | None = None) -> dict[str, Curated]:
    """The seed names, keyed by space-joined SAN line ('' is the start)."""
    raw = json.loads(Path(path or NAMES_PATH).read_text(encoding="utf-8"))
    out: dict[str, Curated] = {}
    for entry in raw["openings"]:
        line = tuple(entry["line"].split())
        out[" ".join(line)] = Curated(
            line=line,
            name=entry["name"],
            source=entry.get("source", "descriptive"),
            note=entry.get("note", ""),
            aliases=tuple(entry.get("aliases", ())),
        )
    return out


# -- your own names --------------------------------------------------------


def rename(db: sqlite3.Connection, line: str, name: str | None, note: str = "") -> None:
    """Name a line yourself, or clear the name with `None`/empty.

    Keyed by the SAN line rather than by FEN, unlike everything else here. A
    name belongs to a *sequence of moves* as it is spoken — "the Kapsarov after
    2...Nh6" — and two move orders that transpose are genuinely two names in
    every published opening index there has ever been.
    """
    key = " ".join(line.split())
    if not name or not name.strip():
        db.execute("DELETE FROM opening_names WHERE line = ?", (key,))
    else:
        db.execute(
            "INSERT INTO opening_names (line, name, note, at) VALUES (?,?,?,?) "
            "ON CONFLICT(line) DO UPDATE SET name = excluded.name, "
            "note = excluded.note, at = excluded.at",
            (key, name.strip(), note.strip(), time.time()),
        )
    db.commit()


def user_names(db: sqlite3.Connection) -> dict[str, tuple[str, str]]:
    """Every name you have written, keyed by line."""
    return {
        row["line"]: (row["name"], row["note"] or "")
        for row in db.execute("SELECT line, name, note FROM opening_names")
    }


# -- classifying a move ----------------------------------------------------


def _offers_material(board: chess.variant.AtomicBoard, move: chess.Move) -> bool:
    """After `move`, can the opponent take and come out ahead?

    The atomic part: a capture destroys the capturer, everything non-pawn
    beside the landing square, and — if it reaches the king — ends the game. So
    "leaves a piece en prise" has to be evaluated by playing the capture out,
    not by counting defenders. Three things disqualify a capture from counting
    as winning material, and each of them is a real atomic position:

    - it explodes the capturer's own king (illegal, and `legal_moves` has
      already dropped it);
    - it wins the game outright for the mover, which is not a gambit but a
      blunder, and is not what this predicate is for;
    - it costs the capturer more than it gains, which is the ordinary case in
      atomic and the reason so little here is actually hanging.
    """
    after = board.copy(stack=False)
    after.push(move)
    if after.is_game_over(claim_draw=True):
        return False
    mover = board.turn
    before = _material(after, mover)
    for reply in after.legal_moves:
        if not after.is_capture(reply):
            continue
        probe = after.copy(stack=False)
        probe.push(reply)
        if probe.is_variant_loss() or probe.is_checkmate():
            continue  # the opponent just wins; that is a blunder, not an offer
        if probe.is_variant_win():
            continue  # the capture loses them the game — no offer accepted
        if _material(probe, mover) <= before - PAWN:
            return True
    return False


def _material(board: chess.Board, color: chess.Color) -> int:
    return material_after(board, color, ATOMIC_PIECE_VALUES)


def _is_sortie(board: chess.Board, move: chess.Move) -> bool:
    """A queen or knight landing in the opponent's half — 2.Qh5 and its family."""
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type not in (chess.QUEEN, chess.KNIGHT):
        return False
    rank = chess.square_rank(move.to_square)
    return rank >= 4 if board.turn == chess.WHITE else rank <= 3


def _is_thrust(board: chess.Board, move: chess.Move) -> bool:
    """A central pawn's double step: the classical way to hit back in the centre."""
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    if chess.square_file(move.to_square) not in (3, 4):   # the d- and e-files
        return False
    return abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) == 2


def _hits_piece(board: chess.variant.AtomicBoard, move: chess.Move) -> bool:
    """After the move, does the moved piece attack an enemy piece it can take?

    "Attacks" has to mean *usefully* attacks: in atomic a queen glaring at a
    defended knight may be unable to take it without dying, and calling that an
    Attack would name half the opening tree after a threat nobody has. So the
    capture is played out and has to leave the mover better off.
    """
    after = board.copy(stack=False)
    after.push(move)
    if after.is_game_over(claim_draw=True):
        return False
    # Look from the mover's side again: push a null move so the attacker moves.
    probe = after.copy(stack=False)
    try:
        probe.push(chess.Move.null())
    except (AssertionError, ValueError):
        return False
    mover = board.turn
    base = _material(probe, mover)
    for follow in probe.legal_moves:
        if follow.from_square != move.to_square or not probe.is_capture(follow):
            continue
        victim = probe.piece_at(follow.to_square)
        if victim is None or victim.piece_type == chess.PAWN:
            continue
        done = probe.copy(stack=False)
        done.push(follow)
        if done.is_variant_win() or _material(done, mover) > base:
            return True
    return False


@dataclass
class Character:
    """What a move *is*, in the terms opening names are built from."""

    gambit: bool = False
    check: bool = False
    sortie: bool = False
    thrust: bool = False
    hits: bool = False
    gain: float | None = None      # win-score points the field gains by it

    @property
    def assertive(self) -> bool:
        """Does the move do something *to* the opponent?

        A central double-step is deliberately not on this list. 4.e4 in a line
        White has been building towards all game is a space-gaining move, and
        naming half the tree "the e4 Attack" would make the suffix meaningless.
        It counts for Black — hitting back in the centre is the classical
        counter and is what `countering` adds it for — and not for White.
        """
        return self.gambit or self.check or self.sortie or self.hits

    @property
    def countering(self) -> bool:
        return self.assertive or self.thrust

    @property
    def evidence(self) -> list[str]:
        out = []
        if self.gambit:
            out.append("offers material")
        if self.check:
            out.append("gives check")
        if self.sortie:
            out.append("sorties into their half")
        if self.thrust:
            out.append("central thrust")
        if self.hits:
            out.append("attacks a piece")
        if self.gain is not None:
            out.append(f"{self.gain:+.1f} pts against the parent line")
        return out


def character(
    board: chess.variant.AtomicBoard, san: str, *, gain: float | None = None
) -> Character:
    """Everything the namer needs to know about one move."""
    move = board.parse_san(san)
    after = board.copy(stack=False)
    after.push(move)
    return Character(
        gambit=_offers_material(board, move),
        check=after.is_check(),
        sortie=_is_sortie(board, move),
        thrust=_is_thrust(board, move),
        hits=_hits_piece(board, move),
        gain=gain,
    )


def kind_of(board: chess.variant.AtomicBoard, san: str, *, gain: float | None = None) -> str:
    """The suffix this move earns. See the table in the module docstring."""
    return classify(board, character(board, san, gain=gain), ply=board.ply())


def classify(board: chess.variant.AtomicBoard, char: Character, *, ply: int) -> str:
    black = board.turn == chess.BLACK
    counter = black and char.countering and char.gain is not None and char.gain >= COUNTER_GAIN
    if counter:
        return "Counter-Gambit" if char.gambit else "Counterattack"
    if char.gambit:
        return "Gambit"
    if char.assertive:
        return "Attack"
    if black and ply == 1:
        return "Defence"
    return "Variation"


# -- naming a line ---------------------------------------------------------


#: How many derived parts a full name shows. Uniqueness within a family is
#: what a name is *for* — "e4 Nf6 Bc4 d5" and "e4 d5" must not both come out
#: as "King's Pawn Opening: d5 Counterattack" — but a name with eight commas in
#: it is not a name either. Three is where the tree stops colliding in this
#: book; anything deeper is elided, and `parts` still carries the whole chain.
NAME_PARTS = 3


@dataclass
class Name:
    """A named line: what to call it, and on whose authority."""

    line: tuple[str, ...]
    family: str                      # the deepest curated ancestor's name
    parts: list[str] = field(default_factory=list)   # derived labels since then
    source: str = "derived"          # user / standard / community / descriptive / derived
    note: str = ""                   # about *this* line; empty if it is derived
    aliases: tuple[str, ...] = ()
    kind: str = ""                   # the suffix word of the last derived part
    evidence: list[str] = field(default_factory=list)
    family_note: str = ""            # about the named ancestor, always
    family_aliases: tuple[str, ...] = ()

    @property
    def variation(self) -> str:
        shown = self.parts[-NAME_PARTS:]
        text = ", ".join(shown)
        return f"…, {text}" if len(self.parts) > len(shown) else text

    @property
    def full(self) -> str:
        """Family, then variation — with one colon, as an index would print it.

        A curated family name may already carry a variation of its own ("…: f6
        Defence, Kapsarov Attack"), and a second colon after it would read as
        two names glued together. Every part after the first separator is
        joined with a comma instead, which is how standard chess writes
        "Sicilian Defence: Najdorf Variation, Poisoned Pawn".
        """
        if not self.family:
            return self.variation or "Starting position"
        if not self.parts:
            return self.family
        joiner = ", " if ":" in self.family else ": "
        return f"{self.family}{joiner}{self.variation}"

    @property
    def short(self) -> str:
        """The last thing that distinguishes this line, for a tree row.

        A tree already draws the ancestors above the row, so repeating them in
        it is noise: under "King's Knight Opening" the child wants to read "f6
        Defence", not its own full path again. For a curated name that means
        the segment after the last separator, since a curated name may carry a
        whole path of its own.
        """
        if self.parts:
            return self.parts[-1]
        tail = self.family.rsplit(": ", 1)[-1]
        return tail.rsplit(", ", 1)[-1]

    @property
    def key(self) -> str:
        return " ".join(self.line)

    def as_dict(self) -> dict[str, object]:
        return {
            "line": list(self.line),
            "key": self.key,
            "full": self.full,
            "short": self.short,
            "family": self.family,
            "variation": self.variation,
            "parts": list(self.parts),
            "source": self.source,
            "note": self.note,
            "aliases": list(self.aliases),
            "kind": self.kind,
            "evidence": self.evidence,
            "family_note": self.family_note,
            "family_aliases": list(self.family_aliases),
        }


def _score_for(entries: list[book.Entry], san: str, color: chess.Color) -> float | None:
    """The field's score for `color` after `san`, or None if it is unplayed."""
    for entry in entries:
        if entry.san == san:
            # `Entry.score` is from the mover's side, and the mover is `color`.
            return entry.score
    return None


def _parent_score(entries: list[book.Entry], color: chess.Color) -> float | None:
    """The field's score for `color` across every move in this position.

    The baseline a Counterattack has to beat. Aggregated over the whole
    position rather than taken from the previous move's row, because that row
    is scored for the *other* player and inverting it would silently import
    every game that left the book at this point.
    """
    games = sum(entry.games for entry in entries)
    if not games:
        return None
    total = sum(entry.score * entry.games for entry in entries)
    return total / games


def name_line(
    db: sqlite3.Connection | None,
    sans: list[str] | tuple[str, ...],
    *,
    names: dict[str, Curated] | None = None,
    yours: dict[str, tuple[str, str]] | None = None,
) -> Name:
    """Name a line of SAN moves.

    Walks forward from the start so that each step is named in the position it
    was actually played in — which is the only place the book statistics and
    the explosion rules can be consulted.
    """
    sans = tuple(sans)
    seeds = names if names is not None else curated()
    mine = yours if yours is not None else (user_names(db) if db is not None else {})

    board = chess.variant.AtomicBoard()
    family = ""
    fam_source = "derived"
    note = ""
    aliases: tuple[str, ...] = ()
    parts: list[str] = []
    kind = ""
    evidence: list[str] = []

    for index, san in enumerate(sans):
        prefix = " ".join(sans[: index + 1])
        entries = book.lookup(db, book.book_fen(board)) if db is not None else []
        try:
            move = board.parse_san(san)
        except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
            break

        # A named ancestor resets the tail: "the Kapsarov Attack" is the whole
        # name of that position, not a suffix on the moves that reached it.
        if prefix in mine:
            family, parts = mine[prefix][0], []
            fam_source, note, aliases = "user", mine[prefix][1], ()
            kind, evidence = "", []
        elif prefix in seeds:
            seed = seeds[prefix]
            family, parts = seed.name, []
            fam_source, note, aliases = seed.source, seed.note, seed.aliases
            kind, evidence = "", []
        else:
            label, kind, evidence = _label(board, san, entries)
            parts.append(label)
        board.push(move)

    # A note and an alias belong to the line they were written about. Carrying
    # them down onto a derived sub-variation made "1.Nf3 f6 2.e4" claim the
    # alias "Main Line" and the f6 Defence's explanation as its own, which is
    # a name for one position wearing the label of another. They stay on the
    # family, and are only *this* line's when this line is the family.
    return Name(
        line=sans,
        family=family,
        parts=parts,
        family_note=note,
        family_aliases=aliases,
        # The authority behind the *whole* name. A derived tail hanging off a
        # curated family is a derived name: the part you would argue about is
        # the part this module made up.
        source="derived" if parts else fam_source,
        note="" if parts else note,
        aliases=() if parts else aliases,
        kind=kind,
        evidence=evidence,
    )


def _label(
    board: chess.variant.AtomicBoard, san: str, entries: list[book.Entry]
) -> tuple[str, str, list[str]]:
    """The derived '<move> <Kind>' label for one move, and why."""
    gain = None
    if entries:
        here = _score_for(entries, san, board.turn)
        baseline = _parent_score(entries, board.turn)
        if here is not None and baseline is not None:
            gain = here - baseline
    char = character(board, san, gain=gain)
    kind = classify(board, char, ply=board.ply())
    return f"{san} {kind}", kind, char.evidence


# -- the tree the UI browses -----------------------------------------------


@dataclass
class TreeNode:
    """One named line in the book, with the statistics that justify naming it."""

    line: tuple[str, ...]
    name: Name
    games: int
    score: float          # for the side that played the last move
    elo: int | None
    children: list[TreeNode] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "line": list(self.line),
            "san": self.line[-1] if self.line else "",
            "ply": len(self.line) - 1,
            "name": self.name.as_dict(),
            "games": self.games,
            "score": self.score,
            "elo": self.elo,
            "children": [child.as_dict() for child in self.children],
        }


def tree(
    db: sqlite3.Connection,
    *,
    depth: int = DEFAULT_DEPTH,
    min_games: int = 2_000,
    min_share: float = 5.0,
    max_children: int = 6,
) -> list[TreeNode]:
    """The named opening tree, as deep as the book supports.

    Thresholds are the same shape as `repertoire._replies` and for the same
    reason: a relative floor keeps a popular position from sprouting a dozen
    sidelines, and an absolute one keeps a rare position from sprouting any.
    """
    seeds = curated()
    mine = user_names(db)

    def grow(board: chess.variant.AtomicBoard, line: tuple[str, ...]) -> list[TreeNode]:
        if len(line) >= depth:
            return []
        entries = book.lookup(db, book.book_fen(board))
        total = sum(entry.games for entry in entries)
        if not total:
            return []
        keep = [
            entry for entry in entries
            if entry.games >= min_games and 100.0 * entry.games / total >= min_share
        ]
        out: list[TreeNode] = []
        for entry in sorted(keep, key=lambda e: -e.games)[:max_children]:
            child = board.copy(stack=False)
            try:
                child.push_san(entry.san)
            except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
                continue
            here = (*line, entry.san)
            out.append(
                TreeNode(
                    line=here,
                    name=name_line(db, here, names=seeds, yours=mine),
                    games=entry.games,
                    score=entry.score,
                    elo=entry.elo,
                    children=grow(child, here),
                )
            )
        return out

    return grow(chess.variant.AtomicBoard(), ())


__all__ = [
    "KINDS",
    "Character",
    "Curated",
    "Name",
    "TreeNode",
    "character",
    "classify",
    "curated",
    "kind_of",
    "name_line",
    "rename",
    "tree",
    "user_names",
]
