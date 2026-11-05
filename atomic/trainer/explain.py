"""Why the engine plays what it plays, in words, from its own search.

The engine's real reason for a move is not a feature of its evaluation. It is
the search: every other move runs into a concrete line, and this one does not.
For an alpha-beta engine that is the *complete* reason, so an explanation built
from it is faithful by construction. The work is compressing a tree of
thousands of nodes into a few sentences a human can check on the board.

This matters more in atomic than in standard chess. The lab measured it:
quiescence alone is worth +203 Elo, and static blast-radius king safety goes
to nothing once quiescence is present (`docs/method-catalog.md`, M1). Atomic
keeps its reasoning in short forcing capture lines, so that is where an
explanation has to look.

Four questions, each answered by one or two searches:

1. **What happens after the move I asked about?** Its refutation line, cut
   to the forcing core (`proof`), with every explosion spelled out: what dies,
   on which side, and whether a king goes.
2. **What was the opponent threatening?** A null move: hand them the turn
   and search. If that is much better for them than the position after the
   engine's move, the move was a defence, and the threat is the explanation.
   This is the usual reason a move "makes no sense". It answers a capture that
   has not happened yet.
3. **What does the engine's move threaten?** The same null-move test one ply
   later.
4. **Which atomic rules does it rest on?** Kings touching, captures that would
   blow up your own pieces, regicidal legality, and the `themes` detectors
   run over the lines.

Nothing here judges a position by itself. Every claim is either a rules fact
(`lab.rules`) or a number from the engine, and the output says when the engine's
preference has no short forcing line behind it, so an explanation never claims
more than the search knows.

`search` is injected, so the same code runs on the trainer's cached
Fairy-Stockfish sessions, on a throttled engine in a test, or on a scripted
fake.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import chess
import chess.variant

from lab.methods import ATOMIC_PIECE_VALUES
from lab.rules import explosion_squares

from . import themes, winrate
from .analysis import Line

#: `(fen, multipv) -> lines`, best first, scored for the side to move.
Search = Callable[[str, int], list[Line]]

#: How far into a variation a forcing core is looked for. Atomic's median
#: game is 30 ply, and the tactics the engine sees within a search horizon are
#: short. Anything that needs more plies than this to show itself is reported
#: as positional rather than dressed up as a combination.
MAX_PROOF = 8

#: Win% a pass has to cost before the opponent's best reply to it counts as a
#: threat. Below this, "if you did nothing they would play X" is true of
#: every position and explains none of them.
THREAT_MIN = 15.0

#: A move within this much win% of the best is not "worse" in any sense worth
#: explaining. The engine's choice between them is noise at its budget.
CLOSE = 5.0

PIECE_NAME = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def colour(color: chess.Color) -> str:
    return "White" if color == chess.WHITE else "Black"


def _value(piece: chess.Piece) -> int:
    return ATOMIC_PIECE_VALUES.get(piece.piece_type, 0)


def _join(words: Sequence[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def _pieces_phrase(color: chess.Color, items: Sequence[tuple[chess.Square, chess.Piece]]) -> str:
    """`White's queen (d1) and bishop (c1)`, most valuable first."""
    ordered = sorted(items, key=lambda item: -_value(item[1]))
    named = [f"{PIECE_NAME[p.piece_type]} ({chess.square_name(sq)})" for sq, p in ordered]
    return f"{colour(color)}'s {_join(named)}"


# -- one ply ----------------------------------------------------------------


def describe_ply(board: chess.variant.AtomicBoard, move: chess.Move) -> dict[str, Any]:
    """Everything a human needs to know about one move, as rules facts.

    `board` is the position *before* the move and is not modified. `delta` is
    the material the move wins for its mover, in centipawns of atomic value
    (`lab.methods.ATOMIC_PIECE_VALUES`): what the explosion takes from the
    opponent minus what it takes from the mover, capturer included.
    """
    mover = board.turn
    san = board.san(move)
    # The capturer stands on its origin square in `board`, and that square can
    # be inside the radius. It dies on the landing square, so it is counted
    # there once and its origin is dropped from the blast.
    blast = explosion_squares(board, move) - chess.SquareSet([move.from_square])
    destroyed: list[tuple[chess.Square, chess.Piece]] = []
    if blast:
        capturer = board.piece_at(move.from_square)
        if capturer is not None:
            destroyed.append((move.to_square, capturer))
        for square in blast:
            piece = board.piece_at(square)
            if piece is not None:
                destroyed.append((square, piece))

    ours = [(sq, p) for sq, p in destroyed if p.color == mover]
    theirs = [(sq, p) for sq, p in destroyed if p.color != mover]
    kings = [p.color for _, p in destroyed if p.piece_type == chess.KING]
    delta = sum(_value(p) for _, p in theirs) - sum(_value(p) for _, p in ours)

    after = board.copy(stack=False)
    after.push(move)

    if kings:
        victim = kings[0]
        square = board.king(victim)
        text = (
            f"{san} explodes {colour(victim)}'s king on "
            f"{chess.square_name(square)}. Game over."
        )
    elif blast:
        if theirs:
            text = f"{san} blows up {_pieces_phrase(not mover, theirs)}"
        else:
            text = f"{san} hits nothing"
        # The capturer always dies, and in atomic that is often the point: a
        # queen that takes a pawn is a queen traded for a pawn. So it is named.
        capturer = next((p for sq, p in ours if sq == move.to_square), None)
        extra = [(sq, p) for sq, p in ours if sq != move.to_square]
        if capturer is not None:
            text += f"; the capturing {PIECE_NAME[capturer.piece_type]} dies with it"
        if extra:
            text += f", and so do {_pieces_phrase(mover, extra)}"
        text += "."
    else:
        text = f"{colour(mover)} plays {san}."
        if after.is_check():
            text = f"{colour(mover)} plays {san}, attacking the king."

    return {
        "uci": move.uci(),
        "san": san,
        "mover": colour(mover).lower(),
        "capture": bool(blast),
        "king": colour(kings[0]).lower() if kings else None,
        "promotion": PIECE_NAME[move.promotion] if move.promotion else None,
        "destroyed": [
            {"square": chess.square_name(sq), "piece": p.symbol(), "color": colour(p.color).lower()}
            for sq, p in destroyed
        ],
        "blast": sorted(chess.square_name(sq) for sq in blast),
        "delta": delta,
        "text": text,
        "fen_before": board.fen(),
        "fen_after": after.fen(),
    }


def walk(fen: str, ucis: Sequence[str], limit: int = MAX_PROOF) -> list[dict[str, Any]]:
    """Describe a variation ply by ply, stopping where it stops being legal.

    A node-limited PV can end mid-air, and a cached one may be stale. Neither
    is an error here: describe what replays and stop.
    """
    board = chess.variant.AtomicBoard(fen)
    out: list[dict[str, Any]] = []
    for uci in list(ucis)[:limit]:
        if board.is_game_over(claim_draw=True):
            break
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in board.legal_moves:
            break
        out.append(describe_ply(board, move))
        board.push(move)
    return out


#: Net material a line has to move, in centipawns of atomic value, before it
#: counts as the reason for anything. A pawn trade six plies deep is part of
#: every variation and explains none of them.
FORCING_SWING = 150

#: How close, in centipawns, a prefix of a line has to come to the line's
#: final material before it is taken as the point where the line is decided:
#: this, or a fifth of the final swing, whichever is larger.
SETTLED = 100

#: A threat that is not a mate has to land within this many plies of the
#: pass. Longer than that, it is a plan the search found with two free moves,
#: not something a human would call a threat.
THREAT_PLIES = 5

#: A forced mate is followed to the explosion however long it is, up to this.
MAX_MATE_PLIES = 24


def proof(
    fen: str, ucis: Sequence[str], mate: int | None = None
) -> tuple[list[dict[str, Any]], bool]:
    """The forcing core of a variation, and whether it has one.

    Forcing means the line ends in a king explosion, or moves at least
    `FORCING_SWING` of material for the side that starts it. It is cut at the
    explosion, or at the last capture or promotion within `MAX_PROOF` plies:
    the point after which the material on the board stops changing, which is
    what the line is *for*. When the engine reports a forced mate, the window
    stretches to reach it, because a mate cut off before the explosion is not
    a reason at all.

    A line that is not forcing comes back as its first two plies, flagged, so
    the caller can say plainly that the engine's reason is positional instead
    of inventing a tactic.
    """
    limit = MAX_PROOF
    if mate is not None:
        limit = min(MAX_MATE_PLIES, max(MAX_PROOF, 2 * abs(mate)))
    plies = walk(fen, ucis, limit)
    for index, ply in enumerate(plies):
        if ply["king"]:
            return plies[: index + 1], True
    events = [i for i, ply in enumerate(plies) if ply["capture"] or ply["promotion"]]
    if not events:
        return plies[:2], False
    core = plies[: events[-1] + 1]
    side = core[0]["mover"]
    final = swing(core, side)
    if abs(final) < FORCING_SWING:
        return plies[:2], False
    # Stop as soon as the damage is done. A PV keeps going after the point is
    # made, and the trade it tacks on afterwards explains nothing.
    for index in events:
        so_far = swing(plies[: index + 1], side)
        slack = max(SETTLED, abs(final) // 5)
        if so_far * final > 0 and abs(so_far) >= abs(final) - slack:
            return plies[: index + 1], True
    return core, True


def swing(plies: Sequence[dict[str, Any]], pov: str) -> int:
    """Net material over a line for `pov` ("white"/"black"), in centipawns."""
    return sum(p["delta"] if p["mover"] == pov else -p["delta"] for p in plies)


def _line_text(plies: Sequence[dict[str, Any]], numbered: bool = True) -> str:
    """The SAN of a line, with move numbers, starting from whoever moves.

    Threat lines start from a position reached by a pass, which no game ever
    contains, so move numbers there would point at a move that does not
    exist. They are written as bare SAN.
    """
    if not plies:
        return ""
    if not numbered:
        return " ".join(ply["san"] for ply in plies)
    board = chess.variant.AtomicBoard(plies[0]["fen_before"])
    number = board.fullmove_number
    parts: list[str] = []
    for index, ply in enumerate(plies):
        white = ply["mover"] == "white"
        if white:
            parts.append(f"{number}.{ply['san']}")
        else:
            parts.append(f"{number}...{ply['san']}" if index == 0 else ply["san"])
            number += 1
    return " ".join(parts)


def _events_text(plies: Sequence[dict[str, Any]]) -> str:
    """The sentences for the plies that change the board."""
    told = [p["text"] for p in plies if p["capture"] or p["king"] or p["promotion"]]
    return " ".join(told)


def _material_text(cp: int, pov: str) -> str:
    pawns = abs(cp) / 100
    if pawns < 0.5:
        return "material ends up level"
    who = colour(chess.WHITE if pov == "white" else chess.BLACK)
    verb = "comes out" if cp > 0 else "ends up"
    return f"{who} {verb} {pawns:.1f} pawns {'ahead' if cp > 0 else 'behind'}"


# -- threats ----------------------------------------------------------------


def null_board(board: chess.variant.AtomicBoard) -> chess.variant.AtomicBoard | None:
    """The position with the turn handed over, if that is a legal position.

    Only used to ask "what would they do if it were their move". En passant
    rights are cleared because they belonged to the move just played. When
    the side to move is in check, passing is not a position at all: the threat
    there is the check itself, and the caller reports it directly.
    """
    if board.is_check() or board.is_game_over(claim_draw=True):
        return None
    flipped = board.copy(stack=False)
    flipped.turn = not board.turn
    flipped.ep_square = None
    if not flipped.is_valid() or flipped.is_game_over(claim_draw=True):
        return None
    return flipped


def threat(
    board: chess.variant.AtomicBoard, search: Search, model: winrate.WinModel
) -> dict[str, Any] | None:
    """What the side *not* to move would do with a free move, and its worth.

    `win` is their win% after the pass. The caller decides whether that is a
    threat by comparing it with what they get in the real game.
    """
    flipped = null_board(board)
    if flipped is None:
        return None
    lines = search(flipped.fen(), 1)
    if not lines:
        return None
    line = lines[0]
    plies, forcing = proof(flipped.fen(), line.pv or (line.uci,), line.mate)
    if not plies:
        return None
    # A threat has to pay the side making it: their opponent's king goes, or
    # they come out ahead. A forcing line where they lose material is the
    # search wandering, not a threat, and is not reported as one.
    side = colour(flipped.turn).lower()
    king = plies[-1]["king"]
    forcing = forcing and (
        (king is not None and king != side)
        or (
            king is None
            and len(plies) <= THREAT_PLIES
            and swing(plies, side) >= FORCING_SWING
        )
    )
    return {
        "uci": line.uci,
        "san": plies[0]["san"],
        "win": line.win_pct(model),
        "mate": line.mate,
        "line": plies,
        "forcing": forcing,
        "start_fen": flipped.fen(),
    }


# -- static notes -----------------------------------------------------------


def _kings_touch(board: chess.Board) -> bool:
    white, black = board.king(chess.WHITE), board.king(chess.BLACK)
    if white is None or black is None:
        return False
    return bool(chess.BB_KING_ATTACKS[white] & chess.BB_SQUARES[black])


def notes(board: chess.variant.AtomicBoard, move: chess.Move) -> list[dict[str, Any]]:
    """Atomic rules the move leans on, read off the move itself."""
    out: list[dict[str, Any]] = []
    mover = board.turn
    san = board.san(move)
    after = board.copy(stack=False)
    after.push(move)

    piece = board.piece_at(move.from_square)
    if piece and piece.piece_type == chess.KING and _kings_touch(after) and not _kings_touch(board):
        out.append({
            "kind": "connected-kings",
            "text": (
                f"{san} puts the kings side by side. A capture next to one king is "
                "also next to the other, so it would explode the capturer's own "
                "king. That makes it illegal, and the king is immune to blasts "
                "while they touch."
            ),
            "squares": [chess.square_name(move.to_square)],
        })

    blast = explosion_squares(board, move) - chess.SquareSet([move.from_square])
    if blast:
        own = [
            (sq, board.piece_at(sq)) for sq in blast
            if sq != move.to_square and board.piece_at(sq) and board.piece_at(sq).color == mover
        ]
        if own:
            out.append({
                "kind": "self-blast",
                "text": (
                    f"{san} is not free: the explosion also destroys "
                    f"{_pieces_phrase(mover, own)}."
                ),
                "squares": [chess.square_name(sq) for sq, _ in own],
            })
        enemy_king = board.king(not mover)
        if enemy_king in blast and (
            board.is_check() or board.is_pinned(mover, move.from_square)
        ):
            out.append({
                "kind": "regicidal",
                "text": (
                    f"{san} would be illegal in standard chess "
                    f"({'it ignores a check' if board.is_check() else 'the piece is pinned'}). "
                    "In atomic it is legal because it explodes the king first."
                ),
                "squares": [chess.square_name(enemy_king)],
            })

    if not blast:
        # Walking a piece out of a capture's blast radius: the shape of most
        # atomic defences, and invisible if you only look at what is attacked.
        enemy = board.copy(stack=False)
        enemy.turn = not mover
        enemy.ep_square = None
        radius = chess.SquareSet()
        for reply in enemy.pseudo_legal_moves:
            if enemy.is_capture(reply):
                radius |= explosion_squares(enemy, reply)
        if (
            piece
            and piece.piece_type != chess.PAWN
            and move.from_square in radius
            and move.to_square not in radius
        ):
            out.append({
                "kind": "defusal",
                "text": (
                    f"{san} takes the {PIECE_NAME[piece.piece_type]} off "
                    f"{chess.square_name(move.from_square)}, which sat inside the blast "
                    f"radius of a capture {colour(not mover)} could make, to a square "
                    "no capture reaches."
                ),
                "squares": [chess.square_name(move.from_square)],
            })
    return out


def _themes_of(fen: str, plies: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    if not plies:
        return []
    try:
        names = themes.detect(fen, [p["uci"] for p in plies])
    except ValueError:
        return []
    # These two describe provenance or length, which says nothing about why.
    skip = {"tb-conversion", "opening-trap", "long"}
    return [{"name": n, "text": themes.CATALOG[n]} for n in names if n not in skip]


# -- the explanation ---------------------------------------------------------


def _find(lines: Sequence[Line], uci: str) -> Line | None:
    return next((line for line in lines if line.uci == uci), None)


def _after(fen: str, uci: str) -> chess.variant.AtomicBoard:
    board = chess.variant.AtomicBoard(fen)
    board.push(chess.Move.from_uci(uci))
    return board


def _score_move(
    fen: str, uci: str, root: Sequence[Line], search: Search, model: winrate.WinModel
) -> tuple[float, tuple[str, ...], int | None]:
    """Mover's win% after `uci`, the continuation after it, and any mate in it.

    From the root MultiPV when the move is among its lines, which is the same
    search the best move came from. Otherwise one search of the position after
    it, flipped, which is the honest number: what the opponent gets.
    """
    found = _find(root, uci)
    if found is not None and found.pv:
        return found.win_pct(model), tuple(found.pv[1:]), found.mate
    after = _after(fen, uci)
    if after.is_game_over(claim_draw=True):
        result = after.result(claim_draw=True)
        mover_white = after.turn == chess.BLACK  # the move was White's
        if result == "1/2-1/2":
            return 50.0, (), None
        return (100.0 if (result == "1-0") == mover_white else 0.0), (), None
    lines = search(after.fen(), 1)
    if not lines:
        return 50.0, (), None
    reply = lines[0]
    return 100.0 - reply.win_pct(model), tuple(reply.pv or (reply.uci,)), reply.mate


def _reason(kind: str, text: str, *, line=None, start_fen=None, squares=None, **extra) -> dict:
    """One reason. `hypothetical` marks a line that starts from a pass, so the
    page can say its first position never happened."""
    out = {"kind": kind, "text": text, "line": line or [], "start_fen": start_fen,
           "squares": squares or [], "hypothetical": False}
    out.update(extra)
    return out


def explain(
    fen: str,
    search: Search,
    *,
    move: str | None = None,
    versus: str | None = None,
    model: winrate.WinModel | None = None,
) -> dict[str, Any]:
    """Explain the engine's move here, or why `move` is worse than it.

    - `move` omitted, or equal to the engine's choice: what the engine's move
      does, and why not `versus` (default: the engine's second choice).
    - `move` something else: why it is worse than the engine's move, then what
      the engine's move does instead.
    """
    model = model or winrate.load()
    board = chess.variant.AtomicBoard(fen)
    if board.is_game_over(claim_draw=True):
        return {"over": True, "fen": fen, "result": board.result(claim_draw=True)}
    mover = colour(board.turn)
    pov = mover.lower()
    opponent = colour(not board.turn)

    root = search(fen, 3)
    if not root:
        raise ValueError("the engine returned no lines for this position")
    best = root[0]
    best_win = best.win_pct(model)
    best_san = board.san(chess.Move.from_uci(best.uci))

    subject = move or best.uci
    if chess.Move.from_uci(subject) not in board.legal_moves:
        raise ValueError(f"illegal move {subject}")
    if versus and chess.Move.from_uci(versus) not in board.legal_moves:
        raise ValueError(f"illegal move {versus}")
    subject_is_best = subject == best.uci

    if subject_is_best:
        other = versus or (root[1].uci if len(root) > 1 else None)
    else:
        other = subject
    reasons: list[dict[str, Any]] = []

    # 1. What the engine's move does, from its own line.
    best_line, best_forcing = proof(fen, best.pv or (best.uci,), best.mate)
    wins_now = bool(best_line) and bool(best_line[0]["king"])
    if wins_now:
        reasons.append(_reason(
            "wins", f"{best_line[0]['text']} {best_san} wins on the spot.",
            line=best_line[:1], start_fen=fen, squares=best_line[0]["blast"]))
    elif best_forcing and len(best_line) > 1:
        gain = swing(best_line, pov)
        ends = best_line[-1]["king"]
        if not ends and gain < 0:
            lead_in = "Even the best line costs material"
        else:
            lead_in = "The engine's line"
        reasons.append(_reason(
            "plan",
            (f"{lead_in}: {_line_text(best_line)}. {_events_text(best_line)} "
             + ("" if ends else f"Over the line {_material_text(gain, pov)}.")).strip(),
            line=best_line, start_fen=fen))

    # 2. What the opponent threatened, and whether the engine's move stops it.
    threat_before = None
    if wins_now:
        pass  # nothing else about the position matters
    elif board.is_check():
        checkers = [chess.square_name(sq) for sq in board.checkers()]
        reasons.append(_reason(
            "check",
            f"{mover}'s king is in check from {_join(checkers)}. In atomic a check "
            "means the king can be captured, which explodes it, so it must be dealt "
            "with unless the reply explodes the enemy king first.",
            squares=checkers))
    else:
        threat_before = threat(board, search, model)
        if threat_before is not None and threat_before["forcing"]:
            relief = threat_before["win"] - (100.0 - best_win)
            if relief >= THREAT_MIN:
                reasons.append(_reason(
                    "parries",
                    f"{opponent} was threatening {_line_text(threat_before['line'], False)}. "
                    f"{_events_text(threat_before['line'])} "
                    f"{best_san} stops it. If {mover} passed, {opponent} would be at "
                    f"{threat_before['win']:.0f}%; after {best_san} they are at "
                    f"{100.0 - best_win:.0f}%.",
                    line=threat_before["line"], start_fen=threat_before["start_fen"],
                    hypothetical=True, win_delta=relief))

    # 3. What the engine's move threatens next.
    after_best = _after(fen, best.uci)
    if not after_best.is_game_over(claim_draw=True):
        created = threat(after_best, search, model)
        if created is not None and created["forcing"]:
            pressure = created["win"] - best_win
            # Already told: the engine's own line plays the threat next.
            already = len(best_line) > 2 and best_line[2]["uci"] == created["uci"]
            if pressure >= THREAT_MIN and not already:
                reasons.append(_reason(
                    "threatens",
                    f"{best_san} threatens {_line_text(created['line'], False)}. "
                    f"{_events_text(created['line'])} "
                    f"{opponent} has to spend their move answering it.",
                    line=created["line"], start_fen=created["start_fen"],
                    hypothetical=True,
                    win_delta=pressure))

    parried = any(r["kind"] == "parries" for r in reasons)
    for note in notes(board, chess.Move.from_uci(best.uci)):
        # A piece leaving *some* blast radius is true of many quiet moves. It
        # is only the reason when there was a real threat to step out of.
        if note["kind"] == "defusal" and not parried:
            continue
        reasons.append(_reason(note["kind"], note["text"], squares=note["squares"]))

    # 4. Why not the other move.
    alternative = None
    if other is not None and other != best.uci:
        other_win, cont, other_mate = _score_move(fen, other, root, search, model)
        other_san = board.san(chess.Move.from_uci(other))
        loss = max(0.0, best_win - other_win)
        after_other = _after(fen, other)
        # The move itself is the first ply of its own refutation: when it is a
        # capture, what its explosion costs is usually the whole story.
        refutation, forcing = proof(fen, (other, *cont), other_mate)
        replies = refutation[1:]
        why: list[dict[str, Any]] = []
        if loss < CLOSE:
            kind = "close"
            text = (
                f"{other_san} is about as good: {other_win:.0f}% against "
                f"{best_win:.0f}% for {best_san}. The engine's choice between "
                "them is within noise at this budget."
            )
        elif forcing and (
            refutation[-1]["king"] == pov
            or (refutation[-1]["king"] is None and swing(refutation, pov) < 0)
        ):
            kind = "refuted"
            ignores = (
                threat_before is not None
                and threat_before["forcing"]
                and bool(replies)
                and replies[0]["uci"] == threat_before["uci"]
            )
            text = (
                f"{other_san} loses {loss:.0f}%: {_line_text(refutation)}. "
                f"{_events_text(refutation)} "
                + (f"{other_san} does nothing about the threat. " if ignores else "")
                + ("" if refutation[-1]["king"]
                   else f"Over the line {_material_text(swing(refutation, pov), pov)}.")
            )
        elif forcing and refutation[-1]["king"] is None:
            kind = "positional"
            text = (
                f"{other_san} loses {loss:.0f}% even though the forcing line "
                f"{_line_text(refutation)} leaves "
                f"{_material_text(swing(refutation, pov), pov)}. The engine's "
                "objection is not material: it dislikes the position that line "
                "leaves behind."
            )
        else:
            kind = "positional"
            text = (
                f"{other_san} loses {loss:.0f}%, but the engine sees no short forcing "
                f"line behind it. The best reply is {_line_text(replies) or '—'}, "
                f"and the difference is positional: it shows up beyond {MAX_PROOF} "
                "plies, or in the evaluation, not in a tactic."
            )
        why.append(_reason(kind, text.strip(), line=refutation, start_fen=fen))
        for note in notes(board, chess.Move.from_uci(other)):
            if note["kind"] == "self-blast":
                why.append(_reason(note["kind"], note["text"], squares=note["squares"]))
        alternative = {
            "uci": other,
            "san": other_san,
            "win": other_win,
            "loss": loss,
            "verdict": winrate.classify(loss),
            "reasons": why,
            # Themes of the punishment, so the "solver" is the opponent.
            "themes": _themes_of(after_other.fen(), replies),
        }

    reasons.sort(key=lambda r: ORDER.get(r["kind"], len(ORDER)))
    summary = _summary(mover, best_san, best_win, reasons, alternative, subject_is_best)

    return {
        "over": False,
        "fen": fen,
        "turn": pov,
        "best": {
            "uci": best.uci,
            "san": best_san,
            "win": best_win,
            "cp": best.cp,
            "mate": best.mate,
            "depth": best.depth,
            "line": best_line,
            "forcing": best_forcing,
            "themes": _themes_of(fen, best_line),
        },
        "subject": subject,
        "subject_is_best": subject_is_best,
        "reasons": reasons,
        "alternative": alternative,
        "summary": summary,
    }


#: The order reasons are shown in, most explanatory first. A defence is a
#: better answer to "why this move" than the line that follows it, because the
#: line is usually just the best of what remains once the threat is stopped.
ORDER = {
    kind: rank for rank, kind in enumerate(
        ["wins", "check", "parries", "threatens", "plan",
         "regicidal", "connected-kings", "defusal", "self-blast"]
    )
}


def _summary(
    mover: str,
    best_san: str,
    best_win: float,
    reasons: Sequence[dict[str, Any]],
    alternative: dict[str, Any] | None,
    subject_is_best: bool,
) -> str:
    """One sentence for the top of the panel. The reasons carry the detail."""
    lead = {
        "wins": f"{best_san} wins immediately.",
        "check": f"{best_san} answers the check.",
        "parries": f"{best_san} is a defence: it stops a threat.",
        "threatens": f"{best_san} creates a threat.",
        "plan": f"{best_san} starts a forcing line.",
        "connected-kings": f"{best_san} uses the touching-kings rule.",
        "defusal": f"{best_san} takes a piece out of a blast radius.",
    }
    first = next((lead[r["kind"]] for r in reasons if r["kind"] in lead), None)
    head = first or f"{best_san} is the engine's choice ({best_win:.0f}% for {mover})."
    if alternative is None:
        return head
    if subject_is_best:
        return f"{head} {alternative['san']} would cost {alternative['loss']:.0f}%."
    if alternative["loss"] < CLOSE:
        return f"{alternative['san']} is fine. The engine slightly prefers {best_san}."
    size = alternative["verdict"] if alternative["verdict"] != "ok" else "a small slip"
    return f"{alternative['san']} costs {alternative['loss']:.0f}% ({size}). {head}"
