/* A chessboard that knows nothing about chess.
 *
 * Every view in the trainer needs a board, and none of them should contain a
 * second opinion about what is legal. This component renders whatever `state`
 * the server sent — squares, pieces, the last move, the explosion — and reports
 * clicks back as UCI strings drawn from `state.legal`, which the server built.
 * If the server did not list a move, this file cannot produce it.
 *
 * That is the same referee rule `lab/server.py` documents, and it is worth
 * keeping in a variant where the rules are the hard part: three implementations
 * of atomic movegen agree on all 38 perft reference counts, and none of them is
 * written in JavaScript.
 *
 * Extras the trainer needs beyond the play page: arbitrary square marks (hint,
 * right, wrong), an arrow layer for "the move you missed", and a static mode
 * for the boards that are illustrations rather than games.
 *
 * Pieces are the cburnett SVGs (see ui/pieces/README.md), not the Unicode
 * chess glyphs this file used to draw. The glyphs were unreadable at board
 * size for a specific reason worth recording: ♔ and ♚ are outline and filled
 * versions of the *same* shape, so telling the colours apart depended entirely
 * on a text-shadow, and every font renders them at a different weight.
 *
 * Input is pointer-based and supports both grammars, because people expect
 * both: click a piece then click a square, or pick it up and drop it. They
 * share one gesture — `pointerdown` selects, and what happens next decides
 * which grammar was meant — so neither is a mode you can be stuck in.
 */

const GLYPH = {
  P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔",
  p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
};

/** FEN letter -> the class that carries its image, e.g. `n` -> `p-bN`. */
const PIECE_CLASS = {};
for (const role of "KQRBNP") {
  PIECE_CLASS[role] = `p-w${role}`;
  PIECE_CLASS[role.toLowerCase()] = `p-b${role}`;
}

/** How far the pointer may wander before a click becomes a drag, in pixels. */
const DRAG_SLOP = 4;

const FILES = "abcdefgh";

/** Unique prefixes so several boards on one page never share element ids. */
let boardCount = 0;

const PIECE_NAME = {
  K: "king", Q: "queen", R: "rook", B: "bishop", N: "knight", P: "pawn",
};

/** What a screen reader says for a square. */
function squareLabel(name, letter) {
  if (!letter) return name;
  const colour = letter === letter.toUpperCase() ? "white" : "black";
  return `${name}, ${colour} ${PIECE_NAME[letter.toUpperCase()]}`;
}

/** The king of the side to move, so "check" can be shown on the board. */
function kingSquare(pieces, turn) {
  const king = turn === "black" ? "k" : "K";
  for (const [square, letter] of Object.entries(pieces)) {
    if (letter === king) return square;
  }
  return null;
}

export function parseFen(fen) {
  const rows = fen.split(" ")[0].split("/");
  const pieces = {};
  rows.forEach((row, i) => {
    let file = 0;
    for (const ch of row) {
      if (ch >= "1" && ch <= "8") { file += +ch; continue; }
      pieces[FILES[file] + (8 - i)] = ch;
      file++;
    }
  });
  return pieces;
}

/** One piece, as an element. The image itself comes from CSS. */
export function pieceNode(letter) {
  const node = document.createElement("span");
  node.className = "piece " + (letter === letter.toUpperCase() ? "w " : "b ")
    + PIECE_CLASS[letter];
  return node;
}

/** Turn a FEN into the subset of `state` this component needs. */
export function staticState(fen) {
  return {
    fen,
    turn: fen.split(" ")[1] === "b" ? "black" : "white",
    legal: {},
    check: false,
    over: false,
    last: null,
    blast: [],
  };
}

export class Board {
  /**
   * @param {HTMLElement} root  container; gets the .board grid inside it
   * @param {object} options
   *   onMove(uci)      called once a click completes a legal move
   *   interactive      whether clicks do anything at all (default true)
   *   flipped          black at the bottom
   */
  constructor(root, options = {}) {
    this.root = root;
    this.onMove = options.onMove || (() => {});
    this.interactive = options.interactive !== false;
    this.flipped = !!options.flipped;
    this.state = null;
    this.selected = null;
    this.marks = {};       // square -> class name
    this.arrows = [];      // {from, to, kind}
    this.frozen = false;   // true while waiting on the server

    this.grid = document.createElement("div");
    this.grid.className = "board";
    this.grid.tabIndex = 0;
    this.grid.setAttribute("role", "grid");
    this.grid.setAttribute(
      "aria-label",
      "Atomic chess board. Arrow keys move the cursor, Enter selects or plays, "
        + "Escape cancels.",
    );
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svg.setAttribute("class", "arrows");
    this.svg.setAttribute("viewBox", "0 0 8 8");
    this.wrap = document.createElement("div");
    this.wrap.className = "board-wrap";
    this.wrap.append(this.grid, this.svg);
    this.root.append(this.wrap);

    this.promo = null;
    this.drag = null;      // {from, piece, node, moved} while a piece is held
    this.press = null;     // what `pointerdown` decided, until `pointerup`

    this.cursor = null;    // keyboard square cursor
    this.uid = `board-${++boardCount}`;

    this.grid.addEventListener("pointerdown", (event) => this._down(event));
    this.grid.addEventListener("pointermove", (event) => this._move(event));
    this.grid.addEventListener("pointerup", (event) => this._up(event));
    this.grid.addEventListener("pointercancel", () => this._cancelDrag());
    this.grid.addEventListener("keydown", (event) => this._key(event));
    this.grid.addEventListener("focus", () => {
      if (!this.cursor && this.state) {
        this.cursor = this.flipped ? "e7" : "e2";
        this.render();
      }
    });
    // A held piece must not double as a text selection or an image drag.
    this.grid.addEventListener("dragstart", (event) => event.preventDefault());
  }

  setState(state, { keepMarks = false } = {}) {
    // A new position under a held piece would leave the drag node orphaned and
    // the ghost pointing at a square that no longer means anything.
    if (this.drag) { this.drag.node.remove(); this.drag = null; this.press = null; }
    this.state = state;
    this.selected = null;
    if (!keepMarks) { this.marks = {}; this.arrows = []; }
    this.render();
  }

  setFlipped(flipped) { this.flipped = flipped; this.render(); }
  setFrozen(frozen) { this.frozen = frozen; this.render(); }

  mark(square, kind) { this.marks[square] = kind; this.render(); }
  clearMarks() { this.marks = {}; this.arrows = []; this.render(); }

  arrow(from, to, kind = "best") {
    this.arrows.push({ from, to, kind });
    this.render();
  }

  get playable() {
    return this.interactive && !this.frozen && this.state && !this.state.over;
  }

  render() {
    if (!this.state) return;
    const pieces = parseFen(this.state.fen);
    const ranks = this.flipped ? [1,2,3,4,5,6,7,8] : [8,7,6,5,4,3,2,1];
    const files = this.flipped ? [...FILES].reverse() : [...FILES];
    const last = this.state.last;
    const targets = this.selected ? (this.state.legal[this.selected] || []) : [];
    const targetSquares = new Set(targets.map((m) => m.slice(2, 4)));
    const inCheck = this.state.check ? kingSquare(pieces, this.state.turn) : null;

    this.grid.textContent = "";
    this.grid.setAttribute(
      "aria-label",
      `Atomic chess board. ${this.state.turn === "black" ? "Black" : "White"} to move`
        + (this.state.check ? ", in check" : "")
        + ". Arrow keys move the cursor, Enter selects or plays, Escape cancels.",
    );
    if (this.cursor) {
      this.grid.setAttribute("aria-activedescendant", `${this.uid}-${this.cursor}`);
    } else {
      this.grid.removeAttribute("aria-activedescendant");
    }
    ranks.forEach((rank, r) => {
      const row = document.createElement("div");
      row.className = "row-cells";
      row.setAttribute("role", "row");
      files.forEach((file, f) => {
        const name = file + rank;
        const sq = document.createElement("div");
        const tone = (r + f) % 2 === 0 ? "light" : "dark";
        sq.className = "sq " + tone;
        sq.id = `${this.uid}-${name}`;
        sq.dataset.square = name;
        sq.setAttribute("role", "gridcell");

        if (last && (name === last.slice(0, 2) || name === last.slice(2, 4))) {
          sq.classList.add("lastmove");
        }
        if (name === this.selected) sq.classList.add("selected");
        if (name === this.cursor) sq.classList.add("cursor");
        if (name === inCheck) sq.classList.add("in-check");
        if (this.state.blast && this.state.blast.includes(name)) sq.classList.add("blast");
        if (this.marks[name]) sq.classList.add("mark-" + this.marks[name]);

        const piece = pieces[name];
        if (piece) {
          const span = pieceNode(piece);
          // The square whose piece is currently in the air keeps its slot — the
          // ghost is what makes "I am holding this one" legible, and removing it
          // would make the board flicker a hole on every pickup.
          if (this.drag && this.drag.from === name) span.classList.add("ghost");
          sq.append(span);
        }
        sq.setAttribute("aria-label", squareLabel(name, piece));
        if (name === this.selected) sq.setAttribute("aria-selected", "true");

        if (targetSquares.has(name)) {
          const hint = document.createElement("div");
          hint.className = "hint" + (piece ? " capture" : "");
          sq.append(hint);
        }
        if (this.playable && this.state.legal[name]) sq.classList.add("grabbable");

        if (f === 0) sq.append(this._coord("rank", rank, tone));
        if (r === 7) sq.append(this._coord("file", file, tone));
        row.append(sq);
      });
      this.grid.append(row);
    });
    this._renderArrows();
  }

  _coord(kind, text, tone) {
    const div = document.createElement("div");
    div.className = "coord " + kind + " on-" + tone;
    div.textContent = text;
    return div;
  }

  /** Keyboard grammar: move a cursor, Enter selects, Enter again plays. */
  _key(event) {
    if (!this.state) return;
    const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0],
                   ArrowUp: [0, 1], ArrowDown: [0, -1] }[event.key];
    if (step) {
      event.preventDefault();
      const [df, dr] = step;
      if (!this.cursor) {
        this.cursor = this.flipped ? "e7" : "e2";
      } else {
        const file = FILES.indexOf(this.cursor[0]) + (this.flipped ? -df : df);
        const rank = +this.cursor[1] + (this.flipped ? -dr : dr);
        if (file >= 0 && file < 8 && rank >= 1 && rank <= 8) {
          this.cursor = FILES[file] + rank;
        }
      }
      this.render();
      return;
    }
    if (event.key === "Escape") {
      event.preventDefault();
      this.selected = null;
      this.render();
      return;
    }
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    if (!this.cursor) {
      this.cursor = this.flipped ? "e7" : "e2";
      this.render();
      return;
    }
    const matches = this._movesTo(this.cursor);
    if (matches.length) { this._play(matches); return; }
    this.selected = this.playable && this.state.legal[this.cursor]
      ? this.cursor
      : null;
    this.render();
  }

  /** Board coordinates of a square's centre, in the 0..8 arrow viewBox. */
  _centre(square) {
    const file = FILES.indexOf(square[0]);
    const rank = +square[1] - 1;
    return this.flipped
      ? { x: 7 - file + 0.5, y: rank + 0.5 }
      : { x: file + 0.5, y: 7 - rank + 0.5 };
  }

  _renderArrows() {
    this.svg.textContent = "";
    for (const { from, to, kind } of this.arrows) {
      const a = this._centre(from);
      const b = this._centre(to);
      // Stop short of the centre so the arrowhead does not cover the piece.
      const dx = b.x - a.x, dy = b.y - a.y;
      const len = Math.hypot(dx, dy) || 1;
      const end = { x: b.x - (dx / len) * 0.32, y: b.y - (dy / len) * 0.32 };
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
      line.setAttribute("x2", end.x); line.setAttribute("y2", end.y);
      line.setAttribute("class", "arrow arrow-" + kind);
      const head = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
      const ux = dx / len, uy = dy / len;
      const px = -uy, py = ux;
      const tip = { x: b.x - ux * 0.1, y: b.y - uy * 0.1 };
      const base = { x: tip.x - ux * 0.28, y: tip.y - uy * 0.28 };
      head.setAttribute("points", [
        `${tip.x},${tip.y}`,
        `${base.x + px * 0.16},${base.y + py * 0.16}`,
        `${base.x - px * 0.16},${base.y - py * 0.16}`,
      ].join(" "));
      head.setAttribute("class", "arrow-head arrow-" + kind);
      this.svg.append(line, head);
    }
  }

  /* ---- input ----
   *
   * One press, three possible meanings, resolved without asking the user to
   * declare which they meant:
   *
   *   press a movable piece            -> select it, and arm a drag
   *   press a square it can move to    -> play that move on release
   *   press the piece already selected -> deselect it on release, unless the
   *                                       press turned into a drag
   *
   * The third is why selection is toggled on *release* rather than on press:
   * picking a piece up and putting it back down where it started is a cancel,
   * and it would be perverse for that gesture to also clear the selection you
   * were about to use.
   */

  _movesTo(square) {
    if (!this.selected) return [];
    return (this.state.legal[this.selected] || [])
      .filter((uci) => uci.slice(2, 4) === square);
  }

  _squareAt(x, y) {
    const found = document.elementFromPoint(x, y);
    const sq = found && found.closest ? found.closest(".sq") : null;
    return sq && this.grid.contains(sq) ? sq.dataset.square : null;
  }

  _play(matches) {
    if (!matches.length) return false;
    if (matches.length > 1) { this._askPromotion(matches); return true; }
    this.selected = null;
    this.render();
    this.onMove(matches[0]);
    return true;
  }

  _down(event) {
    if (event.button > 0 || !this.playable) return;
    const sq = event.target.closest(".sq");
    if (!sq) return;
    const name = sq.dataset.square;
    const targets = this._movesTo(name);
    this.press = { square: name, x: event.clientX, y: event.clientY, targets,
                   wasSelected: this.selected === name };

    // A destination first: a square holding one of your own pieces can never
    // be the target of a legal move, so the two cases cannot both apply.
    if (targets.length) return;
    if (this.state.legal[name]) {
      // Capture before rendering, and onto the grid rather than the square:
      // `render` replaces every square element, so a capture held by the one
      // under the pointer would be dropped the instant the piece lights up —
      // and the rest of the gesture would go wherever the pointer happened to
      // be instead of to this board.
      try { this.grid.setPointerCapture(event.pointerId); } catch { /* no capture */ }
      this.selected = name;
      this.render();
      // The piece is not lifted yet. It becomes a drag only once the pointer
      // moves (`_move`), so a plain click never briefly shows a second copy of
      // the piece hanging off the cursor.
    } else if (this.selected) {
      this.selected = null;
      this.render();
    }
  }

  _startDrag(square, event) {
    const piece = parseFen(this.state.fen)[square];
    if (!piece) return;
    if (this.drag) this.drag.node.remove();   // a lost pointerup, at most once
    const node = pieceNode(piece);
    node.classList.add("piece-drag");
    const size = this.grid.querySelector(".sq").getBoundingClientRect().width;
    node.style.width = node.style.height = `${size}px`;
    document.body.append(node);
    this.drag = { from: square, node };
    this.render();          // the square it left keeps a ghost of it
    this._dragTo(event.clientX, event.clientY);
  }

  _dragTo(x, y) {
    if (!this.drag) return;
    this.drag.node.style.left = `${x}px`;
    this.drag.node.style.top = `${y}px`;
  }

  _move(event) {
    const press = this.press;
    if (!press) return;
    if (!this.drag) {
      const far = Math.hypot(event.clientX - press.x, event.clientY - press.y)
        > DRAG_SLOP;
      // Only a piece that can move gets picked up; a press on a destination
      // square is a click that has not finished, not a drag.
      if (!far || press.targets.length || !this.state.legal[press.square]) return;
      this._startDrag(press.square, event);
    }
    this._dragTo(event.clientX, event.clientY);
    event.preventDefault();
  }

  _cancelDrag() {
    if (this.drag) {
      this.drag.node.remove();
      this.drag = null;
      this.render();
    }
    this.press = null;
  }

  _up(event) {
    const press = this.press;
    const dragged = !!this.drag;   // a drag only exists once the piece was lifted
    const drag = this.drag;
    if (drag) { drag.node.remove(); this.drag = null; }
    this.press = null;
    if (!press || !this.playable) { if (drag) this.render(); return; }

    if (dragged) {
      const dropped = this._squareAt(event.clientX, event.clientY);
      const matches = dropped ? this._movesTo(dropped) : [];
      if (matches.length) { this._play(matches); return; }
      // Dropped on nothing that works. That is a deliberate gesture that
      // failed, so it cancels — unlike a plain click, which leaves the
      // selection standing so the other grammar can finish the move.
      this.selected = dropped === press.square ? this.selected : null;
      this.render();
      return;
    }

    if (press.targets.length) { this._play(press.targets); return; }
    if (press.wasSelected) { this.selected = null; }
    this.render();
  }

  _askPromotion(moves) {
    const backdrop = document.createElement("div");
    backdrop.className = "promo-backdrop";
    const box = document.createElement("div");
    box.className = "promo";
    box.append(Object.assign(document.createElement("div"), {
      textContent: "Promote to",
    }));
    const row = document.createElement("div");
    row.className = "row";
    const white = this.state.turn === "white";
    for (const uci of moves) {
      const role = uci[4];
      const button = document.createElement("button");
      button.append(pieceNode(white ? role.toUpperCase() : role));
      button.onclick = () => {
        backdrop.remove();
        this.selected = null;
        this.render();
        this.onMove(uci);
      };
      row.append(button);
    }
    box.append(row);
    backdrop.append(box);
    backdrop.onclick = (event) => {
      if (event.target === backdrop) { backdrop.remove(); this.selected = null; this.render(); }
    };
    document.body.append(backdrop);
  }
}

export { GLYPH, FILES };
