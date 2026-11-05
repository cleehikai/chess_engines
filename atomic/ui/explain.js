/* "Why?" — the engine's reasons for a move, on the board they happened on.
 *
 * The server does all of the thinking (`trainer/explain.py`): it searches,
 * walks the lines, and says in words what each explosion destroys. This panel
 * lays that out and lets each reason be played through on the board, one ply
 * at a time, with the blast squares lit. It keeps the page's rule: it decides
 * nothing about chess, it only shows what it was sent.
 *
 * A preview borrows the host's board and gives it back exactly as it found it.
 * `onRestore` lets the host redraw whatever it had on top (arrows, marks),
 * since those belong to the host, not to the position.
 */

import { staticState } from "/board.js";

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2).toLowerCase(), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || response.statusText);
  return data;
}

const KIND_LABEL = {
  wins: "wins",
  check: "check",
  parries: "defence",
  threatens: "threat",
  plan: "line",
  refuted: "refuted",
  positional: "positional",
  close: "close",
  "connected-kings": "kings touch",
  defusal: "out of the blast",
  "self-blast": "own pieces",
  regicidal: "atomic legality",
  concepts: "concepts",
};

export class ExplainPanel {
  /**
   * @param {Board} board          the host's board, borrowed for previews
   * @param {object} options
   *   onRestore()  called after a preview hands the board back
   */
  constructor(board, options = {}) {
    this.board = board;
    this.onRestore = options.onRestore || (() => {});
    this.root = el("div", { class: "explain" });
    this.token = 0;
    this.preview = null;    // {frames, at, saved}
    this.root.hidden = true;
  }

  /** Ask the server, and show the answer when it lands. */
  async show({ fen, move = null, versus = null, title = null }) {
    const token = ++this.token;
    this.stopPreview();
    this.root.hidden = false;
    this.root.textContent = "";
    this.root.append(el("div", { class: "sub thinking" },
      "Searching: the position, the move, and what each side threatens…"));
    let result;
    try {
      result = await post("/api/explain", { fen, move, versus });
    } catch (error) {
      if (token !== this.token) return;
      this.root.textContent = "";
      this.root.append(el("div", { class: "sub" }, error.message));
      return;
    }
    if (token !== this.token) return;   // asked about something else since
    this.render(result, title);
  }

  hide() {
    this.token += 1;
    this.stopPreview();
    this.root.hidden = true;
    this.root.textContent = "";
  }

  render(result, title) {
    this.root.textContent = "";
    if (result.over) {
      this.root.append(el("div", { class: "sub" }, `The game is over (${result.result}).`));
      return;
    }
    const best = result.best;
    const alt = result.alternative;
    const head = el("div", { class: "explain-head" },
      title ? el("div", { class: "sub" }, title) : null,
      el("div", { class: "explain-summary" }, result.summary),
      el("div", { class: "explain-moves" },
        el("span", { class: "tag good" }, `engine: ${best.san} · ${best.win.toFixed(0)}%`),
        alt
          ? el("span", { class: "tag " + tagFor(alt.verdict, alt.loss) },
              `${alt.san} · ${alt.win.toFixed(0)}% (−${alt.loss.toFixed(0)})`)
          : null));
    this.root.append(head);

    const whatBest = this.section(`What ${best.san} does`, result.reasons, best.themes);
    const whyNot = alt ? this.section(`Why not ${alt.san}`, alt.reasons, alt.themes) : null;
    // Lead with the question that was asked.
    if (!result.subject_is_best && whyNot) this.root.append(whyNot, whatBest);
    else this.root.append(whatBest, whyNot);

    this.root.append(el("p", { class: "sub explain-foot" },
      `From Fairy-Stockfish's own search, ${num(result.nodes)} nodes per search. `
      + "Each reason is a line the engine actually found. When there is no "
      + "short forcing line, it says so instead of guessing."));
  }

  section(heading, reasons, themes) {
    const box = el("div", { class: "explain-section" }, el("h3", {}, heading));
    if (!reasons.length) {
      box.append(el("div", { class: "sub" },
        "No threat, no forcing line, and no special rule. The engine just rates "
        + "the resulting position highest, for reasons a short search does not "
        + "spell out."));
    }
    for (const reason of reasons) {
      const row = el("div", { class: "reason " + reason.kind },
        el("span", { class: "reason-kind" }, KIND_LABEL[reason.kind] || reason.kind),
        el("span", { class: "reason-text" }, reason.text));
      if (reason.line && reason.line.length && reason.kind !== "close") {
        row.append(el("button", {
          class: "ghost small",
          onClick: () => this.startPreview(reason),
          title: reason.hypothetical
            ? "Starts from a position where the other side has passed: it shows the threat, not a game line."
            : "Step through this line on the board.",
        }, reason.hypothetical ? "Show the threat ▸" : "Show on board ▸"));
      } else if (reason.contributions && reason.contributions.length) {
        row.append(el("div", { class: "concept-bars" + (reason.agrees ? "" : " disagree") },
          reason.contributions.map((part) => el("div", { class: "concept-bar" },
            el("span", { class: "mono " + (part.cp >= 0 ? "good-text" : "bad-text") },
              `${part.cp >= 0 ? "+" : "−"}${(Math.abs(part.cp) / 100).toFixed(1)}`),
            el("span", {}, part.text.replace(/\s*\([+−-]?[\d.]+\)$/, ""))))));
      } else if (reason.squares && reason.squares.length) {
        row.addEventListener("mouseenter", () => this.flash(reason.squares));
        row.addEventListener("mouseleave", () => this.unflash());
      }
      box.append(row);
    }
    if (themes && themes.length) {
      box.append(el("div", { class: "row explain-themes" },
        themes.map((theme) => el("span", { class: "tag", title: theme.text }, theme.name))));
    }
    return box;
  }

  /* ---- previews on the host board ---- */

  framesOf(reason) {
    const frames = [{
      state: staticState(reason.start_fen || reason.line[0].fen_before),
      caption: reason.hypothetical
        ? "If the side to move passed: this position never happened, it shows the threat."
        : "Start of the line.",
    }];
    for (const ply of reason.line) {
      const state = staticState(ply.fen_after);
      state.last = ply.uci;
      state.blast = ply.blast;
      frames.push({ state, caption: ply.text });
    }
    return frames;
  }

  startPreview(reason) {
    const saved = this.preview ? this.preview.saved : {
      state: this.board.state,
      marks: { ...this.board.marks },
      arrows: [...this.board.arrows],
      frozen: this.board.frozen,
    };
    // A previewed position is not the game: a click on it must not play a move.
    this.board.frozen = true;
    this.preview = { frames: this.framesOf(reason), at: 0, saved };
    this.drawPreview();
  }

  step(delta) {
    if (!this.preview) return;
    const at = this.preview.at + delta;
    if (at < 0 || at >= this.preview.frames.length) return;
    this.preview.at = at;
    this.drawPreview();
  }

  drawPreview() {
    const { frames, at } = this.preview;
    const frame = frames[at];
    this.board.setState(frame.state);
    this.controls?.remove();
    this.controls = el("div", { class: "explain-preview" },
      el("div", { class: "row" },
        el("button", { class: "ghost small", onClick: () => this.step(-1), disabled: at === 0 }, "◂"),
        el("span", { class: "mono" }, `${at} / ${frames.length - 1}`),
        el("button", {
          class: "ghost small", onClick: () => this.step(1),
          disabled: at === frames.length - 1,
        }, "▸"),
        el("button", { class: "small", onClick: () => this.stopPreview() }, "Back to the game")),
      el("div", { class: "sub" }, frame.caption));
    this.root.prepend(this.controls);
  }

  stopPreview() {
    if (!this.preview) return;
    const { saved } = this.preview;
    this.preview = null;
    this.controls?.remove();
    this.controls = null;
    if (saved.state) {
      this.board.setState(saved.state);
      this.board.marks = saved.marks;
      this.board.arrows = saved.arrows;
      this.board.frozen = saved.frozen;
      this.board.render();
    }
    this.onRestore();
  }

  /** Arrow keys belong to the preview while one is running. */
  handleKey(event) {
    if (!this.preview) return false;
    if (event.key === "ArrowRight") this.step(1);
    else if (event.key === "ArrowLeft") this.step(-1);
    else if (event.key === "Escape") this.stopPreview();
    else return false;
    event.preventDefault();
    return true;
  }

  flash(squares) {
    if (this.preview) return;
    this.flashed = { ...this.board.marks };
    for (const square of squares) this.board.marks[square] = "hint";
    this.board.render();
  }

  unflash() {
    if (!this.flashed || this.preview) return;
    this.board.marks = this.flashed;
    this.flashed = null;
    this.board.render();
  }
}

function tagFor(verdict, loss) {
  if (verdict === "blunder") return "bad";
  if (verdict === "mistake" || verdict === "inaccuracy") return "warn";
  return loss < 5 ? "good" : "";
}

function num(value) {
  return Number(value || 0).toLocaleString("en-US");
}
