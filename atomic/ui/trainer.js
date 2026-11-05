/* The trainer UI: one page, ten views, hash routing, no dependencies.
 *
 * Every view is a function that renders into <main> and returns an optional
 * teardown. The router owns nothing else — no virtual DOM, no state container —
 * because the server already holds all the state worth holding, and a second
 * copy of it here is a second thing that can be wrong.
 *
 * The rule the whole file obeys: the page never decides anything a chess engine
 * or a database should decide. It does not know what is legal (the server sends
 * `legal`), it does not know a puzzle's solution (it submits moves and is told),
 * and it does not compute an evaluation. It lays things out and it listens.
 */

import { Board, staticState } from "/board.js";
import { ExplainPanel } from "/explain.js";

/* ---------- plumbing ---------- */

const $ = (sel, root = document) => root.querySelector(sel);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "text") node.textContent = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node[key.toLowerCase()] = value;
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, body) {
  const options = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) };
  const response = await fetch(path, options);
  let data;
  try { data = await response.json(); }
  catch { throw new Error(`${response.status} ${response.statusText}`); }
  if (data && data.error) throw new Error(data.error);
  return data;
}

let toastTimer = null;
function toast(message) {
  $("#toast")?.remove();
  const node = el("div", { class: "toast", id: "toast" }, message);
  document.body.append(node);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.remove(), 6000);
}

const pct = (value, digits = 0) =>
  value === null || value === undefined ? "—" : `${value.toFixed(digits)}%`;
const num = (value) => (value === null || value === undefined ? "—" : value.toLocaleString());
const moveNumber = (ply) => `${Math.floor(ply / 2) + 1}${ply % 2 === 0 ? "." : "…"}`;
const dateOf = (ms) => (ms ? new Date(ms).toISOString().slice(0, 10) : "—");
const titleCase = (s) => s.charAt(0).toUpperCase() + s.slice(1);

function judgementMark(judgement) {
  return { blunder: "??", mistake: "?", inaccuracy: "?!", ok: "" }[judgement] || "";
}

function themeTags(list, onClick) {
  return (list || []).map((theme) =>
    el("span", {
      class: "tag" + (onClick ? " accent" : ""),
      style: onClick ? "cursor:pointer" : null,
      title: SUMMARY?.themes?.catalog?.[theme] || theme,
      onClick: onClick ? () => onClick(theme) : null,
    }, theme));
}

function empty(message, hint) {
  return el("div", { class: "empty" },
    el("div", {}, message),
    hint ? el("div", { class: "tiny", style: "margin-top:8px" }, hint) : null);
}

/* ---------- routing ---------- */

const ROUTES = [
  ["dashboard", "Overview", () => viewDashboard()],
  ["puzzles", "Puzzles", (q) => viewPuzzles(q)],
  ["improve", "Improvement", () => viewImprove()],
  ["games", "Games", () => viewGames()],
  ["review", "Review", (q) => viewReview(q)],
  ["play", "Play", (q) => viewPlay(q)],
  ["patterns", "Patterns", () => viewPatterns()],
  // Kept as `book` in the URL — the route id is what every internal link and
  // every bookmark uses — but shown as what it now is: a named index of the
  // openings, with the raw move table one click inside it.
  ["book", "Openings", (q) => viewBook(q)],
  ["traps", "Traps", (q) => viewTraps(q)],
  ["repertoire", "Repertoire", (q) => viewRepertoire(q)],
];

let SUMMARY = null;
let teardown = null;

function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "") || "dashboard";
  const [name, queryString] = raw.split("?");
  return { name: name || "dashboard", query: new URLSearchParams(queryString || "") };
}

function go(name, query = {}) {
  const params = new URLSearchParams(
    Object.entries(query).filter(([, v]) => v !== undefined && v !== null && v !== "")
  );
  const suffix = params.toString();
  location.hash = `#/${name}${suffix ? "?" + suffix : ""}`;
}

function renderTabs(active) {
  const tabs = $("#tabs");
  tabs.textContent = "";
  for (const [name, label] of ROUTES) {
    if (name === "review") continue;   // reached from a game, not from the bar
    tabs.append(el("a", {
      href: `#/${name}`,
      class: name === active ? "active" : "",
    }, label));
  }
}

async function route() {
  const { name, query } = parseHash();
  const entry = ROUTES.find(([key]) => key === name) || ROUTES[0];
  renderTabs(entry[0]);
  if (teardown) { try { teardown(); } catch { /* view already gone */ } teardown = null; }
  const main = $("#view");
  main.textContent = "";
  main.append(el("div", { class: "spinner" }, "loading…"));
  try {
    const node = await entry[2](query);
    main.textContent = "";
    main.append(node);
  } catch (error) {
    main.textContent = "";
    main.append(empty(`Could not load this view: ${error.message}`));
  }
}

/* ---------- shared pieces ---------- */

function statCard(label, value, note) {
  return el("div", { class: "card stat" },
    el("div", { class: "label" }, label),
    el("div", { class: "big" }, value),
    note ? el("div", { class: "sub" }, note) : null);
}

/** Win% over the course of a game, from the mover's side, drawn from White's. */
function evalGraph(plies, onSeek) {
  const width = 600, height = 74;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "evalgraph");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("preserveAspectRatio", "none");
  if (!plies.length) return svg;

  // Everything is drawn White-relative so the line has one meaning end to end.
  const points = plies.map((ply, i) => {
    const white = ply.mover === "white" ? ply.win_before : 100 - ply.win_before;
    return { x: (i / Math.max(plies.length - 1, 1)) * width, y: height - (white / 100) * height, ply };
  });
  const path = points.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const area = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
  area.setAttribute("class", "fill");
  area.setAttribute("points", `0,${height} ${path} ${width},${height}`);
  svg.append(area);

  const zero = document.createElementNS("http://www.w3.org/2000/svg", "line");
  zero.setAttribute("class", "zero");
  zero.setAttribute("x1", 0); zero.setAttribute("x2", width);
  zero.setAttribute("y1", height / 2); zero.setAttribute("y2", height / 2);
  svg.append(zero);

  points.forEach((point, index) => {
    const kind = plies[index].judgement;
    if (kind !== "blunder" && kind !== "mistake") return;
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("class", "dot-" + kind);
    dot.setAttribute("cx", point.x); dot.setAttribute("cy", point.y);
    dot.setAttribute("r", 3.5);
    svg.append(dot);
  });

  const cursor = document.createElementNS("http://www.w3.org/2000/svg", "line");
  cursor.setAttribute("class", "cursor");
  cursor.setAttribute("y1", 0); cursor.setAttribute("y2", height);
  cursor.setAttribute("x1", 0); cursor.setAttribute("x2", 0);
  svg.append(cursor);
  svg.moveCursor = (index) => {
    const x = points[Math.min(index, points.length - 1)]?.x ?? 0;
    cursor.setAttribute("x1", x); cursor.setAttribute("x2", x);
  };
  if (onSeek) {
    svg.onclick = (event) => {
      const box = svg.getBoundingClientRect();
      const ratio = (event.clientX - box.left) / box.width;
      onSeek(Math.round(ratio * (plies.length - 1)));
    };
  }
  return svg;
}

/* ---------- overview ---------- */

async function viewDashboard() {
  const s = SUMMARY = await api("/api/summary");
  const p = s.puzzles;
  const judged = s.judged || {};
  const myMoves = Object.values(judged).reduce((a, b) => a + b, 0);

  const wrap = el("div", {});
  wrap.append(
    el("div", { class: "view-title" },
      el("h1", {}, "Overview"),
      el("div", { class: "sub" }, s.winrate.describe)),
    el("p", { class: "lede" },
      "Everything here is measured on the atomic win% curve fitted in this repo, "
      + "not on the standard-chess one. An atomic centipawn is worth about a "
      + "third of a standard one, so judgements borrowed from standard chess "
      + "would call ordinary play a blunder."),
  );

  wrap.append(el("div", { class: "grid" },
    statCard("Games", num(s.games), `${s.analysed} analysed`),
    statCard("Puzzles", num(p.total),
      `${p.from_games} mined, ${p.from_tablebase} exact`),
    statCard("Solved unaided", `${p.puzzles_solved}/${p.puzzles_attempted || 0}`,
      `${p.attempts} attempts`),
    statCard("Your blunders", num(judged.blunder || 0),
      `${myMoves} of your moves judged`),
  ));

  const phases = el("div", { class: "card" },
    el("h2", {}, "Where your games go wrong"),
    el("table", {},
      el("thead", {}, el("tr", {},
        el("th", {}, "Plies"), el("th", { class: "num" }, "Moves"),
        el("th", { class: "num" }, "Blunders"), el("th", { class: "num" }, "Rate"),
        el("th", { class: "num" }, "Win% lost / move"))),
      el("tbody", {}, s.phases.map((bucket) => el("tr", {},
        el("td", {}, bucket.high === null ? `${bucket.low}+` : `${bucket.low}–${bucket.high}`),
        el("td", { class: "num" }, num(bucket.moves)),
        el("td", { class: "num" }, num(bucket.blunders)),
        el("td", { class: "num" }, pct(bucket.rate, 1)),
        el("td", { class: "num" }, bucket.per_move.toFixed(1)))))),
    el("p", { class: "sub", style: "margin-top:10px" },
      "Monotonic across the game: the opening is where the Elo is."));

  const themeCounts = p.themes || {};
  const themeCard = el("div", { class: "card" },
    el("h2", {}, "Puzzle themes"),
    Object.keys(themeCounts).length
      ? el("div", { class: "chips" }, Object.entries(themeCounts).map(([theme, n]) =>
          el("span", {
            class: "chip", title: s.themes.catalog[theme] || "",
            onClick: () => go("puzzles", { theme }),
          }, `${theme} ${n}`)))
      : el("p", { class: "sub" }, "No puzzles generated yet."),
    el("p", { class: "sub", style: "margin-top:10px" },
      "Detectors are hypotheses. `trainer puzzles themes` reports their firing "
      + "rates; one that never fires or fires on everything is a bug."));

  const quick = el("div", { class: "card" },
    el("h2", {}, "Start here"),
    el("div", { class: "row" },
      el("button", { class: "primary", onClick: () => go("puzzles", { unsolved: "1" }) },
        "Solve unsolved puzzles"),
      el("button", { onClick: () => go("puzzles", { mine: "1" }) }, "From my own games"),
      el("button", { onClick: () => go("improve") }, "What I am bad at"),
      el("button", { onClick: () => go("games") }, "Review a game"),
      el("button", { onClick: () => go("patterns") }, "My repeated mistakes")),
    p.total === 0
      ? el("p", { class: "sub", style: "margin-top:12px" },
          "No puzzles yet. Generate some: ",
          el("code", { class: "mono" }, "py -3.13 -m trainer.cli puzzles generate"),
          " and ",
          el("code", { class: "mono" }, "puzzles endgames"))
      : null);

  wrap.append(el("div", { class: "split", style: "margin-top:14px" },
    el("div", { class: "wide" }, phases, themeCard),
    el("div", { class: "side" }, quick,
      el("div", { class: "card" },
        el("h2", {}, "Corpus"),
        el("table", {}, el("tbody", {},
          el("tr", {}, el("td", {}, "Opening / middle / end"),
            el("td", { class: "num mono" },
              `${p.phases.opening || 0} / ${p.phases.middlegame || 0} / ${p.phases.endgame || 0}`)),
          el("tr", {}, el("td", {}, "Defensive (hold it)"),
            el("td", { class: "num mono" }, num(p.saves))),
          el("tr", {}, el("td", {}, "From your side"),
            el("td", { class: "num mono" }, num(p.mine))),
          el("tr", {}, el("td", {}, "Book"),
            el("td", { class: "num mono" },
              s.book.moves ? `${num(s.book.moves)} moves` : "not built")),
        ))))));
  return wrap;
}

/* ---------- puzzles ---------- */

/* Difficulty bands to filter by, in the Elo units the engine ladder measures
 * (`trainer/difficulty.py`). The boundaries fall between the ladder's rungs
 * rather than on round numbers: an edge inside a rung would claim a resolution
 * the measurement has not got. */
const BANDS = [
  ["", "any"],
  ["|1500", "easy — the weakest level finds it"],
  ["1500|1900", "medium"],
  ["1900|", "hard — needs the top of the ladder"],
];

/* Provenance, in the corpus's own three sources (`puzzles.SOURCES`). `dump` was
 * missing here for as long as the scraper has existed, which made 276 of the
 * 429 puzzles in this corpus unreachable by filter — and, because `game` was
 * *labelled* "mined" on the puzzle card, made it look as though the mined ones
 * were the only thing on offer. Both halves of that are fixed by naming each
 * source after where it came from. */
const SOURCE_LABEL = {
  game: "your own games",
  dump: "mined from rated games",
  tablebase: "tablebase (exact)",
};

const FILTERS = [
  ["source", "Source", [["", "any"], ...Object.entries(SOURCE_LABEL)]],
  ["phase", "Phase", [["", "any"], ["opening", "opening"], ["middlegame", "middlegame"], ["endgame", "endgame"]]],
  ["motive", "Goal", [["", "any"], ["win", "play for the win"], ["save", "hold the draw"]]],
  ["band", "Difficulty", BANDS],
  ["order", "Order", [["recent", "newest"], ["random", "random"],
                      ["easiest", "easiest first"], ["hardest", "hardest first"],
                      ["shortest", "shortest first"], ["longest", "longest first"]]],
];

async function viewPuzzles(query) {
  if (!SUMMARY) SUMMARY = await api("/api/summary");
  const filters = {
    theme: query.get("theme") || "",
    source: query.get("source") || "",
    phase: query.get("phase") || "",
    motive: query.get("motive") || "",
    order: query.get("order") || "random",
    mine: query.get("mine") === "1",
    unsolved: query.get("unsolved") === "1",
    // On unless it is turned off, and the only default here that is not "any":
    // a queue that hands back puzzles you have already sat through is a queue
    // you stop trusting. The redo queue is the `failed` chip, deliberately a
    // separate and explicit request.
    unseen: query.get("unseen") !== "0" && query.get("failed") !== "1",
    failed: query.get("failed") === "1",
    game: query.get("game") || "",
    band: query.get("band") || "",
  };
  const params = new URLSearchParams({ limit: "200", order: filters.order });
  for (const key of ["theme", "source", "phase", "motive", "game"]) {
    if (filters[key]) params.set(key, filters[key]);
  }
  if (filters.band) {
    const [low, high] = filters.band.split("|");
    if (low) params.set("min_rating", low);
    if (high) params.set("max_rating", high);
  }
  if (filters.mine) params.set("mine", "1");
  if (filters.unsolved) params.set("unsolved", "1");
  if (filters.unseen) params.set("unseen", "1");
  if (filters.failed) params.set("failed", "1");

  const listing = await api("/api/puzzles?" + params.toString());
  const wrap = el("div", {});

  /* The filter bar, collapsed.
   *
   * Fourteen controls and a row of theme chips used to sit between the title
   * and the board, permanently. They are the tool you use once to choose a
   * queue and then never look at again for twenty puzzles, which is exactly
   * the shape progressive disclosure is for: the summary line says what is
   * currently selected, so nothing is hidden — only folded. It opens by itself
   * when the filters produce nothing, because then they *are* the task.
   */
  const summary = describeFilters(filters, listing);
  const controls = el("details", { class: "disclosure", open: !listing.puzzles.length },
    el("summary", {},
      el("span", { class: "filter-summary" }, summary),
      el("span", { class: "muted tiny" }, `  ${listing.total} match`)));
  const row = el("div", { class: "row", style: "align-items:flex-end;margin-top:10px" });
  for (const [key, label, options] of FILTERS) {
    row.append(el("div", { class: "field-group", style: "min-width:140px" },
      el("label", { class: "field" }, label),
      el("select", {
        onChange: (event) => go("puzzles", { ...asQuery(filters), [key]: event.target.value }),
      }, options.map(([value, text]) =>
        el("option", { value, selected: filters[key] === value }, text)))));
  }
  row.append(el("div", { class: "row tight", style: "padding-bottom:2px" },
    el("span", {
      class: "chip" + (filters.mine ? " on" : ""),
      onClick: () => go("puzzles", { ...asQuery(filters), mine: filters.mine ? "" : "1" }),
    }, "my side only"),
    el("span", {
      class: "chip" + (filters.unseen ? " on" : ""),
      title: "never attempted — no puzzle comes round twice",
      onClick: () => go("puzzles", {
        ...asQuery(filters), unseen: filters.unseen ? "0" : "1", failed: "",
      }),
    }, "new only"),
    el("span", {
      class: "chip" + (filters.unsolved ? " on" : ""),
      title: "never solved unaided, including ones you have never seen",
      onClick: () => go("puzzles", { ...asQuery(filters), unsolved: filters.unsolved ? "" : "1" }),
    }, "unsolved only"),
    el("span", {
      class: "chip" + (filters.failed ? " on" : ""),
      title: "tried and never solved unaided — the redo queue",
      onClick: () => go("puzzles", {
        ...asQuery(filters),
        failed: filters.failed ? "" : "1",
        // The two are contradictory: a puzzle you failed is one you have seen.
        // Asking for the redo queue is the explicit request that turns the
        // no-repeats default off.
        unseen: filters.failed ? "1" : "0",
      }),
    }, "failed before"),
    filters.theme || filters.game
      ? el("span", { class: "chip on", onClick: () => go("puzzles", { order: filters.order }) },
          `clear ${filters.theme || "game " + filters.game} ✕`)
      : null));
  controls.append(row);

  const themeRow = el("div", { class: "chips", style: "margin-top:10px" },
    Object.entries(SUMMARY.themes.counts || {}).map(([theme, n]) =>
      el("span", {
        class: "chip" + (filters.theme === theme ? " on" : ""),
        title: SUMMARY.themes.catalog[theme] || "",
        onClick: () => go("puzzles", {
          ...asQuery(filters), theme: filters.theme === theme ? "" : theme,
        }),
      }, `${theme} ${n}`)));
  if (themeRow.children.length) {
    controls.append(
      el("div", { class: "sub", style: "margin-top:12px" }, "Themes"), themeRow);
  }

  // The queue was drawn once, so a finished run does not appear in it. Say so
  // rather than reloading: pulling the board out from under a puzzle someone
  // is in the middle of would be a worse bargain than a stale queue.
  const jobPanel = new JobPanel((result) => {
    toast(`Generation finished${result ? " — " + result : ""}. `
      + "Reload the view to solve the new puzzles.");
  }, ["generate", "mine", "endgames", "rate"]);
  const generate = el("details", { class: "disclosure", style: "margin-top:14px" },
    el("summary", {}, "Generate more puzzles"),
    el("div", { style: "margin-top:10px" }, jobPanel.node));

  if (!listing.puzzles.length) {
    wrap.append(
      puzzleHeader(null, listing),
      controls,
      el("div", { style: "margin-top:14px" },
        empty("No puzzles match these filters.",
          filters.unseen
            ? "“new only” is on, so anything you have already attempted is hidden. "
              + "Turn it off to see those again, or generate more below."
            : "Generate some below, or on the command line with "
              + "`trainer puzzles generate` / `mine` / `endgames`.")),
      generate);
    teardown = () => jobPanel.destroy();
    return wrap;
  }

  const session = new PuzzleSession(listing.puzzles, filters);
  wrap.append(puzzleHeader(session, listing), controls, session.node, generate);
  teardown = () => { session.destroy(); jobPanel.destroy(); };
  // Started after the router has attached `wrap`, not before: the first
  // puzzle plays a short setup animation, and running it while the node is
  // still detached would spend it off-screen.
  setTimeout(() => session.start().catch((error) => toast(error.message)), 0);
  return wrap;
}

/* The one line above the board: what you are solving, how it is going, and —
 * folded away — why these puzzles are the ones on offer. The explanation used
 * to be a four-line paragraph in front of every visit; it is worth reading
 * once, so it is available rather than repeated. */
function puzzleHeader(session, listing) {
  return el("div", { class: "puzzle-head" },
    el("div", { class: "view-title" },
      el("h1", {}, "Puzzles"),
      el("div", { class: "sub" }, `${listing.total} in this queue`)),
    session ? session.progress : null,
    el("details", { class: "disclosure quiet" },
      el("summary", {}, "Why these puzzles"),
      el("p", { class: "sub", style: "margin-top:6px;max-width:78ch" },
        "Every puzzle here has exactly one move that works, and none can be "
        + "solved by one-ply greedy material — atomic produces enormous "
        + "evaluation swings that a beginner sees instantly, and those are "
        + "filtered out rather than counted. Roughly half are holds rather than "
        + "wins: the banner under the board says which, before you touch a "
        + "piece.")));
}

/* The collapsed filter bar's summary. Nothing is hidden that is not named
 * here — a fold that loses track of its own state is just a bug. */
function describeFilters(filters, listing) {
  const parts = [];
  const label = (list, value) => (list.find(([key]) => key === value) || [])[1];
  parts.push(label(FILTERS[4][2], filters.order) || filters.order);
  if (filters.source) parts.push(SOURCE_LABEL[filters.source]);
  if (filters.phase) parts.push(filters.phase);
  if (filters.motive) parts.push(label(FILTERS[2][2], filters.motive));
  if (filters.band) parts.push(label(BANDS, filters.band));
  if (filters.theme) parts.push(filters.theme);
  if (filters.game) parts.push("game " + filters.game);
  if (filters.mine) parts.push("my side only");
  if (filters.unseen) parts.push("new only");
  if (filters.unsolved) parts.push("unsolved only");
  if (filters.failed) parts.push("failed before");
  return parts.join(" · ");
}

function asQuery(filters) {
  return {
    theme: filters.theme, source: filters.source, phase: filters.phase,
    motive: filters.motive, order: filters.order, game: filters.game,
    band: filters.band,
    mine: filters.mine ? "1" : "", unsolved: filters.unsolved ? "1" : "",
    // `unseen` defaults to on, so it is the *off* state that has to survive a
    // round trip through the hash; `go` drops empty values.
    unseen: filters.unseen ? "" : "0",
    failed: filters.failed ? "1" : "",
  };
}

/* What the solver is playing *for*, in words.
 *
 * Half this corpus is `save` material — one move holds and everything else
 * loses — and hunting a knockout in one of those is solving a puzzle nobody
 * set. The old UI implied the goal in a verdict line ("Only one move holds")
 * and left the rest to be inferred, which is how a hold gets mistaken for a
 * win that the engine was too weak to see.
 *
 * So the goal is stated, and it carries its evidence: the tablebase's category
 * where there is one, and otherwise the engine's own win% for the solver after
 * best play — which is an opinion, and is labelled as one.
 */
function goalWords(puzzle) {
  const goal = puzzle.goal || {};
  const win = goal.kind === "win";
  const you = titleCase(puzzle.solver);
  if (goal.exact) {
    return {
      short: win ? "Play for the win" : "Play for the draw",
      detail: win
        ? "The tablebase says this position is won. The solution is the only move that keeps it won."
        : `The tablebase says this position is drawn${goal.category && goal.category !== "draw" ? ` (${goal.category})` : ""}. `
          + "There is no win here — the solution is the only move that does not lose.",
      exact: true,
    };
  }
  const at = goal.win === null || goal.win === undefined
    ? null : `${goal.win.toFixed(0)}%`;
  return {
    short: win ? "Play for the win" : "Play to hold",
    detail: win
      ? `The engine puts ${you} at ${at || "a winning score"} after the right move — winning on its judgement, which is not a proof.`
      : `Best play here is worth only ${at || "about half"} to ${you}: the goal is not to lose. `
        + "Do not go hunting for a win — on the engine's judgement there is not one.",
    exact: false,
  };
}

/* How each calibrated engine level did on this puzzle.
 *
 * Shown only once the puzzle is over. Before that it is not quite a spoiler
 * and not quite safe either — "the 1691 level played Nc6 here" is a wrong move
 * rather than the answer, but it is still this board being discussed while you
 * are trying to think about it.
 */
function difficultyProfile(verdict) {
  if (!verdict || !verdict.levels || !verdict.levels.length) return null;
  return el("div", { style: "margin-top:12px" },
    el("div", { class: "sub" }, "Difficulty: " + verdict.describe),
    el("div", { class: "mono tiny muted", style: "margin-top:6px" },
      verdict.levels.map((level) =>
        el("div", {},
          `${level.elo.toFixed(0)} Elo (${level.nodes.toLocaleString()}n): `,
          level.solved
            ? el("span", { class: "good-text" }, "found the line")
            : el("span", { class: "bad-text" },
                `missed at move ${level.found + 1} of ${level.total}`)))),
    verdict.monotone ? null : el("p", { class: "sub", style: "margin-top:6px" },
      "A weaker level found this and a stronger one did not — two rungs of the "
      + "ladder are inside each other's error bars."));
}

/* One sitting at the puzzle board.
 *
 * The whole view is a **line of nodes** — the game that led here, the mistake,
 * the moves you find, and afterwards anything you care to try — with a cursor
 * on it. Everything else follows from that:
 *
 * - `←` and `→` walk the cursor. Backwards always works, including back past
 *   the blunder into the game that produced it, which is how "how did this
 *   arise" gets answered.
 * - Forwards is not restricted and does not need to be: during the puzzle the
 *   line simply *has no nodes past the live position*, because the moves you
 *   have not found are on the server and the page has never been told them.
 *   The referee rule does the work that a disabled key would only pretend to.
 * - The board is playable at the live node while the puzzle is on, and at any
 *   node once it is over — at which point playing a move truncates the line
 *   and continues from there, so exploring an idea is the same operation as
 *   solving, pointed somewhere else.
 *
 * Analysis is off until the puzzle is finished, for the obvious reason.
 */
class PuzzleSession {
  constructor(queue, filters) {
    this.queue = queue;
    this.filters = filters;
    this.index = 0;
    this.results = [];
    this.puzzle = null;

    this.nodes = [];        // {fen, san, uci, blast, state?, kind}
    this.cursor = 0;
    this.startIndex = 0;    // the node the puzzle actually asks about
    this.moveIndex = 0;     // plies of the solution played so far
    this.hints = 0;         // hints taken, over the whole puzzle: scoring
    this.hintLevel = 0;     // how much has been given away about *this* move
    this.hinting = false;
    this.wrong = 0;
    this.finished = false;
    this.analysisOn = false;
    this.startedAt = 0;
    this.busy = false;
    this.analysisToken = 0;
    this.analysisTimer = null;
    this.evalWhite = null;      // White's share of the last evaluation, if any
    this.evalText = "";

    this.boardHost = el("div", { class: "board-lg" });
    this.board = new Board(this.boardHost, {
      onMove: (uci) => this.onBoardMove(uci),
      interactive: true,
    });
    this.evalBar = el("div", { class: "evalbar" },
      el("span", { class: "fill" }), el("i", { class: "even" }),
      el("b", { class: "score" }, ""));
    this.evalBarWhite = $(".fill", this.evalBar);
    this.evalScore = $("b.score", this.evalBar);
    this.turnChip = el("div", { class: "turnchip" }, el("i", {}), el("span", {}, ""));

    this.verdict = el("div", { class: "verdict" }, "…");
    this.goalBox = el("div", { class: "goal" });
    this.instruction = el("div", { class: "sub" }, "");
    this.meta = el("div", {});
    this.progress = el("div", { class: "streak" });
    this.controls = el("div", { class: "row", style: "margin-top:12px" });
    this.lineBox = el("div", { class: "mono tiny muted", style: "margin-top:8px" });
    this.counter = el("span", { class: "right" }, "");
    this.moveList = el("div", { class: "movelist plain" });
    this.analysisBox = el("div", { class: "analysis" });
    this.analysisCard = el("div", { class: "card", hidden: true },
      el("h2", {}, "Engine", el("span", { class: "right tiny muted", id: "an-depth" })),
      this.analysisBox);

    /* What is on screen while you are thinking, and what is not.
     *
     * The old layout put four cards beside the board and kept them all open:
     * the verdict, the engine, this puzzle's provenance and themes, and the
     * session's keyboard map. Only the first is about the move you are trying
     * to find. The rest is extraneous load in Sweller's sense — it competes
     * for attention without contributing to the thing being learned — so it is
     * now disclosed rather than displayed: `meta` opens when the puzzle is
     * over (it is also mildly spoiler-shaped: "endgame, tablebase, one move"
     * narrows a search), `analysisCard` already did, and the keyboard map is a
     * `<details>` that stays shut until somebody wants it.
     *
     * The board and the one sentence telling you what to do are what is left,
     * which is the whole intent.
     */
    this.metaCard = el("div", { class: "card", hidden: true },
      el("h2", {}, "This puzzle"), this.meta);
    this.node = el("div", { class: "split", style: "margin-top:16px" },
      el("div", { class: "wide", style: "flex:0 0 auto" },
        el("div", { class: "board-row" }, this.evalBar, this.boardHost),
        el("div", { class: "card tight" },
          el("div", { class: "row", style: "justify-content:space-between" },
            el("div", { class: "row tight" },
              el("button", { class: "ghost step", onClick: () => this.goto(0) }, "⏮"),
              el("button", { class: "ghost step", onClick: () => this.goto(this.cursor - 1) }, "◀"),
              el("button", { class: "ghost step", onClick: () => this.goto(this.cursor + 1) }, "▶"),
              el("button", { class: "ghost step", onClick: () => this.goto(this.nodes.length - 1) }, "⏭")),
            el("div", { class: "row tight" },
              this.turnChip,
              el("button", { class: "ghost", onClick: () => this.flip() }, "Flip"))),
          this.moveList)),
      el("div", { class: "side" },
        el("div", { class: "card" },
          el("h2", {}, "Your move", this.counter),
          this.verdict, this.goalBox, this.instruction, this.lineBox, this.controls),
        this.analysisCard,
        this.metaCard,
        el("details", { class: "disclosure quiet" },
          el("summary", {}, "Keyboard"),
          el("p", { class: "sub", style: "margin-top:6px" },
            el("span", { class: "kbd" }, "←"), " ", el("span", { class: "kbd" }, "→"),
            " step · ",
            el("span", { class: "kbd" }, "H"), " hint · ",
            el("span", { class: "kbd" }, "S"), " solution · ",
            el("span", { class: "kbd" }, "N"), " next · ",
            el("span", { class: "kbd" }, "F"), " flip · ",
            el("span", { class: "kbd" }, "A"), " analysis")),
      ));

    this.onKey = (event) => {
      if (event.target?.matches?.("input, select, textarea")) return;
      // Held keys autorepeat. Stepping through a line that way is useful;
      // taking four hints because H was held a beat too long is not.
      if (event.repeat && !event.key.startsWith("Arrow")) return;
      if (event.key === "ArrowLeft") this.goto(this.cursor - 1);
      else if (event.key === "ArrowRight") this.goto(this.cursor + 1);
      else if (event.key === "Home") this.goto(0);
      else if (event.key === "End") this.goto(this.nodes.length - 1);
      else {
        const key = event.key.toLowerCase();
        if (key === "h") this.hint();
        else if (key === "s") this.reveal();
        else if (key === "n") this.next();
        else if (key === "f") this.flip();
        else if (key === "a") this.toggleAnalysis();
        else return;
      }
      event.preventDefault();
    };
    document.addEventListener("keydown", this.onKey);
  }

  destroy() {
    document.removeEventListener("keydown", this.onKey);
    clearTimeout(this.analysisTimer);
    this.analysisToken += 1;   // abandon whatever is in flight
  }

  async start() { await this.load(); }

  /* ---- the line ---- */

  get live() { return this.nodes.length - 1; }
  get atLive() { return this.cursor >= this.live; }
  get current() { return this.nodes[this.cursor]; }

  /** Whether the board may be moved on right now, and with what meaning. */
  get playable() {
    if (this.finished) return "explore";
    return this.atLive && this.moveIndex % 2 === 0 ? "solve" : "";
  }

  push(node) {
    this.nodes.push(node);
    this.cursor = this.nodes.length - 1;
  }

  /** Cut the line back to the cursor. Only reachable once the puzzle is over. */
  truncate() { this.nodes = this.nodes.slice(0, this.cursor + 1); }

  async load() {
    const entry = this.queue[this.index];
    clearTimeout(this.analysisTimer);
    this.analysisToken += 1;
    if (!entry) {
      this.verdict.className = "verdict done";
      this.verdict.textContent = "That is the whole set.";
      this.instruction.textContent = "Change the filters, or generate more.";
      this.controls.textContent = "";
      this.nodes = [];
      this.board.setState(staticState("8/8/8/8/8/8/8/8 w - - 0 1"));
      this.moveList.textContent = "";
      this.analysisCard.hidden = true;
      return;
    }
    const puzzle = this.puzzle = await api(`/api/puzzle/${entry.id}`);
    this.moveIndex = 0;
    this.hints = 0;
    this.hintLevel = 0;
    this.wrong = 0;
    this.finished = false;
    this.analysisOn = false;
    this.analysisCard.hidden = true;
    this.metaCard.hidden = true;
    // The previous puzzle's evaluation is not this one's; leave the bar even
    // and blank rather than carrying a stale reading into a new position.
    this.evalWhite = null;
    this.evalText = "";
    this.evalBarWhite.style.height = "50%";
    this.evalScore.textContent = "";
    this.startedAt = Date.now();
    this.board.setFlipped(puzzle.solver === "black");
    this.counter.textContent = `${this.index + 1} / ${this.queue.length}`;
    this.renderGoal();
    this.renderMeta();
    this.renderControls();
    this.lineBox.textContent = "";

    // The line: the game so far (when there is one), then the mistake, then
    // the puzzle position itself. `history` already stops one move short of
    // the mistake, so the two join without overlapping.
    this.nodes = (puzzle.history || []).map((step) => ({
      fen: step.fen, san: step.san, uci: step.uci, blast: step.blast,
      kind: "history",
    }));
    if (puzzle.setup_state && puzzle.setup_san) {
      if (!this.nodes.length && puzzle.before_state) {
        this.nodes.push({
          fen: puzzle.before_state.fen, san: null, uci: null,
          state: puzzle.before_state, kind: "history",
        });
      }
      this.nodes.push({
        fen: puzzle.setup_state.fen, san: puzzle.setup_san, uci: puzzle.setup_uci,
        state: puzzle.state, kind: "setup",
      });
    } else {
      this.nodes.push({
        fen: puzzle.fen, san: null, uci: null, state: puzzle.state, kind: "setup",
      });
    }
    this.startIndex = this.nodes.length - 1;

    if (puzzle.before_state && puzzle.setup_state) {
      // Play the mistake in front of you, so the puzzle arrives with its story
      // attached rather than as a bare position.
      this.cursor = this.startIndex - 1;
      this.board.setFrozen(true);
      this.render();
      this.verdict.className = "verdict";
      this.verdict.textContent = "…";
      this.instruction.textContent =
        `${puzzle.solver === "white" ? "Black" : "White"} plays ${puzzle.setup_san}`;
      await new Promise((resolve) => setTimeout(resolve, 620));
    }
    this.cursor = this.startIndex;
    this.board.setFrozen(false);
    this.verdict.className = "verdict";
    this.verdict.textContent = puzzle.goal?.kind === "draw"
      ? "Only one move holds." : "Find the punishment.";
    this.render();
    this.setInstruction();
  }

  /** The banner that says what winning this puzzle even means. */
  renderGoal() {
    const words = goalWords(this.puzzle);
    this.goalBox.textContent = "";
    this.goalBox.className =
      "goal " + (this.puzzle.goal?.kind === "draw" ? "hold" : "win");
    this.goalBox.append(
      el("div", { class: "goal-head" },
        el("b", {}, words.short),
        el("span", { class: "tag " + (words.exact ? "good" : "") },
          words.exact ? "exact" : "engine")),
      el("div", { class: "sub" }, words.detail));
  }

  /* ---- navigation and rendering ---- */

  goto(index) {
    const target = Math.max(0, Math.min(index, this.nodes.length - 1));
    if (target === this.cursor && this.board.state) return;
    this.cursor = target;
    this.render();
  }

  render() {
    const node = this.current;
    if (!node) return;
    const mode = this.playable;
    // A node fetched from the server carries its legal moves; a replayed one
    // does not, and is drawn read-only until it does. The page never decides
    // legality either way.
    const state = node.state || {
      ...staticState(node.fen), last: node.uci || null, blast: node.blast || [],
    };
    this.board.setState(state);
    this.board.setFrozen(!mode);
    this.showTurn(state);
    this.drawMoves();
    this.updateHint();
    if (mode === "explore" && !node.state) {
      this.hydrate(node).catch(() => { /* stays read-only; no harm */ });
    }
    if (this.analysisOn) this.scheduleAnalysis(); else this.clearAnalysis();
  }

  /** Whose move it is at the cursor.
   *
   * Under the board rather than only in the instruction line, because the
   * instruction speaks about the *puzzle* ("Black to move — two moves to
   * find") and stops being true the moment the line is rewound or explored
   * past. This says what is true of the position on screen, always.
   */
  showTurn(state) {
    const black = state.turn === "black";
    this.turnChip.classList.toggle("black", black);
    $("span", this.turnChip).textContent = state.over
      ? "Game over"
      : `${black ? "Black" : "White"} to move`;
  }

  /** Ask the referee for the full state of a replayed position. */
  async hydrate(node) {
    const fetched = await api("/api/position", { fen: node.fen });
    node.state = { ...fetched, last: node.uci || null, blast: node.blast || [] };
    if (this.current === node) this.render();
  }

  drawMoves() {
    this.moveList.textContent = "";
    this.nodes.forEach((node, index) => {
      if (!node.san) return;
      const ply = this.plyOf(index);
      if (ply % 2 === 0) {
        this.moveList.append(el("span", { class: "n" }, `${ply / 2 + 1}.`));
      } else if (!this.moveList.children.length) {
        this.moveList.append(el("span", { class: "n" }, `${(ply - 1) / 2 + 1}…`));
      }
      this.moveList.append(el("span", {
        class: "mv " + node.kind + (index === this.cursor ? " current" : ""),
        onClick: () => this.goto(index),
        title: node.kind === "explore" ? "your own analysis" : "",
      }, node.san));
    });
    this.moveList.querySelector(".mv.current")?.scrollIntoView({ block: "nearest" });
    if (!this.moveList.children.length) {
      this.moveList.append(el("span", { class: "muted tiny" },
        "No moves before this position — it came from the tablebase."));
    }
  }

  /** Ply number of the move that *reaches* a node, numbered as in the game.
   *
   * The anchor is the puzzle position: it is reached by the opponent's error,
   * which was played at ply `puzzle.ply - 1`. Everything else counts outwards
   * from there, so a puzzle from move 6 of a game says "6" rather than "1".
   */
  plyOf(index) {
    return (this.puzzle.ply ?? 1) - 1 - this.startIndex + index;
  }

  setInstruction() {
    const left = this.puzzle.length - Math.floor(this.moveIndex / 2);
    this.instruction.textContent =
      `${titleCase(this.puzzle.solver)} to move — `
      + (left === 1 ? "one move" : `${left} moves to find`)
      + (this.puzzle.tb_category
          ? ` · tablebase says ${this.puzzle.tb_category}`
          : "");
  }

  /** Say why the board is inert, when it is. */
  updateHint() {
    if (this.finished || this.atLive) {
      this.moveList.classList.remove("rewound");
      $("#rewound")?.remove();
      return;
    }
    if (!$("#rewound")) {
      this.controls.before(el("div", { class: "sub rewound", id: "rewound" },
        "Looking back at the game. ",
        el("a", { href: "#", onClick: (event) => {
          event.preventDefault(); this.goto(this.nodes.length - 1);
        } }, "Back to the puzzle"), " to move."));
    }
  }

  renderMeta() {
    const p = this.puzzle;
    const provenance = SOURCE_LABEL[p.source] || p.source;
    this.meta.textContent = "";
    this.meta.append(
      el("div", { class: "chips", style: "margin-bottom:8px" },
        el("span", { class: "chip static" }, p.phase),
        el("span", { class: "chip static" }, provenance),
        el("span", { class: "chip static" },
          p.goal?.kind === "draw" ? "for a draw" : "for a win"),
        p.rating_label
          ? el("span", {
              class: "chip static",
              title: p.difficulty ? p.difficulty.describe : "",
            }, `${p.rating_label} Elo`)
          : null,
        p.mine ? el("span", { class: "chip static" }, "your side") : null,
        p.solved_before ? el("span", { class: "chip static" }, "solved before") : null),
      el("div", { class: "chips" }, themeTags(p.themes, (theme) =>
        go("puzzles", { ...asQuery(this.filters), theme }))),
      p.source === "game" && p.game_id
        ? el("p", { class: "sub", style: "margin-top:10px" },
            "From ",
            el("a", { href: `#/review?id=${p.game_id}&ply=${p.ply}` }, p.game_id),
            ` at move ${Math.floor((p.ply || 0) / 2) + 1} · `,
            el("a", { href: `https://lichess.org/${p.game_id}`, target: "_blank" }, "lichess"))
        : p.game_id
        ? el("p", { class: "sub", style: "margin-top:10px" },
            "Scraped from a rated game between two other players · ",
            el("a", { href: `https://lichess.org/${p.game_id}`, target: "_blank" }, p.game_id))
        : el("p", { class: "sub", style: "margin-top:10px" },
            "Generated from the Lichess atomic tablebase — the answer is exact, "
            + "not an engine opinion."),
    );
    // `meta.append` is the DOM's, not `el`'s: a null child would be appended
    // as the text "null".
    const profile = this.finished ? difficultyProfile(p.difficulty) : null;
    if (profile) this.meta.append(profile);
  }

  renderControls() {
    this.controls.textContent = "";
    // `Node.append` stringifies null into the text "null"; `el` filters it and
    // `append` does not, so the conditional button is filtered here.
    this.controls.append(...[
      el("button", { onClick: () => this.hint(), disabled: this.finished }, "Hint"),
      el("button", { onClick: () => this.reveal(), disabled: this.finished }, "Solution"),
      el("button", { class: "primary", onClick: () => this.next() }, "Next"),
      this.finished
        ? el("button", {
            class: this.analysisOn ? "" : "ghost",
            onClick: () => this.toggleAnalysis(),
          }, this.analysisOn ? "Analysis on" : "Analysis")
        : null,
      el("button", { class: "ghost", onClick: () => this.replay() }, "Play it out"),
    ].filter(Boolean));
  }

  /* ---- solving ---- */

  onBoardMove(uci) {
    const mode = this.playable;
    if (mode === "solve") return this.tryMove(uci);
    if (mode === "explore") return this.exploreMove(uci);
  }

  async tryMove(uci) {
    if (this.finished || this.busy) return;
    this.busy = true;
    this.board.setFrozen(true);
    // A wrong move leaves the board alone and paints the square it landed on;
    // re-rendering underneath that would wipe the mark before it was seen.
    let rerender = true;
    try {
      const result = await api("/api/puzzle/move", {
        id: this.puzzle.id, index: this.moveIndex, move: uci,
      });
      if (!result.correct) {
        this.wrong += 1;
        rerender = false;
        const played = result.played;
        this.verdict.className = "verdict wrong";
        this.verdict.textContent = `${played.san} is not it.`;
        this.instruction.textContent = played.loss !== null
          ? `That gives up ${played.loss.toFixed(0)} win% against the move that works.`
          : "Try again.";
        this.board.mark(uci.slice(2, 4), "wrong");
        setTimeout(() => { this.board.clearMarks(); }, 900);
        return;
      }
      this.push({
        fen: result.state.fen, san: result.san, uci, state: result.state,
        kind: "solution",
      });
      this.render();
      this.verdict.className = "verdict right";
      this.verdict.textContent = result.san + (result.done ? " — solved." : " — keep going.");
      if (result.reply) {
        await new Promise((resolve) => setTimeout(resolve, 420));
        this.push({
          fen: result.reply.state.fen, san: result.reply.san, uci: result.reply.uci,
          state: result.reply.state, kind: "solution",
        });
        this.render();
        this.instruction.textContent = `They reply ${result.reply.san}.`;
      }
      this.moveIndex += result.reply ? 2 : 1;
      this.hintLevel = 0;    // a new move to find, asked from scratch
      if (result.done) this.finish(true, result.solution);
      else this.setInstruction();
    } catch (error) {
      toast(error.message);
    } finally {
      this.busy = false;
      if (rerender) this.render();
      else this.board.setFrozen(false);
    }
  }

  /** A move played after the puzzle is over: the line continues from here. */
  async exploreMove(uci) {
    if (this.busy) return;
    this.busy = true;
    this.board.setFrozen(true);
    try {
      const next = await api("/api/move", { fen: this.current.fen, move: uci });
      this.truncate();
      this.push({
        fen: next.fen, san: next.san, uci, state: next, kind: "explore",
      });
      this.render();
    } catch (error) {
      toast(error.message);
    } finally {
      this.busy = false;
      this.board.setFrozen(false);
      this.render();
    }
  }

  /** Which piece, then which move — and the level is *per move to find*.
   *
   * It used to be per puzzle: `hints` counts every hint taken, for scoring,
   * and the level was read off that count. On a three-move puzzle that meant
   * one hint on move one, and then the first press on move two — a fresh
   * question the solver had had no help with — handed over the whole move.
   * `hintLevel` resets whenever the line advances; `hints` still counts, so
   * the attempt is scored exactly as before.
   */
  async hint() {
    if (this.finished || this.hinting) return;
    if (!this.atLive) this.goto(this.nodes.length - 1);
    this.hinting = true;
    try {
      this.hints += 1;
      this.hintLevel = Math.min((this.hintLevel || 0) + 1, 2);
      const hint = await api("/api/puzzle/hint", {
        id: this.puzzle.id, index: this.moveIndex, level: this.hintLevel,
      });
      this.board.clearMarks();
      this.board.mark(hint.from, "hint");
      if (hint.to) {
        this.board.arrow(hint.from, hint.to, "hint");
        this.instruction.textContent = "That is the move. Play it.";
      } else {
        this.instruction.textContent = "This piece moves.";
      }
    } catch (error) {
      toast(error.message);
    } finally {
      this.hinting = false;
    }
  }

  async reveal() {
    if (this.finished) return;
    const solution = await api("/api/puzzle/solution", { id: this.puzzle.id });
    // Rewind to the puzzle position and replace whatever was played with the
    // real line, one move at a time so the point is visible rather than stated.
    this.cursor = this.startIndex;
    this.truncate();
    this.finish(false, solution);
    const states = solution.states;
    let step = 0;
    const play = () => {
      if (step >= states.length) return;
      this.push({
        fen: states[step].fen, san: solution.san[step], uci: solution.solution[step],
        state: states[step], kind: "solution",
      });
      this.render();
      step += 1;
      setTimeout(play, 700);
    };
    this.render();
    setTimeout(play, 350);
  }

  finish(solved, solution) {
    if (this.finished) return;
    this.finished = true;
    const clean = solved && this.wrong === 0 && this.hints === 0;
    this.results.push(clean ? "hit" : "miss");
    this.renderStreak();
    this.verdict.className = "verdict " + (clean ? "right" : "done");
    this.verdict.textContent = clean
      ? "Solved."
      : solved ? "Solved, with help." : "Shown.";
    this.instruction.textContent =
      "Move the pieces to try anything you like — the engine follows along.";
    if (solution) this.lineBox.textContent = "Line: " + solution.san.join(" ");
    this.analysisOn = true;
    this.analysisCard.hidden = false;
    this.metaCard.hidden = false;   // where it came from, now that it can't narrow the search
    this.renderControls();
    this.renderMeta();     // the difficulty profile, now that it gives nothing away
    this.render();
    api("/api/puzzle/attempt", {
      id: this.puzzle.id,
      solved: clean,
      hints: this.hints,
      wrong: this.wrong,
      ms: Date.now() - this.startedAt,
    }).catch(() => { /* a lost attempt log is not worth interrupting for */ });
  }

  renderStreak() {
    this.progress.textContent = "";
    for (const result of this.results.slice(-40)) {
      this.progress.append(el("i", { class: result }));
    }
    const hits = this.results.filter((r) => r === "hit").length;
    this.progress.append(el("div", { class: "sub streak-count" },
      `${hits} of ${this.results.length} unaided`));
  }

  next() {
    if (!this.finished && this.puzzle) {
      this.results.push("miss");
      api("/api/puzzle/attempt", {
        id: this.puzzle.id, solved: false, hints: this.hints,
        wrong: this.wrong, ms: Date.now() - this.startedAt,
      }).catch(() => {});
      this.renderStreak();
    }
    this.index += 1;
    this.load().catch((error) => toast(error.message));
  }

  replay() {
    go("play", { fen: this.current?.fen || this.puzzle?.fen, side: this.puzzle.solver });
  }

  /* ---- live analysis ---- */

  toggleAnalysis() {
    if (!this.finished) {
      toast("Analysis is off until the puzzle is finished.");
      return;
    }
    this.analysisOn = !this.analysisOn;
    this.analysisCard.hidden = !this.analysisOn;
    this.renderControls();
    this.render();
  }

  clearAnalysis() {
    clearTimeout(this.analysisTimer);
    this.analysisToken += 1;
    this.evalBar.classList.remove("on");
  }

  /** Debounced: arrowing through a line should not queue an engine run a ply. */
  scheduleAnalysis() {
    clearTimeout(this.analysisTimer);
    const token = ++this.analysisToken;
    const fen = this.current.fen;
    this.analysisBox.classList.add("thinking");
    this.analysisTimer = setTimeout(() => this.deepen(token, fen), 80);
  }

  /** Analyse one position at rising budgets, showing each answer as it lands.
   *
   * One 300k-node search took a third of a second to say anything at all,
   * which reads as a hang; the same third of a second spent as 40k, then
   * 150k, then the rest puts a first opinion on screen in about 60ms and
   * refines it in place. The search cannot be interrupted mid-rung, so the
   * ladder is where abandonment happens: the token is checked between rungs
   * and a cursor that has moved on costs at most the rung in flight.
   *
   * Rungs are cache keys as well as budgets, so a position revisited comes
   * back the whole way instantly.
   */
  async deepen(token, fen) {
    for (const nodes of ANALYSIS_LADDER) {
      let result;
      try {
        result = await api("/api/evaluate", { fen, nodes, multipv: 3 });
      } catch (error) {
        if (token !== this.analysisToken) return;
        this.analysisBox.classList.remove("thinking");
        this.analysisBox.textContent = error.message;
        return;
      }
      if (token !== this.analysisToken) return;   // the cursor moved on
      this.showAnalysis(result, nodes === ANALYSIS_LADDER[ANALYSIS_LADDER.length - 1]);
      if (result.over) return;
    }
  }

  showAnalysis(result, settled) {
    this.analysisBox.classList.remove("thinking");
    this.analysisBox.textContent = "";
    $("#an-depth").textContent = result.over
      ? ""
      : `${num(result.nodes)} nodes${settled ? "" : " · deepening…"}`;
    if (result.over) {
      this.analysisBox.append(el("div", { class: "sub" },
        `The game ends here — ${result.result}.`));
      this.setEvalBar(
        result.result === "1-0" ? 100 : result.result === "0-1" ? 0 : 50,
        result.result === "1/2-1/2" ? "=" : result.result === "1-0" ? "1−0" : "0−1");
      return;
    }
    const best = result.lines[0];
    this.setEvalBar(
      result.turn === "white" ? best.win : 100 - best.win,
      whiteScore(best, result.turn));
    result.lines.forEach((line, rank) => {
      const white = result.turn === "white" ? line.win : 100 - line.win;
      this.analysisBox.append(el("div", { class: "analysis-line" },
        el("span", { class: "score" + (rank ? " alt" : "") }, whiteScore(line, result.turn)),
        el("span", { class: "win" }, `${white.toFixed(0)}%`),
        el("span", { class: "pv" }, (line.pv.length ? line.pv : [line.san]).map((san, at) =>
          el("span", {
            class: "pv-move",
            onClick: () => this.walkInto(
              (line.pv_uci.length ? line.pv_uci : [line.uci]).slice(0, at + 1)),
          }, san)))));
    });
    // Arrows, best first and boldest: the same convention the review board uses.
    this.board.clearMarks();
    result.lines.slice(0, 3).forEach((line, rank) => {
      this.board.arrow(line.uci.slice(0, 2), line.uci.slice(2, 4),
        rank === 0 ? "best" : "alt");
    });
  }

  /** Click a move in a variation to walk the board into it. */
  async walkInto(ucis) {
    if (!this.finished || this.busy) return;
    this.busy = true;
    try {
      this.truncate();
      for (const uci of ucis) {
        // Every move still goes through the referee, even one the engine just
        // suggested: the page is not allowed a second opinion about legality.
        const next = await api("/api/move", { fen: this.current.fen, move: uci });
        this.push({
          fen: next.fen, san: next.san, uci, state: next, kind: "explore",
        });
      }
    } catch (error) {
      toast(error.message);
    } finally {
      this.busy = false;
      this.render();
    }
  }

  /** Flip the board, and the eval bar with it. */
  flip() {
    this.board.setFlipped(!this.board.flipped);
    this.drawEvalBar();
  }

  setEvalBar(whiteWin, text) {
    this.evalWhite = whiteWin;
    this.evalText = text;
    this.drawEvalBar();
  }

  /* The bar is White's share of the position, drawn in White's colour, at
   * White's end of the board.
   *
   * The old version kept the pale fill pinned to the bottom and inverted the
   * *number* when the board was flipped, which meant that with Black at the
   * bottom — the normal case here, since the board is flipped whenever the
   * solver is Black — a position Black was winning drew a bar that was almost
   * entirely white. Every analysis board in existence reads the other way
   * round. Now the fill keeps its meaning and moves to whichever end White is
   * playing from, so "more black" always means Black is better.
   */
  drawEvalBar() {
    if (this.evalWhite === undefined || this.evalWhite === null) return;
    const white = Math.max(0, Math.min(100, this.evalWhite));
    const flipped = this.board.flipped;
    this.evalBar.classList.add("on");
    this.evalBar.classList.toggle("flipped", flipped);
    this.evalBarWhite.style.height = `${white}%`;
    this.evalBar.title = `White ${white.toFixed(0)}%`;
    // The number sits at the winning side's end, which is the convention and
    // is also where there is room. Which end that is depends on the flip, and
    // its colour depends on whether it lands on the pale fill or the dark.
    const whiteAhead = white >= 50;
    this.evalScore.textContent = this.evalText;
    this.evalScore.classList.toggle("low", whiteAhead !== flipped);
    this.evalScore.classList.toggle("on-fill", whiteAhead);
  }
}

//: What the explore board spends, in order. Each rung is shown as it lands and
//: each is its own cache key, so a revisited position comes back at once.
const ANALYSIS_LADDER = [40_000, 150_000, 600_000, 2_000_000];

/** A line's score from White's point of view, the way a board displays it.
 *
 * The engine reports relative to the side to move, and win% alone — which is
 * what this panel used to show — makes two positions with the same winning
 * chances look identical when one is a piece up and the other is mating. Both
 * are shown now: pawns for the shape of the position, win% for what it is
 * worth in atomic, where the two diverge much faster than in chess.
 */
function whiteScore(line, turn) {
  const sign = turn === "white" ? 1 : -1;
  if (line.mate !== null && line.mate !== undefined) {
    const mate = sign * line.mate;
    return (mate > 0 ? "#" : "#−") + Math.abs(mate);
  }
  if (line.cp === null || line.cp === undefined) return "—";
  const pawns = (sign * line.cp) / 100;
  return (pawns > 0 ? "+" : pawns < 0 ? "−" : "") + Math.abs(pawns).toFixed(1);
}

/* ---------- generating more puzzles, from the browser ---------- */

/* The three generators, the arguments worth exposing, and what each one costs.
 * The costs are not decoration: "scrape rated games" is minutes and "tablebase"
 * is rate-limited by somebody else's server, and a button that does not say so
 * is a button that gets pressed by accident. */
const JOB_FORMS = [
  {
    kind: "generate",
    blurb: "Query the analysis cache over your own games. Stage 1 is free — "
      + "every ply of every imported game already has a MultiPV pass on it — "
      + "so this only spends engine time confirming what it finds.",
    fields: [
      ["limit", "Candidates at most", 40, "number"],
      ["nodes", "Confirming nodes", 4000000, "number"],
    ],
  },
  {
    kind: "mine",
    blurb: "Scan rated atomic games from the public Lichess monthly dumps, "
      + "both players 1800+, at a cheap screening budget; survivors go through "
      + "the same confirming pass. Roughly three seconds a game.",
    fields: [
      ["games", "Games to scan", 200, "number"],
      ["min_elo", "Minimum rating", 1800, "number"],
      ["scan_nodes", "Screening nodes", 20000, "number"],
    ],
  },
  {
    kind: "endgames",
    blurb: "Walk the Lichess atomic tablebase and keep the positions where "
      + "exactly one move holds the result. Exact — no engine opinion — but "
      + "rate limited, so it is slow and mostly free after the first run.",
    fields: [
      ["samples", "Positions to start from", 200, "number"],
      ["limit", "Stop after this many puzzles", 0, "number"],
    ],
  },
  {
    kind: "rate",
    blurb: "Give every unrated puzzle to each calibrated level of the M1 "
      + "ladder; its difficulty is the rating of the weakest one that finds the "
      + "whole line. No Fairy-Stockfish and no network — a second a puzzle.",
    fields: [
      ["limit", "Stop after this many puzzles", 0, "number"],
    ],
  },
  {
    kind: "traps",
    blurb: "Walk the opening book and judge every position the field actually "
      + "reaches: a trap is one where at most two moves hold and the rest lose "
      + "outright. Lowering the game floor widens the walk — 2,000 is about 900 "
      + "positions and 90 seconds.",
    fields: [
      ["min_games", "Book games per line", 2000, "number"],
      ["depth", "Depth, in ply", 10, "number"],
      ["scan_nodes", "Screening nodes", 400000, "number"],
    ],
  },
];

/** The generation panel: start a run, watch it, cancel it. */
class JobPanel {
  /** `kinds` narrows the forms to the ones this page is about, because a trap
   *  scan is not a puzzle generator and the puzzle page should not offer it. */
  constructor(onFinished, kinds) {
    this.kinds = kinds || null;
    this.onFinished = onFinished || (() => {});
    this.timer = null;
    this.inputs = new Map();
    this.status = el("div", {});
    this.forms = el("div", { class: "jobforms" });
    this.node = el("div", { class: "card" },
      el("h2", {}, "Generate more puzzles"),
      this.forms, this.status);
    this.drawForms();
    this.poll();
  }

  destroy() { clearTimeout(this.timer); this.timer = null; }

  drawForms() {
    this.forms.textContent = "";
    for (const form of JOB_FORMS) {
      if (this.kinds && !this.kinds.includes(form.kind)) continue;
      const fields = el("div", { class: "row" });
      for (const [key, label, value, type] of form.fields) {
        const input = el("input", { type: "text", value: String(value),
          inputmode: type === "number" ? "numeric" : "text",
          style: "max-width:120px" });
        this.inputs.set(`${form.kind}.${key}`, input);
        fields.append(el("div", { class: "field-group" },
          el("label", { class: "field" }, label), input));
      }
      this.forms.append(el("details", { class: "jobform" },
        el("summary", {}, SUMMARY?.jobs?.[form.kind] || form.kind),
        el("p", { class: "sub", style: "margin:8px 0" }, form.blurb),
        fields,
        el("div", { class: "row", style: "margin-top:10px" },
          el("button", { class: "primary", onClick: () => this.start(form) },
            "Run it"))));
    }
  }

  params(form) {
    const out = {};
    for (const [key] of form.fields) {
      const raw = this.inputs.get(`${form.kind}.${key}`)?.value.trim();
      if (raw) out[key] = Number(raw);
    }
    return out;
  }

  async start(form) {
    try {
      await api("/api/jobs/start", { kind: form.kind, params: this.params(form) });
      this.poll();
    } catch (error) { toast(error.message); }
  }

  async cancel(id) {
    try { await api("/api/jobs/cancel", { id }); this.poll(); }
    catch (error) { toast(error.message); }
  }

  async poll() {
    clearTimeout(this.timer);
    let data;
    try { data = await api("/api/jobs"); }
    catch { return; }
    if (!this.node.isConnected) return;   // the view moved on
    this.draw(data);
    // Only poll while something is running. A finished list does not change on
    // its own, and a timer that never stops is a timer that leaks a view.
    if (data.running) this.timer = setTimeout(() => this.poll(), 1200);
    else if (this.wasRunning) this.onFinished(data.jobs[0]?.result?.summary || "");
    this.wasRunning = data.running;
  }

  draw(data) {
    this.status.textContent = "";
    if (!data.jobs.length) {
      this.status.append(el("p", { class: "sub", style: "margin-top:10px" },
        `${num(data.corpus.total)} puzzles so far — `
        + `${data.corpus.from_games} from your games, `
        + `${data.corpus.from_dump || 0} scraped, `
        + `${data.corpus.from_tablebase} exact. `
        + `${num(data.scrape.games)} rated games scanned.`));
      return;
    }
    for (const job of data.jobs) {
      const bar = job.total
        ? el("div", { class: "bar", style: "margin-top:6px" },
            el("span", { style: `width:${Math.min(100, 100 * job.done / job.total)}%` }))
        : null;
      this.status.append(el("div", { class: "job" },
        el("div", { class: "row", style: "justify-content:space-between" },
          el("div", {},
            el("strong", {}, job.label), " ",
            el("span", { class: "tag " + (job.error ? "bad" : job.running ? "warn" : "good") },
              job.running ? job.stage : job.error ? "failed" : job.stage)),
          job.running
            ? el("button", { class: "ghost", onClick: () => this.cancel(job.id) }, "Stop")
            : null),
        bar,
        el("div", { class: "sub tiny" },
          job.running
            ? `${job.done}/${job.total || "?"} · ${Math.round(job.elapsed)}s${job.message ? " · " + job.message : ""}`
            : job.error || job.result.summary || "done"),
        job.result.stored
          ? el("div", { class: "sub tiny" }, `stored ${job.result.stored} puzzles`)
          : null));
    }
  }
}

/* ---------- improvement ---------- */

/** A skill row: the rate, its interval, and how many puzzles it rests on. */
function skillRow(skill, overall, onClick) {
  const rate = skill.rate === null ? 0 : skill.rate;
  return el("tr", { class: onClick ? "clickable" : "", onClick },
    el("td", {}, skill.key),
    el("td", { class: "num mono" }, `${skill.solved}/${skill.seen}`),
    el("td", { class: "num mono" }, pct(rate, 0)),
    el("td", {},
      // The bar is the interval, not the point estimate: a wide bar is the
      // honest picture of three attempts, and a verdict drawn from the middle
      // of it would not be.
      el("div", { class: "range", title: `${skill.lower.toFixed(0)}–${skill.upper.toFixed(0)}%` },
        el("span", { class: "span", style:
          `left:${skill.lower}%;width:${Math.max(1, skill.upper - skill.lower)}%` }),
        el("i", { class: "point", style: `left:${rate}%` }),
        el("i", { class: "overall", style: `left:${overall}%` }))),
    el("td", {}, el("span", {
      class: "tag " + ({ strong: "good", weak: "bad" }[skill.verdict] || ""),
    }, skill.verdict)));
}

function skillTable(title, skills, overall, note, onClick) {
  if (!skills.length) return null;
  return el("div", { class: "card" },
    el("h2", {}, title),
    el("div", { class: "scroll-x" }, el("table", {},
      el("thead", {}, el("tr", {},
        el("th", {}, "Slice"), el("th", { class: "num" }, "Solved"),
        el("th", { class: "num" }, "Rate"), el("th", {}, "95% interval"),
        el("th", {}, ""))),
      el("tbody", {}, skills.map((skill) =>
        skillRow(skill, overall, onClick ? () => onClick(skill) : null))))),
    note ? el("p", { class: "sub", style: "margin-top:10px" }, note) : null);
}

async function viewImprove() {
  if (!SUMMARY) SUMMARY = await api("/api/summary");
  const data = await api("/api/improvement");
  const o = data.overall;
  const wrap = el("div", {});

  wrap.append(
    el("div", { class: "view-title" }, el("h1", {}, "Improvement"),
      el("div", { class: "sub" }, "what you can see, and what you cannot")),
    el("p", { class: "lede" },
      "Every rate here counts your first attempt at each puzzle, because "
      + "the queue brings failures back and a fourth go at a position you have "
      + "memorised is not evidence about anything. Nothing is called a strength "
      + "or a weakness unless its interval clears your overall rate — with "
      + `fewer than ${o.min_sample} attempts a slice gets an interval and no verdict.`),
  );

  if (!o.seen) {
    wrap.append(empty("No puzzles attempted yet.",
      "Solve a few and this fills in — it is built entirely from your own attempts."));
    wrap.append(el("div", { style: "margin-top:14px" },
      el("button", { class: "primary", onClick: () => go("puzzles", { unsolved: "1" }) },
        "Start solving")));
    return wrap;
  }

  wrap.append(el("div", { class: "grid" },
    statCard("Solved unaided", pct(o.rate, 0),
      `${o.solved} of ${o.seen} puzzles, first sight`),
    statCard("95% interval", `${o.lower.toFixed(0)}–${o.upper.toFixed(0)}%`,
      "on the rate above"),
    statCard("Attempts", num(o.attempts), `${o.seen} distinct puzzles`),
    statCard("Corpus", num(o.corpus), `${o.corpus - o.seen} never tried`),
  ));

  const strong = [];
  const weak = [];
  for (const skill of data.themes) {
    if (skill.verdict === "strong") strong.push(skill);
    if (skill.verdict === "weak") weak.push(skill);
  }

  const headline = el("div", { class: "card" },
    el("h2", {}, "Strong and weak"),
    strong.length || weak.length
      ? el("div", {},
          weak.length
            ? el("p", { class: "sub" }, "Weakest — worth drilling: ",
                ...weak.map((skill) => el("span", {
                  class: "chip", onClick: () => go("puzzles", { theme: skill.key, order: "random" }),
                }, `${skill.key} ${pct(skill.rate, 0)}`)))
            : null,
          strong.length
            ? el("p", { class: "sub", style: "margin-top:8px" }, "Strongest: ",
                ...strong.map((skill) => el("span", { class: "chip static" },
                  `${skill.key} ${pct(skill.rate, 0)}`)))
            : null)
      : el("p", { class: "sub" },
          "No theme's interval separates from your overall rate yet. That is a "
          + "statement about how many puzzles you have done, not about how even "
          + "your play is — keep going and the intervals narrow."));

  const themeNote =
    "Themes overlap: one puzzle can be a `king-blast` and a `blast-fork` at "
    + "once, so these rows are not a partition and their counts do not sum to "
    + "the total.";

  const redo = data.redo.length
    ? el("div", { class: "card" },
        el("h2", {}, "Redo these", el("span", { class: "right tiny muted" },
          `${data.redo.length} failed`)),
        el("p", { class: "sub", style: "margin-bottom:10px" },
          "Puzzles you have tried and never solved unaided."),
        el("div", { class: "row" },
          el("button", { class: "primary",
            onClick: () => go("puzzles", { failed: "1", order: "random" }) },
            "Drill the ones I failed")),
        el("div", { class: "scroll-y", style: "max-height:260px;margin-top:10px" },
          el("table", {}, el("tbody", {}, data.redo.slice(0, 20).map((puzzle) =>
            el("tr", { class: "clickable",
              onClick: () => go("puzzles", { failed: "1", order: "random" }) },
              el("td", { class: "mono tiny" }, puzzle.id.slice(0, 8)),
              el("td", {}, puzzle.phase),
              el("td", { class: "muted tiny" }, (puzzle.themes || []).join(", "))))))))
    : null;

  wrap.append(el("div", { class: "split", style: "margin-top:14px" },
    el("div", { class: "wide" },
      headline,
      skillTable("By theme", data.themes, o.rate, themeNote,
        (skill) => go("puzzles", { theme: skill.key, order: "random" })),
      skillTable("By phase", data.phases, o.rate,
        "Atomic's phases arrive early — the median game is 30 ply — so "
        + "\"opening\" here means the first few moves, where more than half of "
        + "all blunders happen."),
      skillTable("By goal", data.motives, o.rate,
        "`save` is finding the only move that holds; `win` is punishing an "
        + "error. Half this corpus is defensive, which is not how a "
        + "standard-chess puzzle set looks and is what the variant produces."),
      skillTable("By source", data.sources, o.rate,
        "`tablebase` answers are exact, `game` and `dump` are Fairy-Stockfish "
        + "at 4M nodes. A gap between them is worth noticing in either "
        + "direction."),
      skillTable("By measured difficulty", data.difficulty, o.rate,
        "Each row is the rating of the weakest calibrated engine level that "
        + "finds the whole line. If engine-hard and human-hard were the same "
        + "axis your rate would fall down this table; where it does not is "
        + "worth more than the ordering itself. `≤` and `>` are the ends of "
        + "the ladder, where a rating is a bound rather than a reading.")),
    el("div", { class: "side" }, redo,
      el("div", { class: "card tight" },
        el("h2", {}, "Recent form"),
        streakStrip(data.timeline)),
    )));
  return wrap;
}

/** The last 60 attempts as hit/miss ticks, oldest left. */
function streakStrip(timeline) {
  const recent = timeline.slice(-60);
  const strip = el("div", { class: "streak" },
    recent.map((entry) => el("i", { class: entry.solved ? "hit" : "miss" })));
  const hits = recent.filter((entry) => entry.solved).length;
  return el("div", {}, strip,
    el("p", { class: "sub", style: "margin-top:8px" },
      recent.length
        ? `${hits} of the last ${recent.length} attempts solved unaided`
        : "nothing yet"));
}

/* ---------- games ---------- */

async function viewGames() {
  const data = await api("/api/games?limit=300");
  const wrap = el("div", {});
  wrap.append(
    el("div", { class: "view-title" }, el("h1", {}, "Games"),
      el("div", { class: "sub" }, `${data.games.length} imported`)),
    el("p", { class: "lede" },
      "Every analysed game has had a deep pass over every ply, so opening a "
      + "review costs nothing and puzzle mining over it is already paid for."));

  if (!data.games.length) {
    wrap.append(empty("No games imported.",
      "Run `py -3.13 -m trainer.cli import <lichess-username>` then `analyse`."));
    return wrap;
  }

  const search = el("input", {
    type: "text", placeholder: "filter by opponent or id…",
    style: "max-width:280px", oninput: () => draw(),
  });
  const onlyBlunders = el("span", { class: "chip", onClick: () => {
    onlyBlunders.classList.toggle("on"); draw();
  } }, "with blunders");
  const body = el("tbody", {});
  wrap.append(el("div", { class: "card tight" },
    el("div", { class: "row" }, search, onlyBlunders)));

  function draw() {
    const needle = search.value.trim().toLowerCase();
    const blundersOnly = onlyBlunders.classList.contains("on");
    body.textContent = "";
    for (const game of data.games) {
      const opponent = game.my_color === "white" ? game.black : game.white;
      if (needle && !`${game.id} ${game.white} ${game.black}`.toLowerCase().includes(needle)) continue;
      if (blundersOnly && !game.blunders) continue;
      const outcome = game.my_color === null ? "—"
        : game.winner === game.my_color ? "win"
        : game.winner === null ? "draw" : "loss";
      body.append(el("tr", {
        class: "clickable",
        onClick: () => go("review", { id: game.id }),
      },
        el("td", { class: "mono" }, game.id),
        el("td", {}, dateOf(game.created_at)),
        el("td", {}, opponent || "—"),
        el("td", {}, el("span", {
          class: "tag " + (outcome === "win" ? "good" : outcome === "loss" ? "bad" : ""),
        }, outcome)),
        el("td", {}, game.my_color || "—"),
        el("td", {}, game.speed || "—"),
        el("td", { class: "num" }, game.ply),
        el("td", { class: "num" }, game.analysed ? game.blunders : "—"),
        el("td", { class: "num" }, game.puzzles || "—"),
      ));
    }
    if (!body.children.length) {
      body.append(el("tr", {}, el("td", { colspan: 9, class: "muted" }, "nothing matches")));
    }
  }
  draw();

  wrap.append(el("div", { class: "card scroll-x", style: "margin-top:14px" },
    el("table", {},
      el("thead", {}, el("tr", {},
        el("th", {}, "Id"), el("th", {}, "Date"), el("th", {}, "Opponent"),
        el("th", {}, "Result"), el("th", {}, "Side"), el("th", {}, "Speed"),
        el("th", { class: "num" }, "Ply"),
        el("th", { class: "num" }, "Blunders"),
        el("th", { class: "num" }, "Puzzles"))),
      body)));
  return wrap;
}

/* ---------- review ---------- */

async function viewReview(query) {
  const id = query.get("id");
  if (!id) return empty("No game selected.", "Pick one from the Games tab.");
  const data = await api(`/api/game/${id}`);
  const game = data.game;
  const plies = data.plies;

  if (!plies.length) {
    return empty(`Game ${id} has not been analysed.`,
      "Run `py -3.13 -m trainer.cli analyse`.");
  }

  const boardHost = el("div", { class: "board-lg" });
  const board = new Board(boardHost, { interactive: false });
  board.setFlipped(game.my_color === "black");

  const moveList = el("div", { class: "movelist" });
  const detail = el("div", {});
  const graph = evalGraph(plies, (index) => select(index));
  let current = Math.min(Number(query.get("ply") || 0), plies.length - 1);
  const why = new ExplainPanel(board, { onRestore: () => drawBoard() });
  const whyCard = el("div", { class: "card", hidden: true }, el("h2", {}, "Why?"), why.root);

  const mine = plies.filter((p) => game.my_color === null || p.mover === game.my_color);
  const worst = mine.filter((p) => p.loss >= 10).sort((a, b) => b.loss - a.loss);
  const puzzlesByPly = new Map((data.puzzles || []).map((p) => [p.ply, p]));

  function select(index) {
    current = Math.max(0, Math.min(index, plies.length));
    why.hide();
    whyCard.hidden = true;
    drawMoves();
    drawDetail();
    graph.moveCursor(current);
    drawBoard();
  }

  function explainPly(ply) {
    whyCard.hidden = false;
    why.show({ fen: ply.fen, move: ply.uci });
  }

  function drawBoard() {
    const ply = plies[current];
    if (ply) {
      board.setState(staticState(ply.fen));
      if (ply.best_uci && ply.loss >= 10) {
        board.arrow(ply.best_uci.slice(0, 2), ply.best_uci.slice(2, 4), "best");
      }
    } else {
      board.setState(staticState(data.final_fen));
    }
  }

  function drawMoves() {
    moveList.textContent = "";
    for (let i = 0; i < plies.length; i += 2) {
      moveList.append(el("div", { class: "n" }, `${i / 2 + 1}.`));
      for (const index of [i, i + 1]) {
        const ply = plies[index];
        if (!ply) { moveList.append(el("div", {})); continue; }
        moveList.append(el("div", {
          class: "mv " + ply.judgement + (index === current ? " current" : ""),
          onClick: () => select(index),
        },
          el("span", {}, ply.san + judgementMark(ply.judgement)),
          ply.loss >= 10 ? el("span", { class: "cost" }, `−${ply.loss.toFixed(0)}`) : null));
      }
    }
    moveList.querySelector(".mv.current")?.scrollIntoView({ block: "nearest" });
  }

  function drawDetail() {
    const ply = plies[current];
    detail.textContent = "";
    if (!ply) {
      detail.append(el("p", { class: "sub" }, "End of the game."));
      return;
    }
    const mover = titleCase(ply.mover);
    const puzzle = puzzlesByPly.get(ply.ply + 1);
    detail.append(
      el("div", { class: "row", style: "justify-content:space-between" },
        el("div", { class: "big" }, `${moveNumber(ply.ply)} ${ply.san}${judgementMark(ply.judgement)}`),
        el("span", { class: "tag " + (
          ply.judgement === "blunder" ? "bad"
          : ply.judgement === "mistake" ? "warn"
          : ply.judgement === "inaccuracy" ? "warn" : "good") }, ply.judgement)),
      el("table", { style: "margin-top:10px" }, el("tbody", {},
        el("tr", {}, el("td", {}, `Win% for ${mover}`),
          el("td", { class: "num mono" },
            `${ply.win_before.toFixed(1)} → ${ply.win_after.toFixed(1)}`)),
        el("tr", {}, el("td", {}, "Thrown away"),
          el("td", { class: "num mono" }, ply.loss.toFixed(1))),
        el("tr", {}, el("td", {}, "Engine played"),
          el("td", { class: "num mono" }, ply.best_san || "—")),
        el("tr", {}, el("td", {}, "Best vs second"),
          el("td", { class: "num mono" },
            ply.unique_gap === null ? "—" : ply.unique_gap.toFixed(1))))),
      el("div", { class: "row", style: "margin-top:12px" },
        el("button", { class: "primary", onClick: () => explainPly(ply) },
          ply.loss >= 10 ? `Why is ${ply.san} bad?` : `Why ${ply.san}?`),
        el("button", {
          onClick: () => go("play", { fen: ply.fen, side: ply.mover }) },
          "Play it out from here"),
        puzzle
          ? el("button", { onClick: () => go("puzzles", { game: game.id }) }, "Puzzle from this")
          : null),
      el("p", { class: "sub", style: "margin-top:10px" },
        "Replaying from the position before your mistake is the point of the "
        + "review: the engine takes the other side and you find out whether you "
        + "convert it this time."),
    );
  }

  const onKey = (event) => {
    if (event.target.matches("input, select, textarea")) return;
    if (why.handleKey(event)) return;
    if (event.key === "ArrowRight") select(current + 1);
    else if (event.key === "ArrowLeft") select(current - 1);
    else if (event.key === "Home") select(0);
    else if (event.key === "End") select(plies.length - 1);
    else if (event.key.toLowerCase() === "f") board.setFlipped(!board.flipped);
    else return;
    event.preventDefault();
  };
  document.addEventListener("keydown", onKey);
  teardown = () => document.removeEventListener("keydown", onKey);

  const outcome = game.my_color === null ? ""
    : game.winner === game.my_color ? "You won."
    : game.winner === null ? "Drawn." : "You lost.";

  const wrap = el("div", {});
  wrap.append(
    el("div", { class: "view-title" },
      el("h1", {}, `${game.white} (${num(game.white_rating)}) vs ${game.black} (${num(game.black_rating)})`),
      el("div", { class: "sub" },
        `${game.speed || ""} · ${game.status || ""} · ${outcome} · `,
        el("a", { href: `https://lichess.org/${game.id}`, target: "_blank" }, "lichess"))),
    el("p", { class: "lede" },
      `Analysed at ${num(game.nodes)} nodes a ply. `
      + "Arrows mark the move the engine would have played wherever you lost 10 "
      + "win% or more. ",
      el("span", { class: "kbd" }, "←"), " ", el("span", { class: "kbd" }, "→"),
      " step through the game."),
    el("div", { class: "split" },
      el("div", { class: "wide", style: "flex:0 0 auto" }, boardHost),
      el("div", { class: "side" },
        el("div", { class: "card tight" }, graph),
        el("div", { class: "card" }, el("h2", {}, "This move"), detail),
        whyCard,
        el("div", { class: "card" }, el("h2", {}, "Moves"), moveList),
        worst.length
          ? el("div", { class: "card" },
              el("h2", {}, "Where it was thrown"),
              el("table", {}, el("tbody", {}, worst.slice(0, 6).map((ply) =>
                el("tr", { class: "clickable", onClick: () => select(plies.indexOf(ply)) },
                  el("td", { class: "mono" }, `${moveNumber(ply.ply)} ${ply.san}`),
                  el("td", { class: "num mono" }, `−${ply.loss.toFixed(0)}`),
                  el("td", { class: "muted" }, ply.best_san || ""))))))
          : null)));
  select(current);
  return wrap;
}

/* ---------- play ---------- */

async function viewPlay(query) {
  if (!SUMMARY) SUMMARY = await api("/api/summary");
  const startFen = query.get("fen");
  const side = query.get("side") || "white";
  // Two engines on one list: the lab's M1 at each calibrated node budget, and
  // Fairy-Stockfish + the atomic net unthrottled. Older summaries carry only
  // `levels`, so fall back to building the M1 half from those.
  const opponents = SUMMARY.opponents || SUMMARY.levels.map((level) => ({
    id: `m1-${level.nodes}`, engine: "m1", label: level.label,
    detail: level.elo, nodes: level.nodes, movetime: null,
  }));

  const boardHost = el("div", { class: "board-lg" });
  const status = el("div", { class: "verdict" }, "—");
  const evalBox = el("div", { class: "sub" }, "");
  const history = el("div", { class: "movelist" });
  let state = null;
  let busy = false;
  let moves = [];
  // Every move with the position it was played from, so "why" can be asked of
  // any of them after the fact.
  let played = [];

  const levelSelect = el("select", {}, opponents.map((level) =>
    el("option", { value: level.id }, `${level.label} — ${level.detail}`)));
  levelSelect.value = opponents[Math.min(1, opponents.length - 1)].id;
  const opponentOf = () =>
    opponents.find((level) => level.id === levelSelect.value) || opponents[0];
  const sideSelect = el("select", {}, ["white", "black"].map((colour) =>
    el("option", { value: colour, selected: colour === side }, titleCase(colour))));

  const board = new Board(boardHost, { onMove: (uci) => play(uci) });
  board.setFlipped(side === "black");
  const why = new ExplainPanel(board);
  const whyButtons = el("div", { class: "row", style: "margin-top:10px" });
  const whyCard = el("div", { class: "card" }, el("h2", {}, "Why?"), whyButtons, why.root);

  function drawWhy() {
    whyButtons.textContent = "";
    const theirs = [...played].reverse().find((entry) => !entry.mine);
    const mine = [...played].reverse().find((entry) => entry.mine);
    if (theirs) {
      whyButtons.append(el("button", {
        class: "primary",
        onClick: () => why.show({ fen: theirs.fen, move: theirs.uci,
          title: `The engine played ${theirs.san}.` }),
      }, `Why ${theirs.san}?`));
    }
    if (mine) {
      whyButtons.append(el("button", {
        onClick: () => why.show({ fen: mine.fen, move: mine.uci,
          title: `You played ${mine.san}.` }),
      }, `Was ${mine.san} OK?`));
    }
    if (!played.length) {
      whyButtons.append(el("div", { class: "sub" },
        "After a move, ask why the engine played it, or whether yours was any good."));
    }
  }

  function myTurn() {
    return state && !state.over && !busy && state.turn === sideSelect.value;
  }

  function updateStatus() {
    if (!state) return;
    if (state.over) {
      const you = sideSelect.value === "white" ? "1-0" : "0-1";
      status.className = "verdict done";
      status.textContent = state.result === "1/2-1/2" ? "Drawn."
        : state.result === you ? "You win." : "Engine wins.";
      return;
    }
    status.className = "verdict";
    status.textContent = busy ? "Engine thinking…"
      : myTurn() ? "Your move" + (state.check ? " — check" : "")
      : "Engine to move" + (state.check ? " — check" : "");
  }

  function record(uci, next, before, mine) {
    played.push({ fen: before, uci, san: next.san || uci, mine });
    why.hide();
    drawWhy();
    moves.push({ san: next.san || uci, boom: next.blast.length > 0 });
    history.textContent = "";
    for (let i = 0; i < moves.length; i += 2) {
      history.append(el("div", { class: "n" }, `${i / 2 + 1}.`));
      for (const entry of [moves[i], moves[i + 1]]) {
        history.append(el("div", { class: "mv" },
          entry ? entry.san + (entry.boom ? " ✹" : "") : ""));
      }
    }
    history.scrollTop = history.scrollHeight;
  }

  async function engineMove() {
    busy = true; board.setFrozen(true); updateStatus();
    const level = opponentOf();
    try {
      const next = await api("/api/engine", {
        fen: state.fen,
        engine: level.engine,
        nodes: level.nodes || undefined,
        movetime: level.movetime || undefined,
      });
      if (next.last) record(next.last, next, state.fen, false);
      state = next;
      board.setState(state);
    } catch (error) {
      // A missing Fairy-Stockfish binary lands here. Say so and leave the
      // position alone rather than freezing on "Engine thinking…".
      toast(error.message);
    } finally {
      busy = false; board.setFrozen(false); updateStatus();
    }
  }

  async function play(uci) {
    if (!myTurn()) return;
    busy = true; board.setFrozen(true); updateStatus();
    try {
      const next = await api("/api/move", { fen: state.fen, move: uci });
      record(uci, next, state.fen, true);
      state = next;
      board.setState(state);
      busy = false;
      if (!state.over) await engineMove();
    } catch (error) {
      toast(error.message);
    } finally {
      busy = false; board.setFrozen(false); updateStatus();
    }
  }

  async function reset(fen) {
    moves = [];
    played = [];
    why.hide();
    drawWhy();
    history.textContent = "";
    evalBox.textContent = "";
    state = fen
      ? await api("/api/position", { fen })
      : await api("/api/new", {});
    board.setFlipped(sideSelect.value === "black");
    board.setState(state);
    updateStatus();
    if (!state.over && state.turn !== sideSelect.value) await engineMove();
  }

  async function evaluate() {
    evalBox.textContent = "thinking…";
    try {
      const result = await api("/api/evaluate", { fen: state.fen });
      evalBox.textContent = result.over ? "The game is over."
        : result.lines.map((line) =>
            `${line.san} ${line.win.toFixed(0)}%`).join("   ");
    } catch (error) { evalBox.textContent = error.message; }
  }

  const wrap = el("div", {});
  wrap.append(
    el("div", { class: "view-title" }, el("h1", {}, "Play"),
      el("div", { class: "sub" },
        `${SUMMARY.engine}${SUMMARY.fsf ? " · " + SUMMARY.fsf : ""}`)),
    el("p", { class: "lede" },
      startFen
        ? "Replaying a position from your own game. Convert it this time."
        : "Two opponents. The lab's own engine at each rung of the calibrated "
          + "ladder — node budgets with measured Elo attached — or "
          + "Fairy-Stockfish with the atomic NNUE net at full strength, which is "
          + "the gold standard everything else here is measured against."),
    el("div", { class: "split" },
      el("div", { class: "wide", style: "flex:0 0 auto" }, boardHost),
      el("div", { class: "side" },
        el("div", { class: "card" },
          el("h2", {}, "Status"), status, evalBox,
          el("div", { class: "row", style: "margin-top:10px" },
            el("button", { onClick: () => evaluate() }, "What does the engine say?"),
            el("button", { class: "ghost", onClick: () => board.setFlipped(!board.flipped) },
              "Flip"))),
        el("div", { class: "card" },
          el("h2", {}, "Opponent"),
          el("label", { class: "field" }, "Strength"), levelSelect,
          el("label", { class: "field", style: "margin-top:10px" }, "You play"), sideSelect,
          el("div", { class: "row", style: "margin-top:12px" },
            el("button", { class: "primary wide", onClick: () => reset(startFen) },
              startFen ? "Restart this position" : "New game"),
            startFen
              ? el("button", { class: "ghost wide", onClick: () => reset(null) }, "From the start")
              : null),
          el("p", { class: "sub", style: "margin-top:10px" },
            "Elo is on this project's internal scale, anchored by pinning "
            + "Fairy-Stockfish at 64k nodes to 2000. Only differences mean "
            + "anything, and the full-strength Fairy-Stockfish levels carry no "
            + "number at all: the ladder is measured against that engine, so it "
            + "sits above the top of its own scale by construction.")),
        whyCard,
        el("div", { class: "card" }, el("h2", {}, "Moves"), history))));

  const onKey = (event) => {
    if (event.target.matches("input, select, textarea")) return;
    why.handleKey(event);
  };
  document.addEventListener("keydown", onKey);
  teardown = () => document.removeEventListener("keydown", onKey);

  sideSelect.onchange = () => reset(startFen);
  await reset(startFen);
  return wrap;
}

/* ---------- patterns ---------- */

async function viewPatterns() {
  const data = await api("/api/patterns");
  const wrap = el("div", {});
  wrap.append(
    el("div", { class: "view-title" }, el("h1", {}, "Patterns"),
      el("div", { class: "sub" }, "the same mistake, more than once")),
    el("p", { class: "lede" },
      "A single blunder is a story; the same one in five games is a habit. "
      + "Grouped by position and, separately, by move — a move played by rote "
      + "whose correctness depends on what the opponent did shows up only in the "
      + "second grouping."));

  if (!data.positions.length && !data.moves.length) {
    wrap.append(empty("No repeated mistakes found.",
      "That is either good news or a thin corpus — import more games."));
    return wrap;
  }

  const positions = el("div", { class: "grid" });
  for (const cluster of data.positions) {
    const host = el("div", { class: "board-sm" });
    const board = new Board(host, { interactive: false });
    board.setState(staticState(cluster.fen));
    const played = Object.entries(cluster.played)
      .map(([san, n]) => `${san} ×${n}`).join(", ");
    const bookRows = Object.entries(cluster.book)
      .filter(([, info]) => info)
      .map(([san, info]) => el("p", { class: "sub" },
        `${san}: ${info.share.toFixed(0)}% of the field`
        + (info.played ? ` scoring ${info.played.score.toFixed(0)}%` : "")
        + (info.best ? ` · ${info.best.san} scores ${info.best.score.toFixed(0)}% on ${num(info.best.games)}` : "")));
    positions.append(el("div", { class: "card" },
      el("h2", {}, `${moveNumber(cluster.ply)} — ×${cluster.count}`,
        el("span", { class: "right muted" }, `−${cluster.avg_loss.toFixed(0)} win% each`)),
      host,
      el("p", { style: "margin:10px 0 2px" },
        `You played ${played}; the engine plays `,
        el("b", {}, cluster.best_san || "?")),
      bookRows,
      el("div", { class: "row tight", style: "margin-top:8px" },
        el("button", { class: "ghost",
          onClick: () => go("play", { fen: cluster.fen }) }, "Play it out"),
        ...cluster.games.slice(0, 4).map((id) =>
          el("button", { class: "ghost", onClick: () => go("review", { id }) }, id)))));
  }

  const moveRows = data.moves.map((cluster) => el("tr", {},
    el("td", { class: "mono" }, `${moveNumber(cluster.ply)} ${cluster.san}`),
    el("td", { class: "num" }, `×${cluster.count}`),
    el("td", { class: "num" }, cluster.positions),
    el("td", { class: "num" }, `−${cluster.avg_loss.toFixed(0)}`),
    el("td", {}, Object.entries(cluster.alternatives).map(([san, n]) => `${san} ×${n}`).join(", "))));

  wrap.append(
    el("h2", { class: "sub", style: "margin:18px 0 8px" },
      "SAME POSITION, MISHANDLED REPEATEDLY"),
    positions,
    data.moves.length
      ? el("div", { class: "card", style: "margin-top:18px" },
          el("h2", {}, "Same move, wrong in different positions — played by rote"),
          el("table", {},
            el("thead", {}, el("tr", {},
              el("th", {}, "Move"), el("th", { class: "num" }, "Times"),
              el("th", { class: "num" }, "Positions"),
              el("th", { class: "num" }, "Avg cost"), el("th", {}, "Best was"))),
            el("tbody", {}, moveRows)))
      : null);
  return wrap;
}

/* ---------- openings ---------- */

/* Where a name came from, in one word, because it changes how much it is worth
 * arguing with. A name the atomic community uses is a fact about the community;
 * one this program derived from the move is a claim, and a claim should look
 * like one. */
const NAME_SOURCE = {
  user: ["yours", "a name you gave this line"],
  community: ["community", "the name atomic players use for this"],
  standard: ["standard", "the name standard chess gives this exact move order"],
  descriptive: ["coined here", "no established name exists — this one was written for this index"],
  derived: ["derived", "named by rule from the move: see the evidence"],
};

function sourceTag(name) {
  const [label, title] = NAME_SOURCE[name.source] || [name.source, ""];
  return el("span", {
    class: "tag" + (name.source === "user" ? " accent" : ""),
    title: title + (name.evidence?.length ? ` — ${name.evidence.join(", ")}` : ""),
  }, label);
}

/* Naming a line, in place.
 *
 * A prompt() would have been three lines. It is worth more than three lines to
 * be able to see the line you are naming while you name it — that is the whole
 * complaint this feature answers — so the editor opens under the row it
 * belongs to and shows the derived name as the input's placeholder, which
 * makes "keep the derived name" a matter of leaving it alone.
 */
function renameEditor(name, onDone) {
  const input = el("input", {
    type: "text",
    value: name.source === "user" ? name.full : "",
    placeholder: name.full,
    style: "flex:1 1 220px",
  });
  const note = el("input", {
    type: "text", placeholder: "note — what to remember about it (optional)",
    style: "flex:1 1 220px",
  });
  async function save(value) {
    try {
      await api("/api/opening/name", {
        line: name.line.join(" "), name: value, note: note.value,
      });
      onDone();
    } catch (error) { toast(error.message); }
  }
  const box = el("div", { class: "card tight rename" },
    el("div", { class: "sub" }, "Name ", el("b", { class: "mono" },
      name.line.join(" ") || "the starting position")),
    el("div", { class: "row", style: "margin-top:6px" }, input, note),
    el("div", { class: "row tight", style: "margin-top:6px" },
      el("button", { class: "primary", onClick: () => save(input.value) }, "Save"),
      name.source === "user"
        ? el("button", { class: "ghost", onClick: () => save("") }, "Clear my name")
        : null,
      el("button", { class: "ghost", onClick: () => onDone() }, "Cancel")));
  setTimeout(() => input.focus(), 0);
  input.onkeydown = (event) => {
    if (event.key === "Enter") save(input.value);
    if (event.key === "Escape") onDone();
  };
  return box;
}

/** The named tree: one row per line, indented by depth, collapsible. */
function openingRows(nodes, into, depth, refresh) {
  for (const node of nodes) {
    const name = node.name;
    const kids = node.children || [];
    const open = depth < 1;
    const holder = el("div", { class: "opening-kids", hidden: !open });
    const twist = el("span", { class: "twist" + (kids.length ? "" : " leaf") },
      kids.length ? (open ? "▾" : "▸") : "·");
    const row = el("div", {
      class: "opening-row",
      style: `padding-left:${depth * 18}px`,
      onClick: () => {
        if (!kids.length) return;
        holder.hidden = !holder.hidden;
        twist.textContent = holder.hidden ? "▸" : "▾";
      },
    },
      twist,
      el("span", { class: "opening-name" }, name.short || name.full),
      el("span", { class: "mono tiny muted opening-line" }, node.line.join(" ")),
      el("span", { class: "opening-stats" },
        el("span", { class: "num tiny muted" }, num(node.games)),
        el("span", { class: "num tiny muted" }, pct(node.score, 0)),
        sourceTag(name),
        el("button", {
          class: "ghost tiny",
          title: "give this line a name",
          onClick: (event) => {
            event.stopPropagation();
            const existing = row.nextSibling?.classList?.contains("rename");
            document.querySelectorAll(".rename").forEach((n) => n.remove());
            if (!existing) row.after(renameEditor(name, refresh));
          },
        }, "name"),
        el("button", {
          class: "ghost tiny",
          title: "open this position",
          onClick: (event) => {
            event.stopPropagation();
            go("book", { moves: node.line.join(","), mode: "explore" });
          },
        }, "open")));
    into.append(row, holder);
    if (kids.length) openingRows(kids, holder, depth + 1, refresh);
  }
}

async function viewOpeningIndex(query) {
  const depth = Number(query.get("depth") || 4);
  const data = await api(`/api/openings?depth=${depth}&min_games=5000&min_share=8`);
  const wrap = el("div", {});
  const list = el("div", { class: "opening-tree" });
  const refresh = () => route();
  openingRows(data.tree, list, 0, refresh);

  // `el` filters a null child; `Node.append` stringifies it into the text
  // "null", so the conditional card is filtered before it gets there.
  wrap.append(...[
    el("div", { class: "row tight", style: "margin:12px 0" },
      el("span", { class: "sub", style: "align-self:center;margin-right:6px" }, "depth"),
      ...[2, 4, 6, 8].map((option) => el("span", {
        class: "chip" + (option === depth ? " on" : ""),
        onClick: () => go("book", { mode: "index", depth: String(option) }),
      }, String(option)))),
    el("div", { class: "card" }, list),
    data.yours.length
      ? el("div", { class: "card", style: "margin-top:14px" },
          el("h2", {}, "Names you have given"),
          el("table", {}, el("tbody", {}, data.yours.map((entry) => el("tr", {},
            el("td", { class: "mono tiny" }, entry.line || "(start)"),
            el("td", {}, entry.name),
            el("td", { class: "sub" }, entry.note || ""),
            el("td", {}, el("button", {
              class: "ghost tiny",
              onClick: () => go("book", { moves: entry.line.split(" ").join(","), mode: "explore" }),
            }, "open")))))))
      : null,
  ].filter(Boolean));
  return wrap;
}

async function viewBook(query) {
  const mode = query.get("mode") === "explore" || query.get("moves") ? "explore" : "index";
  const stats = SUMMARY?.book || {};
  if (!stats.moves) {
    return empty("No opening book built.",
      "Run `py -3.13 -m trainer.cli book build` — about five minutes over 12 monthly dumps.");
  }
  const head = el("div", {},
    el("div", { class: "view-title" }, el("h1", {}, "Openings"),
      el("div", { class: "sub" },
        `${num(stats.moves)} moves over ${num(stats.positions)} positions, `
        + `from ${num(stats.games)} games`)),
    el("p", { class: "lede" },
      "Atomic has no ECO and no published theory, so the lines are counted from "
      + "the Lichess monthly dumps and the names are built in the grammar "
      + "standard chess uses: a Defence is Black's answer to the first move, a "
      + "Gambit offers material, a Counterattack is Black taking the initiative "
      + "back — and that last one is measured, not asserted. Every name says "
      + "where it came from, and any of them can be replaced with your own."),
    el("div", { class: "row tight" },
      ...[["index", "Named index"], ["explore", "Explore the book"]].map(([key, label]) =>
        el("span", {
          class: "chip" + (key === mode ? " on" : ""),
          onClick: () => go("book", { mode: key }),
        }, label))));

  const body = mode === "index"
    ? await viewOpeningIndex(query)
    : await viewBookExplorer(query);
  return el("div", {}, head, body);
}

async function viewBookExplorer(query) {
  const moves = (query.get("moves") || "").split(",").filter(Boolean);
  const data = await api("/api/book?moves=" + encodeURIComponent(moves.join(",")));
  const wrap = el("div", {});

  const host = el("div", {});
  const board = new Board(host, { interactive: false });
  board.setState(data.state);

  const crumbs = el("div", { class: "row tight" },
    el("span", { class: "chip", onClick: () => go("book", { mode: "explore" }) }, "start"),
    ...moves.map((san, index) =>
      el("span", {
        class: "chip" + (index === moves.length - 1 ? " on" : ""),
        onClick: () => go("book", {
          moves: moves.slice(0, index + 1).join(","), mode: "explore",
        }),
      }, san)));

  const rows = data.entries.map((entry) => el("tr", {
    class: "clickable",
    onClick: () => go("book", {
      moves: [...moves, entry.san].join(","), mode: "explore",
    }),
  },
    el("td", { class: "mono" }, entry.san),
    el("td", { class: "tiny muted" }, data.names?.[entry.san] || ""),
    el("td", { class: "num" }, num(entry.games)),
    el("td", { class: "num" }, pct(entry.share, 1)),
    el("td", { class: "num" }, pct(entry.score, 1)),
    el("td", {}, el("div", { class: "bar" },
      el("span", { style: `width:${Math.max(0, Math.min(100, entry.score))}%` }))),
    el("td", { class: "num" }, entry.elo || "—")));

  const nameCard = el("div", { class: "card" });
  function drawName() {
    const name = data.name;
    nameCard.textContent = "";
    nameCard.append(
      el("h2", {}, "This line", el("span", { class: "right" }, sourceTag(name))),
      el("div", { style: "font-size:15px;font-weight:600" }, name.full),
      name.aliases?.length
        ? el("div", { class: "sub" }, "also called " + name.aliases.join(", "))
        : null,
      name.note ? el("p", { class: "sub", style: "margin-top:6px" }, name.note) : null,
      name.evidence?.length
        ? el("p", { class: "sub", style: "margin-top:6px" },
            "named from: " + name.evidence.join(", "))
        : null,
      // The family's own note, when this line is a variation of it rather than
      // the thing the note was written about.
      !name.note && name.family_note
        ? el("p", { class: "sub", style: "margin-top:8px" },
            el("b", {}, name.family), " — ", name.family_note)
        : null,
      el("div", { class: "row tight", style: "margin-top:8px" },
        el("button", { class: "ghost", onClick: () => {
          const editing = nameCard.querySelector(".rename");
          if (editing) { editing.remove(); return; }
          nameCard.append(renameEditor(name, () => route()));
        } }, "Name this line"),
        moves.length
          ? el("button", { class: "ghost", onClick: () =>
              go("play", { fen: data.fen }) }, "Play from here")
          : null));
  }
  drawName();

  wrap.append(
    crumbs,
    el("div", { class: "split", style: "margin-top:14px" },
      el("div", { class: "wide", style: "flex:0 0 auto" }, host),
      el("div", { class: "side", style: "flex:1 1 420px" },
        nameCard,
        el("div", { class: "card" },
          el("h2", {}, `${titleCase(data.turn)} to move`,
            el("span", { class: "right muted" }, `${num(data.total)} games here`)),
          data.entries.length
            ? el("div", { class: "scroll-x" }, el("table", {},
                el("thead", {}, el("tr", {},
                  el("th", {}, "Move"), el("th", {}, "Becomes"),
                  el("th", { class: "num" }, "Games"),
                  el("th", { class: "num" }, "Share"), el("th", { class: "num" }, "Score"),
                  el("th", {}, ""), el("th", { class: "num" }, "Avg Elo"))),
                el("tbody", {}, rows)))
            : el("p", { class: "sub" }, "Out of book — under the minimum game count.")))));
  return wrap;
}

/* ---------- traps ---------- */

const TRAP_ORDERS = [
  ["bait", "most often walked into"],
  ["severity", "worst if you get it wrong"],
  ["popular", "most played position"],
  ["early", "earliest"],
];

/* One trap on the board, with the punishment on a step control.
 *
 * The order things are shown in is the point. You see the position and what
 * holds; the losing move and its refutation stay behind a button, because a
 * trap you have read the answer to is not preparation, it is trivia. `step`
 * walks the punishment a ply at a time for the same reason the puzzle board
 * plays the blunder in front of you rather than describing it.
 */
class TrapBoard {
  constructor() {
    this.host = el("div", { class: "board-lg" });
    this.board = new Board(this.host, { interactive: false });
    this.title = el("div", { style: "font-size:15px;font-weight:600" });
    this.sub = el("div", { class: "sub" });
    this.holds = el("div", { class: "goal", style: "margin-top:10px" });
    this.punish = el("div", { style: "margin-top:10px" });
    this.moves = el("div", { class: "movelist plain" });
    this.controls = el("div", { class: "row tight", style: "margin-top:10px" });
    // The list lives in the side column rather than under the board: choosing
    // the next trap is the thing you do most on this page, and a chooser you
    // have to scroll past a chessboard to reach is a chooser you stop using.
    this.list = el("div", { class: "traplist" });
    this.node = el("div", { class: "split" },
      el("div", { class: "wide", style: "flex:0 0 auto" },
        this.host,
        el("div", { class: "card tight" }, this.moves)),
      el("div", { class: "side" },
        el("div", { class: "card" }, this.title, this.sub, this.holds, this.controls),
        el("div", { class: "card", hidden: true, id: "trap-punish" }, this.punish),
        el("div", { class: "card tight" },
          el("h2", {}, "All traps", el("span", { class: "right muted", id: "trap-count" })),
          el("div", { class: "scroll-y", style: "max-height:340px" }, this.list))));
    this.trap = null;
    this.step = 0;
  }

  async load(id) {
    this.trap = await api(`/api/trap/${id}`);
    this.step = 0;
    this.draw();
  }

  /** The line to the trapped position, then as much punishment as was asked for. */
  get line() {
    const trap = this.trap;
    return [
      ...trap.history.map((h) => ({ ...h, kind: "history" })),
      ...trap.punishment.slice(0, this.step).map((p, index) => ({
        ...p, kind: index === 0 ? "setup" : "solution",
      })),
    ];
  }

  draw() {
    const trap = this.trap;
    const line = this.line;
    const last = line[line.length - 1];
    this.board.setFlipped(trap.turn === "black");
    this.board.setState(last ? last.state : trap.state);

    this.title.textContent = trap.name.full;
    this.sub.textContent =
      `${titleCase(trap.turn)} to move · ${num(trap.games)} games reached this position`;

    const holds = trap.escapes;
    this.holds.className = "goal " + (holds.length === 1 ? "hold" : "win");
    this.holds.textContent = "";
    this.holds.append(
      el("div", { class: "goal-head" },
        el("b", {}, holds.length === 1
          ? "Exactly one move holds" : `Only ${holds.length} moves hold`),
        el("span", { class: "tag" }, `${trap.severity.toFixed(0)} pts`)),
      el("div", { class: "sub", style: "margin-top:4px" },
        this.step
          ? holds.map((m) => `${m.san} (${m.win.toFixed(0)}%)`).join(", ")
          : "Find them before you look."),
      el("div", { class: "sub", style: "margin-top:4px" },
        `Everything else loses: the best of them, ${trap.losers[0].san}, `
        + `leaves ${titleCase(trap.turn)} at ${trap.win_next.toFixed(0)}%.`),
      el("div", { class: "sub", style: "margin-top:4px" },
        `${trap.bait.toFixed(0)}% of the field walks in `
        + `(${num(trap.bait_games)} of ${num(trap.games)} games).`));

    this.moves.textContent = "";
    line.forEach((step, index) => {
      const ply = index;
      if (ply % 2 === 0) {
        this.moves.append(el("span", { class: "n" }, `${ply / 2 + 1}.`));
      }
      this.moves.append(el("span", { class: "mv " + step.kind }, step.san));
    });

    this.controls.textContent = "";
    const total = trap.punishment.length;
    this.controls.append(...[
      this.step < total
        ? el("button", {
            class: this.step ? "" : "primary",
            onClick: () => { this.step += 1; this.draw(); },
          }, this.step === 0 ? "Show what loses" : "Next move")
        : null,
      this.step
        ? el("button", { class: "ghost", onClick: () => { this.step = 0; this.draw(); } },
            "Back to the position")
        : null,
      el("button", {
        class: "ghost",
        onClick: () => go("play", { fen: trap.fen, side: trap.turn }),
      }, "Play it out"),
      el("button", {
        class: "ghost",
        onClick: () => go("book", { moves: trap.line.join(","), mode: "explore" }),
      }, "In the book"),
    ].filter(Boolean));

    const card = $("#trap-punish", this.node);
    card.hidden = this.step === 0;
    this.punish.textContent = "";
    if (this.step) {
      this.punish.append(
        el("h2", {}, "How it is punished"),
        el("p", { class: "sub" },
          `${trap.losers[0].san} is what the field plays here `
          + `(${num(trap.losers[0].games)} games). The engine's answer:`),
        el("div", { class: "mono", style: "margin-top:6px" },
          trap.punishment.map((step, index) =>
            el("span", {
              class: "pv-move" + (index < this.step ? "" : " muted"),
              onClick: () => { this.step = index + 1; this.draw(); },
            }, step.san + " "))),
        el("p", { class: "sub", style: "margin-top:8px" },
          "Read out of the engine's own principal variation at "
          + `${num(trap.nodes)} nodes — the line its verdict is made of, not a `
          + "second search that might disagree with it."));
    }
  }
}

async function viewTraps(query) {
  const filters = {
    turn: query.get("turn") || "",
    order: query.get("order") || "bait",
    max_ply: query.get("max_ply") || "",
    min_bait: query.get("min_bait") || "",
  };
  const params = new URLSearchParams({ order: filters.order, limit: "120" });
  for (const key of ["turn", "max_ply", "min_bait"]) {
    if (filters[key]) params.set(key, filters[key]);
  }
  const data = await api("/api/traps?" + params.toString());
  const wrap = el("div", {});

  wrap.append(
    el("div", { class: "view-title" }, el("h1", {}, "Traps"),
      el("div", { class: "sub" },
        `${data.stats.traps} found over ${data.stats.scanned} positions judged`)),
    el("p", { class: "lede" },
      "A trap is a book position where at most two moves keep the balance and "
      + "everything else loses outright — the engine's best alternative is at "
      + "least 25 win% worse and leaves you under a third. The last column is "
      + "the only number here that is not an engine opinion: it is the share of "
      + "the field that actually walks in, counted from the dumps. Sorting by it "
      + "is the difference between a critical position and a trap."));

  const scanPanel = new JobPanel(
    (result) => toast(`Scan finished${result ? " — " + result : ""}. `
      + "Reload the view to see what it found."),
    ["traps"],
  );
  const scanner = el("details", { class: "disclosure", style: "margin-top:14px" },
    el("summary", {}, "Scan the book for more"),
    el("div", { style: "margin-top:10px" }, scanPanel.node));

  if (!data.traps.length) {
    wrap.append(
      empty("No traps stored.",
        "Scan the book below, or run `py -3.13 -m trainer.cli traps scan` — "
        + "about 90 seconds over the popular part of the book."),
      scanner);
    teardown = () => scanPanel.destroy();
    return wrap;
  }

  const controls = el("div", { class: "row", style: "align-items:flex-end" },
    el("div", { class: "field-group" },
      el("label", { class: "field" }, "Whose problem"),
      el("select", { onChange: (e) => go("traps", { ...filters, turn: e.target.value }) },
        [["", "either side"], ["white", "White to move"], ["black", "Black to move"]]
          .map(([value, text]) =>
            el("option", { value, selected: filters.turn === value }, text)))),
    el("div", { class: "field-group" },
      el("label", { class: "field" }, "Order"),
      el("select", { onChange: (e) => go("traps", { ...filters, order: e.target.value }) },
        TRAP_ORDERS.map(([value, text]) =>
          el("option", { value, selected: filters.order === value }, text)))),
    el("div", { class: "field-group" },
      el("label", { class: "field" }, "By move"),
      el("select", { onChange: (e) => go("traps", { ...filters, max_ply: e.target.value }) },
        [["", "any"], ["4", "move 2"], ["8", "move 4"], ["12", "move 6"]].map(([value, text]) =>
          el("option", { value, selected: filters.max_ply === value }, text)))));

  const detail = new TrapBoard();
  const listBody = detail.list;
  data.traps.forEach((trap, index) => {
    listBody.append(el("div", {
      class: "trap-row",
      dataset: { id: trap.id },
      onClick: () => select(index),
    },
      el("div", { class: "row tight", style: "justify-content:space-between" },
        el("span", { class: "mono tiny" }, trap.line.join(" ")),
        el("span", { class: "tag" + (trap.bait >= 40 ? " bad" : "") },
          `${trap.bait.toFixed(0)}% walk in`)),
      el("div", { class: "trap-name" }, trap.name.full),
      el("div", { class: "sub" },
        `${titleCase(trap.turn)} to move · `
        + `${trap.escapes.length === 1 ? "one move holds" : trap.escapes.length + " moves hold"}`
        + ` · −${trap.severity.toFixed(0)} otherwise · ${num(trap.games)} games`)));
  });

  function select(index) {
    for (const row of listBody.children) row.classList.remove("on");
    listBody.children[index]?.classList.add("on");
    detail.load(data.traps[index].id).catch((error) => toast(error.message));
  }

  wrap.append(
    el("div", { class: "card tight" }, controls),
    el("div", { style: "margin-top:14px" }, detail.node),
    scanner);
  $("#trap-count", detail.node).textContent = `${data.traps.length} shown`;
  teardown = () => scanPanel.destroy();
  select(0);
  return wrap;
}

/* ---------- repertoire ---------- */

async function viewRepertoire(query) {
  const colour = query.get("color") === "black" ? "black" : "white";
  const data = await api(`/api/repertoire?color=${colour}`);
  const wrap = el("div", {});
  const stats = data.stats;

  wrap.append(
    el("div", { class: "view-title" }, el("h1", {}, "Repertoire"),
      el("div", { class: "sub" },
        `${stats.nodes} nodes to ply ${stats.depth}, ${stats.flagged} flagged`)),
    el("p", { class: "lede" },
      "Derived from the book, then checked by the engine, and grouped under the "
      + "opening each line belongs to — press ", el("b", {}, "name"),
      " on any row to call it whatever you will actually remember. A flagged "
      + "line is the field disagreeing with Fairy-Stockfish, which is worth "
      + "seeing rather than silently resolving — at ply 1 that disagreement is "
      + "the unresolved 1.Nh3-versus-1.Nf3 question."),
    el("div", { class: "row" }, ["white", "black"].map((option) =>
      el("span", {
        class: "chip" + (option === colour ? " on" : ""),
        onClick: () => go("repertoire", { color: option }),
      }, titleCase(option)))));

  if (!stats.nodes) {
    wrap.append(el("div", { style: "margin-top:14px" },
      empty(`No ${colour} repertoire.`,
        "Run `py -3.13 -m trainer.cli repertoire build` then `repertoire validate`.")));
    return wrap;
  }

  const onlyFlagged = el("span", { class: "chip", onClick: () => {
    onlyFlagged.classList.toggle("on"); drawLines();
  } }, "flagged only");
  const body = el("tbody", {});

  function verdictCell(line) {
    if (!line.validated) return el("span", { class: "tag" }, "unchecked");
    if (line.agrees) return el("span", { class: "tag good" },
      `engine agrees (${pct(line.win_move)})`);
    if (line.flagged) return el("span", { class: "tag bad" },
      `leaks ${line.val_loss.toFixed(0)} — engine plays ${line.best_san}`);
    // "by 0" reads as a contradiction; under half a point it is a different
    // move the engine rates the same, which is worth saying in words.
    return el("span", { class: "tag warn" }, line.val_loss < 0.5
      ? `engine plays ${line.best_san}, rates them level`
      : `engine prefers ${line.best_san} by ${line.val_loss.toFixed(0)}`);
  }

  /* Grouped by opening family rather than listed by ply.
   *
   * The flat list was ordered by `(ply, fen)`, which is the order the lines
   * were derived in and not an order anybody thinks in: two branches of the
   * same opening sat ten rows apart and the tenth row of "1.Nf3 f6" looked
   * exactly like the tenth row of "1.e4". Grouping by the name puts a
   * repertoire back into the shape it is actually memorised in — one opening
   * at a time, its variations under it. */
  function drawLines() {
    const flaggedOnly = onlyFlagged.classList.contains("on");
    body.textContent = "";
    const groups = new Map();
    for (const line of data.lines) {
      if (flaggedOnly && !line.flagged) continue;
      const family = line.name.family || "Unnamed lines";
      if (!groups.has(family)) groups.set(family, []);
      groups.get(family).push(line);
    }
    for (const [family, lines] of groups) {
      const flagged = lines.filter((l) => l.flagged).length;
      body.append(el("tr", { class: "group" },
        el("td", { colspan: 5 },
          el("span", { class: "group-name" }, family),
          el("span", { class: "muted tiny" },
            ` · ${lines.length} line${lines.length === 1 ? "" : "s"}`
            + (flagged ? ` · ${flagged} flagged` : "")))));
      for (const line of lines) {
        const name = line.name;
        body.append(el("tr", {
          class: "clickable",
          onClick: () => go("play", { fen: line.fen, side: colour }),
        },
          el("td", {},
            // The last two steps, not just the last: a repertoire has five
            // "Qh5 Attack" nodes in it, and a name that does not tell them
            // apart is not doing the job the whole feature exists for.
            el("div", {}, name.parts?.length
              ? name.parts.slice(-2).join(", ") : name.short),
            el("div", { class: "mono tiny muted" },
              (line.path.join(" ") || "(start)") + " → " + line.san)),
          el("td", { class: "num" }, line.games ? num(line.games) : "—"),
          el("td", { class: "num" }, line.score === null ? "—" : pct(line.score)),
          el("td", {}, verdictCell(line)),
          el("td", {},
            el("button", {
              class: "ghost tiny",
              title: "name this line",
              onClick: (event) => {
                event.stopPropagation();
                const row = event.target.closest("tr");
                if (row.nextSibling?.classList?.contains("rename-row")) {
                  row.nextSibling.remove();
                  return;
                }
                const holder = el("tr", { class: "rename-row" },
                  el("td", { colspan: 5 }, renameEditor(name, () => route())));
                row.after(holder);
              },
            }, "name"))));
      }
    }
    if (!body.children.length) {
      body.append(el("tr", {}, el("td", { colspan: 5, class: "muted" }, "nothing to show")));
    }
  }
  drawLines();

  const coverage = data.coverage;
  const gapRows = data.gaps.map((gap) => el("tr", {},
    el("td", {}, moveNumber(gap.ply)),
    el("td", { class: "num" }, `×${gap.count}`),
    el("td", { class: "num" }, `−${gap.cost.toFixed(0)}`),
    el("td", {}, Object.entries(gap.played).map(([san, n]) => `${san} ×${n}`).join(", ")),
    el("td", {}, el("button", { class: "ghost",
      onClick: () => go("play", { fen: gap.fen, side: colour }) }, "play"))));

  const deviationRows = data.deviations.map((dev) => el("tr", {},
    el("td", {}, moveNumber(dev.ply)),
    el("td", { class: "mono" }, dev.intended),
    el("td", {}, Object.entries(dev.played).map(([san, n]) => `${san} ×${n}`).join(", ")),
    el("td", { class: "num" }, `−${dev.cost.toFixed(0)}`)));

  wrap.append(el("div", { style: "margin-top:14px" },
    el("div", { class: "grid" },
      statCard("Coverage", `${coverage.total ? Math.round(100 * coverage.covered / coverage.total) : 0}%`,
        `${coverage.covered} of ${coverage.total} opening plies`),
      statCard("Engine agrees", `${stats.agreed}/${stats.nodes}`,
        `${stats.validated} validated`),
      statCard("Flagged", String(stats.flagged), "the engine dislikes these")),
    el("div", { class: "card", style: "margin-top:14px" },
      el("h2", {}, "Lines, by opening", el("span", { class: "right" }, onlyFlagged)),
      el("div", { class: "scroll-x" }, el("table", { class: "rep-lines" },
        el("thead", {}, el("tr", {},
          el("th", {}, "Variation"),
          el("th", { class: "num" }, "Games"), el("th", { class: "num" }, "Field score"),
          el("th", {}, "Engine"), el("th", {}, ""))),
        body))),
    data.gaps.length
      ? el("div", { class: "card", style: "margin-top:14px" },
          el("h2", {}, "Gaps — positions you keep reaching with no line"),
          el("table", {},
            el("thead", {}, el("tr", {},
              el("th", {}, "At"), el("th", { class: "num" }, "Times"),
              el("th", { class: "num" }, "Total cost"), el("th", {}, "You played"),
              el("th", {}, ""))),
            el("tbody", {}, gapRows)))
      : null,
    data.deviations.length
      ? el("div", { class: "card", style: "margin-top:14px" },
          el("h2", {}, "Deviations — you had a line and played something else"),
          el("table", {},
            el("thead", {}, el("tr", {},
              el("th", {}, "At"), el("th", {}, "Intended"),
              el("th", {}, "Played"), el("th", { class: "num" }, "Total cost"))),
            el("tbody", {}, deviationRows)))
      : null));
  return wrap;
}

/* ---------- boot ---------- */

async function boot() {
  try {
    SUMMARY = await api("/api/summary");
    $("#engine-name").textContent = SUMMARY.engine;
    const health = $("#health");
    if (!SUMMARY.winrate.fitted) {
      health.className = "health warn";
      health.textContent = "win% curve unfitted — judgements are guesses";
    } else {
      health.textContent = `${SUMMARY.analysed} games analysed · `
        + `${SUMMARY.puzzles.total} puzzles`;
    }
  } catch (error) {
    toast("Could not reach the server: " + error.message);
  }
  window.addEventListener("hashchange", route);
  await route();
}

boot();
