//! Alpha-beta search, specialised for atomic.
//!
//! Three things differ from a standard-chess searcher and all three come from
//! the same source — captures are enormously more consequential:
//!
//! 1. **Immediate-win detection.** Any capture whose blast radius contains the
//!    enemy king ends the game on the spot. `legal_moves()` has already
//!    filtered out the ones that would also kill our own king, so such a move
//!    can be returned without searching it at all. This is cheap and prunes a
//!    lot, because atomic positions are full of them.
//! 2. **Blast-aware ordering instead of MVV-LVA.** The capturer always dies,
//!    so "victim minus attacker" is not even the right sign much of the time.
//!    `eval::blast_gain` scores the whole radius.
//! 3. **No null-move pruning.** The null-move observation ("having the move is
//!    worth something, so if I can pass and still be winning, I am winning") is
//!    weak in a variant where a single quiet-looking capture flips the result.
//!    Leaving it out is a deliberate choice, and a candidate for a later
//!    ablation rather than an oversight.

use shakmaty::variant::Atomic;
use shakmaty::zobrist::Zobrist64;
use shakmaty::{attacks, Color, EnPassantMode, KnownOutcome, Move, Outcome, Position, Square};

use crate::eval::{blast_gain, evaluate, EvalFlags};

pub const MATE: i32 = 30_000;
pub const INFINITY: i32 = 32_000;
const MAX_PLY: usize = 128;
const MAX_QPLY: usize = 32;
/// How often to test the node budget. Checking every node costs more than it
/// saves; 1023 nodes of overshoot is far below the noise floor of a match.
const CHECK_MASK: u64 = 1023;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    mv: Option<Move>,
    score: i32,
    depth: i8,
    bound: Bound,
}

impl TtEntry {
    const EMPTY: TtEntry =
        TtEntry { key: 0, mv: None, score: 0, depth: -1, bound: Bound::Exact };
}

pub struct SearchResult {
    pub best: Option<Move>,
    pub score: i32,
    pub depth: u32,
    pub nodes: u64,
}

pub struct Searcher {
    pub flags: EvalFlags,
    /// Whether to resolve captures at the horizon. Switchable because it is
    /// the search's counterpart to an evaluation term: a capture-only
    /// quiescence covers much more of atomic's tactical space than it does in
    /// standard chess, and turning it off is how we find out which knowledge
    /// lives in the search rather than in the evaluation.
    pub quiescence: bool,
    tt: Vec<TtEntry>,
    mask: usize,
    nodes: u64,
    limit: u64,
    stopped: bool,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: Vec<i32>, // 64 * 64, indexed from * 64 + to
}

impl Searcher {
    pub fn new(flags: EvalFlags, tt_mb: usize) -> Self {
        let bytes = tt_mb.max(1) * 1024 * 1024;
        let mut slots = (bytes / std::mem::size_of::<TtEntry>()).max(1024);
        slots = slots.next_power_of_two() / 2; // stay under the requested size
        Self {
            flags,
            quiescence: true,
            tt: vec![TtEntry::EMPTY; slots],
            mask: slots - 1,
            nodes: 0,
            limit: u64::MAX,
            stopped: false,
            killers: [[None; 2]; MAX_PLY],
            history: vec![0; 64 * 64],
        }
    }

    pub fn clear(&mut self) {
        self.tt.iter_mut().for_each(|e| *e = TtEntry::EMPTY);
        self.history.iter_mut().for_each(|h| *h = 0);
        self.killers = [[None; 2]; MAX_PLY];
    }

    /// Iterative deepening under a node budget.
    ///
    /// The budget is a hard ceiling, not a target: an aborted iteration is
    /// discarded except for its root move, so a method measured at N nodes has
    /// really spent at most N.
    pub fn go(&mut self, pos: &Atomic, max_nodes: u64, max_depth: u32) -> SearchResult {
        self.nodes = 0;
        self.limit = max_nodes;
        self.stopped = false;
        self.killers = [[None; 2]; MAX_PLY];
        self.history.iter_mut().for_each(|h| *h /= 8);

        let mut best = None;
        let mut score = 0;
        let mut completed = 0;

        for depth in 1..=max_depth.min(MAX_PLY as u32 - 2) {
            let (mv, sc) = self.search_root(pos, depth as i32, best);
            if self.stopped {
                // Keep the aborted iteration's root move only if it found one;
                // a partial root search still ordered the best move first.
                if mv.is_some() && completed > 0 {
                    best = mv;
                }
                break;
            }
            best = mv;
            score = sc;
            completed = depth;
            if score.abs() >= MATE - MAX_PLY as i32 {
                break; // forced result found; deeper search cannot improve it
            }
        }

        SearchResult { best, score, depth: completed, nodes: self.nodes }
    }

    fn search_root(
        &mut self,
        pos: &Atomic,
        depth: i32,
        prev_best: Option<Move>,
    ) -> (Option<Move>, i32) {
        if let Some(win) = winning_capture(pos) {
            return (Some(win), MATE - 1);
        }
        let mut moves = self.ordered_moves(pos, prev_best, 0);
        if moves.is_empty() {
            return (None, terminal_score(pos, 0));
        }

        let mut alpha = -INFINITY;
        let mut best_move = Some(moves[0]);
        for (i, mv) in moves.drain(..).enumerate() {
            let mut child = pos.clone();
            child.play_unchecked(mv);
            let score = if i == 0 {
                -self.negamax(&child, depth - 1, -INFINITY, -alpha, 1)
            } else {
                let narrow = -self.negamax(&child, depth - 1, -alpha - 1, -alpha, 1);
                if narrow > alpha {
                    -self.negamax(&child, depth - 1, -INFINITY, -alpha, 1)
                } else {
                    narrow
                }
            };
            if self.stopped {
                return (best_move, alpha);
            }
            if score > alpha {
                alpha = score;
                best_move = Some(mv);
            }
        }
        (best_move, alpha)
    }

    fn negamax(&mut self, pos: &Atomic, depth: i32, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        if self.out_of_budget() {
            return 0;
        }
        self.nodes += 1;

        if pos.is_variant_end() || ply >= MAX_PLY - 1 {
            return terminal_score(pos, ply);
        }
        if depth <= 0 {
            return if self.quiescence {
                self.qsearch(pos, alpha, beta, ply, 0)
            } else {
                self.nodes += 1;
                evaluate(pos, &self.flags)
            };
        }

        let key = u64::from(pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal));
        let slot = (key as usize) & self.mask;
        let mut tt_move = None;
        {
            let entry = self.tt[slot];
            if entry.key == key {
                tt_move = entry.mv;
                if entry.depth as i32 >= depth {
                    let score = from_tt_score(entry.score, ply);
                    match entry.bound {
                        Bound::Exact => return score,
                        Bound::Lower if score >= beta => return score,
                        Bound::Upper if score <= alpha => return score,
                        _ => {}
                    }
                }
            }
        }

        // A capture that catches the enemy king wins immediately, and costs one
        // bitboard test per move to spot.
        if winning_capture(pos).is_some() {
            return MATE - ply as i32;
        }

        let mut moves = self.ordered_moves(pos, tt_move, ply);
        if moves.is_empty() {
            return terminal_score(pos, ply);
        }

        let in_check = pos.is_check();
        let original_alpha = alpha;
        let mut best_score = -INFINITY;
        let mut best_move = None;

        for (i, mv) in moves.drain(..).enumerate() {
            let mut child = pos.clone();
            child.play_unchecked(mv);

            // Late move reduction on quiet moves only. Atomic tactics are
            // overwhelmingly capture-driven, so reducing quiet moves is safer
            // here than in standard chess — but never reduce out of check.
            let quiet = !mv.is_capture() && mv.promotion().is_none();
            let reduction = if depth >= 3 && i >= 4 && quiet && !in_check { 1 } else { 0 };

            let score = if i == 0 {
                -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1)
            } else {
                let narrow =
                    -self.negamax(&child, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);
                if narrow > alpha && (reduction > 0 || narrow < beta) {
                    -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1)
                } else {
                    narrow
                }
            };
            if self.stopped {
                return best_score.max(-INFINITY + 1);
            }
            if score > best_score {
                best_score = score;
                best_move = Some(mv);
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                if quiet {
                    self.remember_quiet(mv, depth, ply);
                }
                break;
            }
        }

        let bound = if best_score <= original_alpha {
            Bound::Upper
        } else if best_score >= beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        let entry = self.tt[slot];
        if entry.key != key || depth >= entry.depth as i32 {
            self.tt[slot] = TtEntry {
                key,
                mv: best_move,
                score: to_tt_score(best_score, ply),
                depth: depth.clamp(0, i8::MAX as i32) as i8,
                bound,
            };
        }
        best_score
    }

    /// Quiescence over captures.
    ///
    /// In atomic almost every tactic is a capture — the king can only die to
    /// one — so a capture-only quiescence covers more of the tactical space
    /// than it does in standard chess. Check evasions are searched in full
    /// because standing pat while in check is unsound.
    fn qsearch(&mut self, pos: &Atomic, mut alpha: i32, beta: i32, ply: usize, qply: usize) -> i32 {
        if self.out_of_budget() {
            return 0;
        }
        self.nodes += 1;

        if pos.is_variant_end() || ply >= MAX_PLY - 1 {
            return terminal_score(pos, ply);
        }
        if winning_capture(pos).is_some() {
            return MATE - ply as i32;
        }
        if qply >= MAX_QPLY {
            return evaluate(pos, &self.flags);
        }

        let in_check = pos.is_check();
        if !in_check {
            let stand_pat = evaluate(pos, &self.flags);
            if stand_pat >= beta {
                return stand_pat;
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
        }

        let mut moves: Vec<Move> =
            if in_check { pos.legal_moves().to_vec() } else { pos.capture_moves().to_vec() };
        if moves.is_empty() {
            return if in_check { terminal_score(pos, ply) } else { alpha };
        }

        let board = pos.board();
        let us = pos.turn();
        moves.sort_by_cached_key(|mv| -blast_gain(board, us, mv.from(), mv.to()));

        let mut best = if in_check { -INFINITY } else { alpha };
        for mv in moves {
            let mut child = pos.clone();
            child.play_unchecked(mv);
            let score = -self.qsearch(&child, -beta, -alpha, ply + 1, qply + 1);
            if self.stopped {
                return best;
            }
            if score > best {
                best = score;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }

    fn out_of_budget(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if self.nodes & CHECK_MASK == 0 && self.nodes >= self.limit {
            self.stopped = true;
        }
        self.stopped
    }

    fn remember_quiet(&mut self, mv: Move, depth: i32, ply: usize) {
        if self.killers[ply][0] != Some(mv) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(mv);
        }
        if let Some(from) = mv.from() {
            let idx = usize::from(from) * 64 + usize::from(mv.to());
            self.history[idx] = (self.history[idx] + depth * depth).min(1 << 20);
        }
    }

    fn ordered_moves(&self, pos: &Atomic, tt_move: Option<Move>, ply: usize) -> Vec<Move> {
        let mut moves: Vec<Move> = pos.legal_moves().to_vec();
        let board = pos.board();
        let us = pos.turn();
        let killers = self.killers[ply.min(MAX_PLY - 1)];
        moves.sort_by_cached_key(|mv| {
            let score = if Some(*mv) == tt_move {
                1 << 28
            } else if mv.is_capture() {
                // Blast gain, not MVV-LVA: a capture that detonates two of our
                // own pieces alongside the victim belongs at the *back*.
                (1 << 20) + blast_gain(board, us, mv.from(), mv.to())
            } else if Some(*mv) == killers[0] {
                1 << 18
            } else if Some(*mv) == killers[1] {
                (1 << 18) - 1
            } else if let Some(from) = mv.from() {
                self.history[usize::from(from) * 64 + usize::from(mv.to())]
            } else {
                0
            };
            -score
        });
        moves
    }
}

/// A legal capture whose blast radius contains the enemy king, if one exists.
///
/// `legal_moves()` has already removed anything that would explode our own
/// king, so such a move is an unconditional, immediate win.
fn winning_capture(pos: &Atomic) -> Option<Move> {
    let board = pos.board();
    let their_king = board.king_of(!pos.turn())?;
    pos.capture_moves().iter().copied().find(|mv| blast_hits(mv.to(), their_king))
}

#[inline]
fn blast_hits(to: Square, target: Square) -> bool {
    to == target || attacks::king_attacks(to).contains(target)
}

/// Score of a position with no moves left, or one the variant has ended.
fn terminal_score(pos: &Atomic, ply: usize) -> i32 {
    let outcome = if pos.is_variant_end() { pos.variant_outcome() } else { pos.outcome() };
    match outcome {
        Outcome::Known(KnownOutcome::Decisive { winner }) => {
            let magnitude = MATE - ply as i32;
            if winner == pos.turn() {
                magnitude
            } else {
                -magnitude
            }
        }
        Outcome::Known(KnownOutcome::Draw) => 0,
        // Not actually over — only reachable when a caller asks for a terminal
        // score at a live node, which the search does not do.
        Outcome::Unknown => 0,
    }
}

/// Mate scores are stored ply-independent so a transposition hit at a different
/// distance from the root still reports the right distance to mate.
#[inline]
fn to_tt_score(score: i32, ply: usize) -> i32 {
    if score >= MATE - MAX_PLY as i32 {
        score + ply as i32
    } else if score <= -(MATE - MAX_PLY as i32) {
        score - ply as i32
    } else {
        score
    }
}

#[inline]
fn from_tt_score(score: i32, ply: usize) -> i32 {
    if score >= MATE - MAX_PLY as i32 {
        score - ply as i32
    } else if score <= -(MATE - MAX_PLY as i32) {
        score + ply as i32
    } else {
        score
    }
}

/// Convenience for callers that want a colour-independent "who is winning".
pub fn score_for_white(pos: &Atomic, score: i32) -> i32 {
    if pos.turn() == Color::White {
        score
    } else {
        -score
    }
}
