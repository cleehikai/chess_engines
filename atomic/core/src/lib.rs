//! Fast atomic chess core, exposed to Python as `atomic_core`.
//!
//! Why this exists: the previous engine in this repo topped out around 12k
//! nodes/sec in pure Python, and that ceiling — not eval quality — decided
//! every experiment it ran. Fairy-Stockfish does 1.2M nps on this machine. A
//! research harness whose methods are 100x slower than the yardstick cannot
//! tell "this idea is worse" apart from "this idea had less compute", so the
//! hot loop lives here.
//!
//! Rules come from `shakmaty::variant::Atomic`, the same implementation Lichess
//! runs in production. Atomic's legality rules are genuinely tricky — a pinned
//! piece *may* move if the move explodes the enemy king, and a capture that
//! would explode your own king is illegal even when it looks winning — so
//! borrowing a battle-tested implementation is worth more than the speed of a
//! hand-rolled one.

pub mod eval;
pub mod search;

use pyo3::exceptions::{PyValueError, PyIndexError};
use pyo3::prelude::*;
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Color, EnPassantMode, Position};
use shakmaty::variant::Atomic;

use eval::EvalFlags;
use search::Searcher;

const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/// An atomic chess position with an undo stack.
#[pyclass(name = "Board")]
#[derive(Clone)]
struct PyBoard {
    pos: Atomic,
    history: Vec<Atomic>,
}

#[pymethods]
impl PyBoard {
    #[new]
    #[pyo3(signature = (fen = None))]
    fn new(fen: Option<&str>) -> PyResult<Self> {
        let fen = fen.unwrap_or(STARTPOS);
        let parsed: Fen = fen
            .parse()
            .map_err(|e| PyValueError::new_err(format!("bad FEN {fen:?}: {e}")))?;
        let pos: Atomic = parsed
            .into_position(CastlingMode::Standard)
            .map_err(|e| PyValueError::new_err(format!("illegal position {fen:?}: {e}")))?;
        Ok(Self { pos, history: Vec::new() })
    }

    /// FEN of the current position.
    fn fen(&self) -> String {
        Fen::from_position(&self.pos, EnPassantMode::Legal).to_string()
    }

    /// Legal moves in UCI notation.
    fn legal_moves(&self) -> Vec<String> {
        self.pos
            .legal_moves()
            .iter()
            .map(|m| UciMove::from_standard(*m).to_string())
            .collect()
    }

    /// True if White is to move.
    fn turn_white(&self) -> bool {
        self.pos.turn() == Color::White
    }

    /// Play a UCI move. Raises if the move is not legal here.
    fn push(&mut self, uci: &str) -> PyResult<()> {
        let parsed: UciMove = uci
            .parse()
            .map_err(|e| PyValueError::new_err(format!("bad UCI {uci:?}: {e}")))?;
        let mv = parsed
            .to_move(&self.pos)
            .map_err(|e| PyValueError::new_err(format!("illegal move {uci:?}: {e}")))?;
        self.history.push(self.pos.clone());
        self.pos.play_unchecked(mv);
        Ok(())
    }

    /// Undo the last move.
    fn pop(&mut self) -> PyResult<()> {
        self.pos = self
            .history
            .pop()
            .ok_or_else(|| PyIndexError::new_err("move stack is empty"))?;
        Ok(())
    }

    fn is_game_over(&self) -> bool {
        self.pos.is_game_over()
    }

    /// True when the game ended by an atomic-specific condition (king exploded)
    /// rather than by checkmate, stalemate or insufficient material.
    fn is_variant_end(&self) -> bool {
        self.pos.is_variant_end()
    }

    /// "1-0", "0-1", "1/2-1/2", or None while the game is running.
    fn result(&self) -> Option<String> {
        if !self.pos.is_game_over() {
            return None;
        }
        let outcome = self.pos.outcome();
        Some(format!("{outcome}"))
    }

    fn ply(&self) -> usize {
        self.history.len()
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __repr__(&self) -> String {
        format!("<atomic_core.Board '{}'>", self.fen())
    }
}

/// Count legal move paths of `depth` plies from `fen`.
///
/// The correctness gate for this whole crate: these numbers must match
/// Fairy-Stockfish's `go perft` exactly on every position in
/// `atomic/tests/perft_positions.json`. If they diverge, nothing measured on
/// top of this core means anything.
#[pyfunction]
#[pyo3(signature = (fen, depth))]
fn perft(fen: &str, depth: u32) -> PyResult<u64> {
    let parsed: Fen = fen
        .parse()
        .map_err(|e| PyValueError::new_err(format!("bad FEN {fen:?}: {e}")))?;
    let pos: Atomic = parsed
        .into_position(CastlingMode::Standard)
        .map_err(|e| PyValueError::new_err(format!("illegal position {fen:?}: {e}")))?;
    Ok(shakmaty::perft(&pos, depth))
}

/// Per-root-move perft breakdown, for bisecting a movegen mismatch.
#[pyfunction]
#[pyo3(signature = (fen, depth))]
fn perft_divide(fen: &str, depth: u32) -> PyResult<Vec<(String, u64)>> {
    let parsed: Fen = fen
        .parse()
        .map_err(|e| PyValueError::new_err(format!("bad FEN {fen:?}: {e}")))?;
    let pos: Atomic = parsed
        .into_position(CastlingMode::Standard)
        .map_err(|e| PyValueError::new_err(format!("illegal position {fen:?}: {e}")))?;
    if depth == 0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for mv in pos.legal_moves().iter() {
        let mut child = pos.clone();
        child.play_unchecked(*mv);
        let count = if depth == 1 { 1 } else { shakmaty::perft(&child, depth - 1) };
        out.push((UciMove::from_standard(*mv).to_string(), count));
    }
    out.sort();
    Ok(out)
}

fn parse_position(fen: &str) -> PyResult<Atomic> {
    let parsed: Fen = fen
        .parse()
        .map_err(|e| PyValueError::new_err(format!("bad FEN {fen:?}: {e}")))?;
    parsed
        .into_position(CastlingMode::Standard)
        .map_err(|e| PyValueError::new_err(format!("illegal position {fen:?}: {e}")))
}

/// M1: alpha-beta over the atomic-native handcrafted evaluation.
///
/// `terms` selects which evaluation terms are live, which is how the ablation
/// experiments are run — `Engine(terms=["material"])` is the material-only
/// control, `Engine(terms=[...all but "king_blast"...])` isolates the
/// blast-radius king-safety term. Passing `None` enables everything.
#[pyclass(name = "Engine")]
struct PyEngine {
    searcher: Searcher,
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (terms = None, tt_mb = 64, quiescence = true))]
    fn new(terms: Option<Vec<String>>, tt_mb: usize, quiescence: bool) -> PyResult<Self> {
        let flags = match terms {
            None => EvalFlags::default(),
            Some(names) => {
                let mut flags = EvalFlags::NONE;
                for name in &names {
                    if !flags.enable(name) {
                        return Err(PyValueError::new_err(format!(
                            "unknown eval term {name:?}; known: material, king_blast, \
                             pawn_shield, mobility, pst"
                        )));
                    }
                }
                flags
            }
        };
        let mut searcher = Searcher::new(flags, tt_mb);
        searcher.quiescence = quiescence;
        Ok(Self { searcher })
    }

    /// The evaluation terms this engine has switched on.
    #[getter]
    fn terms(&self) -> Vec<&'static str> {
        self.searcher.flags.names()
    }

    /// Search `fen` under a node budget.
    ///
    /// Returns `(best_move_uci, score_cp, depth, nodes)` where the score is
    /// from the side to move's point of view. `best_move_uci` is None only if
    /// the position has no legal moves.
    #[pyo3(signature = (fen, nodes = 100_000, depth = 64))]
    fn search(
        &mut self,
        fen: &str,
        nodes: u64,
        depth: u32,
    ) -> PyResult<(Option<String>, i32, u32, u64)> {
        let pos = parse_position(fen)?;
        let result = self.searcher.go(&pos, nodes, depth);
        let best = result
            .best
            .map(|mv| UciMove::from_standard(mv).to_string());
        Ok((best, result.score, result.depth, result.nodes))
    }

    /// Static evaluation in centipawns, from the side to move's point of view.
    fn evaluate(&self, fen: &str) -> PyResult<i32> {
        let pos = parse_position(fen)?;
        Ok(eval::evaluate(&pos, &self.searcher.flags))
    }

    /// Same, but always from White's point of view — easier to eyeball.
    fn evaluate_white(&self, fen: &str) -> PyResult<i32> {
        let pos = parse_position(fen)?;
        let score = eval::evaluate(&pos, &self.searcher.flags);
        Ok(search::score_for_white(&pos, score))
    }

    /// Net material swing of a capture once its explosion resolves.
    ///
    /// Exposed because it is the atomic replacement for static exchange
    /// evaluation and is worth inspecting directly when debugging a blunder.
    fn blast_gain(&self, fen: &str, uci: &str) -> PyResult<i32> {
        let pos = parse_position(fen)?;
        let parsed: UciMove = uci
            .parse()
            .map_err(|e| PyValueError::new_err(format!("bad UCI {uci:?}: {e}")))?;
        let mv = parsed
            .to_move(&pos)
            .map_err(|e| PyValueError::new_err(format!("illegal move {uci:?}: {e}")))?;
        Ok(eval::blast_gain(pos.board(), pos.turn(), mv.from(), mv.to()))
    }

    /// Drop the transposition table and history heuristics. Called between
    /// games so one game cannot leak information into the next.
    fn clear(&mut self) {
        self.searcher.clear();
    }
}

#[pymodule]
fn atomic_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBoard>()?;
    m.add_class::<PyEngine>()?;
    m.add_function(wrap_pyfunction!(perft, m)?)?;
    m.add_function(wrap_pyfunction!(perft_divide, m)?)?;
    m.add("STARTPOS", STARTPOS)?;
    m.add("EVAL_TERMS", vec!["material", "king_blast", "pawn_shield", "mobility", "pst"])?;
    Ok(())
}
