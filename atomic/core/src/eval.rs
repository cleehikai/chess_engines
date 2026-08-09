//! An atomic-native handcrafted evaluation.
//!
//! The design constraint that makes this interesting: **in atomic, king safety
//! is not local.** A king dies when a capture happens on any of the eight
//! squares next to it, by a piece that never attacked the king and may never
//! have come near it. So a standard-chess king-safety term — "count attackers
//! of the king ring, weight by piece type" — measures the wrong quantity. The
//! thing that actually kills you is *your own non-pawn pieces standing next to
//! your own king*, because each one is a fuse the opponent can light.
//!
//! Pawns are immune to the blast, which inverts the other standard-chess
//! intuition: a pawn next to your king is a genuine shield (it cannot detonate,
//! and it occupies a square that a detonatable piece would otherwise sit on),
//! while a knight next to your king is a liability.
//!
//! Every term is behind a flag so the method catalog's ablations are a matter
//! of constructing the evaluator differently, not of editing this file.

use shakmaty::variant::Atomic;
use shakmaty::{attacks, Bitboard, Board, Color, Position, Role, Square};

/// Piece values from Pav (2025), Table 9, normalised to pawn = 100.
///
/// Atomic values are heavily *compressed* relative to standard chess and the
/// bishop is worth clearly more than the knight — the opposite of the usual
/// near-equality. Kept identical to `lab.methods.ATOMIC_PIECE_VALUES` so that
/// the Python and Rust methods are comparable.
pub const fn piece_value(role: Role) -> i32 {
    match role {
        Role::Pawn => 100,
        Role::Knight => 190,
        Role::Bishop => 252,
        Role::Rook => 486,
        Role::Queen => 1018,
        Role::King => 0, // the king is the win condition, not material
    }
}

/// Like [`piece_value`] but with a finite, dominant price on the king.
///
/// Used only when scoring a blast radius: a capture that catches a king is
/// decisive, and the ordering code needs that to outrank every material
/// consideration without special-casing.
pub const fn blast_value(role: Role) -> i32 {
    match role {
        Role::King => 20_000,
        other => piece_value(other),
    }
}

/// Which evaluation terms are switched on. The ablation surface of M1.
#[derive(Clone, Copy, Debug)]
pub struct EvalFlags {
    pub material: bool,
    /// Own non-pawn pieces adjacent to own king, and whether the opponent can
    /// already capture one. The flagship atomic term.
    pub king_blast: bool,
    /// Pawns adjacent to own king: blast-immune, so a real shield.
    pub pawn_shield: bool,
    pub mobility: bool,
    pub pst: bool,
}

impl Default for EvalFlags {
    fn default() -> Self {
        Self { material: true, king_blast: true, pawn_shield: true, mobility: true, pst: true }
    }
}

impl EvalFlags {
    /// All terms off — the starting point for building a flag set by name.
    pub const NONE: EvalFlags = EvalFlags {
        material: false,
        king_blast: false,
        pawn_shield: false,
        mobility: false,
        pst: false,
    };

    /// Enable a term by name. Returns false for an unknown name so the caller
    /// can report a typo rather than silently running a different experiment.
    pub fn enable(&mut self, term: &str) -> bool {
        match term {
            "material" => self.material = true,
            "king_blast" => self.king_blast = true,
            "pawn_shield" => self.pawn_shield = true,
            "mobility" => self.mobility = true,
            "pst" => self.pst = true,
            _ => return false,
        }
        true
    }

    pub fn names(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.material {
            out.push("material");
        }
        if self.king_blast {
            out.push("king_blast");
        }
        if self.pawn_shield {
            out.push("pawn_shield");
        }
        if self.mobility {
            out.push("mobility");
        }
        if self.pst {
            out.push("pst");
        }
        out
    }
}

// Term weights. Deliberately round numbers: they are hypotheses to be tuned by
// measurement later, and pretending to three significant figures before a
// single game has been played would be false precision.
const BOMB_PENALTY: i32 = 22; // per own non-pawn piece adjacent to own king
const LIT_FUSE_PENALTY: i32 = 55; // per enemy attacker of such a piece
const SHIELD_BONUS: i32 = 14; // per own pawn adjacent to own king
const MOBILITY_UNIT: i32 = 4; // per pseudo-legal destination
const TEMPO: i32 = 12;

/// Centralisation, in centipawns, indexed from White's a1. Atomic rewards
/// central activity less than standard chess does — the board empties fast and
/// long-range threats matter more than outposts — so the gradient is shallow.
#[rustfmt::skip]
const CENTRE: [i32; 64] = [
    -20, -12,  -6,  -2,  -2,  -6, -12, -20,
    -12,  -4,   2,   6,   6,   2,  -4, -12,
     -6,   2,   8,  12,  12,   8,   2,  -6,
     -2,   6,  12,  16,  16,  12,   6,  -2,
     -2,   6,  12,  16,  16,  12,   6,  -2,
     -6,   2,   8,  12,  12,   8,   2,  -6,
    -12,  -4,   2,   6,   6,   2,  -4, -12,
    -20, -12,  -6,  -2,  -2,  -6, -12, -20,
];

/// Advancing a pawn matters more in atomic than the raw promotion odds
/// suggest: pawns are the only pieces that survive a blast, so a pawn that
/// reaches the seventh is a threat that cannot be traded off.
#[rustfmt::skip]
const PAWN_RANK: [i32; 8] = [0, 0, 4, 10, 20, 36, 60, 0];

#[inline]
fn sq_index(sq: Square, color: Color) -> usize {
    let i = usize::from(sq);
    if color == Color::White {
        i
    } else {
        i ^ 56 // mirror ranks; files are symmetric in these tables
    }
}

/// Evaluate from the side to move's point of view (negamax convention).
pub fn evaluate(pos: &Atomic, flags: &EvalFlags) -> i32 {
    let board = pos.board();
    let raw = side_score(board, Color::White, flags) - side_score(board, Color::Black, flags);
    let stm = if pos.turn() == Color::White { raw } else { -raw };
    stm + TEMPO
}

fn side_score(board: &Board, us: Color, flags: &EvalFlags) -> i32 {
    let mut score = 0;

    if flags.material {
        for role in [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen] {
            let count = (board.by_color(us) & board.by_role(role)).count() as i32;
            score += count * piece_value(role);
        }
    }

    if flags.king_blast {
        score -= king_blast_penalty(board, us);
    }

    if flags.pawn_shield {
        if let Some(king) = board.king_of(us) {
            let shield = attacks::king_attacks(king) & board.by_color(us) & board.pawns();
            score += SHIELD_BONUS * shield.count() as i32;
        }
    }

    if flags.mobility {
        let ours = board.by_color(us);
        let mut squares = 0i32;
        for sq in ours & !board.pawns() & !board.kings() {
            squares += (board.attacks_from(sq) & !ours).count() as i32;
        }
        score += MOBILITY_UNIT * squares;
    }

    if flags.pst {
        for sq in board.by_color(us) & !board.pawns() & !board.kings() {
            score += CENTRE[sq_index(sq, us)];
        }
        for sq in board.by_color(us) & board.pawns() {
            let rank = sq_index(sq, us) / 8;
            score += PAWN_RANK[rank];
        }
    }

    score
}

/// The atomic king-safety term.
///
/// Counts the fuses next to our king, then asks whether the opponent is
/// already holding a match. Note that the enemy king is excluded from the
/// attacker set: kings cannot capture in atomic, so an enemy king adjacent to
/// one of our pieces is not a threat to detonate it.
pub fn king_blast_penalty(board: &Board, us: Color) -> i32 {
    let Some(king) = board.king_of(us) else {
        return 0; // no king: the game is already over, the search says so
    };
    let ring = attacks::king_attacks(king);
    let fuses = ring & board.by_color(us) & !board.pawns() & !board.kings();
    if fuses.is_empty() {
        return 0;
    }

    let occupied = board.occupied();
    let mut penalty = BOMB_PENALTY * fuses.count() as i32;
    for sq in fuses {
        let attackers = board.attacks_to(sq, !us, occupied) & !board.kings();
        penalty += LIT_FUSE_PENALTY * attackers.count() as i32;
    }
    penalty
}

/// Net material swing, in centipawns, if the given capture is played.
///
/// This is the atomic replacement for MVV-LVA and for static exchange
/// evaluation. Standard chess asks "is the victim worth more than the
/// attacker"; atomic has to ask "what is left standing after the smoke
/// clears", because the capturer always dies too and so does every non-pawn
/// piece — friend or enemy — on the eight adjacent squares.
///
/// `to` is the *landing* square, which is what the blast is centred on. For en
/// passant that is not where the captured pawn stood, but the captured pawn
/// dies regardless.
pub fn blast_gain(board: &Board, us: Color, from: Option<Square>, to: Square) -> i32 {
    let mut gain = 0;

    // Everything non-pawn on the eight neighbouring squares — except the
    // capturer itself, which has left `from` by the time the blast resolves.
    // A short capture (a rook or queen moving one square) starts adjacent to
    // its own landing square, so without this the capturer is charged twice.
    let mut caught = attacks::king_attacks(to) & board.occupied() & !board.pawns();
    if let Some(from) = from {
        caught &= !Bitboard::from_square(from);
    }
    for sq in caught {
        if let Some(piece) = board.piece_at(sq) {
            let value = blast_value(piece.role);
            if piece.color == us {
                gain -= value;
            } else {
                gain += value;
            }
        }
    }

    // The victim on the landing square and the capturer itself both die. (For
    // en passant the landing square is empty and the victim pawn — worth one
    // pawn — is behind it; either way exactly one pawn is lost.)
    if let Some(victim) = board.piece_at(to) {
        gain += blast_value(victim.role);
    } else {
        gain += piece_value(Role::Pawn); // en passant
    }
    if let Some(from) = from {
        if let Some(mover) = board.piece_at(from) {
            gain -= blast_value(mover.role);
        }
    }

    gain
}
