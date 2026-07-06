"""Launch the standalone GUI to play against the style engine.

    python run.py                 # play White vs balanced engine, depth 4
    python run.py --style sacrificer --depth 4 --black
    python run.py --time 5+3      # play a 5+3 blitz game with clocks

Cycle styles, change depth, pick a time control, and toggle fullscreen (F11)
live from the window. In a timed game the engine spends only a fraction of its
remaining time per move, so when low it moves fast (sacrificing depth) rather
than losing on time.
"""
from __future__ import annotations

import argparse

from engine.board import Engine
from style.profiles import STYLE_NAMES, DEFAULT_PROFILE
from ui.gui import ChessGUI, TIME_CONTROLS


# Map friendly CLI names ("5+3", "blitz") to the presets defined in the GUI.
def _time_choices() -> dict[str, object]:
    choices: dict[str, object] = {"none": None, "untimed": None}
    for tc in TIME_CONTROLS:
        if tc is None:
            continue
        base_min = int(tc.base // 60)
        key = f"{base_min}+{int(tc.increment)}"        # e.g. "5+3"
        choices[key] = tc
    return choices


def main() -> None:
    time_choices = _time_choices()
    ap = argparse.ArgumentParser(description="Play the style chess engine.")
    ap.add_argument("--style", choices=STYLE_NAMES, default=DEFAULT_PROFILE,
                    help="engine playing style")
    ap.add_argument("--depth", type=int, default=4,
                    help="search depth (higher = stronger + slower)")
    ap.add_argument("--black", action="store_true",
                    help="play as Black (engine moves first)")
    ap.add_argument("--both", action="store_true",
                    help="pass-and-play: control both sides, engine never moves "
                         "(toggleable live in the window)")
    ap.add_argument("--time", choices=sorted(time_choices), default="none",
                    help="time control, e.g. 1+0, 3+2, 5+3, 10+0, 15+10 "
                         "(default: untimed). You can also switch it in-window.")
    ap.add_argument("--nnue", default=None,
                    help="path to a trained NNUE checkpoint (.pt); omit to use "
                         "the hand-crafted evaluation")
    args = ap.parse_args()

    engine = Engine(style=args.style, nnue_checkpoint=args.nnue)
    gui = ChessGUI(engine, human_is_white=not args.black, depth=args.depth,
                   time_control=time_choices[args.time], both_sides=args.both)
    gui.run()


if __name__ == "__main__":
    main()
