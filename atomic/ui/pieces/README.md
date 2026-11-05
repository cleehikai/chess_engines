# Piece set

`cburnett`, the default Lichess set: Colin M.L. Burnett's chess pieces, taken
from [lichess-org/lila](https://github.com/lichess-org/lila/tree/master/public/piece/cburnett)
(`public/piece/cburnett`), which distributes them under **GPLv2+** — the same
terms as the originals on Wikimedia Commons.

They are vendored rather than fetched at runtime for the reason everything else
in this project is: the trainer serves from `127.0.0.1` off the stdlib
`http.server` and must work with no network at all.

Twelve files, `{w,b}{K,Q,R,B,N,P}.svg`, ~600 bytes each. `ui/app.css` maps each
one onto a `.piece.p-wN`-style class; nothing else refers to them by name, so
swapping in another set is twelve files and no code.
