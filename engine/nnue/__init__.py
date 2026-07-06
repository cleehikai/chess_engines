"""NNUE (Efficiently Updatable Neural Network) evaluation — the strength track.

Modules:
  features.py  — HalfKP feature encoding (board -> sparse indices)
  model.py     — the PyTorch network (feature transformer + small dense head)
  gen_data.py  — generate positions and label them with Stockfish
  train.py     — device-agnostic trainer (CPU now, CUDA on the 5070)
  infer.py     — fast NumPy inference for use inside search
  quantize.py  — export trained weights for infer.py
"""
