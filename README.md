# RookWorld

Re-implementation of https://arxiv.org/pdf/2402.04494 (focus on Figure A6)
- decoder transformer architecture
- 9M parameters (context 78 or 79, 8 heads, 8 layers, embedding dim 256)
  - maybe https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/llama2#transformers.LlamaForSequenceClassification
  - instead of not llm.c?
- Likely LR 4-e4, BS 1024 (4x TPUs V5 @ 95G)
- 40M data samples x 3.19 Epochs / BS 1024 = 125k Steps
  - 14.7% train/test overlap!
- Predictor: base BC (State-Action)
  - Also: SV (State-Value from E)
  - Limited: AV (Action-Value per Legal Move, we only have Top 5 M/E, not all)

Experiments:
- Text Generation (CLM) vs. Text Classification
- Dataset: Lichess Games + Stockfish Selfplay + Optional Puzzles + Optional COT
- Context: 77 without, ~170 with COT
- Multi-Task Tokenization (Arbiter)

Evals:
- Action Accuracy
- Puzzle Accuracy

+ Arbiter -> RookWorld