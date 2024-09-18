# convert data from rook to rook v2 format
# adopt (much of the) data representation from GDM paper https://arxiv.org/pdf/2402.04494

#Board states 𝑠 are encoded as FEN strings which we convert 
#to fixed-length strings of 77 characters where the ASCII-code
#of each character is one token.

#To tokenize them we determine all possible legal actions
#across games, which is 1968, sort them alphanumerically (case-sensitive), and 
#take the action’s index as the token, meaning actions are always described by a
#single token (all details in Appendix A.1).

#The first part of a FEN string encodes the position
#of pieces rank-wise (row-wise). The only change we
#make is that we encode each empty square with a
#‘.’, which always gives us 64 characters for a board.
#The next character denotes the active player (‘w’ or
#‘b’). The next part of the FEN string denotes castling
#availability (up to four characters for King- and Queenside 
#for each color, or ‘-’ for no availability)—we take
#this string and if needed pad it with ‘.’ such that it
#always has length 4. Next are two characters for the
#en passant target, which can be ‘-’ for no target; we
#use the two characters literally or ‘-.’ for no target.
#Finally we have the halfmove clock (up to two digits)
#and the fullmove number (up to three digits); we take
#the numbers as characters and pad them with ‘.’ to
#make sure they are always tokenized into two and
#three characters respectively.

# ROOK v1
#P: 1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33                                          M: e2d3 f2f4 b5b6 a1b1 e2f3      E: -1.27 -3.33 -3.43 -3.11 -3.85           B: e2d3
#P: rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3                                M: g1f3 b1c3 c2c3 f2f4 d2d4      E: 0.8 0.71 0.65 0.58 0.98                 B: d2d4

# ROOK v2
#.r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33.+e2d3. f2f4. b5b6. a1b1. e2f3.+-1.27 -3.33 -3.43 -3.11 -3.85+e2d3[CLS]


def extract_rook(example, field="text"):
    # Behavior Cloning (BC) dataset

    # input sample: {"text": "P: 1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33 M: e2d3 f2f4 b5b6 a1b1 e2f3 E: -1.27 -3.33 -3.43 -3.11 -3.85 B: e2d3"}
    
    # output dataset sample:
    # - text plain: ".r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33."
    # - text cot:   ".r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33.[OPTIONS]e2d3.f2f4.b5b6.a1b1.e2f3[VALUES]-1.27  .-3.33  .-3.43  .-3.11  .-3.85   [ACTION]e2d3"

    # - label plain: "e2d3" -> one-hot-encoded best move
    # - label probas: probabilities for all moves (top5 evals -> rescale for current player -> softmax)

    if field and isinstance(example, dict):
        text = example[field]
    else:
        text = example

    try:
        _, fen, moves, evaluation, best_move = text.split(":")
    except:
        raise ValueError(f"Error processing input string: {text}, {type(text)}")
    
    fen = fen[:-1].strip()
    turn = -1 if fen.split(" ")[1] == "b" else 1

    moves = [m.strip() for m in moves[:-1].split()]
    if len(moves) < 5:
        moves += ["0000"] * (5 - len(moves))
    evaluations = [float(e.strip()) for e in evaluation[:-1].split()]
    if len(evaluations) < 5:
        evaluations += [-999.99*turn] * (5 - len(evaluations))
    best_move = best_move.strip()

    return {"fen": fen, "options": moves, "values": evaluations, "action": best_move}
