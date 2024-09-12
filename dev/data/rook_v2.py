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
#<|policy|>.r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33.+e2d3. f2f4. b5b6. a1b1. e2f3.+-1.27 -3.33 -3.43 -3.11 -3.85+e2d3


import re, json

from tqdm import tqdm


def position_padding(match, padding_char="."):
    return padding_char * int(match.group())

def process_fen(fen):
    position, turn, castling, en_passant, halfmove, fullmove = fen.split(" ")
    # pad position with "." for empty squares, remove numbers and "/"
    position = re.sub(r'\d+', position_padding, position)
    position = position.replace("/", "")
    # left pad castling with "." for 4 characters
    castling = castling.ljust(4, ".")
    # left pad en_passant with "." for 2 characters
    en_passant = en_passant.ljust(2, ".")
    # left pad halfmove with "." for 2 characters
    halfmove = halfmove.ljust(2, ".") + "."
    # left pad fullmove with "." for 3 characters
    fullmove = fullmove.ljust(3, ".")
    return "".join([position, turn, castling, en_passant, halfmove, fullmove])

def process_rook(input_string, delimiter="+", drop_moves=False, drop_evaluation=False):
    try:
        _, fen, moves, evaluation, best_move = input_string.split(":")
    except:
        print("Error processing input string:", input_string)
        return ""
    fen = process_fen(fen[:-1].strip())

    # every move will become a single token, so no padding
    moves = ".".join([m.strip() for m in moves[:-1].split()])
    
    evaluation = ".".join([e.strip().ljust(7, " ") for e in evaluation[:-1].split()]).ljust(40, " ")
    
    # every move will become a single token, so no padding
    best_move = best_move.strip()

    result = delimiter.join(
        ["<|policy|>"+fen] + 
        [moves if not drop_moves else []] + 
        [evaluation if not drop_evaluation else []] + 
        [best_move]
    )
    return result

# FEN Processing Example usage
input_string = "1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33"
result = process_fen(input_string)
print(len(result), result)

input_string = "rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3"
result = process_fen(input_string)
print(len(result), result)

# FULL DATA Processing Example usage
input_string = "P: 1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33                                          M: e2d3 f2f4 b5b6 a1b1 e2f3      E: -1.27 -3.33 -3.43 -3.11 -3.85           B: e2d3"
result = process_rook(input_string)
print(len(result), result)

input_string = "P: rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3                                M: g1f3 b1c3 c2c3 f2f4 d2d4      E: 0.8 0.71 0.65 0.58 0.98                 B: d2d4"
result = process_rook(input_string)
print(len(result), result)

print("-"*100)
#ds = "rook/lichess_train[:4000000].txt"
ds = "rook/rook_train_709k.txt"
#ds = "rook/rook_train_260k.txt"
#ds = "rook/rook_val_500.txt"
with open(ds, "r") as f:
    for line in tqdm(f.readlines()):
        line = line.strip()
        if line and len(line) > 10:
            result = process_rook(line)

