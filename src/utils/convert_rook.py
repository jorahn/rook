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


import re

from datasets import Dataset

from src.utils.common import process_fen


def create_tuple_to_dict_converter(config):
    """
    extract values from tuple based on config
    allow processing of values with functions

    # Configuration 1: Simple mapping
    config1 = {
        "name": 0,
        "age": 1,
        "city": 2
    }

    # Configuration 2: Dropping a field and combining others
    config2 = {
        "full_name": lambda first, last, *rest: f"{first} {last}",
        "location": 2  # Assuming city is at index 2
    }
    """
    def converter(tuple_item):
        result = {}
        for key, value in config.items():
            if callable(value):
                result[key] = value(*tuple_item)
            elif isinstance(value, int):
                result[key] = tuple_item[value]
        return result
    return converter

def create_dict_generator(tuple_generator, config):
    """ create a generator that converts tuples to dictionaries """
    converter = create_tuple_to_dict_converter(config)
    return (converter(t) for t in tuple_generator if t)

def yield_proc_lines(f, proc_fn):
    """ iterate over lines in file-object, process them and yield """
    for line in f:
        yield proc_fn(line.strip())


def extract_rook(example, field="text"):
    if field:
        text = example[field]
    else:
        text = example

    try:
        _, fen, moves, evaluation, best_move = text.split(":")
    except:
        print("Error processing input string:", text)
        return None
    fen = process_fen(fen[:-1].strip())

    # every move will become a single token, so no padding
    moves = [m.strip() for m in moves[:-1].split()]
    
    evaluations = [float(e.strip()) for e in evaluation[:-1].split()]
    
    # every move will become a single token, so no padding
    best_move = best_move.strip()

    result = {"fen": fen, "options": moves, "values": evaluations, "action": best_move}
    return result

def process_cot(record):
    # scale evaluations from (-999.99, 999.99) to (0, 100)
    fen = record["fen"]
    turn = fen.split(" ")[1]
    turn = -1 if turn == "b" else 1
    values = [((turn * e) + 1000) / 20 for e in record["values"]]
    
    # convert to string
    options = ".".join(record["options"]) # no padding, all single tokens in vocab
    values = ".".join([f"{v:.2f}".ljust(5) for v in values])
    action = record["action"]

    return f"{fen}[OPTIONS]{options}[VALUES]{values}[ACTION]{action}"


def make_policy_bc_data(input_file_obj, cot=False, probas=False):
    # Behavior Cloning (BC) dataset

    # input sample: "P: 1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33 M: e2d3 f2f4 b5b6 a1b1 e2f3 E: -1.27 -3.33 -3.43 -3.11 -3.85 B: e2d3"
    
    # output dataset sample:
    # - text plain: ".r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33."
    # - text cot:   ".r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33.[OPTIONS]e2d3.f2f4.b5b6.a1b1.e2f3[VALUES]-1.27  .-3.33  .-3.43  .-3.11  .-3.85   [ACTION]e2d3"

    # - label plain: "e2d3" -> one-hot-encoded best move
    # - label probas: probabilities for all moves (top5 evals -> rescale for current player -> softmax)

    gen_step1 = yield_proc_lines(input_file_obj, extract_rook)

    # extract fen as "text" and best move as "label" from `extract_rook` output
    # append CLS token to "text"

    config = {
        "text": lambda x: x['fen']+"[CLS]",
        "label": lambda x: x["action"],
    }

    if cot:
        # only for LM task, not for CLF
        config = {
            "text": lambda x: process_cot(*x),
        }

    if probas:
        # based on current player (w|b) from FEN
        # rescale top 5 moves & evals to [0, 100] (worst, best)
        # add all remaining possible moves with 0 (worst)
        # and apply softmax
        raise NotImplementedError("Probas not implemented yet")

    dict_gen = create_dict_generator(gen_step1, config)

    # TODO avoid list conversion, use generator
    # ds = Dataset.from_generator(dict_gen) -> cannot pickle
    ds = list(dict_gen)
    return Dataset.from_list(ds)

def make_policy_sv_data():
    raise NotImplementedError("Not implemented yet")

def make_policy_av_data():
    raise NotImplementedError("Not implemented yet")

