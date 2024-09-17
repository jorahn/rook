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

# YoloChess
#{"fen": "5n2/p2b1pk1/1pp2p2/6p1/1PBP1N1p/2P4P/P4PP1/6K1 w - - 0 27", "move": "d2d4", "result": "1-0", "eco": "B15"}

# ROOK v2
#.r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33.+e2d3. f2f4. b5b6. a1b1. e2f3.+-1.27 -3.33 -3.43 -3.11 -3.85+e2d3[CLS]


import re

from datasets import Dataset

from src.tokenizer.tokenizer import SPECIAL_TOKENS
from src.data.common import process_fen


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


def process_rook(input_string):
    try:
        _, fen, moves, evaluation, best_move = input_string.split(":")
    except:
        print("Error processing input string:", input_string)
        return None
    fen = process_fen(fen[:-1].strip())

    # every move will become a single token, so no padding
    moves = ".".join([m.strip() for m in moves[:-1].split()])
    
    evaluation = ".".join([e.strip().ljust(7, " ") for e in evaluation[:-1].split()]).ljust(40, " ")
    
    # every move will become a single token, so no padding
    best_move = best_move.strip()

    return (fen, moves, evaluation, best_move)

def process_yolochess(example):
    return {"text": process_fen(example["fen"]+"[CLS]", "label": example["move"]}

def make_policy_bc_data(ds, debug=False):
    # Behavior Cloning (BC) dataset

    # input sample: {"fen": "5n2/p2b1pk1/1pp2p2/6p1/1PBP1N1p/2P4P/P4PP1/6K1 w - - 0 27", "move": "d2d4", "result": "1-0", "eco": "B15"}
    
    # output dataset sample:
    # - text plain: ".r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33."

    # - label plain: "e2d3" -> one-hot-encoded best move

    # extract fen as "text" and best move as "label" from `process_rook` output
    # append CLS token to "text"

    ds = ds.map(process_yolochess)
    dict_gen = create_dict_generator(tuple_gen, config)

    ds = list(dict_gen)
    if not debug: ds = Dataset.from_list(ds)
    return ds

def make_policy_sv_data():
    raise NotImplementedError("Not implemented yet")

def make_policy_av_data():
    raise NotImplementedError("Not implemented yet")

