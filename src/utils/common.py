import re

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
