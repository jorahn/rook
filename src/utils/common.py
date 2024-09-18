import re
import math


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

def unprocess_fen(custom_fen):
    # Extract components from the custom FEN
    position = custom_fen[:64]
    turn = custom_fen[64]
    castling = custom_fen[65:69].replace(".", "")
    en_passant = custom_fen[69:71].replace(".", "")
    halfmove = custom_fen[71:73].replace(".", "")
    fullmove = custom_fen[73:76].replace(".", "")

    # Process position
    rows = [position[i:i+8] for i in range(0, 64, 8)]
    processed_rows = []
    for row in rows:
        processed_row = ''
        empty_count = 0
        for char in row:
            if char == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    processed_row += str(empty_count)
                    empty_count = 0
                processed_row += char
        if empty_count > 0:
            processed_row += str(empty_count)
        processed_rows.append(processed_row)
    
    position = '/'.join(processed_rows)

    # Combine all components
    return f"{position} {turn} {castling} {en_passant} {halfmove} {fullmove}"

def transform_distribution(x):
    # Normalize to [-1, 1] range
    normalized = (x - 50) / 50
    
    # Apply sigmoid-like transformation
    transformed = math.tanh(100 * normalized)
    
    # Scale back to [0, 100] range
    result = 50 + (transformed * 50)
    
    return result

def process_cot(record, fen_column="fen", options_column="options", values_column="values", move_column="action"):
    # scale evaluations from (-999.99, 999.99) to (0, 100)
    fen = record[fen_column]
    try:
        turn = fen.split(" ")[1]
    except IndexError:
        turn = fen[64]
    turn = -1 if turn == "b" else 1
    values = [((turn * e) + 1000) / 20 for e in record[values_column]]
    values_normalized = [transform_distribution(v) for v in values]

    # convert to string
    options = " ".join(record[options_column]) # no padding, all single tokens in vocab
    values = " ".join([f"{v:.2f}".rjust(6, "-") for v in values_normalized])
    action = record[move_column]
    try:
        fen = process_fen(fen)
    except ValueError:
        pass

    return {"text": f"{fen} [OPTIONS] {options} [VALUES] {values} [ACTION] {action}"}

def process_action_value():
    # return fen+action -> value (binning)
    # Policy: Legal action with maximal predicted expected action-value.
    raise NotImplementedError("Not implemented yet")

def process_state_value():
    # return fen -> value (binning)
    # Policy: Legal action leading to next state with minimal predicted expected state-value for opponent player.
    raise NotImplementedError("Not implemented yet")
