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
