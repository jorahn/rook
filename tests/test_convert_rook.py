from src.utils.common import process_fen, position_padding
from src.utils.convert_rook import extract_rook

import re, io

def test_position_padding():
    assert position_padding(re.match(r'\d+', "12"), ".") == "............"
    assert position_padding(re.match(r'\d+', "3"), ".") == "..."


def test_process_fen():
    input_string = "rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3"
    result = process_fen(input_string)
    assert len(result) == 77
    assert result == "rnbqkbnrpppppppp............P...................PPPP.PPPRNBQKBNRwKQkq-.1..3.."

    input_string = "1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33"
    result = process_fen(input_string)
    assert len(result) == 77
    assert result == ".r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33."

def test_extract_rook():
    input_string = "P: 1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33                                          M: e2d3 f2f4 b5b6 a1b1 e2f3      E: -1.27 -3.33 -3.43 -3.11 -3.85           B: e2d3"
    result = extract_rook(input_string)
    assert result["fen"] == "1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33"
    assert result["action"] == "e2d3"
    assert result["options"] == "e2d3 f2f4 b5b6 a1b1 e2f3".split()
    assert result["values"] == [float(n) for n in "-1.27 -3.33 -3.43 -3.11 -3.85".split()]

