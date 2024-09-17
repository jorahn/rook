from src.utils.common import process_fen, position_padding
from src.utils.convert_rook import process_rook, make_policy_bc_data

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

def test_process_rook():
    input_string = "P: 1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33                                          M: e2d3 f2f4 b5b6 a1b1 e2f3      E: -1.27 -3.33 -3.43 -3.11 -3.85           B: e2d3"
    result = process_rook(input_string)
    result = "+".join(result)
    assert len(result) == 148
    assert result == ".r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33.+e2d3.f2f4.b5b6.a1b1.e2f3+-1.27  .-3.33  .-3.43  .-3.11  .-3.85   +e2d3"

    input_string = "P: rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3                                M: g1f3 b1c3 c2c3 f2f4 d2d4      E: 0.8 0.71 0.65 0.58 0.98                 B: d2d4"
    result = process_rook(input_string)
    result = "+".join(result)
    assert len(result) == 148
    assert result == "rnbqkbnrpppppppp............P...................PPPP.PPPRNBQKBNRwKQkq-.1..3..+g1f3.b1c3.c2c3.f2f4.d2d4+0.8    .0.71   .0.65   .0.58   .0.98    +d2d4"

def test_make_policy_bc_data_plain():
    inputs = io.StringIO("P: 1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33                                          M: e2d3 f2f4 b5b6 a1b1 e2f3      E: -1.27 -3.33 -3.43 -3.11 -3.85           B: e2d3\nP: rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3                                M: g1f3 b1c3 c2c3 f2f4 d2d4      E: 0.8 0.71 0.65 0.58 0.98                 B: d2d4")
    dataset = make_policy_bc_data(inputs, debug=True)
    print(dataset)
    assert len(dataset) == 2
    assert dataset[0] == {"text": ".r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33.[CLS]", "label": "e2d3"}
    assert dataset[1] == {"text": "rnbqkbnrpppppppp............P...................PPPP.PPPRNBQKBNRwKQkq-.1..3..[CLS]", "label": "d2d4"}

def test_make_policy_bc_data_cot():
    inputs = io.StringIO("P: 1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33                                          M: e2d3 f2f4 b5b6 a1b1 e2f3      E: -1.27 -3.33 -3.43 -3.11 -3.85           B: e2d3\nP: rnbqkbnr/pppppppp/8/4P3/8/8/PPPP1PPP/RNBQKBNR w KQkq - 1 3                                M: g1f3 b1c3 c2c3 f2f4 d2d4      E: 0.8 0.71 0.65 0.58 0.98                 B: d2d4")
    dataset = make_policy_bc_data(inputs, cot=True, debug=True)
    assert len(dataset) == 2
    assert dataset[0] == {"text": ".r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33.+e2d3.f2f4.b5b6.a1b1.e2f3+-1.27  .-3.33  .-3.43  .-3.11  .-3.85   [CLS]", "label": "e2d3"}
    assert dataset[1] == {"text": "rnbqkbnrpppppppp............P...................PPPP.PPPRNBQKBNRwKQkq-.1..3..+g1f3.b1c3.c2c3.f2f4.d2d4+0.8    .0.71   .0.65   .0.58   .0.98    [CLS]", "label": "d2d4"}