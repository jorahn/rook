from src.utils.common import process_fen, unprocess_fen

def test_process_fen():
    # Test the function
    standard_fen = "1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33"
    custom_fen = process_fen(standard_fen)
    print(custom_fen)
    assert custom_fen == ".r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33."

def test_unprocess_fen():
    # Test the function
    custom_fen = ".r......p.....k......p..PPp...p...r...P.....B.......KP..R.......w-...-.1..33."
    standard_fen = unprocess_fen(custom_fen)
    assert standard_fen == "1r6/p5k1/5p2/PPp3p1/2r3P1/4B3/4KP2/R7 w - - 1 33"