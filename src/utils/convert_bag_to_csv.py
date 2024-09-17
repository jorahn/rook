from argparse import ArgumentParser

from tqdm import tqdm

from searchless_chess.src import bagz


def extract(source):
  """ 
  Extracts Behavior Cloning data from .bag to .csv format.
  - copy this script to the searchless_chess repository
  - activate the searchless_chess environment
  - cd into the searchless_chess repository
  - export PYTHONPATH=$(pwd)/..
  - Run `python convert_bag_to_csv.py <input.bag> <output.csv>`
  """
  for row in source:
    row_str = row.decode("utf-8")
    split = ord(row_str[0])
    yield (row_str[1:split+1], row_str[split+1:])

if __name__ == "__main__":
  parser = ArgumentParser("Extracts Behavior Cloning data from .bag to .csv format")
  parser.add_argument("filename", type=str, help="Input file name (.bag)")
  parser.add_argument("output", type=str, help="Output file name (.csv)")
  args = parser.parse_args()

  source = bagz.BagReader(args.filename)  
  with open(args.output, "w") as f:
    f.write("FEN,Move\n")
    for row in tqdm(extract(source)):
      f.write(f"{row[0]},{row[1]}\n")
    