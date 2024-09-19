echo "Downloading Big-Bench Checkmate in One task"
wget -nc -O "src/data/checkmate.json" "https://github.com/google/BIG-bench/raw/main/bigbench/benchmark_tasks/checkmate_in_one/task.json"

echo "Downloading Puzzle Test Data from GDM paper"
wget -nc -O "src/data/searchless_puzzles.csv" "https://storage.googleapis.com/searchless_chess/data/puzzles.csv"

echo "Downloading Lichess Puzzle Dataset"
wget -nc -O "src/data/lichess_db_puzzle.csv.zst" "https://database.lichess.org/lichess_db_puzzle.csv.zst"


# Behavioral Cloning Dataset from GDM paper (34GB!)
# after downloading use src/utils/convert_bag_to_csv.py to convert .bag to .csv
# this requires cloning the https://github.com/google-deepmind/searchless_chess repo and installing the necessary dependencies

# wget -nc -O "src/data/searchless_bc_train.bag" "https://storage.googleapis.com/searchless_chess/data/train/behavioral_cloning_data.bag"
# wget -nc -O "src/data/searchless_bc_test.bag" "https://storage.googleapis.com/searchless_chess/data/test/behavioral_cloning_data.bag"
