import random
from typing import List

from graphgps.loader.dataset.tptp import TPTPDataset


def write_indices(indices: List[int], filename: str) -> None:
    with open(filename, "w") as output_file:
        for index in indices:
            output_file.write(str(index) + "\n")


dataset = TPTPDataset("./datasets/TPTP")
indices = list(range(len(dataset)))
random.seed(777)
random.shuffle(indices)
write_indices(indices[-100:], "test.txt")
write_indices(indices[-200:-100], "valid.txt")
write_indices(indices[:-200], "train.txt")
