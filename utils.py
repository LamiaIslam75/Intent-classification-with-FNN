import csv
from typing import Tuple, List, Set


def load_dataset(filename: str) -> Tuple[List[str], List[str], Set[str]]:
    intent = []
    unique_intent = []
    sentences = []
    with open(filename, "r", encoding="latin1") as f:
        data = csv.reader(f, delimiter=",")
        for row in data:
            sentences.append(row[0])
            intent.append(row[1])
    unique_intent = set(intent)
    return sentences, intent, unique_intent
