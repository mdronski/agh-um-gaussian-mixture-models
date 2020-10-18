import os

DATASET_NAMES = [
    't4.8k.txt',
    's1.txt',
    'a1.txt',
    'birch1.txt',
    'birch2.txt',
    'birch3.txt',
    'unbalance.txt',
    'wine.txt',
    'breast.txt',
    'iris.txt',
    'Aggregation.txt'
]

RESOURCES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "resources"
)

DATASET_PATH = os.path.join(RESOURCES_DIR, 'datasets')