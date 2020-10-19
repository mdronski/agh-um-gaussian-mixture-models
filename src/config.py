import os

DATASET_NAMES = [
    't4.8k.txt',
    's1.txt',
    'unbalance.txt',
    'wine.txt',
    'breast.txt',
    'iris.txt',
    'dim064.txt',
    'housec5.txt'
]

BEST_COMPONENT_NUMBER = {
    't4': 6,
    's1': 15,
    'unbalance': 8,
    'wine': 3,
    'breast': 2,
    'iris': 2,
}

RESOURCES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "resources"
)

DATASET_PATH = os.path.join(RESOURCES_DIR, 'datasets')