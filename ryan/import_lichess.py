from datasets import load_dataset
from itertools import islice
import pandas as pd

# Stream the Kaggle-hosted mirror on Hugging Face
ds = load_dataset("Lichess/chess-position-evaluations", split="train", streaming=True)

# Take exactly 100k examples
sample = list(islice(ds, 100_000))

# (Optional) convert to DataFrame
df = pd.DataFrame(sample)