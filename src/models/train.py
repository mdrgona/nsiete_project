import sys
from src.data.load_data import *
from src.models.model import *

sys.path.insert(0, "../")

# load data into correct format

df = load_dataset(filename='../data/Jester-Dataset-ratings.csv')
user_ids, joke_ids, ratings = get_data(df=df, batch_size=10000)

# build a model



