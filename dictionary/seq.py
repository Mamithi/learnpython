import json
import difflib
# from difflib import SequenceMatcher
from difflib import get_close_matches

data = json.load(open("./dic.json"))

# value = SequenceMatcher(None, "rainn", "rain").ratio()
output = get_close_matches("rain", ["help", "mate", "rainy"], n=1, cutoff=0.75)

print(output)

