# import Library
import json
import difflib
from difflib import get_close_matches

data = json.load(open("./dic.json"))

# Function for retrieving definition
def retrieveDefinition(word):
    # Removing case sensitity from the word
    word = word.lower()

    # Check for non existing words
    # 1st elif To make sure the program return the definition of words that start with a capital letter
    # 2nd elif: To make sure the program return the definition of acronyms (e.g USA, NATO)
    if word in data:
        return data[word]
    elif word.title() in data:
        return data[word.title()]
    elif word.upper() in data:
        return data[word.upper()]

        # 3rd elif : To find a similar word
        # -- len > 0 because we can print only when the word has 1 or more close matches
        # -- In the return statement, the last [0] represents teh first element the list of close matches
    elif len(get_close_matches(word, data.keys())) > 0:
        action = input("Did you mean %s instead? [y or n]: " % get_close_matches(word, data.keys())[0])
        # -- if the answer is yes, retrieve definition of suggested word
        if(action == "y"):
            return data[get_close_matches(word, data.keys())[0]]
        elif(action == "n"):
            return ("The word does not exist, yet.")
        else:
            return ("we don't understand your entry. Apologies.")

# input from user
wordUser = input("Enter a word: ")

# Retrieve teh definition using function and prnt the result
output = retrieveDefinition(wordUser)

# If a word has more than one definition, print the recursively
if type(output) == list:
    for item in output:
        print("-", item)

# For words havng single definition
else:
    print("-", output) 