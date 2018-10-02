import random

randon_number = random.randint(1, 9)

guessedNumber = []

while True:
    user_guess = input("Please guess a number between 1 - 9: ")

    guessedNumber.append(user_guess)

    if user_guess == "exit":
        del guessedNumber[-1]
        break
    elif int(user_guess) > randon_number:
        print("Your number is higher than the random number")
    elif int(user_guess) < randon_number:
        print("Your number is lower than the random number")
    else:
        print("You have guessed the correct number")
        break
print("So far you have made " + str(len(guessedNumber)) + " guesses" )