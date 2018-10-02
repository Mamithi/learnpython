word = input("Enter a string to check if its a palindrome: ")

len = len(word) - 1

count = 0

while count <= len :
    if word[count] == word[len-count]:
        if count == len:
            print(word + " is a palindrome")
            break
    else:
        print(word + " is not a palindrome")
        break
    count+=1