number = int(input("Enter the number to check if its odd or even:  "))
result = number % 2

if result == 0:
    print(str(number) + " is even number")
else:
    print(str(number) + " is odd number")
