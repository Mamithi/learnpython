number = int(input("Enter number to check for its divisors: "))

divisorRange = range(1, number+1)

for i in divisorRange :
    if number%i == 0:
        print(str(i) + " is " + str(number) + " divisor")
