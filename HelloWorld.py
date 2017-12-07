from datetime import date

name = input("Enter your name: ")
age = int(input("Enter your age: "))

print("Hello " + name + " You will turn 100yrs by " + str((100-age)+date.today().year))