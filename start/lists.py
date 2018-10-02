import random

a = random.sample(range(50), random.randint(3,10))
b = random.sample(range(50), random.randint(3,10))

c = [num for num in a if num in b]

print(a)
print(b)
print(c)