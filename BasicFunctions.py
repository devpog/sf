friends = ['john', 'pat', 'gary', 'michael']


# Task 1
for i, name in enumerate(friends):
    print("name %d is %s" % (i, name))


# Task 2
# How many friends contain the letter 'a' ?
count_a = 0
for name in friends:
    if 'a' in str(name):
        count_a += 1
print("%f percent of the names contain an 'a'" % (count_a/len(friends)))


# Task 3
# Say hi to all friends
def print_hi(name, greeting='hello'):
    print("%s %s" % (greeting, name))
print(list(map(print_hi, friends)))


# Task 4
# Print sorted names out
print(sorted(friends))


# Task 5
"""
    Calculate the factorial N! = N * (N-1) * (N-2) * ...
"""


def factorial(x):
    """
    Calculate factorial of number
    :param N: Number to use
    :return: x!
    """
    if x < 1:
        return 1
    else:
        return x*factorial(x-1)

print("The value of 5! is", factorial(5))
