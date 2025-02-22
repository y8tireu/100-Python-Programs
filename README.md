# 100 Python Programs

---

## Program 1: Hello World
```python
print("Hello, World!")
```

---

## Program 2: Sum of Two Numbers
```python
a = 5
b = 7
print("Sum:", a + b)
```

---

## Program 3: Factorial Program
```python
def factorial(n):
    return 1 if n == 0 else n * factorial(n-1)

num = 5
print("Factorial of", num, "is", factorial(num))
```

---

## Program 4: Fibonacci Series
```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a, end=" ")
        a, b = b, a + b

fibonacci(10)
```

---

## Program 5: Prime Number Checker
```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

num = 29
print(num, "is prime?" , is_prime(num))
```

---

## Program 6: Reverse a String
```python
s = "Python"
print("Reversed:", s[::-1])
```

---

## Program 7: Palindrome Checker
```python
def is_palindrome(s):
    return s == s[::-1]

word = "radar"
print(word, "is palindrome?", is_palindrome(word))
```

---

## Program 8: Find Largest Number in a List
```python
numbers = [3, 67, 2, 89, 34]
print("Largest number:", max(numbers))
```

---

## Program 9: Find Smallest Number in a List
```python
numbers = [3, 67, 2, 89, 34]
print("Smallest number:", min(numbers))
```

---

## Program 10: Even or Odd
```python
num = 42
print(num, "is", "Even" if num % 2 == 0 else "Odd")
```

---

## Program 11: List Comprehension Example
```python
squares = [x**2 for x in range(10)]
print("Squares:", squares)
```

---

## Program 12: Bubble Sort Implementation
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

print("Sorted:", bubble_sort([64, 34, 25, 12, 22, 11, 90]))
```

---

## Program 13: Merge Sort Implementation
```python
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]
            i += 1; k += 1
        while j < len(R):
            arr[k] = R[j]
            j += 1; k += 1
    return arr

print("Merge Sorted:", merge_sort([38, 27, 43, 3, 9, 82, 10]))
```

---

## Program 14: Binary Search Implementation
```python
def binary_search(arr, target):
    low, high = 0, len(arr)-1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

arr = [1, 3, 5, 7, 9, 11]
print("Index of 7:", binary_search(arr, 7))
```

---

## Program 15: Tower of Hanoi
```python
def tower_of_hanoi(n, source, auxiliary, target):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    tower_of_hanoi(n-1, source, target, auxiliary)
    print(f"Move disk {n} from {source} to {target}")
    tower_of_hanoi(n-1, auxiliary, source, target)

tower_of_hanoi(3, 'A', 'B', 'C')
```

---

## Program 16: Sum of Digits of a Number
```python
num = 12345
total = sum(int(digit) for digit in str(num))
print("Sum of digits:", total)
```

---

## Program 17: Count Vowels in a String
```python
def count_vowels(s):
    vowels = "aeiouAEIOU"
    return sum(1 for char in s if char in vowels)

print("Vowels count:", count_vowels("Hello World"))
```

---

## Program 18: Generate a Random Number
```python
import random
print("Random number (1-100):", random.randint(1, 100))
```

---

## Program 19: Calculate GCD
```python
import math
a, b = 48, 180
print("GCD:", math.gcd(a, b))
```

---

## Program 20: Calculate LCM
```python
def lcm(a, b):
    import math
    return abs(a*b) // math.gcd(a, b)

print("LCM:", lcm(12, 15))
```

---

## Program 21: Check Armstrong Number
```python
def is_armstrong(n):
    digits = [int(d) for d in str(n)]
    power = len(digits)
    return n == sum(d ** power for d in digits)

num = 153
print(num, "is Armstrong?", is_armstrong(num))
```

---

## Program 22: Check Perfect Number
```python
def is_perfect(n):
    return n == sum(i for i in range(1, n) if n % i == 0)

num = 28
print(num, "is perfect?", is_perfect(num))
```

---

## Program 23: Simple Calculator
```python
def calculator(a, b, op):
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        return a / b
    else:
        return "Invalid operator"

print("Calculator:", calculator(10, 5, '+'))
```

---

## Program 24: Temperature Converter (Celsius to Fahrenheit)
```python
def c_to_f(celsius):
    return (celsius * 9/5) + 32

print("25°C =", c_to_f(25), "°F")
```

---

## Program 25: Currency Converter (Simple: USD to EUR)
```python
def usd_to_eur(usd, rate=0.85):
    return usd * rate

print("$100 =", usd_to_eur(100), "EUR")
```

---

## Program 26: Factorization
```python
def factors(n):
    return [i for i in range(1, n+1) if n % i == 0]

print("Factors of 28:", factors(28))
```

---

## Program 27: Check Leap Year
```python
def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

print("2024 is leap year?", is_leap_year(2024))
```

---

## Program 28: Find Area of a Circle
```python
import math
def area_circle(radius):
    return math.pi * radius ** 2

print("Area of circle with radius 5:", area_circle(5))
```

---

## Program 29: Find Area of a Triangle (Heron’s Formula)
```python
import math
def area_triangle(a, b, c):
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))

print("Area of triangle (3, 4, 5):", area_triangle(3, 4, 5))
```

---

## Program 30: Even or Odd Using Lambda
```python
even_or_odd = lambda x: "Even" if x % 2 == 0 else "Odd"
print("17 is", even_or_odd(17))
```

---

## Program 31: Using Map to Square Numbers
```python
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
print("Squared:", squared)
```

---

## Program 32: Using Filter to Filter Even Numbers
```python
numbers = [1, 2, 3, 4, 5, 6]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print("Even numbers:", evens)
```

---

## Program 33: Using Reduce to Multiply List Items
```python
from functools import reduce
numbers = [1, 2, 3, 4]
product = reduce(lambda x, y: x * y, numbers)
print("Product:", product)
```

---

## Program 34: Convert List to Dictionary
```python
keys = ['a', 'b', 'c']
values = [1, 2, 3]
result = dict(zip(keys, values))
print("Dictionary:", result)
```

---

## Program 35: Sort Dictionary by Value
```python
data = {'apple': 3, 'banana': 1, 'cherry': 2}
sorted_data = dict(sorted(data.items(), key=lambda item: item[1]))
print("Sorted dictionary:", sorted_data)
```

---

## Program 36: Count Words in a File
```python
def count_words(filename):
    with open(filename, 'r') as f:
        text = f.read()
    words = text.split()
    return len(words)

# Usage: print("Word count:", count_words("example.txt"))
```

---

## Program 37: Read File and Display Content
```python
def read_file(filename):
    with open(filename, 'r') as f:
        print(f.read())

# Usage: read_file("example.txt")
```

---

## Program 38: Write to a File
```python
def write_file(filename, content):
    with open(filename, 'w') as f:
        f.write(content)

# Usage: write_file("output.txt", "Hello, file!")
```

---

## Program 39: Append to a File
```python
def append_file(filename, content):
    with open(filename, 'a') as f:
        f.write(content + "\n")

# Usage: append_file("output.txt", "Appending new line")
```

---

## Program 40: Delete a File
```python
import os
def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(filename, "deleted")
    else:
        print("File does not exist")

# Usage: delete_file("output.txt")
```

---

## Program 41: Create a Directory
```python
import os
def create_directory(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        print("Directory created")
    else:
        print("Directory already exists")

# Usage: create_directory("new_folder")
```

---

## Program 42: List Files in a Directory
```python
import os
def list_files(dirname):
    files = os.listdir(dirname)
    print("Files:", files)

# Usage: list_files(".")
```

---

## Program 43: Rename a File
```python
import os
def rename_file(old_name, new_name):
    if os.path.exists(old_name):
        os.rename(old_name, new_name)
        print("File renamed")
    else:
        print("File not found")

# Usage: rename_file("old.txt", "new.txt")
```

---

## Program 44: Using Try/Except for Error Handling
```python
try:
    result = 10 / 0
except ZeroDivisionError:
    result = "Cannot divide by zero"
print(result)
```

---

## Program 45: Using Custom Exceptions
```python
class MyError(Exception):
    pass

def check_value(x):
    if x < 0:
        raise MyError("Negative value!")
    return x

try:
    print(check_value(-5))
except MyError as e:
    print("Error:", e)
```

---

## Program 46: Simple Class and Object
```python
class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        print("Hello, my name is", self.name)

p = Person("Alice")
p.greet()
```

---

## Program 47: Inheritance Example
```python
class Animal:
    def speak(self):
        print("Animal sound")

class Dog(Animal):
    def speak(self):
        print("Woof!")

d = Dog()
d.speak()
```

---

## Program 48: Polymorphism Example
```python
class Bird:
    def sound(self):
        print("Tweet")

class Cat:
    def sound(self):
        print("Meow")

def make_sound(animal):
    animal.sound()

make_sound(Bird())
make_sound(Cat())
```

---

## Program 49: Encapsulation Example
```python
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance

    def deposit(self, amount):
        self.__balance += amount

    def get_balance(self):
        return self.__balance

account = BankAccount(100)
account.deposit(50)
print("Balance:", account.get_balance())
```

---

## Program 50: Abstraction Example
```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side * self.side

s = Square(4)
print("Area of square:", s.area())
```

---

## Program 51: Fibonacci Using Recursion
```python
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)

for i in range(10):
    print(fib(i), end=" ")
```

---

## Program 52: Binary to Decimal Conversion
```python
def binary_to_decimal(binary):
    return int(binary, 2)

print("Decimal of 1010:", binary_to_decimal("1010"))
```

---

## Program 53: Decimal to Binary Conversion
```python
def decimal_to_binary(n):
    return bin(n).replace("0b", "")

print("Binary of 10:", decimal_to_binary(10))
```

---

## Program 54: Matrix Addition
```python
def add_matrices(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

A = [[1,2],[3,4]]
B = [[5,6],[7,8]]
print("Matrix Sum:", add_matrices(A, B))
```

---

## Program 55: Matrix Multiplication
```python
def multiply_matrices(A, B):
    result = [[0]*len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

A = [[1,2],[3,4]]
B = [[2,0],[1,2]]
print("Matrix Product:", multiply_matrices(A, B))
```

---

## Program 56: Transpose of a Matrix
```python
def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

matrix = [[1,2,3], [4,5,6]]
print("Transpose:", transpose(matrix))
```

---

## Program 57: Check if Matrix is Symmetric
```python
def is_symmetric(matrix):
    return matrix == [list(row) for row in zip(*matrix)]

matrix = [[1, 2, 3], [2, 4, 5], [3, 5, 6]]
print("Symmetric?", is_symmetric(matrix))
```

---

## Program 58: Create a Multiplication Table
```python
def multiplication_table(n):
    for i in range(1, n+1):
        print(" ".join(f"{i*j:3}" for j in range(1, n+1)))

multiplication_table(5)
```

---

## Program 59: Count Frequency of Elements in a List
```python
def count_frequency(lst):
    freq = {}
    for item in lst:
        freq[item] = freq.get(item, 0) + 1
    return freq

print("Frequency:", count_frequency([1,2,2,3,3,3]))
```

---

## Program 60: Convert JSON to Python Dict
```python
import json
json_str = '{"name": "Alice", "age": 30}'
data = json.loads(json_str)
print("Dictionary:", data)
```

---

## Program 61: Convert Python Dict to JSON
```python
import json
data = {"name": "Bob", "age": 25}
json_str = json.dumps(data)
print("JSON:", json_str)
```

---

## Program 62: Using Regular Expressions
```python
import re
text = "The rain in Spain"
matches = re.findall(r"\bS\w+", text)
print("Matches:", matches)
```

---

## Program 63: Basic Web Scraping
```python
import requests
from bs4 import BeautifulSoup

url = "https://www.example.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
print("Page Title:", soup.title.string)
```

---

## Program 64: API Call Using Requests
```python
import requests
url = "https://api.github.com"
response = requests.get(url)
print("Status Code:", response.status_code)
```

---

## Program 65: Generate a Random Password
```python
import random
import string

def random_password(length=8):
    characters = string.ascii_letters + string.digits + string.punctuation
    return ''.join(random.choice(characters) for _ in range(length))

print("Random Password:", random_password(12))
```

---

## Program 66: Find Duplicates in a List
```python
def find_duplicates(lst):
    seen = set()
    duplicates = set()
    for item in lst:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)

print("Duplicates:", find_duplicates([1,2,3,2,4,5,1]))
```

---

## Program 67: Remove Duplicates from a List
```python
def remove_duplicates(lst):
    return list(dict.fromkeys(lst))

print("Unique List:", remove_duplicates([1,2,3,2,4,5,1]))
```

---

## Program 68: Merge Two Lists
```python
list1 = [1, 2, 3]
list2 = [4, 5, 6]
merged = list1 + list2
print("Merged List:", merged)
```

---

## Program 69: Find Intersection of Two Lists
```python
def intersection(list1, list2):
    return list(set(list1) & set(list2))

print("Intersection:", intersection([1,2,3,4], [3,4,5,6]))
```

---

## Program 70: Flatten a Nested List
```python
def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

print("Flattened:", flatten([[1,2], [3,4], [5]]))
```

---

## Program 71: List Comprehension with Condition
```python
nums = [1, 2, 3, 4, 5, 6]
evens = [x for x in nums if x % 2 == 0]
print("Even numbers:", evens)
```

---

## Program 72: Dictionary Comprehension with Condition
```python
numbers = range(10)
squares = {n: n**2 for n in numbers if n % 2 == 0}
print("Even Squares:", squares)
```

---

## Program 73: Set Operations
```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}
print("Union:", set1 | set2)
print("Intersection:", set1 & set2)
print("Difference:", set1 - set2)
```

---

## Program 74: Create a Decorator
```python
def decorator(func):
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@decorator
def greet(name):
    print("Hello", name)

greet("Alice")
```

---

## Program 75: Create a Generator Function
```python
def my_generator(n):
    for i in range(n):
        yield i * i

for val in my_generator(5):
    print(val)
```

---

## Program 76: Fibonacci Using Generator
```python
def fib_generator(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print("Fibonacci:", list(fib_generator(10)))
```

---

## Program 77: Permutations Using Itertools
```python
import itertools
items = [1, 2, 3]
perms = list(itertools.permutations(items))
print("Permutations:", perms)
```

---

## Program 78: Combinations Using Itertools
```python
import itertools
items = [1, 2, 3, 4]
combs = list(itertools.combinations(items, 2))
print("Combinations:", combs)
```

---

## Program 79: Count Frequency Using Counter
```python
from collections import Counter
lst = [1,2,2,3,3,3]
freq = Counter(lst)
print("Frequency:", freq)
```

---

## Program 80: Using Collections NamedTuple
```python
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print("Point:", p)
```

---

## Program 81: Basic Plotting with Matplotlib
```python
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [10,20,25,30])
plt.title("Simple Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

---

## Program 82: Simple Bar Chart with Matplotlib
```python
import matplotlib.pyplot as plt
categories = ['A', 'B', 'C']
values = [10, 20, 15]
plt.bar(categories, values)
plt.title("Bar Chart")
plt.show()
```

---

## Program 83: Plot Sine Wave with Numpy and Matplotlib
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Sine Wave")
plt.show()
```

---

## Program 84: Linear Regression Using scikit-learn
```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 6, 8])
model = LinearRegression().fit(X, y)
print("Coefficient:", model.coef_[0])
```

---

## Program 85: Decision Tree Classifier Using scikit-learn
```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

X = np.array([[0, 0], [1, 1]])
y = [0, 1]
clf = DecisionTreeClassifier().fit(X, y)
print("Prediction for [2,2]:", clf.predict([[2, 2]]))
```

---

## Program 86: KMeans Clustering Using scikit-learn
```python
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1,2], [1,4], [1,0],
              [10,2], [10,4], [10,0]])
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print("Cluster centers:", kmeans.cluster_centers_)
```

---

## Program 87: Basic Tkinter Window
```python
import tkinter as tk
root = tk.Tk()
root.title("Simple Window")
label = tk.Label(root, text="Hello Tkinter!")
label.pack()
root.mainloop()
```

---

## Program 88: Tkinter Button Click Event
```python
import tkinter as tk

def on_click():
    label.config(text="Button Clicked!")

root = tk.Tk()
root.title("Button Click")
label = tk.Label(root, text="Waiting...")
label.pack()
button = tk.Button(root, text="Click Me", command=on_click)
button.pack()
root.mainloop()
```

---

## Program 89: Simple GUI Calculator with Tkinter
```python
import tkinter as tk

def calculate():
    try:
        result = eval(entry.get())
        label.config(text="Result: " + str(result))
    except Exception as e:
        label.config(text="Error")

root = tk.Tk()
root.title("Calculator")
entry = tk.Entry(root, width=20)
entry.pack()
button = tk.Button(root, text="Calculate", command=calculate)
button.pack()
label = tk.Label(root, text="Result:")
label.pack()
root.mainloop()
```

---

## Program 90: Simple Web Server Using Flask
```python
from flask import Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from Flask!"

if __name__ == "__main__":
    app.run(debug=True)
```

---

## Program 91: Simple REST API Using Flask
```python
from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route("/api", methods=["GET"])
def api():
    data = {"message": "Hello API"}
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
```

---

## Program 92: Download a File Using Requests
```python
import requests
url = "https://www.example.com"
response = requests.get(url)
with open("downloaded_page.html", "w", encoding="utf-8") as f:
    f.write(response.text)
print("File downloaded.")
```

---

## Program 93: Upload a File Using Requests
```python
import requests
url = "http://httpbin.org/post"
files = {'file': open('downloaded_page.html', 'rb')}
response = requests.post(url, files=files)
print("Response:", response.text)
```

---

## Program 94: Sending Email Using smtplib
```python
import smtplib
from email.mime.text import MIMEText

msg = MIMEText("This is a test email.")
msg['Subject'] = "Test"
msg['From'] = "sender@example.com"
msg['To'] = "receiver@example.com"

# Uncomment and configure the following lines to send email
# with smtplib.SMTP('smtp.example.com', 587) as server:
#     server.starttls()
#     server.login("username", "password")
#     server.send_message(msg)
print("Email prepared (sending disabled for safety).")
```

---

## Program 95: Reading CSV File Using csv Module
```python
import csv
with open('data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        print(row)
```

---

## Program 96: Writing CSV File Using csv Module
```python
import csv
data = [["Name", "Age"], ["Alice", 30], ["Bob", 25]]
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)
print("CSV written.")
```

---

## Program 97: Command Line Argument Parsing Using argparse
```python
import argparse

parser = argparse.ArgumentParser(description="Simple Argument Parser")
parser.add_argument("--name", type=str, help="Your name", required=True)
args = parser.parse_args()
print("Hello,", args.name)
```

---

## Program 98: Countdown Timer Using time Module
```python
import time

def countdown(seconds):
    while seconds:
        mins, secs = divmod(seconds, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        seconds -= 1
    print("Time's up!")

countdown(10)
```

---

## Program 99: Simple Multithreading Example
```python
import threading
import time

def print_numbers():
    for i in range(5):
        print(i)
        time.sleep(1)

thread = threading.Thread(target=print_numbers)
thread.start()
thread.join()
```

---

## Program 100: Simple Multiprocessing Example
```python
from multiprocessing import Process
import time

def worker():
    print("Worker process started")
    time.sleep(2)
    print("Worker process finished")

p = Process(target=worker)
p.start()
p.join()
```
