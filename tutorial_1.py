import crypten
import torch

crypten.init()

# Create torch tensor
x = torch.tensor([1.0, 2.0, 3.0])

# Encrypt x
x_enc = crypten.cryptensor(x)

# Decrypt x
x_dec = x_enc.get_plain_text()
crypten.print(x_dec)


# Create python list
y = [4.0, 5.0, 6.0]

# Encrypt x
y_enc = crypten.cryptensor(y)

# Decrypt x
y_dec = y_enc.get_plain_text()
crypten.print(y_dec)

#Arithmetic operations between CrypTensors and plaintext tensors
x_enc = crypten.cryptensor([1.0, 2.0, 3.0])

y = 2.0
y_enc = crypten.cryptensor(2.0)


# Addition
z_enc1 = x_enc + y      # Public
z_enc2 = x_enc + y_enc  # Private
crypten.print("\nPublic  addition:", z_enc1.get_plain_text())
crypten.print("Private addition:", z_enc2.get_plain_text())


# Subtraction
z_enc1 = x_enc - y      # Public
z_enc2 = x_enc - y_enc  # Private
crypten.print("\nPublic  subtraction:", z_enc1.get_plain_text())
print("Private subtraction:", z_enc2.get_plain_text())

# Multiplication
z_enc1 = x_enc * y      # Public
z_enc2 = x_enc * y_enc  # Private
print("\nPublic  multiplication:", z_enc1.get_plain_text())
print("Private multiplication:", z_enc2.get_plain_text())

# Division
z_enc1 = x_enc / y      # Public
z_enc2 = x_enc / y_enc  # Private
print("\nPublic  division:", z_enc1.get_plain_text())
print("Private division:", z_enc2.get_plain_text())

#Construct two example CrypTensors
x_enc = crypten.cryptensor([1.0, 2.0, 3.0, 4.0, 5.0])

y = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
y_enc = crypten.cryptensor(y)

# Print values:
print("x: ", x_enc.get_plain_text())
print("y: ", y_enc.get_plain_text())

# Less than
z_enc1 = x_enc < y      # Public
z_enc2 = x_enc < y_enc  # Private
print("\nPublic  (x < y) :", z_enc1.get_plain_text())
print("Private (x < y) :", z_enc2.get_plain_text())

# Less than or equal
z_enc1 = x_enc <= y      # Public
z_enc2 = x_enc <= y_enc  # Private
print("\nPublic  (x <= y):", z_enc1.get_plain_text())
print("Private (x <= y):", z_enc2.get_plain_text())

# Greater than
z_enc1 = x_enc > y      # Public
z_enc2 = x_enc > y_enc  # Private
print("\nPublic  (x > y) :", z_enc1.get_plain_text())
print("Private (x > y) :", z_enc2.get_plain_text())

# Greater than or equal
z_enc1 = x_enc >= y      # Public
z_enc2 = x_enc >= y_enc  # Private
print("\nPublic  (x >= y):", z_enc1.get_plain_text())
print("Private (x >= y):", z_enc2.get_plain_text())

# Equal
z_enc1 = x_enc == y      # Public
z_enc2 = x_enc == y_enc  # Private
print("\nPublic  (x == y):", z_enc1.get_plain_text())
print("Private (x == y):", z_enc2.get_plain_text())

# Not Equal
z_enc1 = x_enc != y      # Public
z_enc2 = x_enc != y_enc  # Private
print("\nPublic  (x != y):", z_enc1.get_plain_text())
print("Private (x != y):", z_enc2.get_plain_text())


torch.set_printoptions(sci_mode=False)

#Construct example input CrypTensor
x = torch.tensor([0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5])
x_enc = crypten.cryptensor(x)

# Reciprocal
z = x.reciprocal()          # Public
z_enc = x_enc.reciprocal()  # Private
print("\nPublic  reciprocal:", z)
print("Private reciprocal:", z_enc.get_plain_text())

# Logarithm
z = x.log()          # Public
z_enc = x_enc.log()  # Private
print("\nPublic  logarithm:", z)
print("Private logarithm:", z_enc.get_plain_text())

# Exp
z = x.exp()          # Public
z_enc = x_enc.exp()  # Private
print("\nPublic  exponential:", z)
print("Private exponential:", z_enc.get_plain_text())

# Sqrt
z = x.sqrt()          # Public
z_enc = x_enc.sqrt()  # Private
print("\nPublic  square root:", z)
print("Private square root:", z_enc.get_plain_text())

# Tanh
z = x.tanh()          # Public
z_enc = x_enc.tanh()  # Private
print("\nPublic  tanh:", z)
print("Private tanh:", z_enc.get_plain_text())

x_enc = crypten.cryptensor(2.0)
y_enc = crypten.cryptensor(4.0)

a, b = 2, 3

# Normal Control-flow code will raise an error
try:
    if x_enc < y_enc:
        z = a
    else:
        z = b
except RuntimeError as error:
    print(f"RuntimeError caught: \"{error}\"\n")

# Instead use a mathematical expression
use_a = (x_enc < y_enc)
z_enc = use_a * a + (1 - use_a) * b
print("z:", z_enc.get_plain_text())

# Or use the `where` function
z_enc = crypten.where(x_enc < y_enc, a, b)
print("z:", z_enc.get_plain_text())

x_enc = crypten.cryptensor([1.0, 2.0, 3.0])
y_enc = crypten.cryptensor([4.0, 5.0, 6.0])

# Indexing
z_enc = x_enc[:-1]
print("Indexing:\n", z_enc.get_plain_text())

# Concatenation
z_enc = crypten.cat([x_enc, y_enc])
print("\nConcatenation:\n", z_enc.get_plain_text())

# Stacking
z_enc = crypten.stack([x_enc, y_enc])
print('\nStacking:\n', z_enc.get_plain_text())

# Reshaping
w_enc = z_enc.reshape(-1, 6)
print('\nReshaping:\n', w_enc.get_plain_text())