# import the libraries
import crypten
import torch

# initialize crypten
crypten.init()
# Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)

# Constructing CrypTensors with ptype attribute

# arithmetic secret-shared tensors
x_enc = crypten.cryptensor([1.0, 2.0, 3.0], ptype=crypten.mpc.arithmetic)
print("x_enc internal type:", x_enc.ptype)

# binary secret-shared tensors
y = torch.tensor([1, 2, 1], dtype=torch.int32)
y_enc = crypten.cryptensor(y, ptype=crypten.mpc.binary)
print("y_enc internal type:", y_enc.ptype)

from arithmetic_shares import examine_arithmetic_shares

x = examine_arithmetic_shares()
