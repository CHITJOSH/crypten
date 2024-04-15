#import the libraries
import crypten
import torch

#initialize crypten
crypten.init()
#Disables OpenMP threads -- needed by @mpc.run_multiprocess which uses fork
torch.set_num_threads(1)

#Constructing CrypTensors with ptype attribute

#arithmetic secret-shared tensors
x_enc = crypten.cryptensor([1.0, 2.0, 3.0], ptype=crypten.mpc.arithmetic)
print("x_enc internal type:", x_enc.ptype)

#binary secret-shared tensors
y = torch.tensor([1, 2, 1], dtype=torch.int32)
y_enc = crypten.cryptensor(y, ptype=crypten.mpc.binary)
print("y_enc internal type:", y_enc.ptype)

import crypten.mpc as mpc
import crypten.communicator as comm


@mpc.run_multiprocess(world_size=2)
def examine_arithmetic_shares():
    x_enc = crypten.cryptensor([1, 2, 3], ptype=crypten.mpc.arithmetic)

    rank = comm.get().get_rank()
    crypten.print(f"\nRank {rank}:\n {x_enc}\n", in_order=True)


x = examine_arithmetic_shares()


@mpc.run_multiprocess(world_size=2)
def examine_binary_shares():
    x_enc = crypten.cryptensor([2, 3], ptype=crypten.mpc.binary)

    rank = comm.get().get_rank()
    crypten.print(f"\nRank {rank}:\n {x_enc}\n", in_order=True)


x = examine_binary_shares()

from crypten.mpc import MPCTensor


@mpc.run_multiprocess(world_size=2)
def examine_conversion():
    x = torch.tensor([1, 2, 3])
    rank = comm.get().get_rank()

    # create an MPCTensor with arithmetic secret sharing
    x_enc_arithmetic = MPCTensor(x, ptype=crypten.mpc.arithmetic)

    # To binary
    x_enc_binary = x_enc_arithmetic.to(crypten.mpc.binary)
    x_from_binary = x_enc_binary.get_plain_text()

    # print only once
    crypten.print("to(crypten.binary):")
    crypten.print(f"  ptype: {x_enc_binary.ptype}\n  plaintext: {x_from_binary}\n")

    # To arithmetic
    x_enc_arithmetic = x_enc_arithmetic.to(crypten.mpc.arithmetic)
    x_from_arithmetic = x_enc_arithmetic.get_plain_text()

    # print only once
    crypten.print("to(crypten.arithmetic):")
    crypten.print(f"  ptype: {x_enc_arithmetic.ptype}\n  plaintext: {x_from_arithmetic}\n")


z = examine_conversion()


@mpc.run_multiprocess(world_size=3)
def examine_sources():
    # Create a different tensor on each rank
    rank = comm.get().get_rank()
    x = torch.tensor(rank)
    crypten.print(f"Rank {rank}: {x}", in_order=True)

    #
    world_size = comm.get().get_world_size()
    for i in range(world_size):
        x_enc = crypten.cryptensor(x, src=i)
        z = x_enc.get_plain_text()

        # Only print from one process to avoid duplicates
        crypten.print(f"Source {i}: {z}")


x = examine_sources()

