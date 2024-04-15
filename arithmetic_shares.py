# shared_functions.py
import crypten as crypt
import crypten.mpc as mpc1
import crypten.communicator as comm1


@mpc1.run_multiprocess(world_size=2)
def examine_arithmetic_shares():
    x_enc = crypt.cryptensor([1, 2, 3], ptype=crypt.mpc.arithmetic)
    rank = comm1.get().get_rank()
    crypt.print(f"\nRank {rank}:\n {x_enc}\n", in_order=True)
