from typing import List
import random
from phe import paillier
from crypten import CrypTensor
import torch

class MPCTensor(CrypTensor):
    def __init__(self, public_key: paillier.PaillierPublicKey):
        self.public_key = public_key

    @classmethod
    def generate_key_pair(cls, key_length: int = 1024):
        public_key, private_key = paillier.generate_paillier_keypair(n_length=key_length)
        return public_key, private_key

    def encrypt(self, value: int) -> paillier.EncryptedNumber:
        return self.public_key.encrypt(value)

    @staticmethod
    def decrypt(encrypted_number: paillier.EncryptedNumber, private_key: paillier.PaillierPrivateKey) -> int:
        return private_key.decrypt(encrypted_number)

    def __add__(self, other: 'MPCTensor') -> 'MPCTensor':
        if self.public_key != other.public_key:
            raise ValueError("Public keys of operands must be the same.")
        return MPCTensor(self.public_key)

    def __repr__(self):
        return f"MPCTensor"

if __name__ == "__main__":
    # Example usage:
    public_key, private_key = MPCTensor.generate_key_pair()

    value1 = random.randint(100, 999)
    value2 = random.randint(0, 100)

    tensor1 = MPCTensor(public_key)
    tensor2 = MPCTensor(public_key)

    encrypted_sum = tensor1.encrypt(value1) + tensor2.encrypt(value2)
    print("encrypted_sum")
    print(encrypted_sum)
    decrypted_sum = MPCTensor.decrypt(encrypted_sum, private_key)
    print("decrypted_sum")
    print(f"The sum of {value1} and {value2} is: {decrypted_sum}")
