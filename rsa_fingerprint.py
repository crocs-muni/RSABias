"""
Functions to obtain 5p_5q_blum_mod_roca fingerprint from https://crocs.fi.muni.cz/public/papers/privrsa_esorics20 paper
Use RSAFingerprint to obtain fingerprint for single key
Use RSAFingerprintSet to obtain fingerprint and aggregated results for list with multiple keys
"""
import roca
app = roca.RocaFingerprinter()


def extract_bits5(num):
    # Calculate the number of bits required to represent the integer
    num_bits = num.bit_length()
    
    # Extract the five most significant bits (MSBs)
    msb = (num >> (num_bits - 5)) & 0b11111

    # Extract the second least significant bit (LSB)
    second_lsb = (num >> 1) & 0b1
    
    return msb, second_lsb


def extract_bits8(num):
    num_bits = num.bit_length()
    msb = (num >> (num_bits - 8)) & 0b11111111
    second_lsb = (num >> 1) & 0b1
    return msb, second_lsb


def is_decreased_number_divisible_range(num, factor_range):
    # Decrease the number by 1
    decreased_num = num - 1

    # Check if the decreased number is divisible by any factor between 3 and 251
    for factor in factor_range:
        if decreased_num % factor == 0:
            return True
    
    return False


def is_decreased_number_divisible(number):
    ranges = [5, 251, 17863]
    for range_limit in ranges:
        if is_decreased_number_divisible_range(number, range(3, range_limit + 1)):
            return range_limit

    return -1


class RSAFingerprintSet:
    """
    The class accepts list of RSA keys given by primes p, q and modulus n.
    A after running compute_fingerprint() method, aggregate RSA fingerprint results are computed for the whole set
    """
    pqn_list = []
    pqn_fingerprints = []
    is_blum = True  # True if all keys are in the form of Blum prime, False if at least one key is not Blum prime
    at_least_one_roca_fingerprint = False   # by default False, True if at least one key is with ROCA fingerprint
    avoid_factors_max = -1  # -1 if no avoidance of small factors in p-1/q-1, 5/251/17863 if no small factors below given limit are factors

    def __init__(self, pqn: list):
        self.pqn_list = pqn

    def compute_fingerprint(self):
        never_below_17863 = True
        never_below_251 = True
        never_below_5 = True
        for item in self.pqn_list:
            p, q, n = item
            fingerprint = RSAFingerprint(p, q, n)
            fingerprint.compute_fingerprint()
            self.pqn_fingerprints.append(fingerprint)

            if fingerprint.small_divisors_p == 17863 or fingerprint.small_divisors_q == 17863:
                never_below_17863 = False
            if fingerprint.small_divisors_p == 251 or fingerprint.small_divisors_q == 251:
                never_below_251 = False
            if fingerprint.small_divisors_p == 5 or fingerprint.small_divisors_q == 5:
                never_below_5 = False

            if fingerprint.is_roca:
                self.at_least_one_roca_fingerprint = True

            if not fingerprint.is_blum:
                self.is_blum = False

        # all hits
        self.avoid_factors_max = -1
        if never_below_5:
            self.avoid_factors_max = 5
        if never_below_251:
            self.avoid_factors_max = 251
        if never_below_17863:
            self.avoid_factors_max = 17863

    def __str__(self):
        return f"Total keys: {self.pqn_fingerprints}, All are Blum: {self.is_blum}, " \
               f"At least one ROCA: {self.at_least_one_roca_fingerprint}, " \
               f"Avoidance of small factors in primes: {self.avoid_factors_max}"


class RSAFingerprint:
    """
    The class accepts single RSA key given by prime p, q and modulus n.
    A after running compute_fingerprint() method, https://crocs.fi.muni.cz/public/papers/privrsa_esorics20
    fingerprint is computed.
    """
    p = 0
    q = 0
    n = 0
    msb5_p = 0
    msb5_q = 0
    msb8_n = 0
    second_lsb_p = 0
    second_lsb_q = 0
    second_lsb_n = 0
    small_divisors_p = -1
    small_divisors_q = -1
    is_roca = False
    is_blum = False

    def __init__(self, p: str, q: str, n: str):
        self.p = p
        self.q = q
        self.n = n

    def compute_fingerprint(self):
        num_p = int(self.p, 16)
        num_q = int(self.q, 16)
        num_n = int(self.n, 16)
        self.msb5_p, self.second_lsb_p = extract_bits5(num_p)
        self.msb5_q, self.second_lsb_q = extract_bits5(num_q)
        self.msb8_n, self.second_lsb_n = extract_bits8(num_n)

        # Check if modulus is (probabilistically) Blum integer
        if self.second_lsb_p == 1 or self.second_lsb_q == 1:    # if both second lsbs are 1 => n can't be Blum integer
            self.is_blum = False

        # Check small divisors
        self.small_divisors_p = is_decreased_number_divisible(num_p)
        self.small_divisors_q = is_decreased_number_divisible(num_q)

        # Compute ROCA fingerprint
        if app.has_fingerprint_moduli(num_n):
            self.is_roca = True

    def __str__(self):
        return f"{self.msb5_p}, {self.msb5_q}, {self.second_lsb_p}, {self.second_lsb_q}, {self.small_divisors_p}, " \
               f"{self.small_divisors_q}, {self.is_roca}"

