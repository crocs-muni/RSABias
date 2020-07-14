import sympy.ntheory as nt
import sympy.ntheory.modular


def prime_default(modulus_length):
    if modulus_length >= 3968:
        return 1427  # real bound 3968
    if modulus_length >= 1984:
        return 701  # 701 is used from 1984 rather than 2048
    if modulus_length >= 992:
        return 353  # 353 is used from 992 rather than 1024
    if modulus_length >= 512:
        return 167
    return 167  # no data for <512


def primorial_default(modulus_length):
    return nt.primorial(prime_default(modulus_length), False)


def element_order(element, modulus, order, order_decomposition):
    if element == 1:
        return 1  # by definition
    if pow(element, order, modulus) != 1:
        return None  # not an element of the group
    for factor, power in order_decomposition.items():
        for p in range(1, power + 1):
            next_order = order // factor
            if pow(element, next_order, modulus) == 1:
                order = next_order
            else:
                break
    return order


def order_of_generator(generator, primorial):
    totient = nt.totient(primorial)
    totient_decomposition = nt.factorint(totient)
    return element_order(generator, primorial, totient, totient_decomposition)


def discrete_log(element, generator, generator_order,
                 generator_order_decomposition, modulus):
    if pow(element, generator_order, modulus) != 1:
        return None
    moduli = []
    remainders = []
    for prime, power in generator_order_decomposition.items():
        prime_to_power = prime ** power
        order_div_prime_power = generator_order // prime_to_power
        g_dash = pow(generator, order_div_prime_power, modulus)
        h_dash = pow(element, order_div_prime_power, modulus)
        found = False
        for i in range(0, prime_to_power):
            if pow(g_dash, i, modulus) == h_dash:
                remainders.append(i)
                moduli.append(prime_to_power)
                found = True
                break
        if not found:
            return None
    return nt.modular.crt(moduli, remainders, symmetric=False)[0]


generator = 65537

# parameters hard-coded rather than recomputed at import, saves 15 seconds
# primorials = [primorial_default(x) for x in [4096, 2048, 1024, 512]]
primorials = [
    0x48fb0a1f06ed4f92615e4ba12dab29ea5706eb914dd00c67cfec53d10280ce578764fa5344da6c4ee1cfaa1db265e3612889d5220d1faa352d74408df5f348ab8e9a86ee74ecc4cea5613804293255c105e25605b7beb276e168ee74963977ff2e607b1831fd3af4a14554236bf0dfdc283ff55538162bc793adb421ebe02b8c58f13fb8f9274253b52b1a38ee94c50effe05ecf21ce0318c79573d1232191a940f587e2e61a3e2e656a21b26258b20bfe7b2d16defec2f7e106bcde65166254ec2b4fd92b17f3283700a71b9b6e3f954f5606933962d18d629897f7146f37cf84669accf126e5301f056c599f9bc1ef02ea8d17ce2,
    0x7cda79f57f60a9b65478052f383ad7dadb714b4f4ac069997c7ff23d34d075fca08fdf20f95fbc5f0a981d65c3a3ee7ff74d769da52e948d6b0270dd736ef61fa99a54f80fb22091b055885dc22b9f17562778dfb2aeac87f51de339f71731d207c0af3244d35129feba028a48402247f4ba1d2b6d0755baff6,
    0x7923ba25d1263232812ac930e9683ac0b02180c32bae1d77aa950c4a18a4e660db8cc90384a394940593408f192de1a05e1b61673ac499416088382,
    0x924cba6ae99dfa084537facc54948df0c23da044d8cabe0edd75bc6
]
# generator_orders = [order_of_generator(generator, x) for x in primorials]
generator_orders = [
    0x33c07a6215a59c179c46f668d09de25aa7a394ec6d453871b4342d55115c8b8d2086d445a0b8ee950ee16fa5dae8ffd231191d87b5d80,
    0x6d9bc8b22f6a31bd9622dfb3d3b93df4a7f37f1443ad5880236fd6ee5c5e5a80,
    0x34ec47a182e0738e666bc7374dcc9ee620,
    0x220ebcc1b395f710
]
# order_decompositions = [nt.factorint(x) for x in generator_orders]
order_decompositions = [
    {2: 7, 3: 4, 5: 3, 7: 3, 11: 2, 13: 2, 17: 1, 19: 1, 23: 1, 29: 1, 31: 1, 37: 1, 41: 1, 43: 1, 47: 1, 53: 1, 59: 1, 61: 1, 67: 1, 71: 1, 73: 1, 79: 1, 83: 1, 89: 1, 97: 1, 101: 1, 103: 1, 107: 1, 109: 1, 113: 1, 127: 1, 131: 1, 137: 1, 139: 1, 149: 1, 151: 1, 163: 1, 173: 1, 179: 1, 181: 1, 191: 1, 193: 1, 199: 1, 233: 1, 239: 1, 251: 1, 277: 1, 281: 1, 293: 1, 307: 1, 359: 1, 419: 1, 431: 1, 443: 1, 491: 1, 509: 1, 593: 1, 641: 1, 653: 1, 659: 1, 683: 1},
    {2: 7, 3: 4, 5: 3, 7: 2, 11: 1, 13: 2, 17: 1, 19: 1, 23: 1, 29: 1, 31: 1, 37: 1, 41: 1, 43: 1, 47: 1, 53: 1, 61: 1, 67: 1, 71: 1, 73: 1, 79: 1, 83: 1, 89: 1, 97: 1, 101: 1, 103: 1, 107: 1, 113: 1, 127: 1, 131: 1, 139: 1, 163: 1, 173: 1, 179: 1, 191: 1, 233: 1, 239: 1, 251: 1, 281: 1, 293: 1},
    {2: 5, 3: 4, 5: 3, 7: 2, 11: 1, 13: 1, 17: 1, 19: 1, 23: 1, 29: 1, 31: 1, 37: 1, 41: 1, 43: 1, 47: 1, 53: 1, 67: 1, 73: 1, 79: 1, 83: 1, 89: 1, 113: 1, 131: 1, 173: 1},
    {2: 4, 3: 4, 5: 2, 7: 1, 11: 1, 13: 1, 17: 1, 23: 1, 29: 1, 37: 1, 41: 1, 53: 1, 83: 1}
]


def parameters(length, modulus=True):
    if modulus:
        modulus_length = length
    else:
        modulus_length = 2 * length
    # allow tolerance for keys smaller than 512-bit
    breakpoints = [(3968, 0), (1984, 1), (992, 2), (480, 3)]
    for b, i in breakpoints:
        if modulus_length >= b:
            return primorials[i], generator_orders[i], order_decompositions[i]
    raise ValueError('ROCA parameters are unknown for key size < 512')


def logarithm(number, is_modulus):
    length = number.bit_length()
    primorial, order, decomposition = parameters(length, is_modulus)
    log = discrete_log(number, generator, order, decomposition, primorial)
    return int(log) if log else None


def divide(number, is_modulus):
    primorial, __, __ = parameters(number.bit_length(), is_modulus)
    return number // primorial


def fingerprint(number, is_modulus):
    return logarithm(number, is_modulus) is not None
