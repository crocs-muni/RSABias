import json
import base64
import Crypto
from Crypto.PublicKey import RSA


class KeyException(Exception):
    pass


class Key(dict):
    """
    Key is a dictionary that holds information about an RSA key.

    Dictionary key-value pair is name and value of an original key field
    (id, n=modulus, e=pub_exp, p and q=primes, d=priv_exp, t=time)
    or a computed feature Transformation.name(): Transformation.apply().

    The class also handles import from and export to supported formats.
    """

    def export_csv(self, header=None, separator=';', missing=0,
                   format_dict=None):
        """
        Export to CSV format.

        :param header: array of feature names
        :param separator: separator, default ";"
        :param missing: default value for missing values ("0")
        :param format_dict: see Key.export_dict()
        :return: key as a CSV line
        """
        if header is None:
            header = ['id', 'n', 'e', 'p', 'q', 'd', 't']
        values = []
        formatted = self.export_dict(format_dict=format_dict)
        for h in header:
            if h in formatted:
                values.append(str(formatted[h]))
            else:
                values.append(str(missing))
        return separator.join(values)

    @staticmethod
    def base_to_format_dict(base_dict):
        """
        Convert dictionary "feature: base" into "feature: format string".

        :param base_dict: dictionary "feature name: base"
        :return: dictionary "feature name: format string"
        """
        if base_dict is None:
            return None
        format_dict = dict()
        for key, val in base_dict.items():
            if val == 16:
                form = '{:x}'
            elif val == 2:
                form = '{:b}'
            elif val == 10 or val is None:
                # print decimals as number and other values as is
                # adding None to the dictionary would delete the item
                continue
            else:
                raise KeyException('Unsupported base "{}" for "{}"'.format(
                    val, key))
            format_dict[key] = form
        return format_dict

    def export_dict(self, format_dict=None):
        """
        Export key as a dictionary, format values by format_dict.

        :param format_dict: dictionary "feature name: format string"
        :return: the key as a dictionary with formatted values,
                 features without format string are included as objects
        """
        formatted = dict()
        if format_dict is None:
            format_dict = {'n': '{:x}', 'e': '{:x}', 'p': '{:x}', 'q': '{:x}',
                           'd': '{:x}'}
        for key, val in self.items():
            if val is not None and key in format_dict:
                format_str = format_dict[key]
                if isinstance(val, list):
                    formatted[key] = [format_str.format(v) for v in val]
                else:
                    formatted[key] = format_str.format(val)
            else:
                formatted[key] = val
        return formatted

    def export_json_string(self, format_dict=None):
        """
        Export key as a JSON string.

        :param format_dict: dictionary "feature name: format string"
        :return: the key as a JSON string representation of a dictionary
        """
        return json.dumps(self.export_dict(format_dict=format_dict))

    @staticmethod
    def __parse_value(value, base_dict, key_part):
        if not isinstance(value, int) \
                and base_dict is not None \
                and key_part in base_dict:
            if base_dict[key_part] is not None:
                base = base_dict[key_part]
                if value is None:
                    return None
                negative = 1
                if value.startswith('-') and len(value) > 1:
                    value = value[1:]
                    negative = -1
                if ((base == 10 and value.isnumeric())
                        or (base == 16 and value.isalnum())):
                    return int(value, base=base) * negative
                else:
                    raise KeyException('Invalid combination of base and '
                                       'value: base={}, value="{}"'
                                       .format(base, value))
        else:
            return value

    @staticmethod
    def import_dict(key_dict, base_dict=None):
        """
        Import from a dictionary that has strings instead of numbers.

        If the parameters are strings instead of integers, provide
        a dictionary base_dict that specifies the base of each integer.
        If the key is missing, the parameter will be copied as is.
        If the value is None, the parameter will not be copied (unless int).

        :param key_dict: key as a dictionary, numeric values may be strings
        :param base_dict: dictionary "feature name: base"
        :return: key, where numeric values are parsed as integers
        """

        if not key_dict:
            raise KeyException('Key dictionary is None or empty')
        key = Key()
        for key_part, value in key_dict.items():
            if isinstance(value, list):
                pv = [Key.__parse_value(v, base_dict, key_part) for v in value]
            else:
                pv = Key.__parse_value(value, base_dict, key_part)
            key[key_part] = pv
        return key

    @staticmethod
    def import_json(s):
        """
        Import key from JSON string.

        :param s: JSON string representation of a key
        :return: key
        """
        return Key.import_dict(json.loads(s))

    @staticmethod
    def import_alg_test(file):
        """
        Receive a file (JCAlgTest format) and read a single key from it.

        :param file: file object
        :return: a key
        """
        e = None
        p = None
        q = None
        while True:
            line = file.__next__()
            # some files include spaces in the hex string
            line = line.replace(': ', '**').replace('# ', '##') \
                .replace(' ', '') \
                .replace('**', ': ').replace('##', '# ')
            if line.startswith("PUBL: "):
                split = line.split(' ')
                if not split[1].startswith('81'):
                    raise KeyException(
                        'Invalid format "{}", expected "PUBL: 81[LEN*2]..."'
                        .format(line))
                length = int(split[1][2:6], base=16) * 2
                e = int(split[1][6:6 + length], base=16)
            if line.startswith("PRIV: "):
                split = line.split(': ')
                if not split[1].startswith('83'):
                    raise KeyException(
                        'Invalid format "{}", '
                        'expected "PRIV: 83[LEN_BYTES]...84[LEN_BYTES]..."'
                        .format(line))
                len_p_index = 2
                p_index = len_p_index + 4
                length_p = int(split[1][len_p_index:p_index], base=16) * 2
                p = int(split[1][p_index:p_index + length_p], base=16)
                len_q_index = p_index + length_p + 2
                # leftover SW 9000 in the values
                if split[1][p_index + length_p:len_q_index + 2] == '9000':
                    length_p += 4
                    len_q_index += 4
                if split[1][p_index + length_p:len_q_index] != '84':
                    raise KeyException(
                        'Invalid format "{}", '
                        'expected "PRIV: 83[LEN_BYTES]...84[LEN_BYTES]..."'
                        .format(line))
                q_index = len_q_index + 4
                length_q = int(split[1][len_q_index:q_index], base=16) * 2
                q = int(split[1][q_index:q_index + length_q], base=16)
            if line.startswith("#"):
                split = line[2:].split(':')
                id_ = int(split[0])
                t = int(float(split[1]) * 1000)
                n = p * q
                d = int(gmpy2.invert(e, gmpy2.lcm(p-1, q-1)))
                k = {'id': id_, 'n': n, 'e': e, 'p': p, 'q': q, 'd': d, 't': t}
                return Key.import_dict(k)

    @staticmethod
    def import_multi_line(file):
        """
        Receive a file and read a single key from it.

        Parameters are spread over multiple lines:
        Key #x, [P: x, Q: x, d: x,] N: x, e: x
        Private parameters P, Q, d are optional.

        :param file: file object
        :return: key
        """
        k = None
        while True:
            line = file.__next__()
            if line.startswith("Key #"):
                split = line.split('#')
                id_ = int(split[1])
                k = dict()
                k['id'] = id_
            elif line[1:3] == ': ':
                split = line.split(': ')
                key = split[0].lower()
                k[key] = int(split[1], base=16)
                if key == 'e':
                    # originally returning on empty line, but files
                    # are not consistent, return on last parameter (e)
                    return k
            elif k:  # returning on empty line
                return k

    @staticmethod
    def import_multi_line_base64(file, key_par=None):
        """
        Receive a file and read a single key from it

        Parameters are spread over multiple lines:
        'Key: dec', 'Time(ms): dec', ' modulus: base64', ' pub Exp: base64',
        'priv Exp: base64', '  prime1: base64', '  prime2: base64'

        :param file: file object
        :param key_par: mapping for transforming parameter names
        :return: key
        """
        if key_par is None:
            key_par = {'Key': 'id', 'Time(ms)': 't', ' modulus': 'n',
                       ' pub Exp': 'e', 'priv Exp': 'd', '  prime1': 'p',
                       '  prime2': 'q'}
        k = None
        while True:
            line = file.__next__()
            split = line.split(': ')
            if len(split) > 1:
                key = key_par[split[0]]
                if key == 'id':
                    k = dict()
                if key == 'id' or key == 't':
                    k[key] = int(split[1])
                else:
                    k[key] = int(base64.b64decode(split[1]).hex(), base=16)
                if key == 'q':
                    # originally returning on empty line, but files
                    # are not consistent, return on last parameter (q)
                    return k
            elif k:  # returning on empty line
                return k

    @staticmethod
    def import_asn1(der):
        """
        Import key from DER encoded ASN.1 representation.

        :param der: DER encoded ASN.1 string
        :return: key
        """
        return Key.import_standard_format(der)

    @staticmethod
    def import_standard_format(encoded):
        """
        Import key from standard formats (PEM, der)

        :param encoded: encoded key
        :return: key
        """
        key = RSA.importKey(encoded)

        n = int(key.n)
        e = int(key.e)
        p = int(key.p)
        q = int(key.q)
        d = int(key.d)
        assert n == p * q
        # assert (d * e) % gmpy2.lcm((p - 1), (q - 1)) == 1

        return Key.import_dict({'n': n, 'e': e, 'p': p, 'q': q, 'd': d})

    @staticmethod
    def import_tpm_multi_line(file):
        """
        Receive a file and read a single key from it.

        Parameters are spread over multiple lines (TPM tool output):
        exp mod

        :param file: file object
        :return: key
        """
        k = dict()
        while True:
            line = file.__next__()
            split = line.split(' ')
            if line.startswith("exp"):
                k['e'] = int(split[1], base=16)
            elif line.startswith("mod"):
                k['n'] = int(split[1], base=16)
                return k

    @staticmethod
    def import_tpm_modulus(file):
        """
        Receive a file and read a single key from it.

        Only modulus available (TPM tool output):
        Modulus

        :param file: file object
        :return: key
        """
        while True:
            line = file.__next__()
            if line.startswith("Modulus="):
                split = line.split('=')
                split = split[1].split(';')
                k = dict()
                k['n'] = int(split[0], base=16)
                return k

    @staticmethod
    def import_tpm_xml(file):
        """
        Receive a file and read a single XML formatted key from it.

        :param file: file object
        :return: key
        """
        k = dict()
        while True:
            line = file.__next__().strip()
            if line.startswith("<PublicExp"):
                line = file.__next__().strip()
                k['e'] = int(line, base=16)
            elif line.startswith("<Modulus"):
                line = file.__next__().strip()
                k['n'] = int(line, base=16)
                return k
