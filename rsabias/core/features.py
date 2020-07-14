import math
import sympy.ntheory as nt
from collections.abc import Sequence
import decimal
import itertools
import copy
import ast
import random
from collections import OrderedDict
import rsabias.core.roca as roca


class NumericParameter:

    @staticmethod
    def parsed_or_original(number):
        if isinstance(number, NumericParameter):
            return number
        tried = NumericParameter.try_parse(number)
        if tried is None:
            return number
        return tried

    @staticmethod
    def try_parse(number):
        if isinstance(number, NumericParameter):
            return number
        int_ = None
        if isinstance(number, int):
            int_ = number
        elif isinstance(number, str):
            if number.isnumeric():
                int_ = int(number)
            elif number[:-1].isnumeric():
                if number.endswith('!'):
                    int_ = math.factorial(int(number[:-1]))
                elif number.endswith('#'):
                    int_ = nt.primorial(int(number[:-1]), False)
        if int_ is not None:
            return NumericParameter(int_, str(number))
        return None

    def __init__(self, number, string):
        self.int_ = number
        self.str_ = string

    def __int__(self):
        return self.int_

    def __str__(self):
        return self.str_


class Parser:
    """Transform a text or JSON representation of a Transformation"""

    @staticmethod
    def __decode(string):
        if string in [True, False, None]:
            return string
        if string.lower() == 'true':
            return True
        if string.lower() == 'false':
            return False
        if string.lower() == 'null':
            return None
        # if string[0] == '{' and string[-1] == '}':
        #     return Extract(string[1:-1])
        try_numeric = NumericParameter.try_parse(string)
        if try_numeric is not None:
            return try_numeric
        return string

    @staticmethod
    def parse_string(string):
        """Parse a short text representation of a Transformation.

        Format is Name(input,parameter1,parameter2,...)
        - input can be a Transformation or a list of Transformations in []
        - a key part is denoted Extract(key_part_name) and everything
          inside {} is interpreted as a key part (dictionary key)
        """
        # TODO: could be done by eval() if constants are changed to objects
        # parse inside of () and [] recursively, return inside of {} as literal
        special = {'(': ')', '[': ']', '{': '}'}
        split = []
        opened = None
        opened_count = 0
        start = 0
        start_bracket = None
        for i in range(len(string)):
            c = string[i]
            if opened_count == 0 and c in special:
                # open the current type of brackets
                opened = c
                opened_count += 1
                start_bracket = i
            elif opened_count > 0 and c == opened:
                # opening of the same type of brackets
                opened_count += 1
            elif opened_count == 1 and c == special[opened]:
                # closing the top level brackets
                name = string[start:start_bracket]
                inside = string[start_bracket + 1:i]
                if name == '' and opened == '{' and c == '}':
                    # the string is a dictionary key for direct look-up
                    split.append(Extract(inside))
                elif name == '' and opened == '[' and c == ']':
                    # the string is a list of other objects
                    split.append(Parser.parse_string(inside))
                elif name != '' and opened == '(' and c == ')':
                    # the string is a Transformation with name and arguments
                    arguments = Parser.parse_string(inside)
                    split.append(Parser.__from_parameters(name, arguments))
                else:
                    raise TransformationException(
                        'Malformed Transformation "{}"'.format(
                            string[start:i+1]))
                start = i + 1
                opened = None
                start_bracket = None
                opened_count = 0
            elif opened_count > 1 and c == special[opened]:
                # closing inner brackets of the same type
                opened_count -= 1
            elif c == ',' and opened_count == 0:
                # next argument, if brackets are not open
                if start != i:
                    # if the string is not just a comma, it is a literal,
                    # otherwise it would have been taken care of by brackets
                    literal = string[start:i]
                    split.append(Parser.__decode(literal))
                start = i + 1
            elif opened_count == 0 and c in special.values():
                # closing unopened brackets
                raise TransformationException(
                    'Mismatched closing brackets: "{}**{}**{}"'.format(
                        string[:i], c, string[i+1:]))
        if opened_count > 0:
            # unclosed brackets
            raise TransformationException(
                'Missing closing brackets: "{}**{}**"'.format(string, opened))
        if start < len(string):
            # no brackets were closed, so the last part is a literal, if any
            literal = string[start:]
            split.append(Parser.__decode(literal))
        if len(split) == 1:
            return split[0]
        return split

    @staticmethod
    def __from_parameters(name, parameters):
        trans = parameters[0]
        args = parameters[1:]
        # The parameters are not named... To avoid similar code, fake
        # a dictionary by guessing argument positions.
        # Arguments with the same name must be on the same position...
        dict_guess = {'transformation': name}
        options = {'input': trans}
        if len(args) > 0:
            options['part'] = args[0]
            options['count'] = args[0]
            options['value'] = args[0]
            options['roca_modulus'] = args[0]
        if len(args) > 1:
            options['skip'] = args[1]
            options['skip_bottom'] = args[1]
            options['equals'] = args[1]
        if len(args) > 2:
            options['byte_aligned'] = args[2]
            options['skip_top'] = args[2]
        if len(args) > 3:
            options['skip_before'] = args[3]
        if len(args) > 4:
            options['skip_after'] = args[4]
        dict_guess['options'] = options
        return Parser.parse_dict(dict_guess, False)

    @staticmethod
    def parse_dict(d, recursive_inputs=True):
        name = d['transformation']
        opt = d['options']
        input_ = opt.get('input', None)
        if input_ is None:
            input_op = None
        elif recursive_inputs:
            if isinstance(input_, list):
                input_op = [Parser.parse_dict(in_) for in_ in input_]
            elif isinstance(input_, dict):
                input_op = Parser.parse_dict(input_)
            else:
                input_op = Parser.parse_string(input_)
        else:
            input_op = input_
        key_part = opt.get('part', None)
        skip = opt.get('skip', None)
        skip_bottom = opt.get('skip_bottom', None)
        skip_top = opt.get('skip_top', None)
        skip_before = opt.get('skip_before', None)
        skip_after = opt.get('skip_after', None)
        count = opt.get('count', None)
        align_to_bytes = opt.get('byte_aligned', False)
        value = opt.get('value', None)
        equals = opt.get('equals', None)
        combine = opt.get('values_combine', False)
        roca_modulus = opt.get('roca_modulus', True)

        equals = NumericParameter.parsed_or_original(equals)

        op = None

        multi_val_allowed = [Modulo.class_name(), GCD.class_name(), GCDEquals.class_name(),
                             Order.class_name(), Inverse.class_name()]

        if name in multi_val_allowed:
            single_feature = not isinstance(value, list)
            if single_feature:
                value = [value]
            ops = []
            op = None
            for v in value:
                v = NumericParameter.try_parse(v)
                if name == Modulo.class_name():
                    op = Modulo(trans=input_op, divisor=v)
                elif name == GCD.class_name():
                    op = GCD(trans=input_op, value=v)
                elif name == GCDEquals.class_name():
                    op = GCDEquals(trans=input_op, value=v, equals=equals)
                elif name == Order.class_name():
                    op = Order(trans=input_op, number=v)
                elif name == Inverse.class_name():
                    op = Inverse(trans=input_op, number=v)
                else:
                    raise TransformationException(
                        'Unexpectedly processing multiple values for {}'
                        .format(name))
                ops.append(op)
            if not single_feature:
                if combine:
                    op = Combine(ops)
                else:
                    op = Append(ops)
        else:
            value = NumericParameter.parsed_or_original(value)
        if name in multi_val_allowed:
            pass
        elif name == Transformation.class_name():
            op = Transformation(trans=input_op)
        elif name == Join.class_name():
            op = Join(trans=input_op)
        elif name == Append.class_name():
            op = Append(trans=input_op)
        elif name == Combine.class_name():
            op = Combine(trans=input_op)
        elif name == CombineSequences.class_name():
            op = CombineSequences(trans=input_op)
        elif name == Range.class_name():
            op = Range(trans=input_op)
        elif name == Reduce.class_name():
            op = Reduce(trans=input_op)
        elif name == RandomSingle.class_name():
            op = RandomSingle(trans=input_op)
        elif name == MultiReduce.class_name():
            op = MultiReduce(trans=input_op)
        elif name == Max.class_name():
            op = Max(trans=input_op)
        elif name == Min.class_name():
            op = Min(trans=input_op)
        elif name == Sum.class_name():
            op = Sum(trans=input_op)
        elif name == All.class_name():
            op = All(trans=input_op)
        elif name == Any.class_name():
            op = Any(trans=input_op)
        elif name == Equal.class_name():
            op = Equal(trans=input_op)
        elif name == Feature.class_name():
            op = Feature(trans=input_op)
        elif name == Add.class_name():
            op = Add(trans=input_op, value=value)
        elif name == Extract.class_name():
            op = Extract(key_part=key_part)
        elif name == Bits.class_name():
            op = Bits(trans=input_op, count_bits=count, skip_bits=skip)
        elif name == MostSignificantBits.class_name():
            op = MostSignificantBits(trans=input_op, count_bits=count,
                                     skip_bits=skip,
                                     align_to_bytes=align_to_bytes)
        elif name == LeastSignificantBits.class_name():
            op = LeastSignificantBits(trans=input_op, count_bits=count,
                                      skip_bits=skip)
        elif name == LargestEvenDivisor.class_name():
            op = LargestEvenDivisor(trans=input_op)
        elif name == SmallestOddDivisor.class_name():
            op = SmallestOddDivisor(trans=input_op)
        elif name == PrimeDistance.class_name():
            op = PrimeDistance(trans=input_op)
        elif name == PreviousPrimeDistance.class_name():
            op = PreviousPrimeDistance(trans=input_op)
        elif name == NextPrimeDistance.class_name():
            op = NextPrimeDistance(trans=input_op)
        elif name == Constant.class_name():
            op = Constant(constant=value)
        elif name == ROCA.class_name():
            op = ROCA(trans=input_op, on_modulus=roca_modulus)
        elif name == ROCALogarithm.class_name():
            op = ROCALogarithm(trans=input_op, on_modulus=roca_modulus)
        elif name == ROCADivisor.class_name():
            op = ROCADivisor(trans=input_op, on_modulus=roca_modulus)
        elif name == ROCAFingerprint.class_name():
            op = ROCAFingerprint(trans=input_op, on_modulus=roca_modulus)
        elif name == BitLength.class_name():
            op = BitLength(trans=input_op)
        elif name == QuadraticResidue.class_name():
            op = QuadraticResidue(trans=input_op)
        elif name == DistanceToSquare.class_name():
            op = DistanceToSquare(trans=input_op)
        elif name == SerialBits.class_name():
            op = SerialBits(trans=input_op, bits=count,
                            skip_bottom=skip_bottom, skip_top=skip_top,
                            skip_before=skip_before, skip_after=skip_after)
        elif name == OddItems.class_name():
            op = OddItems(trans=input_op)
        elif name == EvenItems.class_name():
            op = EvenItems(trans=input_op)
        elif name == Frequencies.class_name():
            op = Frequencies(trans=input_op)
        elif name == SerialTest.class_name():
            op = SerialTest(trans=input_op, bits=count,
                            skip_bottom=skip_bottom, skip_top=skip_top)
        elif name == MaxSmallestDivisor.class_name():
            op = MaxSmallestDivisor(trans=input_op)
        elif name == ModularFingerprint.class_name():
            op = ModularFingerprint(trans=input_op, value=value)
        elif name == ModularFingerprintSpecial.class_name():
            op = ModularFingerprintSpecial(trans=input_op)
        else:
            raise TransformationException(
                'Name {} not implemented'.format(name))

        return op


def sorted_with_none(s):
    if s is None:
        return None
    if None not in s:
        return sorted(s)
    return [None] + sorted(s - {None})


class Transformation:
    """Transforms a key into feature(s)

    Specifies which computation (feature) will be applied
    on what data (part of a key, result of a transformation).
    """

    @classmethod
    def class_name(cls):
        return cls.__name__

    def __init__(self, trans):
        self.trans = trans
        self.cache_feature = True
        self.__name = None

    def apply(self, key):
        if self.name() in key:
            return key[self.name()]
        feature = self.apply_internal(key)
        if self.cache_feature:
            key[self.name()] = feature
        return feature

    def apply_internal(self, key):
        if self.trans is None:
            return self.feature(key)
        input_ = self.trans.apply(key)
        if input_ is None:
            return None
        # automatically map this transformation if the input transformation
        # produced multiple values
        if isinstance(input_, Sequence):
            return [self.feature(t) for t in input_]
        return self.feature(self.trans.apply(key))

    def feature(self, value):
        return value

    @staticmethod
    def add_feature_to_count(feature, count):
        count[feature] = count.get(feature, 0) + 1

    def count_features(self, features):
        feature_count = dict()
        for f in features:
            self.add_feature_to_count(f, feature_count)
        return feature_count

    def tally(self, features):
        count = self.count_features(features)
        return Distribution(self, count)

    def subspace(self, dimensions):
        if not isinstance(self.trans, Sequence):
            raise TransformationException(
                'Cannot create subspace without multiple transformations')
        if max(dimensions) > len(self.trans) - 1:
            raise TransformationException('Invalid dimensions')
        trans = [copy.deepcopy(self.trans[i])
                 for i in dimensions]
        if len(trans) == 1:
            return trans[0]
        return type(self)(trans)

    def subspace_tally(self, full_tally, dimensions):
        tally = dict()
        if max(dimensions) > len(self.trans) - 1:
            raise TransformationException('Invalid dimensions')
        single = len(dimensions) == 1
        for value_tuple, count in full_tally.items():
            if single:
                reduced_tuple = value_tuple[dimensions[0]]
            else:
                reduced_tuple = tuple(value_tuple[i] for i in dimensions)
            tally[reduced_tuple] = tally.get(reduced_tuple, 0) + count
        return tally

    def name(self):
        if self.__name is not None:
            return self.__name
        trans_params = []
        n_t = self.name_transformations()
        if n_t is not None and n_t != '':
            trans_params.append(n_t)
        n_p = self.name_parameters()
        if n_p is not None and n_p != '':
            trans_params.append(str(n_p))
        name_ = '{}({})'.format(self.class_name(), ','.join(trans_params))
        self.__name = name_
        return self.__name

    def __str__(self):
        return self.name()

    def name_parameters(self):
        return None

    def name_transformations(self):
        if isinstance(self.trans, Sequence):
            return '[{}]'.format(','.join([t.name() for t in self.trans]))
        elif self.trans is not None:
            return self.trans.name()
        return None

    def description(self):
        # TODO
        return self.name()

    def feature_name(self):
        # TODO
        return self.name()

    def range(self, keys=None):
        if keys is not None:
            # best guess - the actual values from a distribution
            return sorted_with_none(keys)
        if self.trans is not None:
            # base transformation does not change the range
            return self.trans.range()
        return None  # not available

    def string(self, value):
        # TODO
        return str(value)

    @staticmethod
    def base():
        # TODO
        """Default base for formatting output, None for non-numeric"""
        return None

    def base_dict(self):
        # TODO add to a single dictionary also the self.trans
        """Dictionary with name(s) to default bases"""
        return {self.name: self.base()}


class TransformationException(Exception):
    pass


class Join(Transformation):
    """Meta-transformation, applies multiple transformations"""

    def __init__(self, trans):
        super().__init__(trans)
        if not isinstance(trans, Sequence):
            raise TransformationException(
                'Join requires multiple transformations')

    def apply_internal(self, key):
        return tuple(t.apply(key) for t in self.trans)

    def range(self, keys=None):
        marginal = self.range_marginal(keys=keys)
        return itertools.product(*marginal)

    def range_marginal(self, keys=None):
        if keys is None:
            return [t.range(None) for t in self.trans]
        return [t.range(set(k)) for t, k in zip(self.trans, keys)]


class Append(Join):
    """Meta-transformation, appends independent features"""

    @staticmethod
    def add_feature_to_count(features, counts):
        for f, c in zip(features, counts):
            c[f] = c.get(f, 0) + 1

    def count_features(self, tuples_list):
        counts = [dict() for _ in self.trans]
        for t in tuples_list:
            self.add_feature_to_count(t, counts)
        return counts

    def tally(self, tuples_list):
        feature_lists = list(zip(*tuples_list))
        dists = [tr.tally(fl)
                 for tr, fl in zip(self.trans, feature_lists)]
        return MultiFeatureDistribution(self, dists)

    def range(self, keys=None):
        return self.range_marginal(keys=keys)


class Combine(Join):
    """Meta-transformation, collects feature combinations"""

    def tally(self, tuples_list):
        return MultiDimDistribution(self, self.count_features(tuples_list))


class CombineSequences(Combine):
    """Like combine, but flattens a list"""

    def apply_internal(self, key):
        lists = [t.apply(key) for t in self.trans]
        return [item for sublist in lists for item in sublist]

    def range(self, keys=None):
        return sorted_with_none(keys)


class Zip(Combine):
    """The underlying transformation makes lists and this zips them"""

    def apply_internal(self, key):
        lists = [t.apply(key) for t in self.trans]
        if None in lists:
            return None
        return list(zip(*lists))

    def range(self, keys=None):
        return sorted_with_none(keys)


# # TODO would it be better to have something that makes a sequence as base?
# # TODO multiple inheritance?
class Range(Join):

    def __init__(self, trans):
        super().__init__(trans)
        if len(trans) > 3 or len(trans) < 2:
            raise Transformation(
                'Range takes 2 or 3 arguments: start, stop, step')

    def feature(self, values):
        start = values[0]
        stop = values[1]
        step = values[2] if len(values) > 2 else 1
        return range(start, stop, step)

    def range(self, keys=None):
        return sorted_with_none(keys)


class Reduce(Transformation):
    """Reduces a sequence of values into a single value

    Examples: SUM, MAX, MIN, ALL, ANY, ALL_EQUAL
    """

    def apply_internal(self, key):
        if isinstance(self.trans, Sequence):
            trans_output = [t.apply(key) for t in self.trans]
        else:
            trans_output = self.trans.apply(key)
        if not isinstance(trans_output, Sequence):
            raise TransformationException(
                'Reduce must be applied on a sequence')
        return self.feature(trans_output)

    def subspace_tally(self, full_tally, dimensions):
        raise TransformationException(
            'Tally of a Reduce cannot be subspaced, because the feature does '
            'not contain all the data needed to recompute a subspace feature')

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        if isinstance(self.trans, Sequence):
            ranges = [t.range() for t in self.trans]
            if None in ranges:
                return None
            maxes = [max(r) for r in ranges]
            mines = [min(r) for r in ranges]
            return range(min(mines), max(maxes))
        else:
            return self.trans.range()


class RandomSingle(Reduce):

    def feature(self, values):
        return values[random.randint(0, len(values) - 1)]


class MultiReduce(Reduce):
    """Reduces a sequence, producing a sequence"""


class Max(Reduce):

    def feature(self, values):
        return max(values)


class Min(Reduce):

    def feature(self, values):
        return min(values)


class Sum(Reduce):

    def feature(self, values):
        return sum(values)

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        if isinstance(self.trans, Sequence):
            ranges = [t.range() for t in self.trans]
            if None in ranges:
                return None
            maxes = [max(r) for r in ranges]
            mines = [min(r) for r in ranges]
            return range(sum(mines), sum(maxes))
        else:
            r = self.trans.range()
            if r is None:
                return None
            return max([sum(el) for el in r])


class All(Reduce):

    def feature(self, values):
        return all(values)

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return [False, True]


class Any(Reduce):

    def feature(self, values):
        return any(values)

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return [False, True]


class Equal(Reduce):

    def feature(self, values):
        first = values[0]
        for v in values:
            if v != first:
                return False
        return True

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return [False, True]


class Feature(Transformation):
    """Extracts features from data.

    Gets a key, applies a transformation level under on the key
    and then transforms the output by its feature

    Examples: MSB, LSB, MOD, GCD, distance, bit-length,
    constant, inverse, order, quadratic residue,
    largest and smallest divisor (via reducing multi-feature)
    """


class Add(Feature):
    """Adds (+) a value"""

    def __init__(self, trans, value):
        super().__init__(trans)
        self.value = value

    def feature(self, value):
        return value + int(self.value)

    def name_parameters(self):
        return str(self.value)

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        r = self.trans.range()
        if r is None:
            return None
        return range(r[0] + self.value, r[-1] + self.value)


class Extract(Feature):
    """Extract a field directly from a key"""

    def __init__(self, key_part):
        super().__init__(None)
        self.key_part = key_part

    def feature(self, key):
        return key.get(self.key_part, None)

    def name(self):
        return '{{{}}}'.format(self.key_part)

    def name_parameters(self):
        return ''

    def name_transformations(self):
        return None

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return None  # unpredictable


# class MultiFeature(Feature):
#     """Extracts a multi-value feature from data
#
#     Examples: factors
#     """
#
#     def extract(self, value):
#         """Extracts a multi-value feature from a value"""
#         pass


class Bits(Feature):
    """Extract a sequence of bits, skipping some"""

    def __init__(self, trans, count_bits, skip_bits):
        super().__init__(trans)
        self.modulo = 2 ** count_bits
        self.count = count_bits
        self.skip = skip_bits

    def name_parameters(self):
        return '{},{}'.format(self.count, self.skip)

    def range(self, keys=None):
        return range(0, 2 ** self.count)


class MostSignificantBits(Bits):
    """Extract bits starting from the most significant bits"""

    def __init__(self, trans, count_bits, skip_bits, align_to_bytes=True):
        super().__init__(trans, count_bits, skip_bits)
        self.shift = skip_bits + count_bits
        self.align_to_bytes = align_to_bytes

    def feature(self, number):
        length = number.bit_length()
        if self.align_to_bytes:
            if number == 0:
                return 0
            length = int(math.ceil(length / 8) * 8)
        if length < self.shift:
            raise TransformationException(
                'Expected {} bits, got {}'.format(self.shift, length))
        number >>= length - self.shift
        number %= self.modulo
        return number

    def name_parameters(self):
        return '{},{},{}'.format(self.count, self.skip, self.align_to_bytes)

    def range(self, keys=None):
        if not self.align_to_bytes and self.skip != 0:
            # if the bits are not aligned to a byte and no bits are
            # skipped at the start, the top bit must be one
            return range(2 ** (self.count - 1), 2 ** self.count)
        return range(0, 2 ** self.count)


class LeastSignificantBits(Bits):
    """Extract bits starting from the least significant bits"""

    def __init__(self, trans, count_bits, skip_bits):
        super().__init__(trans, count_bits, skip_bits)
        self.shift = skip_bits

    def feature(self, number):
        number >>= self.shift
        number %= self.modulo
        return number


def only_coprime(divisor, values):
    candidates = [r for r in range(0, divisor)
                  if math.gcd(r, divisor) != 1]
    full_range = range(0, divisor)
    if set(values).isdisjoint(set(candidates)):
        full_range = sorted(set(full_range) - set(candidates))
    if None in values:
        full_range = [None] + full_range
    return full_range


class Modulo(Feature):

    def __init__(self, trans, divisor):
        super().__init__(trans)
        self.divisor = divisor

    def feature(self, number):
        return number % int(self.divisor)

    def name_parameters(self):
        return '{}'.format(self.divisor)

    def range(self, keys=None):
        if keys is not None:
            # if the value from bottom transformation is a prime,
            # we are not expecting any divisors of the divisor and 0
            # to be in the results, remove them from range too
            return only_coprime(int(self.divisor), keys)
        return range(0, self.divisor)


class LargestEvenDivisor(Feature):
    """Largest even divisor (given as a power of 2)"""

    def __init__(self, trans):
        super().__init__(trans)

    def feature(self, value):
        power = 1
        while value % 2 ** power == 0:
            power += 1
        return power - 1

    def range(self, keys=None):
        if keys is None:
            return range(0, 128)  # estimate
        if None not in keys:
            return range(0, max(keys) + 1)
        return [None] + list(range(0, max(set(keys) - {None})))


class SmallestOddDivisor(Feature):
    """Smallest odd divisor.

    Smallest even divisor is always 2 or nothing.

    Useful to detect what primes are avoided in p-1
    or to look for incremental search starting point.
    """

    def __init__(self, trans):
        super().__init__(trans)
        # OpenSSL avoids primes up to 17863 in p-1 (about 2**14.125)
        end = 2 ** 15
        # Using a shortcut for 251 and 17863 with GCD, order descending!
        self.gcd_primes = [
            (nt.primorial(17863, nth=False), nt.primerange(17864, end)),
            (nt.primorial(251, nth=False), nt.primerange(252, end))
        ]
        self.primes = nt.primerange(3, end)

    def feature(self, value):
        prime_range = self.primes
        for gcd, primes in self.gcd_primes:
            if math.gcd(value, gcd) == 0:
                prime_range = primes
                break
        for p in prime_range:
            if value % p == 0:
                return p
        return 0

    def range(self, keys=None):
        if keys is None:
            return self.primes
        if None not in keys:
            return range(0, max(keys) + 1)
        return [None] + list(range(0, max(set(keys) - {None})))


class GCD(Feature):

    def __init__(self, trans, value):
        super().__init__(trans)
        self.value = value

    def feature(self, value):
        return math.gcd(value, int(self.value))

    def name_parameters(self):
        return '{}'.format(self.value)

    def range(self, keys=None):
        if keys is not None and None in keys:
            return [None] + list(range(1, int(self.value) + 1))
        return range(1, int(self.value) + 1)


class GCDEquals(Feature):

    def __init__(self, trans, value, equals):
        super().__init__(trans)
        self.value = value
        self.equals = equals

    def feature(self, value):
        return math.gcd(value, int(self.value)) == int(self.equals)

    def name_parameters(self):
        return '{},{}'.format(self.value, self.equals)

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return [False, True]


class PrimeDistance(Feature):

    def __init__(self, trans):
        super().__init__(trans)

    def range(self, keys=None):
        if keys is None:
            return range(0, 5000)  # estimate
        none_in_keys = None in keys
        keys = set(keys) - {None}
        # id the starting values are consistently odd or even,
        # the distances will be consistently even or odd
        all_even = all([k % 2 == 0 for k in keys])
        all_odd = all([k % 2 == 1 for k in keys])
        if all_even:
            r = range(2, max(keys) + 1, 2)
        elif all_odd:
            r = range(1, max(keys) + 1, 2)
        else:
            r = range(0, max(keys) + 1)
        if none_in_keys:
            return [None] + list(r)
        return r


class PreviousPrimeDistance(PrimeDistance):

    def __init__(self, trans):
        super().__init__(trans)

    def feature(self, number):
        return number - nt.prevprime(number)


class NextPrimeDistance(PrimeDistance):

    def __init__(self, trans):
        super().__init__(trans)

    def feature(self, number):
        return nt.nextprime(number) - number


class Constant(Feature):

    def __init__(self, constant):
        super().__init__(None)
        self.constant = constant

    def feature(self, value=None):
        return int(self.constant)

    def name_parameters(self):
        return str(self.constant)

    def range(self, keys=None):
        return [self.constant]


class ROCA(Feature):

    def __init__(self, trans, on_modulus):
        super().__init__(trans)
        self.on_modulus = on_modulus

    def name_parameters(self):
        return self.on_modulus

    def range(self, keys=None):
        if keys is not None and None in keys:
            return [None, False, True]
        return [False, True]


class ROCALogarithm(ROCA):
    # TODO also store the fingerprint value

    def feature(self, value):
        return roca.logarithm(value, self.on_modulus)

    def range(self, keys=None):
        if keys == [None]:
            return [None]
        if keys is not None:
            keys = sorted_with_none(keys)
            for p in reversed(roca.primorials):
                if keys[-1] < p:
                    if None in keys:
                        return [None] + list(range(0, p))
                    return range(0, p)
        return range(0, roca.primorials[0])


class ROCADivisor(ROCA):

    def feature(self, value):
        return roca.divide(value, self.on_modulus)

    def range(self, keys=None):
        return sorted_with_none(keys)


class ROCAFingerprint(ROCA):
    # TODO store the logarithm if called first

    def feature(self, value):
        return roca.fingerprint(value, self.on_modulus)

    def range(self, keys=None):
        if keys is not None and None in keys:
            return [None, False, True]
        return [False, True]


class BitLength(Feature):

    def feature(self, value):
        return value.bit_length()

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return None  # unpredictable


class Order(Feature):
    # TODO how to implement without the factorization (or phi)

    def __init__(self, trans, number):
        super().__init__(trans)
        self.number = number

    def feature(self, value):
        return None

    def name_parameters(self):
        return self.number

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return None  # depends on the modulus (value)


class Inverse(Feature):

    def __init__(self, trans, number):
        super().__init__(trans)
        self.number = number

    def feature(self, value):
        is_prime = nt.isprime(value)
        if is_prime:
            return pow(int(self.number), value - 2, value)
        return pow(int(self.number), nt.totient(value) - 1, value)

    def name_parameters(self):
        return self.number

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return None  # depends on the modulus (value)


class QuadraticResidue(Feature):

    def __init__(self, trans, limit=65537):
        super().__init__(trans)
        # the limit is not really needed, as most results will be very low
        self.candidates = list(range(2, limit + 1))

    @staticmethod
    def is_quad_residue(number, prime):
        """Euler's criterion"""
        return pow(number, (prime - 1) // 2, prime) == 1

    def feature(self, prime):
        for candidate in self.candidates:
            if self.is_quad_residue(candidate, prime):
                return candidate
        return 0

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return self.candidates


class DistanceToSquare(Feature):

    def feature(self, number):
        # increase precision, since we are handling large numbers
        decimal.getcontext().prec = 310
        root = int(decimal.Decimal(number).sqrt())
        smaller_square = root**2
        bigger_square = (root + 1)**2
        return [number - smaller_square, bigger_square - number]

    def range(self, keys=None):
        if keys is not None:
            return [sorted_with_none(k) for k in zip(*keys)]
        return None


class OddItems(MultiReduce):

    def feature(self, values):
        return values[::2]

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return None


class EvenItems(MultiReduce):

    def feature(self, values):
        return values[1::2]

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return None


class Frequencies(Feature):

    def __init__(self, trans):
        super().__init__(trans)

    def apply_internal(self, key):
        return self.feature(self.trans.apply_internal(key))

    def feature(self, values):
        if values is None:
            return None
        count = dict()
        for f in values:
            count[f] = count.get(f, 0) + 1
        return count

    @staticmethod
    def add_feature_to_count(feature, count):
        for k, v in feature.items():
            count[k] = count.get(k, 0) + v

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return None


class SerialBits(Feature):

    def __init__(self, trans, bits, skip_bottom, skip_top,
                 skip_before=0, skip_after=0):
        super().__init__(trans)
        self.bits = bits
        self.skip_bottom = skip_bottom
        self.skip_top = skip_top
        self.skip_before = skip_before
        self.skip_after = skip_after
        self.modulo = 2 ** bits
        self.minimum = 2 ** (skip_before + bits + skip_after + skip_top - 1)

    def feature(self, value):
        features = []
        num = value
        num >>= self.skip_bottom
        while num > self.minimum:
            num >>= self.skip_before
            f = num % self.modulo
            features.append(f)
            num >>= self.bits + self.skip_after
        return features

    def name_parameters(self):
        return '{},{},{},{},{}'.format(
            self.bits, self.skip_bottom, self.skip_top, self.skip_before,
            self.skip_after)

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return None  # TODO returns a list of values with known range


class SerialTest(Frequencies):

    def __init__(self, trans, bits, skip_bottom, skip_top):
        self.bits = bits
        self.skip_bottom = skip_bottom
        self.skip_top = skip_top
        half_first = bits // 2
        half_second = bits - half_first
        serial_first = SerialBits(trans, half_first, skip_bottom, skip_top,
                                  skip_before=0, skip_after=half_second)
        serial_second = SerialBits(trans, half_second, skip_bottom, skip_top,
                                   skip_before=half_first, skip_after=0)
        # zip it by pairs and compute the frequencies of pairs
        combine = Zip([serial_first, serial_second])
        super().__init__(combine)

    def name_parameters(self):
        return '{},{},{}'.format(self.bits, self.skip_bottom, self.skip_top)

    def range(self, keys=None):
        if keys is not None:
            return sorted_with_none(keys)
        return None  # TODO can be drawn


class MaxSmallestDivisor(Max):
    """The largest value of the smallest divisor from a range.

    Intended to find the likely starting candidate of incremental search.
    """

    def __init__(self, trans):
        if not isinstance(trans, Range):
            raise TransformationException(
                'MaxSmallestDivisor requires a range transformation')
        # using automatic "map"
        smallest_divisors = SmallestOddDivisor(trans)
        super().__init__(smallest_divisors)

    def range(self, keys=None):
        return self.trans.range(keys=keys)


class ModularFingerprint(Feature):
    """P-1/Q-1 values are biased, they do not have small odd divisors.

    Receive P and return True if P-1 has no small odd divisors
    up to the defined value. Use in combination with All for P and Q.

    OpenSSL: P-1 and Q-1 not divisible by 3 to 17863
    Some NXP: 3 to 251
    Some G&D: 3 and 5
    """

    def __init__(self, trans, value):
        super().__init__(trans)
        self.value = value
        self.product = nt.primorial(int(value), nth=False) // 2

    def feature(self, value):
        return math.gcd(value - 1, int(self.product)) == 1

    def name_parameters(self):
        return '{}'.format(str(self.value))

    def range(self, keys=None):
        return [False, True]


class ModularFingerprintSpecial(Feature):
    """P-1/Q-1 values are biased, they do not have small odd divisors.

    Combines ModularFingerprint into a single feature with 4 levels.

    Receive P and return one of (1, 5, 251, 17863) if P-1 has no small
    odd divisors up to that value. Use in combination with Min for P and Q.

    OpenSSL: P-1 and Q-1 not divisible by 3 to 17863
    Some NXP: 3 to 251
    Some G&D: 3 and 5
    Other: return 1
    """

    def __init__(self, trans):
        super().__init__(trans)
        self.values = [17863, 251, 5]
        self.products = [nt.primorial(v, nth=False) // 2 for v in self.values]

    def feature(self, value):
        for v, p in zip(self.values, self.products):
            if math.gcd(value - 1, p) == 1:
                return v
        return 1

    def name_parameters(self):
        return None

    def range(self, keys=None):
        return self.values + [1]


def marginalize(dimensions, tally):
    marginal = [dict() for _ in range(dimensions)]
    for value_tuple, count in tally.items():
        for v, i in zip(value_tuple, range(dimensions)):
            marginal[i][v] = marginal[i].get(v, 0) + count
    if len(marginal) == 1:
        return marginal[0]
    return marginal


def normed_dictionary(d):
    normed = dict()
    sum_ = sum(d.values())
    for k, v in d.items():
        normed[k] = float(v) / sum_
    return normed


class Distribution:

    def __init__(self, trans, tally):
        self.name = trans.name()
        self.counts = tally
        self.trans = trans
        self.description = trans.description()
        self.feature_name = trans.feature_name()

    @property
    def plotable(self):
        if isinstance(self.range, list) or isinstance(self.range, range):
            return self.range[-1] == self.range[0] \
                   or self.range[-1] - self.range[0] <= 2**16
        return False

    @property
    def range(self):
        return self.trans.range(self.counts.keys())

    def distance(self, other, l_k_norm):
        """
        Distance to compare two distributions
        :param other: other distribution to compare to
        :param l_k_norm: e.g. 1=Manhattan, 2=Euclidean, 1/x=fractional norm
        :return: distance
        """
        if self.name != other.name:
            raise DistributionException(
                'Cannot compare distribution of {} to {}'.format(
                    self.name, other.name))
        keys = set(self.counts.keys()).union(set(other.counts.keys()))
        normed_a = normed_dictionary(self.counts)
        normed_b = normed_dictionary(other.counts)
        if l_k_norm == 1:
            return self.distance_manhattan(normed_a, normed_b, keys)
        else:
            return self.distance_general(normed_a, normed_b, keys, l_k_norm)

    @staticmethod
    def distance_manhattan(normed_a, normed_b, keys):
        dist_sum = sum([abs(normed_a.get(k, 0) - normed_b.get(k, 0))
                        for k in keys])
        return dist_sum

    @staticmethod
    def distance_general(normed_a, normed_b, keys, l_k_norm):
        dist_sum = sum([abs(normed_a.get(k, 0) - normed_b.get(k, 0))
                        ** l_k_norm
                        for k in keys])
        return dist_sum ** (1.0 / l_k_norm)

    def vlc(self):
        if not self.plotable:
            return None
        values = list(self.range)
        extra = set(self.counts) - set(values)
        if len(extra) > 0:
            if len(self.counts) == 1:
                values = list(extra)
            else:
                values = list(extra) + values
        labels = [self.trans.string(v) for v in values]
        counts = [0 for _ in values]
        for value, count in self.counts.items():
            counts[values.index(value)] = count
        if None in values:
            values = copy.deepcopy(labels)
        return values, labels, counts

    def vlc_overlay(self):
        return None

    def heat_map(self):
        return None

    def encoded_range(self):
        if isinstance(self.range, range):
            return 'range({},{},{})'.format(
                self.range.start, self.range.stop, self.range.step)
        return self.range

    def __dict__(self):
        return {'name': self.name,
                'range': self.encoded_range(),
                'counts': self.counts}

    def __str__(self):
        return str(self.__dict__())

    def export_dict(self, dist_type='distribution'):
        # All we need to reconstruct a Distribution is
        # the Transformation and the feature counts
        return {'type': dist_type, 'name': self.trans.name(),
                'counts': self.counts}

    def export_dict_json_safe(self):
        # JSON only supports strings as dictionary keys
        safe_counts = OrderedDict()
        key_eval = []
        key_eval_all = True
        for k, v in self.counts.items():
            if not isinstance(k, str):
                k = str(k)
                key_eval.append(k)
            else:
                key_eval_all = False
            safe_counts[k] = v
        dictionary = self.export_dict()
        dictionary['counts'] = safe_counts
        if key_eval_all:
            dictionary['eval'] = 'all'
        elif key_eval:
            dictionary['eval'] = key_eval
        return dictionary

    @staticmethod
    def import_dict(dictionary, transformation=None):
        dist_type = dictionary.get('type', 'distribution')
        counts = dictionary['counts']
        if transformation is None:
            transformation = Parser.parse_string(dictionary['name'])
        if dist_type == 'multi_dim_distribution':
            return MultiDimDistribution(transformation, counts)
        if dist_type == 'multi_feature_distribution':
            return MultiFeatureDistribution(transformation, counts)
        return Distribution(transformation, counts)

    @staticmethod
    def __import_counts(counts, key_eval, transformation):
        if isinstance(counts, list):
            return [Distribution.import_dict_json_safe(c, t)
                    for c, t in zip(counts, transformation.trans)]
        if key_eval == 'all':
            eval_dict = {}
            for k, v in counts.items():
                eval_dict[ast.literal_eval(k)] = v
            counts = eval_dict
        elif key_eval:
            for k in key_eval:
                v = counts.pop(k)
                counts[ast.literal_eval(k)] = v
        return counts

    @staticmethod
    def import_dict_json_safe(dictionary, transformation=None):
        """
        if transformation and not (transformation.name() == dictionary['name']):
            raise DistributionException(
                'Distribution was computed with a different transformation'
                ' "{}" vs "{}"'
                    .format(transformation.name(), dictionary['name']))
        """
        counts = dictionary['counts']
        key_eval = dictionary.get('eval', None)
        counts = Distribution.__import_counts(counts, key_eval, transformation)
        dictionary['counts'] = counts
        return Distribution.import_dict(dictionary, transformation)

    @staticmethod
    def same_type(dists):
        types = [type(t.trans) for t in dists]
        return len(set(types)) == 1

    @staticmethod
    def _overlay(dists):
        if not Distribution.same_type(dists):
            return None
        if not all([d.plotable for d in dists]):
            return None
        vlc_list = [d.vlc() for d in dists]
        val, lab, cou = zip(*vlc_list)
        val_set = set()
        largest = 0
        largest_len = 0
        for v_l, i in zip(val, range(len(val))):
            for v in v_l:
                val_set.add(v)
                if len(val[i]) > largest_len:
                    largest = i
                    largest_len = len(val[i])
        if len(val_set) != len(val[largest]):
            return None
        return val, cou, val[largest], lab[largest]

    @staticmethod
    def sum(trans, dists):
        names = set(d.name for d in dists)
        if len(names) != 1:
            raise DistributionException('Cannot sum different distributions')
        name = names.pop()
        if name != trans.name():
            raise DistributionException('Distribution was computed '
                                        'using a different Transformation')
        counts = [d.counts for d in dists]
        total = dict()
        for count in counts:
            for k, v in count.items():
                total[k] = total.get(k, 0) + v
        summed = type(dists[0])(trans, total)
        return summed


class MultiFeatureDistribution(Distribution):
    """Distribution with multiple features, e.g. P modulo 3, 5, 7"""

    def __init__(self, trans, tally):
        super().__init__(trans, tally)
        self.counts = tally
        self.count = len(self.counts)
        self.transforms = trans.trans
        self.descriptions = [t.description for t in self.counts]
        self.feature_names = [t.feature_name for t in self.counts]

    @property
    def plotable(self):
        return [d.plotable for d in self.counts]

    @property
    def range(self):
        return None

    def distance(self, other, l_k_norm):
        if self.name != other.name:
            raise DistributionException(
                'Cannot compare distribution of {} to {}'.format(
                    self.name, other.name))
        return sum(s.distance(o, l_k_norm) for s, o
                   in zip(self.counts, other.counts))

    def vlc(self):
        return None

    def vlc_overlay(self):
        return self._overlay(self.counts)

    def export_dict_json_safe(self):
        exp = self.export_dict()
        exp['counts'] = [d.export_dict_json_safe() for d in self.counts]
        return exp

    def encoded_range(self):
        return None

    def export_dict(self, dist_type='multi_feature_distribution'):
        return super().export_dict(dist_type=dist_type)


class MultiDimDistribution(Distribution):
    """Distribution with multiple dimensions, e.g. MSB of P vs Q"""

    def __init__(self, trans, tally):
        super().__init__(trans, tally)
        transforms = trans.trans
        self.dimensions = len(transforms)
        marginal = marginalize(self.dimensions, tally)
        self.marginal = [Distribution(tr, ta)
                         for tr, ta in zip(transforms, marginal)]
        self.transforms = transforms
        self.descriptions = [t.description for t in self.marginal]
        self.feature_names = [t.feature_name for t in self.marginal]

    @property
    def plotable(self):
        if isinstance(self.range, itertools.product):
            all_plotable = all([m.plotable for m in self.marginal])
            if not all_plotable:
                return False
            range_size = [len(m.range) for m in self.marginal]
            comb = 1
            for r in range_size:
                comb *= r
            return comb < 2**12
        return super().plotable

    def subspace(self, dimensions):
        trans = self.trans.subspace(dimensions)
        tally = self.trans.subspace_tally(self.counts, dimensions)
        if len(dimensions) == 1:
            return Distribution(trans, tally)
        return MultiDimDistribution(trans, tally)

    def vlc_overlay(self):
        return self._overlay(self.marginal)

    def heat_map(self):
        vs = []
        ls = []
        cs = []
        subspace = range(self.dimensions)
        # TODO smallest partitions
        divider = len(subspace) // 2
        first = subspace[:divider]
        second = subspace[divider:]
        sides = [self.subspace(first), self.subspace(second)]
        for m in sides:
            vlc = m.vlc()
            if vlc is None:
                return None
            v, l, c = vlc
            vs.append(v)
            ls.append(l)
            cs.append(c)
        heat_map = [[float('nan') for _ in vs[0]] for _ in vs[1]]
        index = [{}, {}]
        for dim in range(2):
            i = 0
            for r in vs[dim]:
                index[dim][r] = i
                i += 1
        for key_tuple, value in self.counts.items():
            x = key_tuple[:divider]
            y = key_tuple[divider:]
            heat_map[index[1][y]][index[0][x]] = value
        return heat_map, vs, ls, cs

    def encoded_range(self):
        if isinstance(self.range, itertools.product):
            return 'product({})'.format([m.encoded_range()
                                         for m in self.marginal])
        return super().encoded_range()

    def __dict__(self):
        d = super().__dict__()
        d['dimensions'] = self.dimensions
        d['marginal'] = [str(m) for m in self.marginal]
        return d

    def export_dict(self, dist_type='multi_dim_distribution'):
        return super().export_dict(dist_type=dist_type)


class DistributionException(Exception):
    pass
