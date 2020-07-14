"""
Each source in a data set has file(s) that contain keys and meta info file.

To read from a dataset, supply the path to a meta file to DataSet.prepare().
To read multiple data sets, use DataSet.find().

Then get a DataSetIterator using DataSet.iterator().

To output keys, use Writer.import_meta() with the proper DataSetMeta.

Example usage can be found in tasks.convert().

"""
import json
import copy
import csv
import os
import gzip
import io
import hashlib


import rsabias.core.key as key


class DataSetException(Exception):
    pass


class DataSetMeta:
    """ Data set metadata - source, details, list of files, ... """

    class File:
        """ A file in the data set, for a single source and key length. """

        def __init__(self, name, records, digest):
            self.name = name  # file name
            self.records = records  # number of records (keys)
            self.digest = digest  # hash of the decompressed content

        @staticmethod
        def import_dict(d):
            name = d.get('name', None)
            records = d.get('records', None)
            digest = d.get('sha256', None)
            return DataSetMeta.File(name, records, digest)

        def export_dict(self):
            return {'name': self.name, 'records': self.records,
                    'sha256': self.digest}

    class Details:
        """ Detailed information about a data set. """

        def __init__(self, base_dict, bitlen, category, compressed, fips_mode, 
                     ds_format, header, name, public_only, separator, version, group=None):
            self.base_dict = base_dict  # dictionary "feature name: base"
            self.bitlen = bitlen  # binary length of the modulus (e.g., 2048)
            self.category = category  # source type (e.g., Library, Card, HSM)
            self.compressed = compressed  # compressed with GZIP or plain text
            self.fips_mode = fips_mode  # FIPS mode of a library was active
            self.format = ds_format  # data set format (e.g., "json", "csv")
            self.header = header  # CSV header as a list of strings
            self.name = name  # name of the source (e.g. "OpenSSL")
            self.public_only = public_only  # only public keys available
            self.separator = separator  # CSV separator (e.g., ";", ",")
            self.version = version  # source version (e.g., "1.0.2g")
            self.group = group

        @staticmethod
        def import_dict(d):
            base_dict = d.get('base_dict', None)
            bitlen = d.get('bitlen', None)
            category = d.get('category', None)
            compressed = d.get('compressed', None)
            fips_mode = d.get('fips_mode', None)
            ds_format = d.get('format', None)
            header = d.get('header', None)
            name = d.get('name', None)
            public_only = d.get('public_only', None)
            separator = d.get('separator', None)
            version = d.get('version', None)
            group = d.get('group', None)
            return DataSetMeta.Details(base_dict, bitlen, category, compressed,
                                       fips_mode, ds_format, header, name,
                                       public_only, separator, version, group)

        def export_dict(self):
            return {'base_dict': self.base_dict, 'bitlen': self.bitlen,
                    'category': self.category, 'compressed': self.compressed,
                    'fips_mode': self.fips_mode, 'format': self.format,
                    'header': self.header, 'name': self.name,
                    'public_only': self.public_only,
                    'separator': self.separator, 'version': self.version, 'group': self.group}

    # TODO opaque notes, currently undefined properties are not copied over
    def __init__(self, ds_type, files, details):
        self.type = ds_type
        self.files = files
        self.details = details

    @staticmethod
    def import_dict(meta):
        ds_type = meta.get('type', None)
        files = [DataSetMeta.File.import_dict(f)
                 for f in meta.get('files', [])]
        details = DataSetMeta.Details.import_dict(meta.get('details', []))
        return DataSetMeta(ds_type, files, details)

    def export_dict(self):
        return {'type': self.type,
                'files': [f.export_dict() for f in self.files],
                'details': self.details.export_dict()}

    def count_records(self):
        total = 0
        for f in self.files:
            total += f.records
        return total

    def source(self):
        """
        Return the name of the source as a tuple.

        (Category, Name, Version [FIPS], Bit length [PUBLIC])

        :return: a tuple representing the name of the source
        """
        is_fips = ' FIPS' if self.details.fips_mode else ''
        is_public = ' PUBLIC' if self.details.public_only else ''
        cat = self.details.category
        return (cat if cat else 'No category',
                self.details.name,
                self.details.version + is_fips,
                str(self.details.bitlen) + is_public)

    def source_path(self):
        """
        Return the typical directory path for the data set.

        :return: relative directory path, where the data set should be found
        """
        parts = self.source()
        path = ''
        for p in parts:
            # strip dot to avoid making hidden directories
            path = os.path.join(path, str(p).lstrip('.'))
        return path

    def get_full_name(self):
        fps_mode = ' '
        if self.details.fips_mode is True:
            fps_mode = ' FIPS '
        return self.details.category + ' ' + self.details.name + ' ' + self.details.version + fps_mode + str(self.details.bitlen)


class DataSet:
    """
    Data set (of keys) that can be instantiated from metadata.

    Provides transparent access to keys in different file types
    using a forward only iterator.
    """

    def __init__(self, meta):
        self.meta = meta

    @staticmethod
    def import_meta(meta, path=None):
        """
        Create from meta data (DataSetMeta).

        :param meta: DataSetMeta meta data
        :param path: TODO ??
        :return: DataSet
        """
        if not meta.type:
            raise DataSetException('Missing file type:\n{}'.format(meta))
        if meta.type == 'reference':
            return ReferenceDataSet(meta, path)
        raise DataSetException(
            'Unsupported data set type "{}"'.format(meta.type))

    @staticmethod
    def __find_paths(start, filename):
        parent_paths = []
        for root, dirs, files in os.walk(start):
            if filename in files:
                parent_paths.append(root)
        return parent_paths

    @staticmethod
    def prepare(path, meta_filename):
        """
        Create DataSet from path with the the metadata file

        :param path: directory path to the meta file
        :param meta_filename: name of the meta file (e.g., "meta.json")
        :return: DataSet
        """
        meta_path = os.path.join(path, meta_filename)
        with open(meta_path) as fp:
            meta = DataSetMeta.import_dict(json.load(fp))
        return DataSet.import_meta(meta, path)

    @staticmethod
    def find(start, meta_name='meta.json'):
        """
        Finds all data sets starting from a root path.

        :param start: root directory path
        :param meta_name: file name of the meta file
        :return: list of DataSet
        """
        parent_paths = DataSet.__find_paths(start, meta_name)
        return [DataSet.prepare(path, meta_name) for path in parent_paths]

    def iterator(self, prime_wise=False):
        """
        Get an iterator for the data set.

        :return: iterator
        """
        pass

    def __str__(self):
        return str(vars(self))

    def export_json(self):
        return json.dumps(vars(self), sort_keys=False, indent=4)


class CatchIterator:
    """
    Iterator wrapper that catches exceptions when keys cannot be parsed.
    """

    def __init__(self, iter, name):
        self.iter = iter
        self.name = name

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                return self.iter.__next__()
            except key.KeyException as err:
                print('Malformed key in dataset {}'.format(self.name))
                print(err.args)


class ReferenceDataSet(DataSet):
    """
    Reference data set of keys generated from a source.
    """

    def __init__(self, meta, path):
        DataSet.__init__(self, meta)
        self.path = path

    def iterator(self, prime_wise=False):
        """
        Get an iterator for the data set.

        :return: iterator
        """
        format_ = self.meta.details.format
        bases = self.meta.details.base_dict
        cmp = self.meta.details.compressed
        files = self.meta.files
        path = self.path
        sep = self.meta.details.separator
        sep = sep if sep else ';'
        it = None
        if format_ == 'csv':
            it = CSVFileIterator(files, self.path, bases, compressed=cmp, separator=sep)
        if format_ == 'json':
            it = JSONFileIterator(files, path, bases, compressed=cmp)
        if format_ == 'jcalgtest':
            it = AlgTestFileIterator(files, path, bases, compressed=cmp)
        if format_ == 'multiline':
            it = MultiLineFileIterator(files, path, bases, compressed=cmp)
        if format_ == 'multilinebase64':
            it = MultiLineBase64FileIterator(files, path, bases, compressed=cmp)
        if format_ == 'asn1':
            it = ASN1FileIterator(files, path, bases, compressed=cmp)
        if format_ == 'pem':
            it = PEMFileIterator(files, path, bases, compressed=cmp)
        if format_ == 'tpm_multiline':
            it = TPMMultiLineFileIterator(files, path, bases, compressed=cmp)
        if format_ == 'tpm_modulus':
            it = TPMModulusFileIterator(files, path, bases, compressed=cmp)
        if format_ == 'tpm_xml':
            it = TPMXMLFileIterator(files, path, bases, compressed=cmp)
        if it is None:
            raise DataSetException('Unsupported DS format "{}"'.format(format_))
        if prime_wise:
            return SinglePrimeIterator(it)
        return it


class HashIO(io.BytesIO):
    """
    Wrapper for hashing files as they are read/written.

    Hashing should happen on the text files, since compression
    may result in different binary files.
    """

    def __init__(self, file):
        super().__init__()
        self.file = file
        self.hash = hashlib.sha256()

    def read(self, size=-1):
        read_buf = self.file.read(size)
        if read_buf:
            self.hash.update(read_buf)
        return read_buf

    def read1(self, size=-1):
        read_buf = self.file.read1(size)
        if read_buf:
            self.hash.update(read_buf)
        return read_buf

    def readinto(self, b):
        read = self.file.readinto(b)
        if read > 0:
            self.hash.update(b[0:read])
        return read

    def readinto1(self, b):
        read = self.file.readinto1(b)
        if read > 0:
            self.hash.update(b[0:read])
        return read

    def write(self, b):
        written = self.file.write(b)
        if written > 0:
            self.hash.update(b[0:written])
        return written

    def __next__(self):
        next_ = self.file.__next__()
        self.hash.update(next_)
        return next_

    def digest(self):
        return self.hash.hexdigest()

    def close(self):
        self.file.close()


class DataSetIterator:
    """ Iterate over a data set. """

    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        pass


class FileDataSetIterator(DataSetIterator):
    """ Iterate over data saved in files. """

    def __init__(self, files, path, base_dict, binary=False, compressed=True,
                 check_hash=True, separator=None):
        super().__init__()
        self.files = copy.deepcopy(files)  # list of files from DataSetMeta
        self.files.reverse()
        self.current_file = None  # currently open file
        self.path = path  # TODO
        self.base_dict = base_dict  # dictionary "feature name: base"
        self.binary = binary  # True if binary data, False if text
        self.compressed = compressed  # True if GZIP compressed
        self.check_hash = check_hash  # if True, check hash when reading
        self.separator = separator  # CSV separator

    def __next__(self):
        if self.current_file is None:
            if not self.files:  # empty list
                raise StopIteration
            next_file = self.files.pop()
            filename = os.path.join(self.path, next_file.name)
            file = open(filename, mode='rb')

            self.hash = self.check_hash and next_file.digest
            if self.hash:
                self.original_hash = next_file.digest
            self.hash_file = None

            if self.compressed:
                self.current_file = gzip.GzipFile(fileobj=file, mode='rb')
                if self.hash:
                    self.hash_file = HashIO(self.current_file)
                    self.current_file = self.hash_file
            else:
                if self.hash:
                    file = HashIO(file)
                    self.hash_file = file
                if self.binary:
                    self.current_file = file
                else:
                    self.current_file = io.TextIOWrapper(file,
                                                         encoding='utf-8')
            self.reader = self.init_reader(self.current_file)
        try:
            key_dict = self.reader.__next__()
        except StopIteration:
            self.current_file.close()
            self.current_file = None
            if self.hash_file:
                if self.original_hash != self.hash_file.digest():
                    raise DataSetException('Hashes of files differ')
            return self.__next__()
        return key.Key.import_dict(key_dict, self.base_dict)

    def init_reader(self, file):
        """
        Initialize the underlying reader.

        :param file: the underlying file
        :return: Reader
        """
        return file


class CSVFileIterator(FileDataSetIterator):
    """ Iterate over data saved in CSV files. """

    def init_reader(self, file):
        if self.binary or self.compressed:
            self.current_file = io.TextIOWrapper(self.current_file,
                                                 encoding='utf-8')
        return csv.DictReader(self.current_file, delimiter=self.separator)


class SinglePrimeIterator(DataSetIterator):
    """
    Iterates over individual primes rather than keys.

    Key has to be a dictionary. Each prime will be called 'p' for simplicity.
    """

    def __init__(self, iterator):
        super().__init__()
        self.iterator = iterator
        self.cached_prime = None

    def __next__(self):
        if self.cached_prime is None:
            k = self.iterator.__next__()
            extra_params = ['batch', 'id']
            p = key.Key({'p': k['p']})
            q = key.Key({'p': k['q']})
            for ep in extra_params:
                if ep in k:
                    p[ep] = k[ep]
                    q[ep] = k[ep]
            self.cached_prime = q
            return p
        q = self.cached_prime
        self.cached_prime = None
        return q


class Reader:
    """
    Holds state for an iterator, works on single files.

    An iterator can correspond to multiple files (hence Readers).
    """

    def __init__(self, file):
        self.file = file
        self.curr = None

    def __iter__(self):
        return self


class JSONReader(Reader):
    """ Read JSON line from a file. """

    def __next__(self):
        return json.loads(self.file.__next__())


class JSONFileIterator(FileDataSetIterator):
    """ Iterate over data saved in files, where every line is JSON. """

    def init_reader(self, file):
        return JSONReader(file)


class AlgTestReader(Reader):
    """ Read keys in JCAlgTest format. """

    def __next__(self):
        return key.Key.import_alg_test(self.file)


class AlgTestFileIterator(FileDataSetIterator):
    """ Iterate over data saved in output of AlgTest utility. """

    def init_reader(self, file):
        return AlgTestReader(file)


class MultiLineReader(Reader):
    """ Read keys spread over multiple lines. """

    def __next__(self):
        return key.Key.import_multi_line(self.file)


class MultiLineFileIterator(FileDataSetIterator):
    """ Iterate over keys saved over multiple lines. """

    def init_reader(self, file):
        return MultiLineReader(file)


class MultiLineBase64Reader(Reader):
    """ Read keys spread over multiple lines, with values in base64. """

    def __next__(self):
        return key.Key.import_multi_line_base64(self.file)


class MultiLineBase64FileIterator(FileDataSetIterator):
    """
    Iterate over keys saved over multiple lines.

    The field names are different than MultiLineFileDataSetIterator
    and the values are in base64 and not hexadecimal
    """

    def init_reader(self, file):
        return MultiLineBase64Reader(file)


class ASN1Reader(Reader):
    """ Read ASN.1 encoded keys. """

    def __init__(self, file):
        super().__init__(file)
        self.content = file.read()
        self.file_size = len(self.content)
        self.read = 0
        self.id = 0

    def __next__(self):
        if self.read >= self.file_size:
            raise StopIteration
        length = int(self.content[self.read + 2:self.read + 4].hex(),
                     base=16) + 4
        der = self.content[self.read:self.read + length]
        self.read += length
        k = key.Key.import_asn1(der)
        k['id'] = self.id
        self.id += 1
        return k


class ASN1FileIterator(FileDataSetIterator):
    """ Iterate over keys saved as ASN.1 DER object without delimiters. """

    def __init__(self, files, path, base_dict, compressed):
        super().__init__(files, path, base_dict, binary=True,
                         compressed=compressed)

    def init_reader(self, file):
        return ASN1Reader(file)


class PEMReader(Reader):
    """ Read PEM keys spread over multiple lines. """

    def __next__(self):
        lines = str()
        while True:
            line = self.file.__next__()
            lines += line
            if line.startswith('-----END'):
                break
        return key.Key.import_standard_format(lines)


class PEMFileIterator(FileDataSetIterator):
    """ Iterate over PEM keys saved over multiple lines. """

    def init_reader(self, file):
        return PEMReader(file)


class TPMMultiLineReader(Reader):

    def __next__(self):
        return key.Key.import_tpm_multi_line(self.file)


class TPMMultiLineFileIterator(FileDataSetIterator):
    """ Iterate over keys saved over multiple lines. """

    def init_reader(self, file):
        return TPMMultiLineReader(file)


class TPMModulusReader(Reader):

    def __next__(self):
        return key.Key.import_tpm_modulus(self.file)


class TPMModulusFileIterator(FileDataSetIterator):
    """ Iterate over keys saved over multiple lines. """

    def init_reader(self, file):
        return TPMModulusReader(file)


class TPMXMLReader(Reader):

    def __next__(self):
        return key.Key.import_tpm_xml(self.file)


class TPMXMLFileIterator(FileDataSetIterator):
    """ Iterate over keys saved over multiple lines. """

    def init_reader(self, file):
        return TPMXMLReader(file)


class Writer:
    """
    Writes keys to output. Requires DataSetMeta and path.
    Optionally saves distributions (possibly only distributions and no keys).
    """

    def __init__(self, root_path, meta, skip_writing=False):
        """
        Handles output of keys and distributions.

        :param root_path: path to the root output directory
        :param meta: DataSetMeta, used e.g. to compute rest of path
        :param skip_writing: if True, keys are not written to output
                             (used to only save distributions)
        """
        self.path = None
        self.meta = meta
        self.skip_writing = skip_writing
        path = os.path.join(root_path, meta.source_path())
        self.__created_files = []
        # list of created files that should be deleted on error
        while not self.path:
            try:
                os.makedirs(path)
                self.path = path
                self.__created_files.append(path)
            except FileExistsError:
                path += '+'
        self.written = 0

        if self.skip_writing:
            return

        self.file_name = os.path.join(self.path, self.default_file_name())
        if self.meta.details.compressed:
            self.file_name += '.gz'
            self.file = gzip.open(self.file_name, mode='wb')
        else:
            self.file = open(self.file_name, mode='wb')
        self.__created_files.append(self.file_name)
        self.file = HashIO(self.file)

    def save_distributions(self, dists):
        """
        Save distributions to the correct path.

        :param dists: Distribution
        :return:
        """
        json_safe = dists.export_dict_json_safe()
        dist_out_path = os.path.join(self.path, 'dist.json')
        with open(dist_out_path, mode='w') as f:
            f.writelines(json.dumps(json_safe, sort_keys=False, indent=4))
        self.__created_files.append(dist_out_path)

    def save_meta(self):
        """
        Save a copy of the meta file to the correct path.

        :return:
        """
        meta_out_path = os.path.join(self.path, 'meta.json')
        with open(meta_out_path, mode='w') as f:
            f.writelines(json.dumps(self.meta.export_dict(),
                                    sort_keys=True, indent=4))
        self.__created_files.append(meta_out_path)

    def close(self):
        """
        Close the Writer. Saves meta files, closes files, etc.

        :return:
        """
        if self.skip_writing:
            self.save_meta()
            return

        self.file.close()
        digest = self.file.digest()

        old_path = self.file_name
        old_name = os.path.basename(old_path)
        new_name = old_name.replace(old_name.split('.')[0], digest[0:16])
        new_path = os.path.join(os.path.dirname(old_path), new_name)
        os.rename(old_path, new_path)
        self.__created_files.remove(old_path)
        self.__created_files.append(new_path)

        file_info = DataSetMeta.File(new_name, self.written, digest)
        self.meta.files = [file_info]
        self.save_meta()
        self.__created_files = []

    def cleanup(self):
        """
        Perform a clean up of created files, if unsuccessful.

        :return:
        """
        for f in reversed(self.__created_files):
            os.remove(f)

    def write(self, string):
        """
        Write a string representation of key to file.

        :param string: string to be written
        :return:
        """
        self.written += 1
        if self.skip_writing:
            return
        line = '{}\n'.format(string)
        encoded = line.encode('utf-8')
        self.file.write(encoded)

    @staticmethod
    def default_file_name():
        return 'keys.txt'

    @staticmethod
    def import_meta(root_path, old_meta, out_format=None, compress=None,
                    header=None, separator=None, new_bases=None,
                    skip_writing=False):
        """
        Create a writer from DataSetMeta.

        Differences from source data set are supplied with other parameters.

        :param root_path: output root directory path
        :param old_meta: original DataSetMeta
        :param out_format: output format (e.g., "json", "csv")
        :param compress: if True, output is compressed with GZIP
        :param header: CSV header
        :param separator: CSV separator
        :param new_bases: output dictionary "feature name: base"
        :param skip_writing: if True, no keys are written to output
                             (used to only output distributions)
        :return: Writer of the correct type
        """
        new_meta = copy.deepcopy(old_meta)
        if out_format is not None:
            new_meta.details.format = out_format
        if compress is not None:
            new_meta.details.compressed = compress
        if header is not None:
            new_meta.details.header = header
        if separator is not None:
            new_meta.details.separator = separator
        if new_bases is not None:
            for k, v in new_bases.items():
                if v is not None:
                    new_meta.details.base_dict[k] = v
        output_format = new_meta.details.format
        if output_format == 'json':
            return JSONWriter(root_path, new_meta, skip_writing=skip_writing)
        elif output_format == 'csv':
            return CSVWriter(root_path, new_meta, header, separator,
                             skip_writing=skip_writing)
        else:
            return Writer(root_path, new_meta, skip_writing=skip_writing)


class CSVWriter(Writer):
    """
    Writing CSV output. Preferably use Writer.import_meta() instead.
    """

    def __init__(self, path, meta, header, separator=';', skip_writing=False):
        """
        Use Writer.import_meta() instead, that handles meta and distributions.

        :param path:
        :param meta:
        :param header:
        :param separator:
        :param skip_writing:
        """
        super().__init__(path, meta, skip_writing=skip_writing)
        if header is None:
            header = ['id', 'n', 'e', 'p', 'q', 'd', 't']
        if separator is None:
            separator = ';'
        self.header = header
        super().write(separator.join(header))
        # do not count header as record
        self.written = 0
        self.format_dict = key.Key.base_to_format_dict(meta.details.base_dict)

    def write(self, k):
        if 'id' not in k:
            k['id'] = self.written
        k_export = k.export_csv(self.header, format_dict=self.format_dict)
        super().write(k_export)

    @staticmethod
    def default_file_name():
        return 'keys.csv'


class JSONWriter(Writer):
    """
    Writing JSON output. Preferably use Writer.import_meta() instead.
    """

    def __init__(self, path, meta, skip_writing=False):
        """
        Use Writer.import_meta() instead, that handles meta and distributions.

        :param path:
        :param meta:
        :param skip_writing:
        """
        super().__init__(path, meta, skip_writing=skip_writing)
        self.format_dict = key.Key.base_to_format_dict(meta.details.base_dict)

    def write(self, k):
        if 'id' not in k:
            k['id'] = self.written
        k_export = k.export_json_string(format_dict=self.format_dict)
        super().write(k_export)

    @staticmethod
    def default_file_name():
        return 'keys.json'
