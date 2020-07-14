# Baby dataset

This folder contains a baby dataset that can be used to experiment with our tool. The dataset contains 70 distinct RSA keys of bitlength 512 gathered from 7 different smart card models. 

Such dataset is technically loaded by the [dataset class](https://github.com/matusn/RSABias/blob/master/rsabias/dataset.py). The keys can be distributed across multiple folders, and can be optionally compressed. Each record in the dataset -- a list of keys originating from the same source -- is represented by a `meta.json` file. Example of such file is presented below, together with comments of various properties:

### meta.json example

```json
{
    "details": {
        "base_dict": { # what records are taken for each of the keys. 
            "d": 16, # 16 suggests hexadecimal record
            "e": 16,
            "id": 10, # 10 suggests decimal record
            "n": 16,
            "p": 16,
            "q": 16,
            "t": 10
        },
        "bitlen": 512, # length of the keys. Should be same for all keys.
        "category": "Card", # category of the key. Invent your own categories if you want
        "compressed": true, # whether the keys are compressed or not
        "fips_mode": false, # whether the keys are in FIPS mode or not
        "format": "json", # json or csv
        "header": null, # If the files have header or not. 
        "name": "Athena", # Name of the source
        "public_only": false, # Set to true if the primes are not available
        "separator": null, # only if csv is used. Specifies a separator
        "version": "IDProtect" # Version of the source
    },
    "files": [ # Each source can be distributed across multiple files
        {
            "name": "131a196326d1cdd1.json.gz",
            "records": 10, # how many keys in a file.
            "sha256": "131a196326d1cdd17629d6a68fe59ab07037cf35e361d4f9fb4b582153b04cbe"
        }
    ],
    "type": "reference"
}
```

