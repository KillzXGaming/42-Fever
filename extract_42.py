# extracts the '42' archive from Mario Tennis Fever.
# based on the original Mario Golf Super Rush extract_42 script.
# Supports only 1.0 and 1.1 (ensure the correct fst.json is used)

import json
import os
import numpy as np

CHUNK_RANGE = 64 * 1024 * 1024 
KEY_XOR = [0x11111111, 0x22222223, 0x33333333, 0x44444445,
           0x55555555, 0x66666667, 0x77777777, 0x88888889]

def crc_table(poly):
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            if c & 1:
                c = (c >> 1) ^ poly
            else:
                c >>= 1
        table.append(c)
    return np.array(table, dtype=np.uint32)

# keys 0-3 normal CRC-32 table, keys 4-7 use CRC-32C
crc_tables = [crc_table(0xEDB88320), crc_table(0x82F63B78)]

def crc_of_words(words, table, seed):
    crc = np.full(len(words), seed, dtype=np.uint32)
    for shift in range(0, 64, 8):
        byte = ((words >> np.uint64(shift)) & np.uint64(0xFF)).astype(np.uint32)
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc

def decrypt(data, key):
    variant = key & 7
    table = crc_tables[variant >> 2]
    factor = np.uint64(((((key & 0xFFFFFFFE) ^ KEY_XOR[variant]) & 0xFFFFFFFF) << 32) | 1)

    words = np.frombuffer(data, dtype="<u8").reshape(-1, 2)
    crc_word = words[:, 0]
    other_word = words[:, 1]

    if variant & 2:
        crc_word, other_word = other_word, crc_word

    crc = crc_of_words(crc_word, table, key).astype(np.uint64)
    product = factor * crc

    new_crc = product ^ ~(crc_word << np.uint64(32))
    new_other = other_word ^ (product >> 3)

    plain = np.empty_like(words)
    if variant & 1:
        plain[:, 0] = new_other
        plain[:, 1] = new_crc
    else:
        plain[:, 0] = new_crc
        plain[:, 1] = new_other

    return plain.tobytes(), int(new_other.sum(dtype=np.uint64))

basedir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(basedir, "fst.json"), "r") as f:
    fst = json.load(f)

with open(os.path.join(basedir, "42"), "rb") as arc:
    for filename, info in fst.items():
        print(f"{filename}...")

        out_path = os.path.join(basedir, "42_output", filename)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            continue

        arc.seek(info["offset"])

        size = info["size"]
        padded = (size + 15) & ~15
        key = int(info["key"], 16)

        data = bytearray()
        total = 0

        for pos in range(0, padded, CHUNK_RANGE):
            chunk = arc.read(min(CHUNK_RANGE, padded - pos))
            plain, part = decrypt(chunk, key)
            data += plain
            total = (total + part) & 0xFFFFFFFFFFFFFFFF

        del data[size:]

        # checksum test
        assert ((total >> 32) ^ total) & 0xFFFFFFFF == info["crc"]

        with open(out_path, "wb") as f:
            f.write(data)
