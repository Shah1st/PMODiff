import lmdb

p = "data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
db = lmdb.open(p, subdir=False, readonly=True, lock=False, readahead=False, meminit=False)
with db.begin() as txn:
    n = sum(1 for _ in txn.cursor().iternext(values=False))
print("LMDB key count:", n)
