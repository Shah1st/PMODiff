source ~/.bashrc
conda activate tagmol
cd /mnt/scratch/users/andrij/PMODiff

python - << 'EOF'
import os

path = "./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb"
print("cwd:", os.getcwd())
print("path repr:", repr(path))
print("exists?", os.path.exists(path))
print("isfile?", os.path.isfile(path))

print("\nDirectory listing of ./data/:")
for name in os.listdir("data"):
    print(" ", repr(name))
EOF

