#!/bin/bash

# 跑进去环境cathAplhaflow
# 检查是否提供了参数
if [ $# -eq 0 ]; then
    echo "错误: 未提供目录路径作为参数。"
    exit 1
fi

# 获取传入的目录路径
directory=$1

# 尝试切换到指定目录
cd "$directory" 



DAT_DIR="query_dat"
TARGET_DAT_DIR="target_dat"

NP=60
MPI="/usr/bin/mpirun"

echo "generate pdb_list"
ls $TARGET_DAT_DIR | perl -pe 's/\.dat//' > pdb.list
echo "generate fasta"
../bin/dat2fasta.pl $TARGET_DAT_DIR < pdb.list  | awk -v RS=">" -v FS="\n" -v ORS="" ' { if ($2) print ">"$0 } ' > pdb.fasta
# ../bin/dat2fasta.pl $TARGET_DAT_DIR < pdb.list  | awk -v RS=">" -v FS="\n" -v ORS="" ' { if ($2) print ">"$0 } ' > pdb.fasta


echo "generate blast"
makeblastdb  -in pdb.fasta -out pdb.blast -dbtype prot > makeblastdb.stdout 2> makeblastdb.stderr

# ls $TARGET_DAT_DIR | perl -pe 's/\.dat//' > pdb.list
# ../bin/dat2fasta.pl $TARGET_DAT_DIR < pdb.list  | awk -v RS=">" -v FS="\n" -v ORS="" ' { if ($2) print ">"$0 } ' > pdb.fasta
# makeblastdb  -in pdb.fasta -out pdb.blast -dbtype prot > makeblastdb.stdout 2> makeblastdb.stderr


for file in $DAT_DIR/*.dat; do
    # 获取文件名（去除扩展名）
    cd1_id=$(basename "$file" .dat)
    echo $cd1_id
    # 执行命令并将标准输出和标准错误输出重定向到对应的文件
    ../bin/dali.pl --np $NP --MPIRUN_EXE $MPI --cd1 "$cd1_id" --db pdb.list --TITLE systematic --dat1 $DAT_DIR --dat2 $TARGET_DAT_DIR > "systematic_${cd1_id}.stdout" 2> "systematic_${cd1_id}.stderr"
    
    # 等待命令完成
    wait
done