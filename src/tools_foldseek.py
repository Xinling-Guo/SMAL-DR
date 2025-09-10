import subprocess
import shutil
import logging
import argparse
import sys
import subprocess
sys.path.append("/mnt/sdb4/protein_gen/ProDomain/cath-alphaflow-main/")
from cath_alphaflow.io_utils import get_csv_dictwriter
from cath_alphaflow.models.domains import AFDomainID, LURSummary, pLDDTSummary, AFChainID
from cath_alphaflow.constants import DEFAULT_CIF_SUFFIX, DEFAULT_DSSP_SUFFIX, DEFAULT_FS_QUERYDB_NAME,DEFAULT_FS_QUERYDB_SUFFIX
import os
import sys
import csv
import subprocess
import time
import re
import pandas as pd
from pathlib import Path
from Bio.PDB import MMCIF2Dict
from Bio.PDB import MMCIFParser, PDBParser ,DSSP
from cath_alphaflow.io_utils import yield_first_col
from cath_alphaflow.settings import get_default_settings
from cath_alphaflow.errors import ArgumentError
from tempfile import TemporaryDirectory
import glob
from cath_alphaflow.settings import get_default_settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import time
config = get_default_settings()
logging.basicConfig(level=logging.DEBUG)
FS_BINARY_PATH = config.FS_BINARY_PATH
FS_TMP_PATH = config.FS_TMP_PATH
FS_OVERLAP = config.FS_OVERLAP
DEFAULT_FS_COV_MODE = "0" # overlap over query and target
DEFAULT_FS_ALIGNER = "2" # 3di+AA (fast, accurate)
DEFAULT_FS_FORMAT_OUTPUT = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,qlen,tstart,tend,tlen,qcov,tcov,bits,evalue,qca,tca,alntmscore,qtmscore,ttmscore,u,t,lddt,lddtfull,prob"

def convert_fasta_to_mmseqs2_db(fasta_file, db_name="./fasta_db"):
    """将FASTA文件转换为MMseqs2数据库"""
    db_path = db_name + ".db"
   
    subprocess.run(
        [
            MM_PATH,
            "createdb",
            fasta_file,
            db_path,
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )

    print(f"Database created: {db_path}")

def create_mmseqs2_index(db_name):
    """为数据库创建索引"""
    index_name = f"{db_name}_index"
    db_Name = f"{db_name}.db"
    subprocess.run(
        [
            MM_PATH,
            "createindex",
            db_Name,
            index_name,
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )
    print(f"Index for '{db_name}' created successfully.")
    

   
def run_mmseqs2_search(query_db, target_db, result_file,tmp_dir):
    """运行MMseqs2的搜索"""
    assert os.path.exists(query_db), f"Query DB {query_db} does not exist."
    assert os.path.exists(target_db), f"Target DB {target_db} does not exist."
    output_file = result_file
    subprocess.run(
        [
            MM_PATH,
            "search",
            query_db,
            target_db,
            result_file,
            tmp_dir,
            "--threads",
            "1",  # 可以根据需要调整线程数
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )

    print("MMseqs2 search done")    

def get_mmseqs2_results(query_db, target_db, results_search_dir, output_file):
    """转换比对结果为可读格式"""
    
    result = subprocess.run(
    [
        MM_PATH,
        "convertalis",
        query_db,
        target_db,
        results_search_dir,
        output_file,
    ],
    capture_output=True,
    text=True,
    )
    if result.returncode != 0:
        print("Error:", result.stderr)
    else:
        print("Output:", result.stdout)
    print(f"Results converted and saved in '{output_file}'.")
    
    
def extract_aligned_regions_for_tm_align(query_seq, target_seq, alignment):
    """
    Extract aligned residue intervals from query and target sequences based on the alignment string.
    
    :param query_seq: The query sequence string (e.g., sequence from Chain_1)
    :param target_seq: The target sequence string (e.g., sequence from Chain_2)
    :param alignment: The alignment string (containing ":" and "." representing the alignment)
    :return: List of tuples containing aligned regions as (query_start, query_end, target_start, target_end)
    """
    aligned_query_regions = []
    aligned_target_regions = []
    
    regions = []
    start = None

    for i, char in enumerate(alignment):
        if char in [":","."]:
            if start is None:
                start = i  # mark the start of a new non-dash segment
        else:
            if start is not None:
                if i - start == 1:
                    regions.append(str(start))  # single position
                else:
                    regions.append(f"{start}-{i - 1}")  # range of positions
                start = None
    
    # Handle the case where the last segment ends without a '-'
    if start is not None:
        regions.append(str(start))
    
    
        # Convert positions into a list of actual indices
    for pos in regions:
        if '-' in pos:  # it's a range
            start, end = map(int, pos.split('-'))
            count_start = sum(1 for c in query_seq[:start+1] if c != '-')
            count_end = sum(1 for c in query_seq[:end+1] if c != '-')
            aligned_query_regions.append(f"{count_start}-{count_end}")
        else:  # single position
            index = int(pos)
            count = sum(1 for c in query_seq[:index+1] if c != '-')
            aligned_query_regions.append(str(count))
    
    for pos in regions:
        if '-' in pos:  # it's a range
            start, end = map(int, pos.split('-'))
            count_start = sum(1 for c in target_seq[:start+1] if c != '-')
            count_end = sum(1 for c in target_seq[:end+1] if c != '-')
            aligned_target_regions.append(f"{count_start}-{count_end}")
        else:  # single position
            index = int(pos)
            count = sum(1 for c in target_seq[:index+1] if c != '-')
            aligned_target_regions.append(str(count))
    
    
    
    
    
    return aligned_query_regions, aligned_target_regions

def parse_TMalign_results(txt_file): 
    """
    解析 TM-align 的 txt 文件，提取相关信息
    :param txt_file: TM-align 输出的 txt 文件路径
    :return: 提取的相关信息，字典格式
    """
    with open(txt_file, 'r') as f:
        content = f.read()
    with open(txt_file, 'r') as f:
        # 读取文件所有行，存储在列表中
        content2 = f.readlines()

    # 使用正则表达式提取信息
    protein1 = re.search(r"Name of Chain_1: .*/([^/]+)\.pdb", content)
    protein2 = re.search(r"Name of Chain_2: (.+)", content)
    length_chain1 = re.search(r"Length of Chain_1: (\d+) residues", content)
    length_chain2 = re.search(r"Length of Chain_2: (\d+) residues", content)
    aligned_length = re.search(r"Aligned length= (\d+)", content)
    rmsd = re.search(r"RMSD=\s+(\S+)", content)
    tm_score1 = re.search(r"TM-score= (\S+) \(if normalized by length of Chain_1", content)
    tm_score2 = re.search(r"TM-score= (\S+) \(if normalized by length of Chain_2", content)

    # 如果没有找到相应的字段，则返回None
    if not all([protein1, protein2, length_chain1, length_chain2, aligned_length, rmsd, tm_score1, tm_score2]):
        return None

    # 获取提取的值
    protein1 = os.path.basename(protein1.group(1)).replace(".pdb", "")
    protein2 = os.path.basename(protein2.group(1)).replace(".pdb", "")
    length_chain1 = int(length_chain1.group(1))
    length_chain2 = int(length_chain2.group(1))
    aligned_length = int(aligned_length.group(1))
    tm_score1 = float(tm_score1.group(1))
    tm_score2 = float(tm_score2.group(1))
    if rmsd:
        # 去掉可能的逗号并转换为浮点数
        rmsd = float(rmsd.group(1).strip(','))
    else:
        print("RMSD not found.")

    # 计算两个TM-score的平均值作为weight
    ave_weight = (tm_score1 + tm_score2) / 2
    weight = max(tm_score1 , tm_score2)

    
    align_keyword = "denotes residue pairs of d"
    i_align = 0
    for i, line in enumerate(content2):
        if align_keyword in line:
            i_align = i  # 返回行号，行号是从 0 开始的
    if i_align > 0 :
        query_seq = content2[i_align+1]
        target_seq = content2[i_align+3]
        alignment = content2[i_align+2]
        aligned_regions = extract_aligned_regions_for_tm_align(query_seq, target_seq, alignment)
    else:
        print(f"{protein1}vs{protein2} has no alinged sequence")

    # 返回结果
    return {
        "source": protein1,
        "target": protein2,
        "weight": weight,
        "ave_weight": ave_weight,
        "aligned_length": aligned_length,
        "rmsd": rmsd,
        "length_chain1": length_chain1,
        "length_chain2": length_chain2,
        "tm_score1": tm_score1,
        "tm_score2": tm_score2,
        "query_Equivalences": aligned_regions[0],
        "target_Equivalences": aligned_regions[1]
    }

def parse_TMalign_results0(txt_file):
    """
    解析 TM-align 的 txt 文件，提取相关信息
    :param txt_file: TM-align 输出的 txt 文件路径
    :return: 提取的相关信息，字典格式
    """
    with open(txt_file, 'r') as f:
        content = f.read()

    # 使用正则表达式提取信息
    #protein1 = re.search(r"Name of Chain_1: (.+)", content)
    protein1 = re.search(r"Name of Chain_1: .*/([^/]+)\.pdb", content)
    protein2 = re.search(r"Name of Chain_2: (.+)", content)
    length_chain1 = re.search(r"Length of Chain_1: (\d+) residues", content)
    length_chain2 = re.search(r"Length of Chain_2: (\d+) residues", content)
    aligned_length = re.search(r"Aligned length= (\d+)", content)
    rmsd = re.search(r"RMSD=\s+(\S+)", content)
    tm_score1 = re.search(r"TM-score= (\S+) \(if normalized by length of Chain_1", content)
    tm_score2 = re.search(r"TM-score= (\S+) \(if normalized by length of Chain_2", content)

    # 如果没有找到相应的字段，则返回None
    if not all([protein1, protein2, length_chain1, length_chain2, aligned_length, rmsd, tm_score1, tm_score2]):
        return None

    # 获取提取的值
    protein1 = os.path.basename(protein1.group(1)).replace(".pdb", "")
    protein2 = os.path.basename(protein2.group(1)).replace(".pdb", "")
    length_chain1 = int(length_chain1.group(1))
    length_chain2 = int(length_chain2.group(1))
    aligned_length = int(aligned_length.group(1))
    tm_score1 = float(tm_score1.group(1))
    tm_score2 = float(tm_score2.group(1))
    if rmsd:
        # 去掉可能的逗号并转换为浮点数
        rmsd = float(rmsd.group(1).strip(','))
        #print(f"RMSD: {rmsd}")
    else:
        print("RMSD not found.")

    # 计算两个TM-score的平均值作为weight
    weight = (tm_score1 + tm_score2) / 2

    # 返回结果
    return {
        "source": protein1,
        "target": protein2,
        "weight": weight,
        "aligned_length": aligned_length,
        "rmsd": rmsd,
        "length_chain1": length_chain1,
        "length_chain2": length_chain2,
        "tm_score1": tm_score1,
        "tm_score2": tm_score2
    }
def run_TM_align(TM_align_bin, structure1, structure2, output_dir):
    """
    使用 TM-align 计算两个蛋白质结构文件的相似性。
    :param structure1: 第一个结构文件路径
    :param structure2: 第二个结构文件路径
    :param output_dir: 输出结果存放目录
    :return: TM-align 输出文件的路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 运行 TM-align
        output_file = os.path.join(output_dir, f"{os.path.basename(structure1)}_vs_{os.path.basename(structure2)}.txt")
        
        # 调试输出构建的命令
        cmd = [TM_align_bin, structure1, structure2]

        # 使用 subprocess 执行命令，避免使用 shell=True
        with open(output_file, "w") as f_out:
            subprocess.run(cmd, stdout=f_out, stderr=subprocess.PIPE, check=True)
        
        return (structure1, structure2, output_file)  # 返回结构体文件路径和输出文件路径
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running TM-align: {e}")
        return None


def run_TM_align_batch_target_query_parallel(TM_align_bin, query_structure_dir, target_structure_dir, output_dir, batch_size=1000, num_workers=64):
    """
    比较指定目录下的所有查询蛋白质结构文件和目标蛋白质结构文件，计算它们两两之间的 TM-score，分批次运行。
    :param query_structure_dir: 查询蛋白质结构文件存放目录
    :param target_structure_dir: 目标蛋白质结构文件存放目录
    :param output_dir: 结果输出目录
    :param batch_size: 每批次运行的任务数
    :param num_workers: 并行工作进程数
    """
    # 获取查询目录中的所有文件
    query_files = [f for f in os.listdir(query_structure_dir) if f.endswith('.pdb')]  # 假设是 PDB 格式文件
    query_paths = [os.path.join(query_structure_dir, f) for f in query_files]

    # 获取目标目录中的所有文件
    target_files = [f for f in os.listdir(target_structure_dir) if f.endswith('.pdb')]  # 假设是 PDB 格式文件
    target_paths = [os.path.join(target_structure_dir, f) for f in target_files]

    # 生成所有查询文件和目标文件的两两组合
    all_combinations = [(query, target) for query in query_paths for target in target_paths]

    # 将所有组合分成多个批次
    batches = [all_combinations[i:i + batch_size] for i in range(0, len(all_combinations), batch_size)]

    # 使用多进程并行处理每个批次
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for batch in batches:
            # 每个批次都运行一个 TM-align 实例
            futures.append(executor.submit(run_TM_align_batch, TM_align_bin, batch, output_dir))
            logging.debug(f"Running TM-align for batch ")
        
        # 等待所有批次完成
        for future in futures:
            future.result()

    print(f"TM-align calculations completed. Results are stored in {output_dir}.")

def run_TM_align_batch_parallel(TM_align_bin, structure_dir, output_dir, batch_size=5000, num_workers=100):
    """
    比较指定目录下的所有蛋白质结构文件，计算它们两两之间的 TM-score，分批次运行。
    :param structure_dir: 蛋白质结构文件存放目录
    :param output_dir: 结果输出目录
    :param batch_size: 每批次运行的任务数
    :param num_workers: 并行工作进程数
    """
    # 获取目录中的所有文件
    structure_files = [f for f in os.listdir(structure_dir) if f.endswith('.pdb')]  # 假设是 PDB 格式文件
    structure_paths = [os.path.join(structure_dir, f) for f in structure_files]

    # 生成所有的两两组合
    all_combinations = list(itertools.combinations(structure_paths, 2))

    # 将所有组合分成多个批次
    batches = [all_combinations[i:i + batch_size] for i in range(0, len(all_combinations), batch_size)]

    # 使用多进程并行处理每个批次
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for batch in batches:
            # 每个批次都运行一个 TM-align 实例
            futures.append(executor.submit(run_TM_align_batch, TM_align_bin, batch, output_dir))
            logging.debug(f"runming TM-align for bacth ")
        
        # 等待所有批次完成
        for future in futures:
            future.result()

    print(f"TM-align calculations completed. Results are stored in {output_dir}.")

def run_TM_align_batch(TM_align_bin, batch, output_dir):
    """
    处理一个批次的比对任务
    :param TM_align_bin: TM-align 可执行文件路径
    :param batch: 一批次的比对任务（包含多个结构文件对）
    :param output_dir: 输出结果存放目录
    """
    # 对当前批次中的每一对结构文件进行比对
    for structure1, structure2 in batch:
        run_TM_align(TM_align_bin, structure1, structure2, output_dir)
def extract_TMalign_to_csv(input_dir, output_csv):
    """
    解析指定目录下的所有 TM-align 结果文件，并将数据存入 CSV 文件
    :param input_dir: 存放 TM-align 结果文件的目录
    :param output_csv: 输出的 CSV 文件路径
    """
    # 获取所有 TXT 文件
    tmalign_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    tmalign_data = []

    # 遍历所有文件，解析并提取数据
    for txt_file in tmalign_files:
        txt_file_path = os.path.join(input_dir, txt_file)
        result =  parse_TMalign_results0(txt_file_path)
        if result:
            tmalign_data.append(result)

    # 将结果存储为 DataFrame
    df = pd.DataFrame(tmalign_data)

    # 保存为 CSV 文件
    df.to_csv(output_csv, index=False)
    print(f"TM-align results saved to {output_csv}")




def run_foldseek_pad(fs_querydb, fs_padded_querydb, fs_rawdata="./fs_query_structures.raw", fs_results="./fs_query_results.m8", tmp_dir=FS_TMP_PATH, cov_mode=DEFAULT_FS_COV_MODE, coverage=FS_OVERLAP, alignment_type=DEFAULT_FS_ALIGNER, fs_bin_path=FS_BINARY_PATH):
    """
    使用 Foldseek 执行查询数据库与目标数据库的比对，并启用 GPU 加速（如果填充数据库已经准备好）。

    参数:
    fs_querydb (str): 查询数据库路径。
    fs_targetdb (str): 目标数据库路径。
    fs_rawdata (str): 存储原始结果的路径。
    fs_results (str): 存储最终结果的路径。
    tmp_dir (str): 临时目录。
    cov_mode (str): 覆盖模式。
    coverage (str): 覆盖率。
    alignment_type (str): 对齐类型。
    fs_bin_path (str): Foldseek 二进制路径。

    返回:
    None
    """


    # 运行 Foldseek 的搜索命令，启用 GPU 加速
    subprocess.run(
        [
            fs_bin_path,
            "search",
            fs_querydb,  # 使用填充后的查询数据库
            fs_padded_querydb,  # 使用填充后的目标数据库
            fs_rawdata,
            "-s", "9.5",  # Sensitivity 参数
            "--cov-mode", str(cov_mode),  # 覆盖模式
            "--num-iterations", "3",  # 迭代次数
            "-c", str(coverage),  # 覆盖率
            "--alignment-type", 
            "2",  # 对齐方式
            "--gpu", "1",  # 启用 GPU 加速
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # 将结果转换为指定格式
    subprocess.run(
        [
            fs_bin_path,
            "convertalis",
            fs_padded_querydb,  # 使用填充后的查询数据库
            fs_padded_targetdb,  # 使用填充后的目标数据库
            fs_rawdata,
            fs_results,
            "--format-output",
            DEFAULT_FS_FORMAT_OUTPUT,
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # 删除临时文件
    files_to_remove = glob.glob(f"{fs_rawdata}*")
    if files_to_remove:
        for file in files_to_remove:
            os.unlink(file)

    print("Foldseek run completed with GPU acceleration.")


def run_foldseek_pairwise(
    fs_querydb,            # Query foldseek database
    fs_targetdb,           # Target foldseek database
    fs_rawdata="./fs_query_structures.raw",
    fs_results="./fs_query_results.m8",
    tmp_dir="./tmp",
    cov_mode="1",          # Coverage mode, 1 is usually recommended
    coverage="0.2",        # Minimum coverage
    alignment_type="2",    # 0=local(3Di), 1=TM-align, 2=3Di+AA local
    fs_bin_path="foldseek",# Path to foldseek binary
    max_seqs="1000"           # Only return top-1 result for each query
):
    """
    Run Foldseek for pairwise structure comparison (1:1 matching).
    Useful for comparing 10k query-target structure pairs.
    """
    assert str(fs_rawdata) != ''
    assert str(fs_querydb) != ''
    assert str(fs_targetdb) != ''

    # Step 1: Run foldseek search
    subprocess.run(
        [
            "nice", "-n", "0", "ionice", "-c", "2", "-n", "0",
            fs_bin_path, "search",
            fs_querydb, fs_targetdb, fs_rawdata, tmp_dir,
            "-s", "9.5",
            "--cov-mode", str(cov_mode),
            "-c", str(coverage),
            "--alignment-type", str(alignment_type),
            "--num-iterations", "3",
            "-e", "0.1",
            "--max-seqs", str(max_seqs)  # <<—— 限制为 top-1 匹配
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(
        ["nice", "-n", "0", "ionice", "-c", "2", "-n", "0",
            fs_bin_path,
            "convertalis",
            fs_querydb,
            fs_targetdb,
            fs_rawdata,
            fs_results,
            "--format-output",
            DEFAULT_FS_FORMAT_OUTPUT,
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )
    # Step 2: Convert alignment results to m8 or tsv

    # Step 3: Clean up temporary alignment files
    for file in glob.glob(f"{fs_rawdata}*"):
        os.unlink(file)

    print("Foldseek pairwise run completed.")
def run_foldseek1(fs_querydb, fs_targetdb, fs_rawdata="./fs_query_structures.raw", fs_results="./fs_query_results.m8", tmp_dir=FS_TMP_PATH, cov_mode=DEFAULT_FS_COV_MODE, coverage=FS_OVERLAP, alignment_type=DEFAULT_FS_ALIGNER, fs_bin_path=FS_BINARY_PATH):
    "Run Foldseek Query DB against Target DB"
    #alignment_type = 1  0：3Di Gotoh-Smith-Waterman（局部，不推荐），1：TMalign（全局，慢），2：3Di+AA Gotoh-Smith-Waterman（局部，默认）
    #ensure_dir(fs_results)
    assert str(fs_rawdata) != ''
    subprocess.run(
        [ "nice", "-n", "0", "ionice", "-c", "2", "-n", "0",
            fs_bin_path,
            "search",
            fs_querydb,
            fs_targetdb,
            fs_rawdata,
            tmp_dir,
            "-s",
            "9.5",
            "--cov-mode",
            str(cov_mode),
            "--num-iterations",
            "1",
            "-c",
            "0.2",
            "--alignment-type",
            "1",
            "-e",
            "0.1",
            "--max-seqs", 
            "10000000" ,
            "--threads", "1"  # 单线程运行
            #"--seed-sub-mat", "aa:3di.out,nucl:3di.out"     # 固定随机种子
            
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(
        ["nice", "-n", "0", "ionice", "-c", "2", "-n", "0",
            fs_bin_path,
            "convertalis",
            fs_querydb,
            fs_targetdb,
            fs_rawdata,
            fs_results,
            "--format-output",
            DEFAULT_FS_FORMAT_OUTPUT,
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )
    files_to_remove = glob.glob(f"{fs_rawdata}*")
    if files_to_remove:
        for file in files_to_remove:
            os.unlink(file)
    print("run foldseek done")
def run_foldseek(fs_querydb, fs_targetdb, fs_rawdata="./fs_query_structures.raw", fs_results="./fs_query_results.m8", tmp_dir=FS_TMP_PATH, cov_mode=DEFAULT_FS_COV_MODE, coverage=FS_OVERLAP, alignment_type=DEFAULT_FS_ALIGNER, fs_bin_path=FS_BINARY_PATH):
    "Run Foldseek Query DB against Target DB"
    #alignment_type = 1  0：3Di Gotoh-Smith-Waterman（局部，不推荐），1：TMalign（全局，慢），2：3Di+AA Gotoh-Smith-Waterman（局部，默认）
    #ensure_dir(fs_results)
    assert str(fs_rawdata) != ''
    subprocess.run(
        [ "nice", "-n", "0", "ionice", "-c", "2", "-n", "0",
            fs_bin_path,
            "search",
            fs_querydb,
            fs_targetdb,
            fs_rawdata,
            tmp_dir,
            "-s",
            "9.5",
            "--cov-mode",
            str(cov_mode),
            "--num-iterations",
            "3",
            "-c",
            "0.2",
            "--alignment-type",
            "2",
            "-e",
            "0.1",
            #"--threads", "1" ,
            "--max-seqs", 
            "10000000" 
             #"-a"
            
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(
        ["nice", "-n", "0", "ionice", "-c", "2", "-n", "0",
            fs_bin_path,
            "convertalis",
            fs_querydb,
            fs_targetdb,
            fs_rawdata,
            fs_results,
            "--format-output",
            DEFAULT_FS_FORMAT_OUTPUT,
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )
    files_to_remove = glob.glob(f"{fs_rawdata}*")
    if files_to_remove:
        for file in files_to_remove:
            os.unlink(file)
    print("run foldseek done")
    
    
def run_foldseek_with_error_handling(fs_querydb, fs_targetdb, fs_rawdata="./fs_query_structures.raw", 
                                    fs_results="./fs_query_results.m8", tmp_dir=FS_TMP_PATH, 
                                    cov_mode=DEFAULT_FS_COV_MODE, coverage=FS_OVERLAP, 
                                    alignment_type=DEFAULT_FS_ALIGNER, fs_bin_path=FS_BINARY_PATH):
    """运行Foldseek并处理可能的错误"""
    assert str(fs_rawdata) != ''
    try:
        # 第一步：prefilter步骤
        result = subprocess.run(
            [ "nice", "-n", "0", "ionice", "-c", "2", "-n", "0",
                fs_bin_path,
                "search",
                fs_querydb,
                fs_targetdb,
                fs_rawdata,
                tmp_dir,
                "-s",
                "9.5",
                "--cov-mode",
                str(cov_mode),
                "--num-iterations",
                "3",
                "-c",
                "0.2",
                "--alignment-type",
                "2",
                "-e",
                "0.1",
                "--max-seqs", 
                "10000000" 
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=False,  # 改为False以捕获错误
            timeout=300   # 添加超时时间，避免永久挂起
        )
        
        # 检查是否有"Prefilter died"错误
        stderr_output = result.stderr.decode('utf-8', errors='ignore')
        if "Error: Prefilter died" in stderr_output or "Error: Kmer matching step died" in stderr_output:
            print(f"预过滤步骤失败，可能没有结构匹配: {stderr_output}")
            # 创建一个空的结果文件
            with open(fs_results, 'w') as f:
                # 写入一个空的结果头部
                f.write("query\ttarget\tqstart\ttstart\tqend\tend\tevalue\tgapopen\tpident\talnlen\tmismatch\tqcov\ttcov\tqlen\ttlen\tqaln\ttaln\tcfweight\n")
            return False
        
        if result.returncode != 0:
            print(f"搜索命令失败，返回码: {result.returncode}, 错误: {stderr_output}")
            return False
            
        # 第二步：convertalis步骤
        result = subprocess.run(
            ["nice", "-n", "0", "ionice", "-c", "2", "-n", "0",
                fs_bin_path,
                "convertalis",
                fs_querydb,
                fs_targetdb,
                fs_rawdata,
                fs_results,
                "--format-output",
                DEFAULT_FS_FORMAT_OUTPUT,
            ],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            check=False
        )
        
        if result.returncode != 0:
            stderr_output = result.stderr.decode('utf-8', errors='ignore')
            print(f"转换结果失败，返回码: {result.returncode}, 错误: {stderr_output}")
            return False
            
        # 清理临时文件
        files_to_remove = glob.glob(f"{fs_rawdata}*")
        if files_to_remove:
            for file in files_to_remove:
                try:
                    os.unlink(file)
                except Exception as e:
                    print(f"删除文件 {file} 时出错: {e}")
        
        print("foldseek运行成功")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"运行foldseek超时")
        return False
    except Exception as e:
        print(f"运行foldseek时发生错误: {e}")
        return False
def run_foldseek0(fs_querydb, fs_targetdb, fs_rawdata="./fs_query_structures.raw", fs_results="./fs_query_results.m8", tmp_dir=FS_TMP_PATH, cov_mode=DEFAULT_FS_COV_MODE, coverage=FS_OVERLAP, alignment_type=DEFAULT_FS_ALIGNER, fs_bin_path=FS_BINARY_PATH):
    "Run Foldseek Query DB against Target DB"
    #alignment_type = 1  0：3Di Gotoh-Smith-Waterman（局部，不推荐），1：TMalign（全局，慢），2：3Di+AA Gotoh-Smith-Waterman（局部，默认）
    #ensure_dir(fs_results)
    assert str(fs_rawdata) != ''
    subprocess.run(
        [ "nice", "-n", "0", "ionice", "-c", "2", "-n", "0",
            fs_bin_path,
            "search",
            fs_querydb,
            fs_targetdb,
            fs_rawdata,
            tmp_dir,
            "-s",
            "9.5",
            "--cov-mode",
            str(cov_mode),
            "--num-iterations",
            "3",
            "-c",
            "0.2",
            "--alignment-type",
            "2",
            "-e",
            "0.1",
            "--max-seqs", 
            "10000000" 
            
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(
        ["nice", "-n", "0", "ionice", "-c", "2", "-n", "0",
            fs_bin_path,
            "convertalis",
            fs_querydb,
            fs_targetdb,
            fs_rawdata,
            fs_results,
            "--format-output",
            DEFAULT_FS_FORMAT_OUTPUT,
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )
    files_to_remove = glob.glob(f"{fs_rawdata}*")
    if files_to_remove:
        for file in files_to_remove:
            os.unlink(file)
    print("run foldseek done")

def prepare_db_for_gpu_search_for_foldseek(fs_querydb_file, fs_padded_db_file):
    """
    为GPU搜索准备Foldseek数据库，填充数据库以支持GPU搜索。

    参数:
    fs_querydb_path (str): 输入的Foldseek数据库路径（已通过createdb创建的数据库）。
    fs_padded_db_path (str): 输出填充后的数据库路径，用于GPU搜索。
    fs_bin_path (str): Foldseek二进制文件的路径，默认为FS_BINARY_PATH。

    返回:
    None
    """
    # 确保输出目录存在
    
    # 使用foldseek命令生成GPU兼容的填充数据库
    subprocess.run(
        [
            FS_BINARY_PATH,
            "makepaddedseqdb",  # 将数据库格式化为适用于GPU搜索的格式
            fs_querydb_file,    # 输入的Foldseek数据库
            fs_padded_db_file   # 输出的填充后的数据库
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )
    print(f"GPU-compatible padded database created at {fs_padded_db_file}.")


def convert_pdb_to_foldseek_db(pdb_dir, fs_db_dir, fs_db_name="fs_db"):
    fs_querydb_path = os.path.join(fs_db_dir, fs_db_name+".db")
    ensure_dir(fs_db_dir)
    subprocess.run(
        [
            FS_BINARY_PATH,
            "createdb",
            "--input-format",
            "1",
            pdb_dir,
            fs_querydb_path,
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )
def ensure_file(file_path):
    """确保文件存在，如果文件不存在则创建一个空文件"""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass  # 创建一个空文件
        print(f"File created: {file_path}")
    else:
        print(f"File already exists: {file_path}")
# 创建保存目录
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
         
def process_fasta_and_pdb_data(raw_data_dir, raw_keyword):
    """处理FASTA和PDB数据，提取有效的序列和文件"""
    
    # 定义路径
    fasta_file = os.path.join(raw_data_dir, f"{raw_keyword}_fasta_filtered.fasta")
    pdb_data_dir = os.path.join(raw_data_dir, f"pdb_data_filtered/")
    cif_data_dir = os.path.join(raw_data_dir, f"cif_data_filtered/")
    pdb_source_file = os.path.join(raw_data_dir, f"{raw_keyword}_pdb_source.csv")
    new_dir = os.path.join(raw_data_dir, f"protein_relation/")
    new_pdb_dir = os.path.join(new_dir, f"pdb_data/")
    new_cif_dir = os.path.join(new_dir, f"cif_data/")

    # 创建目标文件夹
    ensure_dir(new_dir)
    ensure_dir(new_pdb_dir)
    ensure_dir(new_cif_dir)
    # 读取CSV文件并过滤
    uniprot_data = pd.read_csv(pdb_source_file)
    filtered_data = uniprot_data[uniprot_data['PDB Source'] != 'Not Found']
    new_id_file = os.path.join(new_dir, f"{raw_keyword}_uniprot_id.csv")
    filtered_data.to_csv(new_id_file, index=False)
    id_list = filtered_data['UniProt ID'].tolist()

    # 复制以id_list开头的文件
    
    for protein_id in id_list:
        # 处理PDB文件
        pdb_file = os.path.join(pdb_data_dir, f"{protein_id}.pdb")
        if os.path.exists(pdb_file):
            shutil.copy(pdb_file, new_pdb_dir)

        # 处理CIF文件
        cif_file = os.path.join(cif_data_dir, f"{protein_id}.cif")
        if os.path.exists(cif_file):
            shutil.copy(cif_file, new_cif_dir)
    
    # 提取对应的FASTA序列
    new_fasta_file = os.path.join(new_dir, f"{raw_keyword}_fasta.fasta")
    ensure_file(new_fasta_file)
    with open(fasta_file, 'r') as f, open(new_fasta_file, 'w') as new_fasta:
        record = False
        for line in f:
            if line.startswith(">sp|") or line.startswith(">tr|"):
                # 提取ID
                start = line.find('|') + 1
                end = line.find('|', start)
                if end != -1:
                    seq_id = line[start:end].strip()  # 提取ID
                    print(seq_id)
                    if any(seq_id.startswith(id_) for id_ in id_list):
                        new_fasta.write(line)  # 写入FASTA头
                        record = True  # 开始记录序列
                        print(f"{seq_id}: in")
                        id_list.remove(seq_id)  # 移除已处理的ID
                    else:
                        record = False
                        print(f"{seq_id}: not in")
                else:
                    record = False
            elif record:
                new_fasta.write(line)  # 只写入与id_list匹配的序列

    print(f"Filtered FASTA file saved to: {new_fasta_file}")


    print(f"Filtered FASTA file saved to: {new_fasta_file}")#
MM_PATH = "/mnt/sdb4/conda/envs/cathAlphaflow/bin/mmseqs"
