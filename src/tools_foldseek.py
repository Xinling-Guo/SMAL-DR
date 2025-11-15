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
    """Convert FASTA file to MMseqs2 database"""
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
    """Create index for database"""
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
    """Run MMseqs2 search"""
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
            "1",  # Can adjust thread count as needed
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )

    print("MMseqs2 search done")    

def get_mmseqs2_results(query_db, target_db, results_search_dir, output_file):
    """Convert alignment results to readable format"""
    
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
    Parse TM-align txt file and extract relevant information
    :param txt_file: TM-align output txt file path
    :return: Extracted relevant information in dictionary format
    """
    with open(txt_file, 'r') as f:
        content = f.read()
    with open(txt_file, 'r') as f:
        # Read all lines of file, store in list
        content2 = f.readlines()

    # Use regular expressions to extract information
    protein1 = re.search(r"Name of Chain_1: .*/([^/]+)\.pdb", content)
    protein2 = re.search(r"Name of Chain_2: (.+)", content)
    length_chain1 = re.search(r"Length of Chain_1: (\d+) residues", content)
    length_chain2 = re.search(r"Length of Chain_2: (\d+) residues", content)
    aligned_length = re.search(r"Aligned length= (\d+)", content)
    rmsd = re.search(r"RMSD=\s+(\S+)", content)
    tm_score1 = re.search(r"TM-score= (\S+) \(if normalized by length of Chain_1", content)
    tm_score2 = re.search(r"TM-score= (\S+) \(if normalized by length of Chain_2", content)

    # If corresponding fields not found, return None
    if not all([protein1, protein2, length_chain1, length_chain2, aligned_length, rmsd, tm_score1, tm_score2]):
        return None

    # Get extracted values
    protein1 = os.path.basename(protein1.group(1)).replace(".pdb", "")
    protein2 = os.path.basename(protein2.group(1)).replace(".pdb", "")
    length_chain1 = int(length_chain1.group(1))
    length_chain2 = int(length_chain2.group(1))
    aligned_length = int(aligned_length.group(1))
    tm_score1 = float(tm_score1.group(1))
    tm_score2 = float(tm_score2.group(1))
    if rmsd:
        # Remove possible commas and convert to float
        rmsd = float(rmsd.group(1).strip(','))
    else:
        print("RMSD not found.")

    # Calculate average of two TM-scores as weight
    ave_weight = (tm_score1 + tm_score2) / 2
    weight = max(tm_score1 , tm_score2)

    
    align_keyword = "denotes residue pairs of d"
    i_align = 0
    for i, line in enumerate(content2):
        if align_keyword in line:
            i_align = i  # Return line number, line numbers start from 0
    if i_align > 0 :
        query_seq = content2[i_align+1]
        target_seq = content2[i_align+3]
        alignment = content2[i_align+2]
        aligned_regions = extract_aligned_regions_for_tm_align(query_seq, target_seq, alignment)
    else:
        print(f"{protein1}vs{protein2} has no aligned sequence")

    # Return result
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
    Parse TM-align txt file and extract relevant information
    :param txt_file: TM-align output txt file path
    :return: Extracted relevant information in dictionary format
    """
    with open(txt_file, 'r') as f:
        content = f.read()

    # Use regular expressions to extract information
    #protein1 = re.search(r"Name of Chain_1: (.+)", content)
    protein1 = re.search(r"Name of Chain_1: .*/([^/]+)\.pdb", content)
    protein2 = re.search(r"Name of Chain_2: (.+)", content)
    length_chain1 = re.search(r"Length of Chain_1: (\d+) residues", content)
    length_chain2 = re.search(r"Length of Chain_2: (\d+) residues", content)
    aligned_length = re.search(r"Aligned length= (\d+)", content)
    rmsd = re.search(r"RMSD=\s+(\S+)", content)
    tm_score1 = re.search(r"TM-score= (\S+) \(if normalized by length of Chain_1", content)
    tm_score2 = re.search(r"TM-score= (\S+) \(if normalized by length of Chain_2", content)

    # If corresponding fields not found, return None
    if not all([protein1, protein2, length_chain1, length_chain2, aligned_length, rmsd, tm_score1, tm_score2]):
        return None

    # Get extracted values
    protein1 = os.path.basename(protein1.group(1)).replace(".pdb", "")
    protein2 = os.path.basename(protein2.group(1)).replace(".pdb", "")
    length_chain1 = int(length_chain1.group(1))
    length_chain2 = int(length_chain2.group(1))
    aligned_length = int(aligned_length.group(1))
    tm_score1 = float(tm_score1.group(1))
    tm_score2 = float(tm_score2.group(1))
    if rmsd:
        # Remove possible commas and convert to float
        rmsd = float(rmsd.group(1).strip(','))
        #print(f"RMSD: {rmsd}")
    else:
        print("RMSD not found.")

    # Calculate average of two TM-scores as weight
    weight = (tm_score1 + tm_score2) / 2

    # Return result
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
    Use TM-align to calculate similarity between two protein structure files.
    :param structure1: First structure file path
    :param structure2: Second structure file path
    :param output_dir: Output result storage directory
    :return: TM-align output file path
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Run TM-align
        output_file = os.path.join(output_dir, f"{os.path.basename(structure1)}_vs_{os.path.basename(structure2)}.txt")
        
        # Debug output of constructed command
        cmd = [TM_align_bin, structure1, structure2]

        # Use subprocess to execute command, avoid using shell=True
        with open(output_file, "w") as f_out:
            subprocess.run(cmd, stdout=f_out, stderr=subprocess.PIPE, check=True)
        
        return (structure1, structure2, output_file)  # Return structure file paths and output file path
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running TM-align: {e}")
        return None


def run_TM_align_batch_target_query_parallel(TM_align_bin, query_structure_dir, target_structure_dir, output_dir, batch_size=1000, num_workers=64):
    """
    Compare all query protein structure files and target protein structure files in specified directories, calculate TM-scores between all pairs, run in batches.
    :param query_structure_dir: Query protein structure file storage directory
    :param target_structure_dir: Target protein structure file storage directory
    :param output_dir: Result output directory
    :param batch_size: Number of tasks per batch
    :param num_workers: Number of parallel worker processes
    """
    # Get all files in query directory
    query_files = [f for f in os.listdir(query_structure_dir) if f.endswith('.pdb')]  # Assume PDB format files
    query_paths = [os.path.join(query_structure_dir, f) for f in query_files]

    # Get all files in target directory
    target_files = [f for f in os.listdir(target_structure_dir) if f.endswith('.pdb')]  # Assume PDB format files
    target_paths = [os.path.join(target_structure_dir, f) for f in target_files]

    # Generate all pairwise combinations of query files and target files
    all_combinations = [(query, target) for query in query_paths for target in target_paths]

    # Split all combinations into multiple batches
    batches = [all_combinations[i:i + batch_size] for i in range(0, len(all_combinations), batch_size)]

    # Use multi-processing to process each batch in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for batch in batches:
            # Run a TM-align instance for each batch
            futures.append(executor.submit(run_TM_align_batch, TM_align_bin, batch, output_dir))
            logging.debug(f"Running TM-align for batch ")
        
        # Wait for all batches to complete
        for future in futures:
            future.result()

    print(f"TM-align calculations completed. Results are stored in {output_dir}.")

def run_TM_align_batch_parallel(TM_align_bin, structure_dir, output_dir, batch_size=5000, num_workers=100):
    """
    Compare all protein structure files in specified directory, calculate TM-scores between all pairs, run in batches.
    :param structure_dir: Protein structure file storage directory
    :param output_dir: Result output directory
    :param batch_size: Number of tasks per batch
    :param num_workers: Number of parallel worker processes
    """
    # Get all files in directory
    structure_files = [f for f in os.listdir(structure_dir) if f.endswith('.pdb')]  # Assume PDB format files
    structure_paths = [os.path.join(structure_dir, f) for f in structure_files]

    # Generate all pairwise combinations
    all_combinations = list(itertools.combinations(structure_paths, 2))

    # Split all combinations into multiple batches
    batches = [all_combinations[i:i + batch_size] for i in range(0, len(all_combinations), batch_size)]

    # Use multi-processing to process each batch in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for batch in batches:
            # Run a TM-align instance for each batch
            futures.append(executor.submit(run_TM_align_batch, TM_align_bin, batch, output_dir))
            logging.debug(f"Running TM-align for batch ")
        
        # Wait for all batches to complete
        for future in futures:
            future.result()

    print(f"TM-align calculations completed. Results are stored in {output_dir}.")

def run_TM_align_batch(TM_align_bin, batch, output_dir):
    """
    Process a batch of alignment tasks
    :param TM_align_bin: TM-align executable file path
    :param batch: A batch of alignment tasks (containing multiple structure file pairs)
    :param output_dir: Output result storage directory
    """
    # Perform alignment for each structure file pair in current batch
    for structure1, structure2 in batch:
        run_TM_align(TM_align_bin, structure1, structure2, output_dir)
def extract_TMalign_to_csv(input_dir, output_csv):
    """
    Parse all TM-align result files in specified directory and store data in CSV file
    :param input_dir: Directory storing TM-align result files
    :param output_csv: Output CSV file path
    """
    # Get all TXT files
    tmalign_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    tmalign_data = []

    # Iterate through all files, parse and extract data
    for txt_file in tmalign_files:
        txt_file_path = os.path.join(input_dir, txt_file)
        result =  parse_TMalign_results0(txt_file_path)
        if result:
            tmalign_data.append(result)

    # Store results as DataFrame
    df = pd.DataFrame(tmalign_data)

    # Save as CSV file
    df.to_csv(output_csv, index=False)
    print(f"TM-align results saved to {output_csv}")




def run_foldseek_pad(fs_querydb, fs_padded_querydb, fs_rawdata="./fs_query_structures.raw", fs_results="./fs_query_results.m8", tmp_dir=FS_TMP_PATH, cov_mode=DEFAULT_FS_COV_MODE, coverage=FS_OVERLAP, alignment_type=DEFAULT_FS_ALIGNER, fs_bin_path=FS_BINARY_PATH):
    """
    Use Foldseek to perform query database vs target database alignment, and enable GPU acceleration (if padded database is ready).

    Parameters:
    fs_querydb (str): Query database path.
    fs_targetdb (str): Target database path.
    fs_rawdata (str): Path to store raw results.
    fs_results (str): Path to store final results.
    tmp_dir (str): Temporary directory.
    cov_mode (str): Coverage mode.
    coverage (str): Coverage.
    alignment_type (str): Alignment type.
    fs_bin_path (str): Foldseek binary path.

    Returns:
    None
    """


    # Run Foldseek search command, enable GPU acceleration
    subprocess.run(
        [
            fs_bin_path,
            "search",
            fs_querydb,  # Use padded query database
            fs_padded_querydb,  # Use padded target database
            fs_rawdata,
            "-s", "9.5",  # Sensitivity parameter
            "--cov-mode", str(cov_mode),  # Coverage mode
            "--num-iterations", "3",  # Number of iterations
            "-c", str(coverage),  # Coverage
            "--alignment-type", 
            "2",  # Alignment method
            "--gpu", "1",  # Enable GPU acceleration
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # Convert results to specified format
    subprocess.run(
        [
            fs_bin_path,
            "convertalis",
            fs_padded_querydb,  # Use padded query database
            fs_padded_targetdb,  # Use padded target database
            fs_rawdata,
            fs_results,
            "--format-output",
            DEFAULT_FS_FORMAT_OUTPUT,
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # Delete temporary files
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
            "--max-seqs", str(max_seqs)  # <<—— Limit to top-1 match
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
    #alignment_type = 1  0：3Di Gotoh-Smith-Waterman (local, not recommended), 1：TMalign (global, slow), 2：3Di+AA Gotoh-Smith-Waterman (local, default)
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
            "--threads", "1"  # Run with single thread
            #"--seed-sub-mat", "aa:3di.out,nucl:3di.out"     # Fixed random seed
            
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
    #alignment_type = 1  0：3Di Gotoh-Smith-Waterman (local, not recommended), 1：TMalign (global, slow), 2：3Di+AA Gotoh-Smith-Waterman (local, default)
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
    """Run Foldseek and handle possible errors"""
    assert str(fs_rawdata) != ''
    try:
        # Step 1: prefilter step
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
            check=False,  # Change to False to capture errors
            timeout=300   # Add timeout to avoid permanent hanging
        )
        
        # Check for "Prefilter died" error
        stderr_output = result.stderr.decode('utf-8', errors='ignore')
        if "Error: Prefilter died" in stderr_output or "Error: Kmer matching step died" in stderr_output:
            print(f"Prefilter step failed, possibly no structure matches: {stderr_output}")
            # Create an empty result file
            with open(fs_results, 'w') as f:
                # Write an empty result header
                f.write("query\ttarget\tqstart\ttstart\tqend\tend\tevalue\tgapopen\tpident\talnlen\tmismatch\tqcov\ttcov\tqlen\ttlen\tqaln\ttaln\tcfweight\n")
            return False
        
        if result.returncode != 0:
            print(f"Search command failed, return code: {result.returncode}, error: {stderr_output}")
            return False
            
        # Step 2: convertalis step
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
            print(f"Result conversion failed, return code: {result.returncode}, error: {stderr_output}")
            return False
            
        # Clean temporary files
        files_to_remove = glob.glob(f"{fs_rawdata}*")
        if files_to_remove:
            for file in files_to_remove:
                try:
                    os.unlink(file)
                except Exception as e:
                    print(f"Error deleting file {file}: {e}")
        
        print("foldseek run successful")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"foldseek run timeout")
        return False
    except Exception as e:
        print(f"Error running foldseek: {e}")
        return False
def run_foldseek0(fs_querydb, fs_targetdb, fs_rawdata="./fs_query_structures.raw", fs_results="./fs_query_results.m8", tmp_dir=FS_TMP_PATH, cov_mode=DEFAULT_FS_COV_MODE, coverage=FS_OVERLAP, alignment_type=DEFAULT_FS_ALIGNER, fs_bin_path=FS_BINARY_PATH):
    "Run Foldseek Query DB against Target DB"
    #alignment_type = 1  0：3Di Gotoh-Smith-Waterman (local, not recommended), 1：TMalign (global, slow), 2：3Di+AA Gotoh-Smith-Waterman (local, default)
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
    Prepare Foldseek database for GPU search, pad database to support GPU search.

    Parameters:
    fs_querydb_path (str): Input Foldseek database path (database already created via createdb).
    fs_padded_db_path (str): Output padded database path for GPU search.
    fs_bin_path (str): Foldseek binary file path, default is FS_BINARY_PATH.

    Returns:
    None
    """
    # Ensure output directory exists
    
    # Use foldseek command to generate GPU-compatible padded database
    subprocess.run(
        [
            FS_BINARY_PATH,
            "makepaddedseqdb",  # Format database for GPU search
            fs_querydb_file,    # Input Foldseek database
            fs_padded_db_file   # Output padded database
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
    """Ensure file exists, create empty file if file doesn't exist"""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass  # Create an empty file
        print(f"File created: {file_path}")
    else:
        print(f"File already exists: {file_path}")
# Create save directory
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
         
def process_fasta_and_pdb_data(raw_data_dir, raw_keyword):
    """Process FASTA and PDB data, extract valid sequences and files"""
    
    # Define paths
    fasta_file = os.path.join(raw_data_dir, f"{raw_keyword}_fasta_filtered.fasta")
    pdb_data_dir = os.path.join(raw_data_dir, f"pdb_data_filtered/")
    cif_data_dir = os.path.join(raw_data_dir, f"cif_data_filtered/")
    pdb_source_file = os.path.join(raw_data_dir, f"{raw_keyword}_pdb_source.csv")
    new_dir = os.path.join(raw_data_dir, f"protein_relation/")
    new_pdb_dir = os.path.join(new_dir, f"pdb_data/")
    new_cif_dir = os.path.join(new_dir, f"cif_data/")

    # Create target folders
    ensure_dir(new_dir)
    ensure_dir(new_pdb_dir)
    ensure_dir(new_cif_dir)
    # Read CSV file and filter
    uniprot_data = pd.read_csv(pdb_source_file)
    filtered_data = uniprot_data[uniprot_data['PDB Source'] != 'Not Found']
    new_id_file = os.path.join(new_dir, f"{raw_keyword}_uniprot_id.csv")
    filtered_data.to_csv(new_id_file, index=False)
    id_list = filtered_data['UniProt ID'].tolist()

    # Copy files starting with id_list
    
    for protein_id in id_list:
        # Process PDB files
        pdb_file = os.path.join(pdb_data_dir, f"{protein_id}.pdb")
        if os.path.exists(pdb_file):
            shutil.copy(pdb_file, new_pdb_dir)

        # Process CIF files
        cif_file = os.path.join(cif_data_dir, f"{protein_id}.cif")
        if os.path.exists(cif_file):
            shutil.copy(cif_file, new_cif_dir)
    
    # Extract corresponding FASTA sequences
    new_fasta_file = os.path.join(new_dir, f"{raw_keyword}_fasta.fasta")
    ensure_file(new_fasta_file)
    with open(fasta_file, 'r') as f, open(new_fasta_file, 'w') as new_fasta:
        record = False
        for line in f:
            if line.startswith(">sp|") or line.startswith(">tr|"):
                # Extract ID
                start = line.find('|') + 1
                end = line.find('|', start)
                if end != -1:
                    seq_id = line[start:end].strip()  # Extract ID
                    print(seq_id)
                    if any(seq_id.startswith(id_) for id_ in id_list):
                        new_fasta.write(line)  # Write FASTA header
                        record = True  # Start recording sequence
                        print(f"{seq_id}: in")
                        id_list.remove(seq_id)  # Remove processed ID
                    else:
                        record = False
                        print(f"{seq_id}: not in")
                else:
                    record = False
            elif record:
                new_fasta.write(line)  # Only write sequences matching id_list

    print(f"Filtered FASTA file saved to: {new_fasta_file}")


    print(f"Filtered FASTA file saved to: {new_fasta_file}")#
MM_PATH = "/mnt/sdb4/conda/envs/cathAlphaflow/bin/mmseqs"
