import pandas as pd
import requests
from Bio import PDB
import argparse
import shutil
import itertools
import sys
from Bio.PDB import PDBParser, PDBIO
import os
from Bio import pairwise2
sys.path.append("/mnt/sdb4/protein_gen/Cas9_work/code")
sys.path.append("/mnt/sdb4/protein_gen/Cas9_domain_work/code")
import HNH_domain_get_from_TED_OSS as HdgfTO
from tqdm import tqdm
import concurrent.futures
import shutil
import subprocess
import datetime
import logging
logging.getLogger("urllib3").setLevel(logging.WARNING)
import Dali_domain_in_AFDBcluster as dda
import re
import csv
import get_protein_relation as gpr
import get_protein_relation_network as gprn
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from Bio.Align import substitution_matrices

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def extract_pdb_segment(input_pdb, output_pdb, range_str, chain_id="A"):
    """
    Extract multiple specified segment fragments from a PDB file and save to a new file

    Parameters:
    - input_pdb: Input PDB file path
    - output_pdb: Output PDB file path
    - range_str: Target fragment range string, format like "17-114_220-312"
    - chain_id: Target chain ID (default 'A')
    """
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", input_pdb)
        io = PDB.PDBIO()

        # Split range string
        ranges = range_str.split('_')
        valid_residues = []

        for r in ranges:
            start_res, end_res = map(int, r.split('-'))

            # Extract chain
            model = structure[0]  # Use only the first model
            chain = model[chain_id]

            for residue in chain:
                res_id = residue.get_id()[1]
                if start_res <= res_id <= end_res:
                    valid_residues.append(residue)

        class SelectResidues(PDB.Select):
            def accept_residue(self, residue):
                return residue in valid_residues

        io.set_structure(structure)
        io.save(output_pdb, select=SelectResidues())

        # print(f"✅ Extraction completed! Saved to {output_pdb}")
        return True
    except Exception as e:
        print(f"Error processing {input_pdb}: {e}. Skipping...")
        return False


def process_dali_results(input_path, output_file, csv1_path, csv2_path):
    # Regular expression pattern
    pattern = re.compile(
        r'^(\d+):\s+(\S+)\s+(\S+)\s+(\d+)\s*-\s*(\d+)\s*<=>\s*(\d+)\s*-\s*(\d+).*$'
    )
    
    # Build mapping dictionary
    def build_mapping(file_path):
        mapping = {}
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    key = row[1].strip()
                    value = row[0].strip()
                    mapping[key] = value
        return mapping

    # Build mapping dictionaries
    csv1_map = build_mapping(csv1_path)
    csv2_map = build_mapping(csv2_path)

    # Get all .txt files
    input_files = [f for f in os.listdir(input_path) if f.endswith('.txt')]

    # Write processed results
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        # CSV header
        writer.writerow(["source", "source_id", "sourcerange", "target", "target_id", "targetrange"])

        # Process each input file
        for file_path in input_files:
            with open(os.path.join(input_path, file_path), 'r') as f_in:
                in_section = False
                for line in f_in:
                    line = line.strip()

                    # Enter target section
                    if line == '# Structural equivalences':
                        print(f"Processing in {file_path}: In Structural equivalences section")
                        in_section = True
                        continue
                    # Exit target section
                    if line == '# Translation-rotation matrices':
                        print(f"Processing in {file_path}: In Translation-rotation matrices section")
                        in_section = False
                        continue

                    if in_section:
                        match = pattern.match(line)
                        if match:
                            # Extract各部分信息
                            source = match.group(2)
                            sourcerange = f"{match.group(4)}-{match.group(5)}"
                            target = match.group(3)
                            targetrange = f"{match.group(6)}-{match.group(7)}"

                            # Map source column
                            source_id = csv1_map.get(source.split("-")[0], source)
                            # Map target column
                            target_id = csv2_map.get(target.split("-")[0], target)

                            # Write in CSV format
                            writer.writerow([source, source_id, sourcerange, target, target_id, targetrange])

    print(f"Processing and mapping completed. Output written to {output_file}")

def map_and_merge(csv1_path, csv2_path, input_csv, output_csv):
    """
    Map source and target columns through csv1 and csv2 respectively, and generate six-column output
    :param csv1_path: source mapping file path (key,value)
    :param csv2_path: target mapping file path (key,value)
    :param input_csv: input CSV file path (four columns: source,sourcerange,target,targetrange)
    :param output_csv: output CSV file path (six columns: source,source_id,target,target_id,sourcerange,targetrange)
    """
    # Build mapping dictionary
    def build_mapping(file_path):
        mapping = {}
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    key = row[1].strip()
                    value = row[0].strip()
                    mapping[key] = value
        return mapping

    csv1_map = build_mapping(csv1_path)
    csv2_map = build_mapping(csv2_path)
    # print(build_mapping)
    # Process input file
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header row
        writer.writerow([
            'source', 'source_id','sourcerange',
            'target', 'target_id','targetrange'
        ])
        
        for row in reader:
            if len(row) != 4:
                continue  # Skip malformed rows
            
            source, sourcerange, target, targetrange = row
            # print(source)
            # break
            # Map source column
            # print(csv1_map.get(source.split("-")[0]))
            source_id = csv1_map.get(source.split("-")[0], source)
            # Map target column
            target_id = csv2_map.get(target.split("-")[0], target)
            
            writer.writerow([
                source, source_id,sourcerange,
                target, target_id,targetrange
            ])


def merge_target_ranges_parts(input_csv, output_csv,split,clas):
    """
    Merge range intervals for the same source and target, and delete ranges that do not meet conditions based on target range
    :param input_csv: Input CSV file path
    :param output_csv: Output CSV file path
    """
    merged_data = {}

    # Read and process input file
    with open(input_csv, 'r') as f_in:
        reader = csv.reader(f_in)
        next(reader)  # Skip header row
        print(clas)
        for row in reader:
            if len(row) != 6:
                continue  # Skip malformed rows
            if clas not in row[1].split("_")[-1]:
                # print(row[1].split("_")[-1],clas)
                continue

            source = row[1]#.split("_")[0]
            target = row[4]
            # print(row[1])
            # Parse source range
            try:
                src_start, src_end = map(int, row[2].split('-'))
            except:
                continue  # Skip invalid range

            # Parse target range
            try:
                tgt_start, tgt_end = map(int, row[5].split('-'))
            except:
                continue  # Skip invalid range

            # Update merged data
            key = (source, target)
            if key not in merged_data:
                merged_data[key] = []
            merged_data[key].append({
                'src_start': src_start,
                'src_end': src_end,
                'tgt_start': tgt_start,
                'tgt_end': tgt_end
            })
    # print(merged_data)
    # Sort and merge ranges for each source and target combination
    final_merged_data = {}
    for (source, target), ranges in merged_data.items():
        ranges.sort(key=lambda x: x['tgt_start'])  # Sort by target start position
        if not ranges:
            continue
        merged_ranges = []
        current_range = ranges[0]
        for i in range(1, len(ranges)):
            if ranges[i]['tgt_start'] - current_range['tgt_end'] <= split:
                # Consider as continuous interval, update current range
                current_range['src_start'] = min(current_range['src_start'], ranges[i]['src_start'])
                current_range['src_end'] = max(current_range['src_end'], ranges[i]['src_end'])
                current_range['tgt_start'] = min(current_range['tgt_start'], ranges[i]['tgt_start'])
                current_range['tgt_end'] = max(current_range['tgt_end'], ranges[i]['tgt_end'])
            else:
                # Difference exceeds 20, separate as new interval
                merged_ranges.append(current_range)
                current_range = ranges[i]
        # Add last range
        merged_ranges.append(current_range)

        for merged_range in merged_ranges:
            final_key = (source, target, merged_range['tgt_start'], merged_range['tgt_end'])
            final_merged_data[final_key] = {
                'src_start': merged_range['src_start'],
                'src_end': merged_range['src_end'],
                'tgt_start': merged_range['tgt_start'],
                'tgt_end': merged_range['tgt_end']
            }
    # print(final_merged_data)
    # Write output file
    # Write output file, and add sourcerange_abs column
    with open(output_csv, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['source', 'sourcerange', 'target', 'targetrange', 'sourcerange_abs'])

        for (source, target, _, _), data in final_merged_data.items():
            # Extract absolute position of source (e.g., 961-1121)
            try:
                abs_start, abs_end = map(int, source.split('_')[1].split('-'))
            except:
                abs_start = 0  # fallback
            # Calculate absolute range
            abs_src_start = abs_start + data['src_start'] - 1
            abs_src_end = abs_start + data['src_end'] - 1

            writer.writerow([
                source,
                f"{data['src_start']}-{data['src_end']}",
                target,
                f"{data['tgt_start']}-{data['tgt_end']}",
                f"{abs_src_start}-{abs_src_end}"
            ])

def add_chopping_column(input_csv, output_csv):
    """
    Add targetcheck column to the fourth column position, with the format as target_prefix + '_' + chopping_check.
    
    Parameters:
    input_csv: Input CSV file path
    output_csv: Output CSV file path
    """
    # Read the CSV file
    df = pd.read_csv(input_csv)
    # Remove duplicate rows
    df = df.drop_duplicates()
    # df = df.sort_values(['weight', 'alntmscore'], ascending=[False, False]).groupby(['source', 'target']).first().reset_index()
    # print(f"Successfully removed duplicate rows in {input_csv}.")

    # Define the processing function
    def process_row(row):
        try:
            target_cp = row['target'].split('_')[1]
            # target_prefix = target_parts[0] if target_parts else pd.NA
            # chopping = row['chopping_check']
            return f"{target_cp}" if not pd.isna(row['target']) else pd.NA
        except Exception as e:
            print(f"Error processing row: {row}, Error message: {e}")
            return pd.NA
    
    # Create a new column and insert it at the fourth position
    df.insert(2, 'chopping', df.apply(process_row, axis=1))
    
    # # Save the result
    df.to_csv(output_csv, index=False)
    print(f"Successfully added target_check column at the fourth position, result saved to: {output_csv}")

def merge_csv_files(csv1_path, csv2_path, output_path):
    # Read two CSV files
    df1 = pd.read_excel(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # Perform split operation on df1's source column
    # df1['source'] = df1['source'].apply(lambda x: str(x)[:-4])
    df1['source'] = df1['source'].apply(lambda x: str(x).replace('_A_', '_'))
    df1['targetID'] = df1['target'].apply(lambda x: str(x).split("-")[1])
    df2['target'] = df2['target'].apply(lambda x: str(x).split("_")[0])

    # Define the processing function
    # def process_row_add_chopping(row):
    #     try:
    #         target_cp = row['target'].split('_')[1]
    #         # target_prefix = target_parts[0] if target_parts else pd.NA
    #         # chopping = row['chopping_check']
    #         return f"{target_cp}" if not pd.isna(row['target']) else pd.NA
    #     except Exception as e:
    #         print(f"Error processing row: {row}, Error message: {e}")
    #         return pd.NA
    
    # # Create a new column and insert it at the fourth position
    # df2.insert(2, 'chopping', df2.apply(process_row_add_chopping, axis=1))

    # Delete rows in df2 where the first 5 characters of source and target are the same
    # df2 = df2[df2['source'].str.split("_").str[0] != df2['target'].str.split("_").str[0]]

    # Rename columns of csv2 (except source and target)
    csv2_columns = df2.columns.tolist()
    # print(csv2_columns)
    rename_dict = {col: col + '_dalisearch' for col in csv2_columns if col not in ['source', 'target']}
    # print(rename_dict)
    # print(df2.columns)

    df2_renamed = df2.rename(columns=rename_dict)
    # Perform merge operation, here using target_check to correspond with target, source with source
    merged_df = pd.merge(
        df1,
        df2_renamed,
        left_on=['source', 'targetID'],
        right_on=['source', 'target'],
        how='left'
    )

    # Delete redundant target column (since target_check has been used for matching)
    # merged_df = merged_df.drop(columns=['target_y'])

    # Define column names for the first few columns
    first_columns = ['source', 'targetID','target_x','target_y', 
                     'chopping_check', 'chopping_dalisearch']

    # Get remaining column names
    remaining_columns = [col for col in merged_df.columns if col not in first_columns]

    # Arrange columns in required order
    final_columns = first_columns + remaining_columns
    merged_df = merged_df[final_columns]

    # Use dictionary to specify column name mapping
    # Use dictionary to specify column name mapping
    new_column_names = {'target_x': 'target','weight_dalisearch':'FS_weight_dalisearch','alntmscore_dalisearch':'TM_weight_dalisearch'}
    merged_df = merged_df.rename(columns=new_column_names)

    merged_df = merged_df.drop(columns=['targetID','target_y'])

    print(merged_df.columns)

    # Save result
    merged_df.to_csv(output_path, index=False)
    print(f"Successfully merged files, result saved to: {output_path}")

def download_pdb(pdb_id, pdb_dir, pdb_url_template="https://files.rcsb.org/download/{pdb_id}.pdb"):
    """Download PDB file with retry mechanism"""
    pdb_url = pdb_url_template.format(pdb_id=pdb_id)
    save_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")

    if os.path.exists(save_path):
        print(f"PDB file {pdb_id} already exists. Skipping download.")
    else:
        
        print(pdb_url)
        attempts = 0
        while attempts < MAX_RETRIES:
            try:
                response = requests.get(pdb_url, timeout=20)
                response.raise_for_status()
                with open(save_path, "w") as f:
                    f.write(response.text)
                #print(f"Downloaded {pdb_id} to {save_path}.")
                break
            except requests.exceptions.RequestException as e:
                attempts += 1
                print(f"Error downloading PDB file {pdb_id} (Attempt {attempts}/{MAX_RETRIES}): {e}")
                if attempts < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"Failed to download PDB file {pdb_id} after {MAX_RETRIES} attempts.")
# Unified download, then write only once, will lose data, so download one fasta and save one file,


def download_pdb_urllib(pdb_id, pdb_dir, pdb_url_template="https://alphafold.ebi.ac.uk/files/AF-{pdb_id}-F1-model_v4.pdb"):
    """Download PDB using urllib (more compatible)"""
    pdb_url = pdb_url_template.format(pdb_id=pdb_id)
    save_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")

    if os.path.exists(save_path):
        print(f"PDB file {pdb_id} already exists. Skipping download.")
        return

    attempts = 0
    while attempts < 5:
        try:
            urllib.request.urlretrieve(pdb_url, save_path)
            print(f"Downloaded {pdb_id} to {save_path}.")
            return
        except Exception as e:
            attempts += 1
            print(f"Error downloading {pdb_id} (Attempt {attempts}/5): {e}")
            time.sleep(5)

    print(f"Failed to download {pdb_id} after 5 attempts.")
    
import urllib.request
def download_pdb_from_AFDB_parallel(hnh_cath_in_ted_file, output_dir):
    """
    Download PDB data and FASTA sequences from AlphaFold database based on TED IDs.

    Args:
        hnh_cath_in_ted_file (str): Path to the CSV file containing TED ID and chopping information.
        output_dir (str): Directory to save the downloaded PDB files.
        fasta_output_dir (str): Directory to save the downloaded FASTA sequences.

    Returns:
        None
    """
    try:
        
        # Load the HNH_cath_in_ted file
        df = pd.read_csv(hnh_cath_in_ted_file)
        if 'target' not in df.columns:
            raise ValueError("Column 'ted_id' not found in HNH_cath_in_ted file.")

        # Extract unique PDB IDs from the ted_id column
        pdb_ids = (
            df['target']
            .dropna()  # Drop NaN values
            .astype(str)  # Convert to string
            .apply(lambda x: x.split('-')[1])  # Split and extract the first part
            .unique()  # Get unique values
        )

        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        
        # Use thread pool to download PDB files in parallel
        with ThreadPoolExecutor() as pdb_executor:
            # Submit each PDB download task
            pdb_futures = [pdb_executor.submit(download_pdb_urllib, pdb_id, output_dir, pdb_url_template = "https://alphafold.ebi.ac.uk/files/AF-{pdb_id}-F1-model_v4.pdb" ) for pdb_id in pdb_ids]
            
            # Wait for all PDB download tasks to complete
            for future in as_completed(pdb_futures):
                future.result()  # Get return result of each thread, although we don't need it now
                
        


    except Exception as e:
        print(f"An error occurred: {e}")
        
def combine_fasta_files(fasta_output_dir, all_fasta_output_file):
    """Combine all individual FASTA files into one output file."""
    with open(all_fasta_output_file, "w") as all_fasta_out:
        for fasta_file in os.listdir(fasta_output_dir):
            fasta_file_path = os.path.join(fasta_output_dir, fasta_file)
            if fasta_file.endswith(".fasta"):
                with open(fasta_file_path, "r") as fasta_in:
                    temp_content = fasta_in.readlines()
                    all_fasta_out.writelines(temp_content)
                    if len(temp_content) ==1:
                        print(len(temp_content))
    print(f"All FASTA files combined into {all_fasta_output_file}.")

import subprocess

def run_dali_shell(assemble_dali_script_path, Dali_work_dir):
    try:
        # Define parameters to pass to the script
        param1 = Dali_work_dir
        # Build command list including script path and parameters
        command = ['nice', '-n', '-20', 'sh', assemble_dali_script_path, param1]
        # Run CSH script and pass parameters
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # Print script's standard output
        print("Script standard output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        print("Error output:")
        print(e.stderr)
   
   
def process_entry_split_pdb(entry, pdb_dir,domain_pdbs_path):
    pdb_id_str, range_str, failed_downloads = entry
    # pdb_id = pdb_id_str.split('-')[1]
    pdb_id = pdb_id_str
    try:
        range_parts = range_str.split(',')# Split range string by underscore
        # print(range_str)
        for part in range_parts:
            out_pdb_path = os.path.join(domain_pdbs_path, f"{pdb_id}_{part}.pdb")
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            if os.path.exists(out_pdb_path):
                continue

            success = extract_pdb_segment(pdb_file, out_pdb_path, part)
            if not success:
                failed_downloads.append(pdb_id_str)
                # break
                continue
            else:
                print(f"extract_pdb_segment {pdb_id}_{part}")
    except Exception as e:
        print(f"Unexpected error processing {pdb_id_str}: {e}. Skipping...")
        failed_downloads.append(pdb_id_str)

def process_dali_fstm_results_250703(dali_result_path, dali_result_fstm_path, output_path):
    # Read CSV files
    dali_result = pd.read_csv(dali_result_path)
    dali_result_fstm = pd.read_csv(dali_result_fstm_path)
    

    # Append sourcerange and targetrange from dali_result to dali_result_fstm
    merged = dali_result_fstm.rename(columns={'weight': 'FS_weight_Dali', 'alntmscore': 'TM_weight_Dali'})
    
    # Rename sourcerange and targetrange columns
    merged = merged.rename(columns={'sourcerange': 'sourcerange_Dali', 'targetrange': 'targetrange_Dali'})
    merged['sourcerange_Dali'] = merged['source'].str.split('_').str[-1] 
    merged['targetrange_Dali'] = merged['target'].str.split('_').str[-1]
    
    # Rearrange column order, move source, target, FS_weight_Dali, TM_weight_Dali, sourcerange_Dali and targetrange_Dali to front
    columns_order = ['source', 'target', 'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali'] + \
                    [col for col in merged.columns if col not in ['source', 'target', 'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali']]
    merged = merged[columns_order]
    
    # Save result as new CSV file
    merged.to_csv(output_path, index=False)
    
    print(f"Processing completed, result saved to {output_path}")
    

def process_dali_fstm_results(dali_result_path, dali_result_fstm_path, output_path):
    # Read CSV files
    dali_result = pd.read_csv(dali_result_path)
    dali_result_fstm = pd.read_csv(dali_result_fstm_path)
    
    # Modify column names of dali_result_fstm
    dali_result_fstm = dali_result_fstm.rename(columns={'weight': 'FS_weight_Dali', 'alntmscore': 'TM_weight_Dali'})
    
    # Generate unique ID (combination of source and target columns as unique ID)
    dali_result['id'] = dali_result['source'] + '_' + dali_result['target']+'_'+dali_result['targetrange']
    dali_result_fstm['id'] = dali_result_fstm['source'] + '_' + dali_result_fstm['target']
    
    # Append sourcerange and targetrange from dali_result to dali_result_fstm
    merged = pd.merge(dali_result_fstm, dali_result[['id', 'sourcerange', 'targetrange']], on='id', how='left')
    
    # Rename sourcerange and targetrange columns
    merged = merged.rename(columns={'sourcerange': 'sourcerange_Dali', 'targetrange': 'targetrange_Dali'})
    
    # Rearrange column order, move source, target, FS_weight_Dali, TM_weight_Dali, sourcerange_Dali and targetrange_Dali to front
    columns_order = ['source', 'target', 'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali'] + \
                    [col for col in merged.columns if col not in ['source', 'target', 'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali']]
    merged = merged[columns_order]
    
    # Save result as new CSV file
    merged.to_csv(output_path, index=False)
    
    print(f"Processing completed, result saved to {output_path}")
    

def split_target_domains_from_dali_results(dali_results_pro, Target_domains_dali, pdb_dir, max_chunksize=100, max_workers=30):
    """
    Process CSV file chunk by chunk and utilize multi-processing

    :param dali_results_pro: Input CSV file path
    :param Target_domains_dali: Output directory
    :param max_chunksize: Size of each chunk
    :param max_workers: Maximum number of worker processes
    """
    ensure_dir(Target_domains_dali)
    # Read CSV file and process by chunks
    chunks = pd.read_csv(dali_results_pro, chunksize=max_chunksize, sep=',')
    
    # Process CSV file chunk by chunk
    for j, chunk in enumerate(tqdm(chunks, desc="Processing CSV chunks")):
        print(f"\n\n=====================Processing chunk {j + 1}=====================\n\n")
        failed_downloads = []
        
        # Prepare entry list
        entries = [(row["target"], row["targetrange"], failed_downloads) for _, row in chunk.iterrows() if pd.notna(row["targetrange"])]

        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Use multi-processing to process entries
                futures = []
                for entry in entries:
                    future = executor.submit(process_entry_split_pdb2, entry, pdb_dir, Target_domains_dali)
                    futures.append(future)

                # Wait for all processes to complete and check for errors
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # Get result (will raise exception if any)
                    except Exception as e:
                        print(f"Error processing entry: {e}")

        except Exception as e:
            print(f"Error processing chunk {j + 1}: {e}")
            continue

    print(f"Processing of {dali_results_pro} completed.")

def split_query_domains_from_dali_results(dali_results_pro, Target_domains_dali, pdb_dir, max_chunksize=5, max_workers=30):
    """
    Process CSV file chunk by chunk and utilize multi-processing

    :param dali_results_pro: Input CSV file path
    :param Target_domains_dali: Output directory
    :param max_chunksize: Size of each chunk
    :param max_workers: Maximum number of worker processes
    """
    ensure_dir(Target_domains_dali)
    # Read CSV file and process by chunks
    all_info = pd.read_csv(dali_results_pro)
    unique_info = all_info.drop_duplicates(subset=["source", "sourcerange_abs"])
    filtered_info = unique_info[unique_info['source'].str.contains('Q99ZW2') | unique_info['source'].str.contains('J7RUA')]
    #chunks = pd.read_csv(dali_results_pro, chunksize=max_chunksize, sep=',')
    chunks = np.array_split(filtered_info, max_chunksize)
    # Process CSV file chunk by chunk
    for j, chunk in enumerate(tqdm(chunks, desc="Processing CSV chunks")):
        print(f"\n\n=====================Processing chunk {j + 1}=====================\n\n")
        failed_downloads = []
        
        # Prepare entry list
        entries = [(row["source"], row["sourcerange_abs"], failed_downloads) for _, row in chunk.iterrows() if pd.notna(row["sourcerange_abs"])] 
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Use multi-processing to process entries
                futures = []
                for entry in entries:
                    future = executor.submit(process_entry_split_pdb2, entry, pdb_dir, Target_domains_dali)
                    futures.append(future)

                # Wait for all processes to complete and check for errors
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # Get result (will raise exception if any)
                    except Exception as e:
                        print(f"Error processing entry: {e}")

        except Exception as e:
            print(f"Error processing chunk {j + 1}: {e}")
            continue

    print(f"Processing of {dali_results_pro} completed.")

def process_entry_split_pdb2(entry, pdb_dir, domain_pdbs_path):
    pdb_id_str, range_str, failed_downloads = entry
    pdb_id = pdb_id_str
    try:
        range_parts = range_str.split(',')  # Split range string by underscore
        for part in range_parts:
            out_pdb_path = os.path.join(domain_pdbs_path, f"{pdb_id}_{part}.pdb")
            pdb_file = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            if os.path.exists(out_pdb_path):
                continue

            success = extract_pdb_segment(pdb_file, out_pdb_path, part)
            if not success:
                failed_downloads.append(pdb_id_str)
                continue
            else:
                print(f"Extracted {pdb_id}_{part}")
    except Exception as e:
        print(f"Unexpected error processing {pdb_id_str}: {e}. Skipping...")
        failed_downloads.append(pdb_id_str)

def get_new_candidate_info(csv_file, excel_file, output_file):
    # Read CSV file as DataFrame df1
    df1 = pd.read_csv(csv_file)
    
    # Read two specified sheets (sp-2 and sa-2) from Excel file
    df2_sp = pd.read_excel(excel_file, sheet_name='sp-2')
    df2_sa = pd.read_excel(excel_file, sheet_name='sa-2')
    
    # Modify source and target columns in df2_sp, split by '_' and take first and last parts, construct new ID
    df2_sp['source_id'] = df2_sp['source'].apply(lambda x: '_'.join(x.split('_')[:1] + x.split('_')[-1:]) if pd.notna(x) and x != '' else None)
    df2_sp['target_id'] = df2_sp['target'].apply(lambda x: x if pd.notna(x) and x != '' else None)
    
    # Modify source and target columns in df2_sa, split by '_' and take first and last parts, construct new ID
    df2_sa['source_id'] = df2_sa['source'].apply(lambda x: '_'.join(x.split('_')[:1] + x.split('_')[-1:]) if pd.notna(x) and x != '' else None)
    df2_sa['target_id'] = df2_sa['target'].apply(lambda x: x if pd.notna(x) and x != '' else None)
    
    # Modify source and target columns in df1, split by '_' and take first and last parts, construct new ID
    df1['source_id'] = df1['source'].apply(lambda x: '_'.join(x.split('_')[:1] + x.split('_')[-1:]) if pd.notna(x) and x != '' else None)
    df1['target_id'] = df1['target'].apply(lambda x: x if pd.notna(x) and x != '' else None)
    
    # Find corresponding rows in df1 based on pair id from source_id and target_id in df2_sp, get df3_sp
    df3_sp = pd.merge(df2_sp[['source_id', 'target_id']], df1, left_on=['source_id', 'target_id'], right_on=['source_id', 'target_id'], how='left')[df1.columns]
    
    # Find corresponding rows in df1 based on pair id from source_id and target_id in df2_sa, get df3_sa
    df3_sa = pd.merge(df2_sa[['source_id', 'target_id']], df1, left_on=['source_id', 'target_id'], right_on=['source_id', 'target_id'], how='left')[df1.columns]
    
    # Create a new Excel file, save df1, df3_sp and df3_sa to different sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='df1', index=False)
        df3_sp.to_excel(writer, sheet_name='df3_sp', index=False)
        df3_sa.to_excel(writer, sheet_name='df3_sa', index=False)

    print(f"Processing completed, result saved to {output_file}")

def process_candidate_with_dali0(candidate_file, dali_check_file, output_file):
    """
    Read candidate and Dali_check_fstm files, perform merge and processing

    :param candidate_file: Input candidate CSV file path
    :param dali_check_file: Input Dali_check_fstm CSV file path
    :param output_file: Output file path after processing
    """
    # Read CSV files
    candidate = pd.read_csv(candidate_file)
    dali_check = pd.read_csv(dali_check_file)

    # Generate unique id column
    candidate['new_source'] = candidate['source'].apply(lambda x: '_'.join((x.split('_')[0],x.split('_')[-1]))  if isinstance(x, str) else '')
    candidate['new_target'] = candidate['target'].apply(lambda x: x.split('-')[1] if isinstance(x, str) else '')
    candidate['id'] = candidate['new_source'] + '_' + candidate['new_target']
    
    dali_check['new_source'] = dali_check['source'].apply(lambda x: '_'.join((x.split('_')[0],x.split('_')[-1]))  if isinstance(x, str) else '')
    dali_check['new_target'] =dali_check['target'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else '')
    dali_check['id'] = dali_check['new_source'] + '_' + dali_check['new_target']


    # First initialize required columns as empty
    candidate['FS_weight_Dali'] = None
    candidate['TM_weight_Dali'] = None
    candidate['sourcerange_Dali'] = None
    candidate['targetrange_Dali'] = None
    candidate['start_dif'] = None
    candidate['end_dif'] = None
    candidate['FS_dif'] = None
    candidate['TM_dif'] = None

    # Process candidate data row by row
    for idx, row in candidate.iterrows():
        # Find unique id corresponding to candidate
        candidate_id = row['id']
        
        # Find all rows with same id in Dali_check_fstm
        matching_rows = dali_check[dali_check['id'] == candidate_id]

        if len(matching_rows) == 1:
            # If only one row, directly add columns to candidate
            dali_row = matching_rows.iloc[0]
        elif len(matching_rows) > 1:
            # If multiple rows, first select rows where targetrange_Dali is not empty
            non_na_rows = matching_rows[matching_rows['targetrange_Dali'].notna()]
            
            if not non_na_rows.empty:
                # If rows with non-empty targetrange_Dali exist, select first row
                dali_row = non_na_rows.iloc[0]
            else:
                # If no rows with non-empty targetrange_Dali exist, select row with maximum FS_weight_Dali
                dali_row = matching_rows.loc[matching_rows['FS_weight_Dali'].idxmax()]

        # Calculate start_dif and end_dif
        chopping_check = row['chopping_check']
        targetrange_Dali = dali_row['target'].split('_')[1]
        
        chopping_start, chopping_end = map(int, chopping_check.split('-'))
        targetrange_start, targetrange_end = map(int, targetrange_Dali.split('-'))

        start_dif = chopping_start - targetrange_start
        end_dif = targetrange_end - chopping_end

        # Calculate FS_dif and TM_dif
        FS_weight = row['FS_weight']
        TM_weight = row['TM_weight']
        FS_weight_Dali = dali_row['FS_weight_Dali']
        TM_weight_Dali = dali_row['TM_weight_Dali']
        
        FS_dif = FS_weight_Dali - FS_weight
        TM_dif = TM_weight_Dali - TM_weight

        # Update row data in candidate
        candidate.at[idx, 'FS_weight_Dali'] = dali_row['FS_weight_Dali']
        candidate.at[idx, 'TM_weight_Dali'] = dali_row['TM_weight_Dali']
        candidate.at[idx, 'sourcerange_Dali'] = dali_row['sourcerange_Dali']
        candidate.at[idx, 'targetrange_Dali'] =  dali_row['target'].split('_')[1]
        candidate.at[idx, 'start_dif'] = start_dif
        candidate.at[idx, 'end_dif'] = end_dif
        candidate.at[idx, 'FS_dif'] = FS_dif
        candidate.at[idx, 'TM_dif'] = TM_dif

    # Move specific columns to front
    columns_order = [
        'source', 'target', 'FS_weight', 'TM_weight', 'chopping_check', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif'
    ] + [col for col in candidate.columns if col not in [
        'source', 'target', 'FS_weight', 'TM_weight', 'chopping_check', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif'
    ]]
    
    # Rearrange order
    candidate = candidate[columns_order]

    # Save processed data to CSV file
    candidate.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    
def process_candidate_with_dali0(candidate_file, dali_check_file, output_file):
    """
    Read candidate and Dali_check_fstm files, perform merge and processing

    :param candidate_file: Input candidate CSV file path
    :param dali_check_file: Input Dali_check_fstm CSV file path
    :param output_file: Output file path after processing
    """
    # Read CSV files
    candidate = pd.read_csv(candidate_file)
    dali_check = pd.read_csv(dali_check_file)

    # Generate unique id column
    candidate['new_source'] = candidate['source'].apply(lambda x: '_'.join((x.split('_')[0],x.split('_')[-1]))  if isinstance(x, str) else '')
    candidate['new_target'] = candidate['target'].apply(lambda x: x.split('-')[1] if isinstance(x, str) else '')
    candidate['id'] = candidate['new_source'] + '_' + candidate['new_target']
    
    dali_check['new_source'] = dali_check['source'].apply(lambda x: '_'.join((x.split('_')[0],x.split('_')[-2]))  if isinstance(x, str) else '')
    dali_check['new_target'] =dali_check['target'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else '')
    dali_check['id'] = dali_check['new_source'] + '_' + dali_check['new_target']


    # First initialize required columns as empty
    candidate['FS_weight_Dali'] = None
    candidate['TM_weight_Dali'] = None
    candidate['sourcerange_Dali'] = None
    candidate['targetrange_Dali'] = None
    candidate['start_dif'] = None
    candidate['end_dif'] = None
    candidate['FS_dif'] = None
    candidate['TM_dif'] = None

    # Process candidate data row by row
    for idx, row in candidate.iterrows():
        # Find unique id corresponding to candidate
        candidate_id = row['id']
        
        # Find all rows with same id in Dali_check_fstm
        matching_rows = dali_check[dali_check['id'] == candidate_id]

        if len(matching_rows) == 1:
            # If only one row, directly add columns to candidate
            dali_row = matching_rows.iloc[0]
        elif len(matching_rows) > 1:
            # If multiple rows, first select rows where targetrange_Dali is not empty
            non_na_rows = matching_rows[matching_rows['targetrange_Dali'].notna()]
            
            if not non_na_rows.empty:
                # If rows with non-empty targetrange_Dali exist, select first row
                dali_row = non_na_rows.iloc[0]
            else:
                # If no rows with non-empty targetrange_Dali exist, select row with maximum FS_weight_Dali
                dali_row = matching_rows.loc[matching_rows['FS_weight_Dali'].idxmax()]

        # Calculate start_dif and end_dif
        chopping_check = row['chopping_check']
        targetrange_Dali = dali_row['target'].split('_')[1]
        
        chopping_start, chopping_end = map(int, chopping_check.split('-'))
        targetrange_start, targetrange_end = map(int, targetrange_Dali.split('-'))

        start_dif = chopping_start - targetrange_start
        end_dif = targetrange_end - chopping_end

        # Calculate FS_dif and TM_dif
        FS_weight = row['FS_weight']
        TM_weight = row['TM_weight']
        FS_weight_Dali = dali_row['FS_weight_Dali']
        TM_weight_Dali = dali_row['TM_weight_Dali']
        
        FS_dif = FS_weight_Dali - FS_weight
        TM_dif = TM_weight_Dali - TM_weight

        # Update row data in candidate
        candidate.at[idx, 'FS_weight_Dali'] = dali_row['FS_weight_Dali']
        candidate.at[idx, 'TM_weight_Dali'] = dali_row['TM_weight_Dali']
        candidate.at[idx, 'sourcerange_Dali'] = dali_row['sourcerange_Dali']
        candidate.at[idx, 'targetrange_Dali'] =  dali_row['target'].split('_')[1]
        candidate.at[idx, 'start_dif'] = start_dif
        candidate.at[idx, 'end_dif'] = end_dif
        candidate.at[idx, 'FS_dif'] = FS_dif
        candidate.at[idx, 'TM_dif'] = TM_dif

    # Move specific columns to front
    columns_order = [
        'source', 'target', 'FS_weight', 'TM_weight', 'chopping_check', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif'
    ] + [col for col in candidate.columns if col not in [
        'source', 'target', 'FS_weight', 'TM_weight', 'chopping_check', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif'
    ]]
    
    # Rearrange order
    candidate = candidate[columns_order]

    # Save processed data to CSV file
    candidate.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")   
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from difflib import SequenceMatcher
from Bio.Align import substitution_matrices
def calculate_protein_sequence_similarity(seq1, seq2):
    """
    Calculate similarity between two protein sequences using BLOSUM62 scoring matrix and normalized by self-alignment
    
    Parameters:
        seq1 (str): First protein sequence
        seq2 (str): Second protein sequence
        
    Returns:
        float: Normalized similarity score (0-1 range)
    """
    # Parameter check
    if not isinstance(seq1, str) or not isinstance(seq2, str):
        return 0.0
        
    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0
    
    # Load BLOSUM62 matrix
    matrix = substitution_matrices.load("BLOSUM62")
    
    try:
        # Use BLOSUM62 matrix for global alignment
        # Set gap_open=-10, gap_extend=-0.5 as common parameters
        alignments = pairwise2.align.globalds(seq1, seq2, 
                                            matrix, -10, -0.5,
                                            one_alignment_only=True)
        
        if not alignments:
            return 0.0
            
        # Get alignment score
        alignment_score = alignments[0].score
        
        # Calculate self-alignment scores for normalization
        self_score1 = pairwise2.align.globalds(seq1, seq1, 
                                            matrix, -10, -0.5,
                                            one_alignment_only=True)[0].score
        self_score2 = pairwise2.align.globalds(seq2, seq2, 
                                            matrix, -10, -0.5,
                                            one_alignment_only=True)[0].score
        
        # Normalized score (use smaller of two self-alignment scores for normalization)
        normalized_score = alignment_score / min(self_score1, self_score2)
        
        # Ensure score is in 0-1 range
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        return round(normalized_score, 4)
        
    except Exception as e:
        print(f"Error calculating sequence similarity: {e}")
        return 0.0
def replace_subsequence(orig_seq: str, start: int, end: int, new_subseq: str) -> str:
    """Replace [start-1:end] of orig_seq with new_subseq"""
    return orig_seq[:start-1] + new_subseq + orig_seq[end:]
def augment_dali_with_check_250703(dali_csv: str,
                            check_csv: str,
                            fasta_dict: dict[str, SeqRecord],
                            output_csv: str):
    # 1. Read
    dali_df  = pd.read_csv(dali_csv)
    check_df = pd.read_csv(check_csv)

    # 2. Construct unified ID
    dali_df['id'] = (
        dali_df['source']
        .astype(str)
        .str.split('_')
        .str[0]
        + '_'
        + dali_df['target']
        .astype(str)
        .str.split('-')
        .str[1]
    )

    check_df['id'] = (
        check_df['source']
        .astype(str)
        .str.split('_')
        .str[0]
        + '_'
        + check_df['target']
        .astype(str)
        .str.split('-')
        .str[1]
    )

    # 3. Pre-prepare new columns in dali_df (all set to NaN/None)
    new_cols = [
        # Original check fields
        'chopping_check','FS_weight_Check','TM_weight_Check',
        # Difference _Check
        'start_dif_Check','end_dif_Check','FS_dif_Check','TM_dif_Check',
        # Difference _Dali_Check
        'start_dif_Dali_Check','end_dif_Dali_Check',
        'FS_dif_Dali_Check','TM_dif_Dali_Check',
        # Domain sequences & similarity
        'source_domain_seq_Check','target_domain_seq_Check',
        'assemble_seq_Check','domain_seq_sim_Check'
    ]
    for c in new_cols:
        dali_df[c] = None

    # 4. Build a quick access map for check_df
    check_df = check_df.drop_duplicates(subset='id', keep='first')
    check_map = check_df.set_index('id').to_dict(orient='index')

    # 5. Iterate through dali_df, match and calculate
    for idx, row in dali_df.iterrows():
        rid = row['id']
        info = check_map.get(rid)
        if info is None:
            continue

        # —— 4.1 Directly merge fields
        dali_df.at[idx, 'chopping_check']    = info.get('chopping_check')
        dali_df.at[idx, 'FS_weight_Check']   = info.get('FS_weight_Check')
        dali_df.at[idx, 'TM_weight_Check']   = info.get('TM_weight_Check')

        # —— 4.2 Calculate _Check differences
        try:
            # Original chopping
            cs, ce = map(int, row['chopping'].split('-'))
            # check chopping
            cks, cke = map(int, info['chopping_check'].split('-'))
            dali_df.at[idx, 'start_dif_Check'] = cs - cks
            dali_df.at[idx, 'end_dif_Check']   = cke - ce
        except Exception:
            pass

        for metric in ['FS','TM']:
            base = row[f'{metric}_weight']
            chk  = info.get(f'{metric}_weight_Check')
            if pd.notnull(base) and pd.notnull(chk):
                dali_df.at[idx, f'{metric}_dif_Check'] = chk - base

        # —— 4.3 Calculate _Dali_Check differences
        # Require fields chopping_Dali, FS_weight_Dali, TM_weight_Dali
        try:
            ds, de = map(int, row['chopping_Dali'].split('-'))
            cs, ce = map(int, info['chopping_check'].split('-'))
            dali_df.at[idx, 'start_dif_Dali_Check'] = ds - cs
            dali_df.at[idx, 'end_dif_Dali_Check']   = ce - de
        except Exception:
            pass

        for metric in ['FS','TM']:
            dval = row.get(f'{metric}_weight_Dali')
            cval = info.get(f'{metric}_weight_Check')
            if pd.notnull(dval) and pd.notnull(cval):
                dali_df.at[idx, f'{metric}_dif_Dali_Check'] = dval - cval

        # —— 4.4 Extract domain sequences & calculate similarity (Check)
        # Fixed source range 770-905
        src_id = row['source'].split('_')[0]
        tgt_id = row['target'].split('-')[1]
        rec_src = fasta_dict.get(src_id)
        rec_tgt = fasta_dict.get(tgt_id)
        chop_ck = info.get('chopping_check')
        if rec_src and rec_tgt and isinstance(chop_ck, str):
            seq_src = str(rec_src.seq)
            seq_tgt = str(rec_tgt.seq)
            # Domain fragments
            ds, de = 770, 905
            tks, tke = map(int, chop_ck.split('-'))
            dom_src = seq_src[ds-1:de]
            dom_tgt = seq_tgt[tks-1:tke]
            # Write
            dali_df.at[idx, 'source_domain_seq_Check'] = dom_src
            dali_df.at[idx, 'target_domain_seq_Check'] = dom_tgt
            # Assemble & similarity
            dali_df.at[idx, 'assemble_seq_Check']   = replace_subsequence(seq_src, ds, de, dom_tgt)
            dali_df.at[idx, 'domain_seq_sim_Check'] = calculate_protein_sequence_similarity(dom_src, dom_tgt)
    # 6. Column rearrangement: put new_cols after chopping column
    cols = list(dali_df.columns)
    insert_pos = cols.index('chopping') + 1
    front = cols[:insert_pos]
    rest = [c for c in cols[insert_pos:] if c not in new_cols]
    dali_df = dali_df[front + new_cols + rest]
    
    # 6. Output
    dali_df['unique_ID'] = 'SpMut_' + (dali_df.index + 1).astype(str).str.zfill(4)
    dali_df.insert(0, 'unique_ID', dali_df.pop('unique_ID'))
    dali_df.to_csv(output_csv, index=False)
    
    print(f"Done! Output saved to {output_csv}")
def process_candidate_with_dali_250703(candidate_file, dali_check_file, csv1_file, fasta_dict, output_file):
    """
    Based on original process_candidate, add domain sequence and similarity information:
      - source_domain_seq_Dali
      - target_domain_seq_Dali
      - assemble_seq
      - domain_seq_sim_Dali
      - protein_seq_sim_Dali

    :param fasta_dict: A dict, key is fasta_records key, value is BioPython SeqRecord
    """
    # --- 1. Read CSV ---
    candidate = pd.read_csv(candidate_file, low_memory=False)
    dali_check = pd.read_csv(dali_check_file)
    csv1 = pd.read_csv(csv1_file)

    # --- 2. Original ID generation and filtering logic ---
    csv1['id'] = (csv1['source'] + '_' + 
                  csv1['sourcerange_abs'].astype(str) + '_' + 
                  csv1['target'] + '_' + 
                  csv1['targetrange'].astype(str))
    dali_check['id2'] = dali_check['source'] + '_' + dali_check['target']
    valid_ids = set(csv1['id'])
    dali_check = dali_check[dali_check['id2'].isin(valid_ids)]

    # New: Split source/target ID fragments, generate mergeable "id"
    dali_check['new_source'] = dali_check['source'].apply(
        lambda x: '_'.join((x.split('_')[0], x.split('_')[-2]))
        if isinstance(x, str) else ''
    )
    dali_check['new_target'] = dali_check['target'].apply(
        lambda x: x.split('_')[0] if isinstance(x, str) else ''
    )
    dali_check['id'] = dali_check['new_source'] + '_' + dali_check['new_target']
    dali_check = (dali_check
                  .sort_values('tlen', ascending=False)
                  .drop_duplicates(subset=['id'])
                  .reset_index(drop=True))

    # --- 3. candidate new ID ---
    candidate['new_source'] = candidate['source'].apply(
        lambda x: '_'.join((x.split('_')[0], x.split('_')[-1])) if isinstance(x, str) else ''
    )
    candidate['new_target'] = candidate['target'].apply(
        lambda x: x.split('-')[1] if isinstance(x, str) else ''
    )
    candidate['id'] = candidate['new_source'] + '_' + candidate['new_target']

    # --- 4. Initialize new columns ---
    for col in [
        'FS_weight_Dali', 'TM_weight_Dali',
        'sourcerange_Dali', 'targetrange_Dali',
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif',
        # New domain information
        'source_domain_seq_Dali', 'target_domain_seq_Dali',
        'assemble_seq_Dali', 'domain_seq_sim_Dali', 'protein_seq_sim_Dali'
    ]:
        candidate[col] = None

    # --- 5. Row-by-row matching and calculation ---
    for idx, row in candidate.iterrows():
        """
        if 'Sp' in row['wet_ID'] :
            print(f"Row {idx} is Sp38")
        else:
            continue
        """
        print(f"{row['id']}: starting!!!!")
        
        cid = row['id']
        match = dali_check[dali_check['id'] == cid]
        if match.empty:
            continue
        d = match.iloc[0]

        # Basic weights and differences
        candidate.at[idx, 'FS_weight_Dali'] = d['FS_weight_Dali']
        candidate.at[idx, 'TM_weight_Dali'] = d['TM_weight_Dali']
        candidate.at[idx, 'sourcerange_Dali'] = d['source'].split('_')[-1]
        candidate.at[idx, 'targetrange_Dali'] = d['target'].split('_')[-1]

        # Calculate differences
        chopping = row['chopping']
        sr = d['source'].split('_')[-1]  # e.g. "100-200"
        tr = d['target'].split('_')[1]  # e.g. "50-150"
        c_start, c_end = map(int, (chopping.split('-')[0], chopping.split('-')[-1]))
        t_start, t_end = map(int, tr.split('-'))
        candidate.at[idx, 'start_dif'] = c_start - t_start
        candidate.at[idx, 'end_dif'] = t_end - c_end
        candidate.at[idx, 'FS_dif'] = d['FS_weight_Dali'] - row['FS_weight']
        candidate.at[idx, 'TM_dif'] = d['TM_weight_Dali'] - row['TM_weight']

        # --- New: domain sequences and similarity calculation ---
        # Get source and target sequences from fasta_dict
        src_id = row['new_source'].split('_')[0]
        tgt_id = row['new_target']
        src_rec: SeqRecord = fasta_dict.get(src_id)
        tgt_rec: SeqRecord = fasta_dict.get(tgt_id)
        if src_rec and tgt_rec:
            src_seq = str(src_rec.seq)
            tgt_seq = str(tgt_rec.seq)

            # Extract domain fragments
            ds, de = map(int, sr.split('-'))
            ts, te = map(int, tr.split('-'))
            src_dom = src_seq[ds-1:de]
            tgt_dom = tgt_seq[ts-1:te]

            # Calculate similarity
            dom_sim = 0
            prot_sim = 0
            #dom_sim = calculate_protein_sequence_similarity(src_dom, tgt_dom)
            #prot_sim = calculate_protein_sequence_similarity(src_seq, tgt_seq)

            # Assemble new sequence
            assembled = replace_subsequence(src_seq, ds, de, tgt_dom)

            # Write back
            candidate.at[idx, 'source_domain_seq_Dali'] = src_dom
            candidate.at[idx, 'target_domain_seq_Dali'] = tgt_dom
            candidate.at[idx, 'domain_seq_sim_Dali'] = dom_sim
            candidate.at[idx, 'protein_seq_sim_Dali'] = prot_sim
            candidate.at[idx, 'assemble_seq_Dali'] = assembled
            print(f"{d['target']}: OKOKOKOKOKOKKO!!!!")
    # --- 6. Adjust column order and output ---
    front = [
        'source','target','FS_weight','TM_weight',
        'FS_weight_Dali','TM_weight_Dali',
        'sourcerange_Dali','targetrange_Dali',
        'start_dif','end_dif','FS_dif','TM_dif',
        'source_domain_seq_Dali','target_domain_seq_Dali',
        'domain_seq_sim_Dali','protein_seq_sim_Dali','assemble_seq_Dali'
    ]
    rest = [c for c in candidate.columns if c not in front]
    candidate = candidate[front + rest]
    candidate.to_csv(output_file, index=False)
    print(f"Processing completed, saved to {output_file}")
    
    
def process_candidate_with_dali_25070300(candidate_file, dali_check_file, csv1_file, output_file):
    """
    Read candidate, Dali_check_fstm and csv1 files, perform merge and processing

    :param candidate_file: Input candidate CSV file path
    :param dali_check_file: Input Dali_check_fstm CSV file path
    :param csv1_file: Input csv1 CSV file path, containing source, sourcerange, target, targetrange, sourcerange_abs columns
    :param output_file: Output file path after processing
    """
    # Read CSV files
    candidate = pd.read_csv(candidate_file)
    dali_check = pd.read_csv(dali_check_file)
    csv1 = pd.read_csv(csv1_file)

    # Create unique ID for csv1
    csv1['id'] = csv1['source'] + '_' + csv1['sourcerange_abs'].astype(str) + '_' + csv1['target'] + '_' + csv1['targetrange'].astype(str)
    
    # Create id2 for dali_check
    dali_check['id2'] = dali_check['source'] + '_'+ dali_check['target']
    
    # Only keep rows where id2 exists in csv1's id
    valid_ids = set(csv1['id'])
    dali_check = dali_check[dali_check['id2'].isin(valid_ids)]
    
        
    dali_check['new_source'] = dali_check['source'].apply(lambda x: '_'.join((x.split('_')[0],x.split('_')[-2]))  if isinstance(x, str) else '')
    dali_check['new_target'] =dali_check['target'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else '')
    dali_check['id'] = dali_check['new_source'] + '_' + dali_check['new_target']

    
    # For duplicate dali_check['id'], only keep the row with maximum TM_weight_Dali
    #dali_check = dali_check.sort_values('TM_weight_Dali', ascending=False).drop_duplicates(subset=['id']).reset_index(drop=True)
    dali_check = dali_check.sort_values('tlen', ascending=False).drop_duplicates(subset=['id']).reset_index(drop=True)

    # Generate unique id column for matching
    candidate['new_source'] = candidate['source'].apply(lambda x: '_'.join((x.split('_')[0],x.split('_')[-1])) if isinstance(x, str) else '')
    candidate['new_target'] = candidate['target'].apply(lambda x: x.split('-')[1] if isinstance(x, str) else '')
    candidate['id'] = candidate['new_source'] + '_' + candidate['new_target']
    
    # First initialize required columns as empty
    candidate['FS_weight_Dali'] = None
    candidate['TM_weight_Dali'] = None
    candidate['sourcerange_Dali'] = None
    candidate['targetrange_Dali'] = None
    candidate['start_dif'] = None
    candidate['end_dif'] = None
    candidate['FS_dif'] = None
    candidate['TM_dif'] = None
    # Add

    # Process candidate data row by row
    for idx, row in candidate.iterrows():
        # Find unique id corresponding to candidate
        candidate_id = row['id']
        
        # Find all rows with same id in Dali_check_fstm
        matching_rows = dali_check[dali_check['id'] == candidate_id]

        if len(matching_rows) >= 1:
            # If matching rows exist, select first row (already sorted by TM_weight_Dali)
            dali_row = matching_rows.iloc[0]
            
            try:
                # Calculate start_dif and end_dif
                #"""
                chopping_check = row['chopping']
                #chopping_check = row['chopping_check'] # Calculate check later
                targetrange_Dali = dali_row['target'].split('_')[1]
                
                #chopping_start, chopping_end = map(int, chopping_check.split('-'))
                chopping_start, chopping_end = map(int, (chopping_check.split('-')[0], chopping_check.split('-')[-1]))
                targetrange_start, targetrange_end = map(int, targetrange_Dali.split('-'))

                start_dif = chopping_start - targetrange_start
                end_dif = targetrange_end - chopping_end

                # Calculate FS_dif and TM_dif
                FS_weight = row['FS_weight']
                TM_weight = row['TM_weight']
                FS_weight_Dali = dali_row['FS_weight_Dali']
                TM_weight_Dali = dali_row['TM_weight_Dali']
                
                FS_dif = FS_weight_Dali - FS_weight
                TM_dif = TM_weight_Dali - TM_weight
                #"""
                # Update row data in candidate
                candidate.at[idx, 'FS_weight_Dali'] = dali_row['FS_weight_Dali']
                candidate.at[idx, 'TM_weight_Dali'] = dali_row['TM_weight_Dali']
                candidate.at[idx, 'sourcerange_Dali'] = dali_row['source'].split('_')[-1]
                candidate.at[idx, 'targetrange_Dali'] = dali_row['target'].split('_')[1]
                candidate.at[idx, 'start_dif'] = start_dif
                candidate.at[idx, 'end_dif'] = end_dif
                candidate.at[idx, 'FS_dif'] = FS_dif
                candidate.at[idx, 'TM_dif'] = TM_dif
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue

    # Move specific columns to front
    columns_order = [
        'source', 'target', 'FS_weight', 'TM_weight', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif'
    ] + [col for col in candidate.columns if col not in [
        'source', 'target', 'FS_weight', 'TM_weight', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif'
    ]]
    
    # Rearrange order
    candidate = candidate[columns_order]

    # Save processed data to CSV file
    candidate.to_csv(output_file, index=False)
    print(f"Processing completed, result saved to {output_file}")

def copy_directory(src_dir, dest_dir):
    """
    Copy a directory and its contents to another directory

    :param src_dir: Source directory path
    :param dest_dir: Destination directory path
    """
    try:
        shutil.copytree(src_dir, dest_dir)
        print(f"Directory '{src_dir}' successfully copied to '{dest_dir}'")
    except Exception as e:
        print(f"Copy failed: {e}")

def add_sequence_2_after_check(csv1_path, fasta_dict, output_path):
    # 1. Read csv1 file
    csv1 = pd.read_csv(csv1_path)
    
    # 2. Process fasta dictionary
    fasta_records = fasta_dict
    uniprot_to_len = {}
    uniprot_info = {}
    uniprot_seq = {}
    for key in fasta_records.keys():
        record = fasta_records[key]
        uniprot_id = record.id.split('|')[1]
        info = record.description
        uniprot_to_len[uniprot_id] = len(record.seq)
        uniprot_info[uniprot_id] = info
        uniprot_seq[uniprot_id] = str(record.seq)

    def get_additional_info(row):
        # Initialize return values
        target_domain_seq = ""
        source_domain_seq = ""
        combine_seq = ""
        domain_seq_sim = 0.0
        
        try:
            # Process target part
            if pd.notna(row['targetrange_Dali']) and row['targetrange_Dali'] != '':
                target = row['target']
                target_chopping = row['targetrange_Dali']
                target_id = target.split('-')[1]
                target_seq = uniprot_seq.get(target_id, "")
                
                if target_seq:
                    start_res, end_res = map(int, target_chopping.split('-'))
                    target_domain_seq = target_seq[start_res-1:end_res]
            
            # Process source part
            if pd.notna(row['sourcerange_Dali']) and row['sourcerange_Dali'] != '':
                source = row['source']
                source_chopping = row['sourcerange_Dali']
                source_id = source.split('_')[0]
                source_seq = uniprot_seq.get(source_id, "")
                
                if source_seq:
                    start_res, end_res = map(int, source_chopping.split('-'))
                    source_domain_seq = source_seq[start_res-1:end_res]
                    
                    # Calculate sequence similarity
                    if target_domain_seq and source_domain_seq:
                        domain_seq_sim = calculate_protein_sequence_similarity(
                            target_domain_seq, source_domain_seq
                        )
                    
                    # Generate combine_seq
                    if source_seq and target_domain_seq:
                        src_start, src_end = map(int, source_chopping.split('-'))
                        combine_seq = (
                            source_seq[:src_start-1] + 
                            target_domain_seq + 
                            source_seq[src_end:]
                        )
        
        except Exception as e:
            print(f"Error processing row {row.name}: {e}")
        
        return pd.Series([
            target_domain_seq,
            source_domain_seq,
            combine_seq,
            domain_seq_sim
        ], index=[
            'target_domain_seq_Dali',
            'source_domain_seq_Dali',
            'combine_seq_Dali',
            'domain_seq_sim_dali'
        ])
    
    # Apply function
    csv1[['target_domain_seq_Dali', 'source_domain_seq_Dali', 
          'combine_seq_Dali', 'domain_seq_sim_dali']] = csv1.apply(get_additional_info, axis=1)
    
    # Adjust column order
    columns_order = [
        'source', 'target', 'FS_weight', 'TM_weight',  
        'target_domain_seq_Dali', 'source_domain_seq_Dali',
        'combine_seq_Dali', 'domain_seq_sim_dali',
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
    ] + [col for col in csv1.columns if col not in [
        'source', 'target', 'FS_weight', 'TM_weight', 
        'target_domain_seq_Dali', 'source_domain_seq_Dali',
        'combine_seq_Dali', 'domain_seq_sim_dali',
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
    ]]
    
    csv1 = csv1[columns_order]
    csv1.to_csv(output_path, index=False)
    print(f"Output file saved to: {output_path}")
    
def calculate_protein_sequence_similarity(seq1, seq2):
    """
    Calculate similarity between two protein sequences using BLOSUM62 scoring matrix and normalized by self-alignment
    
    Parameters:
        seq1 (str): First protein sequence
        seq2 (str): Second protein sequence
        
    Returns:
        float: Normalized similarity score (0-1 range)
    """
    # Parameter check
    if not isinstance(seq1, str) or not isinstance(seq2, str):
        return 0.0
        
    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0
    
    # Load BLOSUM62 matrix
    matrix = substitution_matrices.load("BLOSUM62")
    
    try:
        # Use BLOSUM62 matrix for global alignment
        # Set gap_open=-10, gap_extend=-0.5 as common parameters
        alignments = pairwise2.align.globalds(seq1, seq2, 
                                            matrix, -10, -0.5,
                                            one_alignment_only=True)
        
        if not alignments:
            return 0.0
            
        # Get alignment score
        alignment_score = alignments[0].score
        
        # Calculate self-alignment scores for normalization
        self_score1 = pairwise2.align.globalds(seq1, seq1, 
                                            matrix, -10, -0.5,
                                            one_alignment_only=True)[0].score
        self_score2 = pairwise2.align.globalds(seq2, seq2, 
                                            matrix, -10, -0.5,
                                            one_alignment_only=True)[0].score
        
        # Normalized score (use smaller of two self-alignment scores for normalization)
        normalized_score = alignment_score / min(self_score1, self_score2)
        
        # Ensure score is in 0-1 range
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        return round(normalized_score, 4)
        
    except Exception as e:
        print(f"Error calculating sequence similarity: {e}")
        return 0.0    
    
def add_sequence_2_after_check0(csv1_path, fasta_dict, output_path):
    # 1. Read csv1 and csv2 files
    csv1 = pd.read_csv(csv1_path)
    
    # 3. Read fasta file once and extract UniProt ID and sequence length
    #with open(fasta_path, "r") as fasta_in:
    #    fasta_records = list(SeqIO.parse(fasta_in, "fasta"))
    fasta_records = fasta_dict
    # Convert FASTA records to dictionary, key as UniProt ID, value as sequence length
    uniprot_to_len = {}
    uniprot_info = {}
    uniprot_seq = {}
    for key in fasta_records.keys():
        # Extract UniProt ID: assume FASTA ID format as "tr|X6BLS5|X6BLS5_9HYPH"
        record = fasta_records[key]
        uniprot_id = record.id.split('|')[1]  # Get UniProt ID part by splitting
        info = record.description
        uniprot_to_len[uniprot_id] = len(record.seq)
        uniprot_info[uniprot_id] = info
        uniprot_seq[uniprot_id] = str(record.seq)




    
    # 4. Merge data to csv1
    def get_additional_info(row):
        # Initialize default return values
        target_domain_seq = []
        
        # Check if targetrange_Dali is empty
        if pd.notna(row['targetrange_Dali']) and row['targetrange_Dali'] != '':
            try:
                # Extract uniprot_id from target (assume format as target-xxxx, take part after '-')
                target = row['target']
                chopping_check = row['targetrange_Dali']
                target_id = target.split('-')[1]

                # Get sequence length of UniProt ID
                target_seq = uniprot_seq.get(target_id, "")
                
                if target_seq:  # Check if target_seq exists and is not empty
                    # Parse range
                    start_res, end_res = map(int, chopping_check.split('-'))
                    # Extract domain sequence
                    target_domain_seq = target_seq[start_res-1:end_res]
                    
            except (IndexError, ValueError, AttributeError) as e:
                # Handle possible errors: split failure, int conversion failure, etc.
                print(f"Error processing row {row.name}: {e}")
                target_domain_seq = []
        
        return pd.Series([target_domain_seq], index=['target_domain_seq'])
    
    # 5. Apply get_additional_info function to each row in csv1

    csv1[['target_domain_seq_Dali']] = csv1.apply(get_additional_info, axis=1)
    
    
    # Move specific columns to front
    columns_order = [
        'source', 'target', 'FS_weight', 'TM_weight', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif','target_domain_seq_Dali'
    ] + [col for col in csv1.columns if col not in [
        'source', 'target', 'FS_weight', 'TM_weight', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif','target_domain_seq_Dali'
    ]]
    # Adjust column order
    csv1 = csv1[columns_order]
    
    # 6. Save merged csv1
    csv1.to_csv(output_path, index=False)
    print(f"Output file saved to: {output_path}")

import shutil

import shutil

def organize_query_domains_files(query_path, out_dir):
    # Ensure query_path and out_dir exist
    if not os.path.exists(query_path):
        print(f"Path {query_path} does not exist!")
        return []
    
    if not os.path.exists(out_dir):
        print(f"Path {out_dir} does not exist! Will create this directory.")
        os.makedirs(out_dir)

    new_paths = []  # Used to save new path of each file
    
    # Traverse all files in query_path
    for filename in os.listdir(query_path):
        if filename.endswith(".pdb"):
            # Get full path of each file
            pdb_file_path = os.path.join(query_path, filename)
            
            # Get filename (without extension)
            folder_name = os.path.splitext(filename)[0]
            
            # Create new folder path
            new_folder_path = os.path.join(out_dir, folder_name)
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            
            # Construct new file path
            new_file_path = os.path.join(new_folder_path, filename)
            
            # Copy pdb file to new folder
            shutil.copy(pdb_file_path, new_file_path)
            
            # Add new path to return list
            new_paths.append(new_file_path)
            print(f"Moved {filename} to folder {new_folder_path}")
    
    print("All files have been organized!")
    
    # Return all new paths
    return new_paths
def fstm_1file(query_file,  fs_results_dirs, domain_threshold, fs_targetdb_name, fs_targetdb_path):
    """Process a single query file with FoldSeek"""
    # Step 1: Extract relevant paths and names
    fs_querydb_name = os.path.splitext(os.path.basename(query_file))[0]  # Simplified path handling
    fs_querydb_path = os.path.join(fs_results_dirs, f"{fs_querydb_name}/")
    fs_result_file = os.path.join(fs_results_dirs, f"{domain_threshold}_{fs_querydb_name}.m8")
    fstm_dali_result = os.path.join(fs_results_dirs, f"{fs_querydb_name}_{domain_threshold}_fstm.csv")

    # Step 2: Ensure directories exist
    ensure_dir(fs_querydb_path)
    
    # Step 3: Convert PDB to FoldSeek database
    gpr.convert_pdb_to_foldseek_db(query_file, fs_querydb_path, fs_querydb_name)
    
    # Step 4: Run FoldSeek
    ensure_dir(fs_results_dirs+'/TMP/')
    Tmp_dir = os.path.join(fs_results_dirs, 'TMP', f"{domain_threshold}_{fs_querydb_name}")  # Cleaner path

    gpr.run_foldseek(
        fs_querydb_path + fs_querydb_name + ".db", 
        fs_targetdb_path + fs_targetdb_name + ".db",
        fs_results=fs_result_file,
        tmp_dir=Tmp_dir
    )
    
    # Step 5: Create cytoscape network
    gprn.create_fs_cytoscape_network(fs_result_file, fstm_dali_result)

    return fstm_dali_result

def fstm_1file0(query_file, Data_dir, fs_results_dirs, domain_threshold, all_New_query_dirs, fs_targetdb_name, fs_targetdb_path, Target_domains_dali):
    # Step 1: Extract relevant paths and names
    fs_querydb_name = os.path.splitext(query_file)[0].split('/')[-1]
    fs_querydb_path = os.path.join(Data_dir, f"FS_results/{fs_querydb_name}/")
    fs_querypdb_dir = query_file
    fs_result_file = os.path.join(fs_results_dirs, f"{domain_threshold}_{fs_querydb_name}.m8")
    fstm_dali_result = os.path.join(Data_dir, f"FS_results/{fs_querydb_name}_{domain_threshold}_fstm.csv")

    # Step 2: Ensure directories exist
    ensure_dir(fs_querydb_path)
    
    # Step 3: Convert PDB to FoldSeek database (gpr.convert_pdb_to_foldseek_db)
    gpr.convert_pdb_to_foldseek_db(fs_querypdb_dir, fs_querydb_path, fs_querydb_name)
    
    # Step 4: Run FoldSeek for query and target database (gpr.run_foldseek)
    ensure_dir(fs_results_dirs+'/TMP/')
    Tmp_dir = fs_results_dirs+'/TMP/'+str(domain_threshold)+os.path.splitext(query_file)[0].split('/')[-1]

    gpr.run_foldseek(fs_querydb_path + fs_querydb_name + ".db", fs_targetdb_path + fs_targetdb_name + ".db",
                     fs_results=fs_result_file,  tmp_dir=Tmp_dir)
    
    # Step 5: Create cytoscape network (gprn.create_fs_cytoscape_network)
    gprn.create_fs_cytoscape_network(fs_result_file, os.path.join(fs_results_dirs, f"{fs_querydb_name}_{domain_threshold}_fstm.csv"))

    return fstm_dali_result  # Return the path of the result file to be processed
import multiprocessing
from functools import partial
def get_fstm_1b1_para(all_New_query_dirs,  domain_threshold, fs_results_dirs, fs_targetdb_name, fs_targetdb_path):
    """Process all queries and merge results"""
    # Step 1: Initialize
    combined_df = pd.DataFrame()
    all_fstm_dali_result = os.path.join(fs_results_dirs, f"All_1b1_{domain_threshold}_fstm.csv")

    # Step 2: Filter unprocessed queries
    unprocessed_queries = []
    for query_file in all_New_query_dirs:
        query_name = os.path.splitext(os.path.basename(query_file))[0]
        expected_csv = os.path.join(fs_results_dirs, f"{query_name}_{domain_threshold}_fstm.csv")
        
        if os.path.exists(expected_csv) and os.path.getsize(expected_csv) > 0:
            print(f"Skipping processed query: {query_name}")
            continue
        unprocessed_queries.append(query_file)

    # Step 3: Process uncompleted queries
    if unprocessed_queries:
        partial_func = partial(
            fstm_1file,
            fs_results_dirs=fs_results_dirs,
            domain_threshold=domain_threshold,
            fs_targetdb_name=fs_targetdb_name,
            fs_targetdb_path=fs_targetdb_path
        )
        num_processes = 30
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.map(partial_func, unprocessed_queries)
        print(f"Processed {len(unprocessed_queries)} new queries.")

    # Step 4: Merge results
    for query_file in all_New_query_dirs:
        query_name = os.path.splitext(os.path.basename(query_file))[0]
        result_csv = os.path.join(fs_results_dirs, f"{query_name}_{domain_threshold}_fstm.csv")
        
        if os.path.exists(result_csv):
            combined_df = pd.concat([combined_df, pd.read_csv(result_csv)], ignore_index=True)

    # Step 5: Save final results
    combined_df.to_csv(all_fstm_dali_result, index=False)
    print(f"Merged results saved to {all_fstm_dali_result} (total: {len(combined_df)} entries)")
    return all_fstm_dali_result
    
def get_fstm_1b1(all_New_query_dirs, domain_threshold, fs_results_dirs, fs_targetdb_name, fs_targetdb_path):
    # Step 1: Initialize combined DataFrame
    combined_df = pd.DataFrame()
    all_fstm_dali_result = os.path.join(fs_results_dirs, f"All_1b1_{domain_threshold}_fstm.csv")

    # Step 1: Initialize combined DataFrame
    combined_df = pd.DataFrame()
    all_fstm_dali_result = os.path.join(fs_results_dirs, f"All_1b1_{domain_threshold}_fstm.csv")

    # Step 2: Run FoldSeek for each query file sequentially
    fstm_results = []
    for query_file in all_New_query_dirs:
        # Call the fstm_1file function sequentially (instead of using executor)
        fstm_dali_result = fstm_1file(query_file, fs_results_dirs, domain_threshold,  fs_targetdb_name, fs_targetdb_path)
        fstm_results.append(fstm_dali_result)
    
    # Step 3: Merge the CSV results sequentially
    for fstm_dali_result in fstm_results:
        df = pd.read_csv(fstm_dali_result)
        # Merge data frame (keeping column names only from the first file)
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # Step 4: Save the merged results to a single CSV file
    combined_df.to_csv(all_fstm_dali_result, index=False)
    print(f"All results have been merged and saved to {all_fstm_dali_result}")

import os
import pandas as pd
import glob
import shutil
import os
import pandas as pd
import glob
import shutil
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import uuid
import time


# Function: Parallel process single query PDB
def process_query_pdb(query_pdb, fs_querypdb_base_dir, main_query_dir, query_db_dir):
    """Process single query PDB file"""
    # Create unique database name
    query_db_name = query_pdb.replace(".", "_").replace("/", "_")
    
    # Copy PDB file to query directory
    src_query_path = os.path.join(fs_querypdb_base_dir, query_pdb)
    
    # Create separate folder for this PDB
    single_query_dir = os.path.join(main_query_dir, "single_pdbs", query_db_name)
    os.makedirs(single_query_dir, exist_ok=True)
    single_query_path = os.path.join(single_query_dir, query_pdb)
    
    if os.path.exists(src_query_path):
        shutil.copy(src_query_path, single_query_path)
        
        # Create database for this PDB
        db_output_dir = os.path.join(query_db_dir, query_db_name)
        os.makedirs(db_output_dir, exist_ok=True)
        gpr.convert_pdb_to_foldseek_db(single_query_dir, db_output_dir, query_db_name)
        
        return query_pdb, os.path.join(db_output_dir, f"{query_db_name}.db")
    else:
        print(f"Warning: Cannot find query PDB file: {src_query_path}")
        return query_pdb, None
    
# Function: Parallel process single target PDB
def process_target_pdb(target_pdb, fs_targetpdb_base_dir, main_target_dir, target_db_dir):
    """Process single target PDB file"""
    # Create unique database name
    target_db_name = target_pdb.replace(".", "_").replace("/", "_")
    
    # Copy PDB file to target directory
    src_target_path = os.path.join(fs_targetpdb_base_dir, target_pdb)
    
    # Create separate folder for this PDB
    single_target_dir = os.path.join(main_target_dir, "single_pdbs", target_db_name)
    os.makedirs(single_target_dir, exist_ok=True)
    single_target_path = os.path.join(single_target_dir, target_pdb)
    
    if os.path.exists(src_target_path):
        shutil.copy(src_target_path, single_target_path)
        
        # Create database for this PDB
        db_output_dir = os.path.join(target_db_dir, target_db_name)
        os.makedirs(db_output_dir, exist_ok=True)
        gpr.convert_pdb_to_foldseek_db(single_target_dir, db_output_dir, target_db_name)
        
        return target_pdb, os.path.join(db_output_dir, f"{target_db_name}.db")
    else:
        print(f"Warning: Cannot find target PDB file: {src_target_path}")
        return target_pdb, None

def process_pdb_pair(args):
    """Process single PDB pair"""
    pair_info, query_db_map, target_db_map, pair_result_dir, fs_tmp_path, worker_id = args
    index, row, query_pdb, target_pdb = pair_info
    
    # Get database paths
    query_db_path = query_db_map[query_pdb]
    target_db_path = target_db_map[target_pdb]
    
    # Create unique result filename
    result_name = f"{row['source']}_{row['sourcerange_abs']}__vs__{row['target']}_{row['targetrange']}"
    result_name = result_name.replace(".", "_").replace("/", "_")
    
    # Create unique temporary directory for each worker process
    unique_tmp_dir = os.path.join(fs_tmp_path, f"worker_{worker_id}_{uuid.uuid4().hex}")
    os.makedirs(unique_tmp_dir, exist_ok=True)
    
    # Create unique result file path
    result_m8 = os.path.join(pair_result_dir, f"{result_name}.m8")
    
    try:
        # Run foldseek, using unique temporary directory and original data files
        gpr.run_foldseek_with_error_handling(
            query_db_path, 
            target_db_path,
            fs_rawdata=os.path.join(unique_tmp_dir, f"{result_name}.raw"),
            fs_results=result_m8,
            tmp_dir=unique_tmp_dir
        )
        
        # Clean up temporary directory
        shutil.rmtree(unique_tmp_dir, ignore_errors=True)
        
        return result_m8
    except Exception as e:
        print(f"Error processing pair {index}: {e}")
        # Ensure cleanup of temporary directory
        shutil.rmtree(unique_tmp_dir, ignore_errors=True)
        return None

# Create a process for each batch
def process_batch(args):
    """Process a batch of PDB pairs"""
    batch_idx, batch, query_db_map, target_db_map, pair_result_dir, fs_tmp_path = args
    batch_results = []
    
    for i, pair_info in enumerate(batch):
        # Pass worker process ID to create unique temporary directory
        result = process_pdb_pair((pair_info, query_db_map, target_db_map, pair_result_dir, fs_tmp_path, batch_idx))
        if result:
            batch_results.append(result)
            
        # Output progress periodically
        if (i + 1) % 100 == 0:
            print(f"Batch {batch_idx}: Processed {i+1}/{len(batch)} pairs")
    
    return batch_results

      
def process_pdb_pairs_from_csv_parallel(
    csv_path, 
    fs_querypdb_base_dir, 
    fs_targetpdb_base_dir, 
    fs_results_dir,
    fs_tmp_path,
    num_processes=64  # Default use 64 processes, can be adjusted according to actual situation
):
    """
    Parallel process PDB pairs from CSV, first deduplicate to create databases, then run foldseek query for each pair in parallel
    
    Parameters:
    csv_path: Path to 10_dali_results_pro.csv
    fs_querypdb_base_dir: Base directory for query PDB files
    fs_targetpdb_base_dir: Base directory for target PDB files
    fs_results_dir: Directory to save result files
    fs_tmp_path: Base path for foldseek temporary directory
    num_processes: Number of parallel processes
    """
    start_time = time.time()
    
    # Ensure result directory exists
    os.makedirs(fs_results_dir, exist_ok=True)
    
    # Create main folders for query and target
    main_query_dir = os.path.join(fs_results_dir, "all_queries")
    main_target_dir = os.path.join(fs_results_dir, "all_targets")
    os.makedirs(main_query_dir, exist_ok=True)
    os.makedirs(main_target_dir, exist_ok=True)
    
    print(f"Reading CSV file: {csv_path}")
    # Read CSV file
    df = pd.read_csv(csv_path, encoding='latin1')
    print(f"CSV file contains {len(df)} rows of data")
    
    # Extract all unique query and target PDB files
    unique_query_pdbs = set()
    unique_target_pdbs = set()
    
    # Extract all unique PDB files from CSV
    for index, row in df.iterrows():
        # Check if targetrange length meets minimum length requirement
        try:
            # Assume targetrange format as "start-end"
            target_range = row['targetrange'].split('-')
            if len(target_range) == 2:
                start_pos = int(target_range[0])
                end_pos = int(target_range[1])
                if end_pos - start_pos < 15:
                    # If length less than 15, skip this row
                    print(f"Skipping row {index}: targetrange length {end_pos - start_pos} less than 15")
                    continue
        except (ValueError, IndexError) as e:
            # Handle exception cases, e.g., targetrange format incorrect
            print(f"Error processing targetrange for row {index}: {e}")
            continue
        
        # Only process this row when targetrange length meets requirement
        query_pdb = f"{row['source']}_{row['sourcerange_abs']}.pdb"
        target_pdb = f"{row['target']}_{row['targetrange']}.pdb"
        
        unique_query_pdbs.add(query_pdb)
        unique_target_pdbs.add(target_pdb)
    
    print(f"Found {len(unique_query_pdbs)} unique query PDB files")
    print(f"Found {len(unique_target_pdbs)} unique target PDB files")
    
    # Create directory for query databases
    query_db_dir = os.path.join(main_query_dir, "db")
    os.makedirs(query_db_dir, exist_ok=True)
    
    # Create directory for target databases
    target_db_dir = os.path.join(main_target_dir, "db")
    os.makedirs(target_db_dir, exist_ok=True)
    

    # Use process pool to process query PDBs in parallel
    print("Starting parallel processing of query PDB files...")
    # Create partial application function, pre-binding some parameters
    process_query_partial = partial(
        process_query_pdb,
        fs_querypdb_base_dir=fs_querypdb_base_dir,
        main_query_dir=main_query_dir,
        query_db_dir=query_db_dir
    )
    
    with mp.Pool(processes=min(num_processes, len(unique_query_pdbs))) as pool:
        query_results = pool.map(process_query_partial, unique_query_pdbs)
    
    # Build mapping from query PDB to database path
    query_db_map = {pdb: db_path for pdb, db_path in query_results if db_path is not None}
    print(f"Successfully processed {len(query_db_map)} query PDB files")
    
    # Use process pool to process target PDBs in parallel
    print("Starting parallel processing of target PDB files...")
    
    # Create partial application function, pre-binding some parameters
    process_target_partial = partial(
        process_target_pdb,
        fs_targetpdb_base_dir=fs_targetpdb_base_dir,
        main_target_dir=main_target_dir,
        target_db_dir=target_db_dir
    )
    
    with mp.Pool(processes=min(num_processes, len(unique_target_pdbs))) as pool:
        target_results = pool.map(process_target_partial, unique_target_pdbs)
    
    # Build mapping from target PDB to database path
    target_db_map = {pdb: db_path for pdb, db_path in target_results if db_path is not None}
    print(f"Successfully processed {len(target_db_map)} target PDB files")
    
    # Prepare CSV row batches for parallel processing
    # Prepare CSV row batches for parallel processing
    pair_data = []
    for index, row in df.iterrows():
        # Check if targetrange length meets minimum length requirement
        try:
            # Assume targetrange format as "start-end"
            target_range = row['targetrange'].split('-')
            if len(target_range) == 2:
                start_pos = int(target_range[0])
                end_pos = int(target_range[1])
                if end_pos - start_pos < 15:
                    # If length less than 15, skip this row
                    print(f"!!!!!run foldsek skipping row {index}: targetrange length {end_pos - start_pos} less than 15")
                    continue
        except (ValueError, IndexError) as e:
            # Handle exception cases, e.g., targetrange format incorrect
            print(f"Error processing targetrange for row {index}: {e}")
            continue
        
        # Only process this row when targetrange length meets requirement
        print(f"!!!!!run foldsek will process row {index}: targetrange length {end_pos - start_pos} greater than 15")
        query_pdb = f"{row['source']}_{row['sourcerange_abs']}.pdb"
        target_pdb = f"{row['target']}_{row['targetrange']}.pdb"
        
        # Only add valid PDB pairs
        if query_pdb in query_db_map and target_db_map.get(target_pdb):
            pair_data.append((index, row, query_pdb, target_pdb))
    
    # Create result directory
    pair_result_dir = os.path.join(fs_results_dir, "pair_results")
    os.makedirs(pair_result_dir, exist_ok=True)
    
    # Split pair data into batches to avoid creating too many processes
    batch_size = max(1, len(pair_data) // num_processes)
    batches = [pair_data[i:i + batch_size] for i in range(0, len(pair_data), batch_size)]
    
    print(f"Starting parallel processing of {len(pair_data)} PDB pairs, divided into {len(batches)} batches...")
    
    # Use process pool to process PDB pairs in parallel
    all_m8_files = []
    
    # Prepare batch processing parameters
    batch_args = [
        (i, batch, query_db_map, target_db_map, pair_result_dir, fs_tmp_path)
        for i, batch in enumerate(batches)
    ]
    
    with mp.Pool(processes=min(num_processes, len(batches))) as pool:
        batch_results = pool.map(process_batch, batch_args)
    
    # Flatten result list
    for results in batch_results:
        all_m8_files.extend([r for r in results if r])
    
    print(f"Successfully processed {len(all_m8_files)} PDB pairs")
    
    # Merge all .m8 files
    print("Starting to merge result files...")
    merged_m8 = os.path.join(fs_results_dir, "merged_results.m8")
    with open(merged_m8, 'w') as outfile:
        # Directly merge all file contents, no special handling for header lines
        for m8_file in all_m8_files:
            try:
                with open(m8_file, 'r') as infile:
                    # Directly copy entire file content
                    outfile.write(infile.read())
            except Exception as e:
                print(f"Error merging file {m8_file}: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"All results merged to: {merged_m8}")
        print(f"Total processing time: {total_time:.2f} seconds")
        return merged_m8

def process_pdb_pairs_from_csv_optimized(
    csv_path, 
    fs_querypdb_base_dir, 
    fs_targetpdb_base_dir, 
    fs_results_dir,
    fs_tmp_path
):
    """
    Process PDB pairs from CSV file, first deduplicate to create databases, then run foldseek query for each pair, and merge results
    
    Parameters:
    csv_path: Path to 10_dali_results_pro.csv
    fs_querypdb_base_dir: Base directory for query PDB files
    fs_targetpdb_base_dir: Base directory for target PDB files
    fs_results_dir: Directory to save result files
    fs_tmp_path: foldseek temporary directory
    """
    # Ensure result directory exists
    os.makedirs(fs_results_dir, exist_ok=True)
    
    # Create main folders for query and target
    main_query_dir = os.path.join(fs_results_dir, "all_queries")
    main_target_dir = os.path.join(fs_results_dir, "all_targets")
    os.makedirs(main_query_dir, exist_ok=True)
    os.makedirs(main_target_dir, exist_ok=True)
    
    # Read CSV file
    df = pd.read_csv(csv_path, encoding='latin1')
    
    # Extract all unique query and target PDB files
    unique_query_pdbs = set()
    unique_target_pdbs = set()
    
    # Create mapping dictionaries to track PDB files and their database paths
    query_db_map = {}
    target_db_map = {}
    
    # Extract all unique PDB files from CSV
    for index, row in df.iterrows():
        query_pdb = f"{row['source']}_{row['sourcerange_abs']}.pdb"
        target_pdb = f"{row['target']}_{row['targetrange']}.pdb"
        
        unique_query_pdbs.add(query_pdb)
        unique_target_pdbs.add(target_pdb)
    
    print(f"Found {len(unique_query_pdbs)} unique query PDB files")
    print(f"Found {len(unique_target_pdbs)} unique target PDB files")
    
    # Process all unique query PDB files
    query_db_dir = os.path.join(main_query_dir, "db")
    os.makedirs(query_db_dir, exist_ok=True)
    
    for query_pdb in unique_query_pdbs:
        # Create unique database name
        query_db_name = query_pdb.replace(".", "_").replace("/", "_")
        
        # Copy PDB file to query directory
        src_query_path = os.path.join(fs_querypdb_base_dir, query_pdb)
        dst_query_dir = os.path.join(main_query_dir, "pdbs")
        os.makedirs(dst_query_dir, exist_ok=True)
        dst_query_path = os.path.join(dst_query_dir, query_pdb)
        
        if os.path.exists(src_query_path):
            shutil.copy(src_query_path, dst_query_path)
            
            # Create separate folder for this PDB
            single_query_dir = os.path.join(main_query_dir, "single_pdbs", query_db_name)
            os.makedirs(single_query_dir, exist_ok=True)
            single_query_path = os.path.join(single_query_dir, query_pdb)
            shutil.copy(src_query_path, single_query_path)
            
            # Create database for this PDB
            gpr.convert_pdb_to_foldseek_db(single_query_dir, query_db_dir+'/'+query_db_name, query_db_name)
            
            # Record database path corresponding to this PDB
            query_db_map[query_pdb] = os.path.join(query_db_dir+'/'+query_db_name, f"{query_db_name}.db")
            
            print(f"Processing query PDB: {query_pdb}")
        else:
            print(f"Warning: Cannot find query PDB file: {src_query_path}")
    
    # Process all unique target PDB files
    target_db_dir = os.path.join(main_target_dir, "db")
    os.makedirs(target_db_dir, exist_ok=True)
    
    for target_pdb in unique_target_pdbs:
        # Create unique database name
        target_db_name = target_pdb.replace(".", "_").replace("/", "_")
        
        # Copy PDB file to target directory
        src_target_path = os.path.join(fs_targetpdb_base_dir, target_pdb)
        dst_target_dir = os.path.join(main_target_dir, "pdbs")
        os.makedirs(dst_target_dir, exist_ok=True)
        dst_target_path = os.path.join(dst_target_dir, target_pdb)
        
        if os.path.exists(src_target_path):
            shutil.copy(src_target_path, dst_target_path)
            
            # Create separate folder for this PDB
            single_target_dir = os.path.join(main_target_dir, "single_pdbs", target_db_name)
            os.makedirs(single_target_dir, exist_ok=True)
            single_target_path = os.path.join(single_target_dir, target_pdb)
            shutil.copy(src_target_path, single_target_path)
            
            # Create database for this PDB
            gpr.convert_pdb_to_foldseek_db(single_target_dir, target_db_dir+'/'+target_db_name, target_db_name)
            
            # Record database path corresponding to this PDB
            target_db_map[target_pdb] = os.path.join(target_db_dir+'/'+target_db_name, f"{target_db_name}.db")
            
            print(f"Processing target PDB: {target_pdb}")
        else:
            print(f"Warning: Cannot find target PDB file: {src_target_path}")
    
    # List to store all m8 file paths
    all_m8_files = []
    
    # Now iterate through each row of CSV, run foldseek for each PDB pair
    for index, row in df.iterrows():
        query_pdb = f"{row['source']}_{row['sourcerange_abs']}.pdb"
        target_pdb = f"{row['target']}_{row['targetrange']}.pdb"
        
        # Check if databases exist for this PDB pair
        if query_pdb in query_db_map and target_pdb in target_db_map:
            query_db_path = query_db_map[query_pdb]
            target_db_path = target_db_map[target_pdb]
            
            # Create result directory for this pair
            pair_result_dir = os.path.join(fs_results_dir, f"pair_results")
            os.makedirs(pair_result_dir, exist_ok=True)
            
            # Create unique result filename
            result_name = f"{row['source']}_{row['sourcerange_abs']}__vs__{row['target']}_{row['targetrange']}"
            result_name = result_name.replace(".", "_").replace("/", "_")
            result_m8 = os.path.join(pair_result_dir, f"{result_name}.m8")
            
            # Run foldseek
            gpr. run_foldseek_with_error_handling(
                query_db_path, 
                target_db_path,
                fs_rawdata=os.path.join(pair_result_dir, f"{result_name}.raw"),
                fs_results=result_m8,
                tmp_dir=fs_tmp_path
            )
            
            # Add result to list
            all_m8_files.append(result_m8)
            print(f"Completed processing pair {index}: {query_pdb} vs {target_pdb}")
        else:
            print(f"Warning: Cannot find database for pair {index}: {query_pdb} or {target_pdb}")
    
    # Merge all .m8 files
    merged_m8 = os.path.join(fs_results_dir, "merged_results.m8")
    with open(merged_m8, 'w') as outfile:
        # Write header row of first file (if any)
        if all_m8_files:
            with open(all_m8_files[0], 'r') as first_file:
                header = first_file.readline()
                outfile.write(header)
        
        # Merge contents of all files
        for m8_file in all_m8_files:
            with open(m8_file, 'r') as infile:
                # Skip header row (if any)
                next(infile, None)
                # Write remaining content
                for line in infile:
                    outfile.write(line)
    
    print(f"All results merged to: {merged_m8}")
    return merged_m8


def importpl_pdb_target_parallel(Dali_bin,pdb_dir, dat_dir, work_path, max_workers=8, batch_size=10):
    print("Starting import process with additional verification for dat_dir consistency.")
    print(f"max_workers =  {max_workers}")
    print(f"batch_size =  {batch_size}")
    
    os.makedirs(dat_dir, exist_ok=True)
    os.makedirs(work_path, exist_ok=True)
    
    # CSV file path
    csv_file = os.path.join(work_path, "dat", "importpl_target_ids.csv")
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Get PDB files to process
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    print(f"pdb_files =  {len(pdb_files)}")
    
    all_pdb_ids = list(generate_pdb_ids_target(len(pdb_files)))
    print(f"all_pdb_ids =  {len(all_pdb_ids)}")
    
    # Read records from CSV file
    processed_files = {}
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                processed_files[row[0]] = row[1]
    print(f"processed_files =  {len(processed_files)}")
    
    # Check matching
    mismatch_found = False
    for pdb_file, pdb_id in zip(pdb_files, all_pdb_ids):
        pdb_name = pdb_file.split('.')[0]
        if pdb_name in processed_files and processed_files[pdb_name] != pdb_id:
            print(f"Warning: Mismatch found in CSV for {pdb_name}. Expected ID: {pdb_id}, Found ID: {processed_files[pdb_name]}")
            mismatch_found = True

    if mismatch_found:
        print("Warning: Some records in the CSV file do not match the expected file-ID correspondence.")
    
    # Check files in dat_dir
    existing_dat_files = {f[:-5] for f in os.listdir(dat_dir) if f.endswith("A.dat")}
    verified_files = set()
    
    for pdb_name, pdb_id in processed_files.items():
        if pdb_id in existing_dat_files:
            verified_files.add(pdb_name)
        else:
            print(f"Warning: {pdb_name} with ID {pdb_id} is in CSV but not found in dat_dir. It will be reprocessed.")
    
    # Filter out files to process
    pdb_files_to_process = []
    pdb_ids_to_process = []
    for pdb_file, pdb_id in zip(pdb_files, all_pdb_ids):
        pdb_name = pdb_file.split('.')[0]
        if pdb_name not in verified_files:
            pdb_files_to_process.append(pdb_file)
            pdb_ids_to_process.append(pdb_id)
    print(f"pdb_ids_to_process =  {len(pdb_ids_to_process)}")
    print(f"Total files to process: {len(pdb_files_to_process)}")
    if not pdb_files_to_process:
        print("All files have already been processed.")
        return
    
    start_time = time.time()
    
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Original_PDB_File", "Generated_PDB_ID"])
        
        with tqdm(total=len(pdb_files_to_process), desc="Processing") as pbar:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:     
                # Submit tasks in batches
                for batch_start in range(0, len(pdb_files_to_process), batch_size):
                    batch_files = pdb_files_to_process[batch_start: batch_start + batch_size]
                    batch_ids = pdb_ids_to_process[batch_start: batch_start + batch_size]
                    futures = {} 
                    for pdb_file, pdb_id in zip(batch_files, batch_ids):
                        future = executor.submit(
                            import_single_pdb, Dali_bin,
                            pdb_file, pdb_id, pdb_dir, dat_dir, work_path
                        )
                        futures[future] = (pdb_file, pdb_id)
                    
                    # Process completed tasks
                    for future in as_completed(futures):
                        pdb_file, pdb_id = futures[future]
                        try:
                            result = future.result()
                            if result:
                                writer.writerow([pdb_file.split('.')[0], pdb_id])
                        except Exception as e:
                            print(f"Error processing {pdb_file}: {e}")
                        finally:
                            pbar.update(1)
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    
def generate_pdb_ids_target(n):
    """ Generator that creates unique 4-character IDs based on detailed combinations of uppercase, lowercase, and digits """

    uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lowercase = 'abcdefghijklmnopqrstuvwxyz'
    digits = '123456789'  # Note: first digit cannot be 0, other positions can be 0

    count = 0

    # 1. All uppercase letter combinations
    for combo in itertools.product(uppercase, repeat=4):
        yield "".join(combo)
        count += 1
        if count >= n:
            return

    # 2. Combinations containing 1 lowercase letter
    for pos in range(4):
        for combo in itertools.product(uppercase, repeat=3):
            for l in lowercase:
                temp = list(combo)
                temp.insert(pos, l)
                yield "".join(temp)
                count += 1
                if count >= n:
                    return

    # 3. Combinations containing 1 digit
    for pos in range(4):
        for combo in itertools.product(uppercase, repeat=3):
            for d in digits:
                temp = list(combo)
                temp.insert(pos, d)
                if pos != 0 or d != '0':  # First position cannot be 0
                    yield "".join(temp)
                    count += 1
                    if count >= n:
                        return

    # 4. Combinations containing 2 lowercase letters
    for pos1, pos2 in itertools.combinations(range(4), 2):
        for combo in itertools.product(uppercase, repeat=2):
            for l1, l2 in itertools.product(lowercase, repeat=2):
                temp = list(combo)
                for p, l in zip([pos1, pos2], [l1, l2]):
                    temp.insert(p, l)
                yield "".join(temp)
                count += 1
                if count >= n:
                    return

    # 5. Combinations containing 2 digits
    for pos1, pos2 in itertools.combinations(range(4), 2):
        for combo in itertools.product(uppercase, repeat=2):
            for d1, d2 in itertools.product(digits, repeat=2):
                temp = list(combo)
                for p, d in zip([pos1, pos2], [d1, d2]):
                    temp.insert(p, d)
                if pos1 != 0 or d1 != '0':  # First position cannot be 0
                    if pos2 != 0 or d2 != '0':  # First position cannot be 0
                        yield "".join(temp)
                        count += 1
                        if count >= n:
                            return

    # 6. Combinations containing 3 lowercase letters
    for pos1, pos2, pos3 in itertools.combinations(range(4), 3):
        for combo in itertools.product(uppercase, repeat=1):
            for l1, l2, l3 in itertools.product(lowercase, repeat=3):
                temp = list(combo)
                for p, l in zip([pos1, pos2, pos3], [l1, l2, l3]):
                    temp.insert(p, l)
                yield "".join(temp)
                count += 1
                if count >= n:
                    return

    # 7. Combinations containing 3 digits
    for pos1, pos2, pos3 in itertools.combinations(range(4), 3):
        for combo in itertools.product(uppercase, repeat=1):
            for d1, d2, d3 in itertools.product(digits, repeat=3):
                temp = list(combo)
                for p, d in zip([pos1, pos2, pos3], [d1, d2, d3]):
                    temp.insert(p, d)
                if pos1 != 0 or d1 != '0':
                    if pos2 != 0 or d2 != '0':
                        if pos3 != 0 or d3 != '0':
                            yield "".join(temp)
                            count += 1
                            if count >= n:
                                return

    # 8. All lowercase letter combinations
    for combo in itertools.product(lowercase, repeat=4):
        yield "".join(combo)
        count += 1
        if count >= n:
            return

    # 9. All digit combinations
    for combo in itertools.product(digits + '0', repeat=4):
        if combo[0] != '0':  # First digit cannot be 0
            yield "".join(combo)
            count += 1
            if count >= n:
                return

    # 10. Complex combinations of uppercase letters, lowercase letters and digits
    # 10.1 2 uppercase letters, 1 lowercase letter, 1 digit
    for uppercase_pos in itertools.combinations(range(4), 2):
        remaining_pos = [i for i in range(4) if i not in uppercase_pos]
        for lowercase_pos, digit_pos in itertools.permutations(remaining_pos, 2):
            for u1, u2 in itertools.product(uppercase, repeat=2):
                for l in lowercase:
                    for d in digits:
                        temp = [''] * 4
                        temp[uppercase_pos[0]] = u1
                        temp[uppercase_pos[1]] = u2
                        temp[lowercase_pos] = l
                        temp[digit_pos] = d
                        if temp[0] != '0':  # First position cannot be 0
                            yield "".join(temp)
                            count += 1
                            if count >= n:
                                return

    # 10.2 2 lowercase letters, 1 uppercase letter, 1 digit
    for lowercase_pos in itertools.combinations(range(4), 2):
        remaining_pos = [i for i in range(4) if i not in lowercase_pos]
        for uppercase_pos, digit_pos in itertools.permutations(remaining_pos, 2):
            for l1, l2 in itertools.product(lowercase, repeat=2):
                for u in uppercase:
                    for d in digits:
                        temp = [''] * 4
                        temp[lowercase_pos[0]] = l1
                        temp[lowercase_pos[1]] = l2
                        temp[uppercase_pos] = u
                        temp[digit_pos] = d
                        if temp[0] != '0':
                            yield "".join(temp)
                            count += 1
                            if count >= n:
                                return

    # 10.3 2 digits, 1 uppercase letter, 1 lowercase letter
    for digit_pos in itertools.combinations(range(4), 2):
        remaining_pos = [i for i in range(4) if i not in digit_pos]
        for uppercase_pos, lowercase_pos in itertools.permutations(remaining_pos, 2):
            for d1, d2 in itertools.product(digits, repeat=2):
                for u in uppercase:
                    for l in lowercase:
                        temp = [''] * 4
                        temp[digit_pos[0]] = d1
                        temp[digit_pos[1]] = d2
                        temp[uppercase_pos] = u
                        temp[lowercase_pos] = l
                        if temp[0] != '0':
                            yield "".join(temp)
                            count += 1
                            if count >= n:
                                return                   


# Single PDB file import logic
def import_single_pdb(Dali_bin,pdb_file, pdb_id, pdb_dir, dat_dir, work_path):
    short_work_dir = os.path.join(work_path, f"dali_temp_{pdb_id}")
    os.makedirs(short_work_dir, exist_ok=True)

    pdb_path = os.path.join(pdb_dir, pdb_file)
    short_pdb_path = os.path.join(short_work_dir, f"{pdb_id}.pdb")
    shutil.copyfile(pdb_path, short_pdb_path)

    short_dat_dir = os.path.join(short_work_dir, "dat")
    os.makedirs(short_dat_dir, exist_ok=True)

    # Build and run import.pl command
    cmd = [
        os.path.join(Dali_bin, "import.pl"), 
        "--pdbfile", f"./{pdb_id}.pdb",  
        "--pdbid", pdb_id, 
        "--dat", "./dat"
    ]

    original_dir = os.getcwd()
    os.chdir(short_work_dir)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error importing {pdb_file}: {result.stderr}")
            #print(f"Successfully imported {pdb_file}")
    finally:
        os.chdir(original_dir)

    for file in os.listdir(short_dat_dir):
        shutil.move(os.path.join(short_dat_dir, file), dat_dir)

    shutil.rmtree(short_work_dir)
    return pdb_file, pdb_id

def importpl_pdb_query(Dali_bin,pdb_dir, dat_dir, work_path):
    """ Imports PDB files into Dali internal format using import.pl script and logs to CSV """
    
    # Ensure Dali data directory exists
    ensure_dir(dat_dir)  
    
    # Create a short path working directory for running import.pl
    short_work_dir = os.path.join(work_path, "dali_work/query")
    if not os.path.exists(short_work_dir):
        os.makedirs(short_work_dir)
    
    # Create ID generator
    pdb_id_generator = generate_pdb_ids_query()

    # Get all PDB files
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    csv_file = os.path.join(work_path,"dat","importpl_query_ids.csv")
    # Open CSV file to record original filename and newly generated ID
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Original_PDB_File", "Generated_PDB_ID"])  # CSV header row
        
        for pdb_file in pdb_files:
            pdb_path = os.path.join(pdb_dir, pdb_file)
            pdb_id = next(pdb_id_generator)  # Get next unique ID from generator
            
            # Copy PDB file to short path working directory and rename
            short_pdb_path = os.path.join(short_work_dir, f"{pdb_id}.pdb")
            shutil.copyfile(pdb_path, short_pdb_path)
            
            # Ensure dat_dir is also switched to a relatively short path
            short_dat_dir = os.path.join(short_work_dir, "dat")
            if not os.path.exists(short_dat_dir):
                os.makedirs(short_dat_dir)

            # Build and run import.pl command
            cmd = [
                os.path.join(Dali_bin, "import.pl"), 
                "--pdbfile", f"./{pdb_id}.pdb",  # Use relative path
                "--pdbid", pdb_id, 
                "--dat", "./dat"  # Relative path simplified to "./dat"
            ]
            
            # Switch to short path directory for command execution
            original_dir = os.getcwd()
            os.chdir(short_work_dir)
            try:
                subprocess.run(cmd)
            finally:
                # After completion, switch back to original working directory
                os.chdir(original_dir)
            
            # Copy generated results back to original dat_dir
            for file in os.listdir(short_dat_dir):
                shutil.move(os.path.join(short_dat_dir, file), dat_dir)
            
            # Record original filename and generated PDB ID to CSV file
            writer.writerow([pdb_file.split('.')[0], pdb_id])

    print("PDB files have been imported and logged successfully.")
def generate_pdb_ids_query():
    """ Generator that creates unique 4-character IDs in the desired sequence: 0A00, 0A01, ..., 9z99, 00AA, etc. """
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    uletters ='0'  # Extended character set
    digits = '0123456789'
    
    # 1. Generate starting from 0A00, 0A01, ..., 9z99
    for digit in uletters:
        for letter in letters:
            for num in range(100):  # Generate numbers from 000 to 999
                yield f"{digit}{letter}{num:02d}"  # Format as 0A000, 0A001, ..., 9z999

    # 2. Generate 00AA, 00AB, ..., 99zz
    for digit1 in uletters:
        for digit2 in digits:
            for letter1 in letters:
                for num in range(10):  # Generate numbers from 00 to 99
                    yield f"{digit1}{digit2}{letter1}{num:01d}"  # Format as 00AA00, 00AA01, ..., 99zz99

    # 3. Generate 0AAA, 0AAB, ..., 9zzz
    for digit in uletters:
        for letter1 in letters:
            for letter2 in letters:
                for letter3 in letters:
                    yield f"{digit}{letter1}{letter2}{letter3}"  # Format as 0AAA0, 0AAA1, ..., 9zzz9
