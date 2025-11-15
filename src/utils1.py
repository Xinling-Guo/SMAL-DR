import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logging
import requests
import pandas as pd
import os
import json
import csv
import re
import shutil
from Bio.PDB import PDBList
import sys
import glob
from Bio import PDB
import subprocess
from Bio import pairwise2
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MAX_RETRIES = 5
RETRY_DELAY = 5  # Delay time in seconds
TED_info_URL =  "https://ted.cathdb.info/api/v1"

def create_session_with_retries(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 503, 504)):
    """Create a session with retry mechanism"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def download_uniprot_data_pages(ipr_id, output_file, max_retries=3, delay=2):
    """Download UniProt data with retry mechanism"""
    base_url = f"https://rest.uniprot.org/uniprotkb/search?query={ipr_id}+AND+reviewed:true&format=tsv&fields=accession,id,protein_name,cc_domain,ft_domain,organism_name,length&size=500"
    
    session = create_session_with_retries(retries=max_retries)
    
    next_page_url = base_url
    attempt = 0
    
    while attempt < max_retries:
        try:
            with open(output_file, 'w') as f:
                while next_page_url:
                    logger.info(f"Fetching: {next_page_url}")
                    response = session.get(next_page_url, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.text
                        if not data.strip():
                            break
                        f.write(data)
                        
                        # Check if there is next page
                        link_header = response.headers.get('Link')
                        if link_header and 'rel="next"' in link_header:
                            next_page_url = link_header.split(';')[0].strip('<>')
                        else:
                            next_page_url = None
                    else:
                        logger.warning(f"Failed to download data. Status code: {response.status_code}")
                        break
            
            logger.info(f"Downloaded data saved to {output_file}")
            return True
            
        except requests.exceptions.RequestException as e:
            attempt += 1
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.error(f"All {max_retries} attempts failed for {ipr_id}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False
    
    return False

def download_interpro_data(ipr_id, output_file, max_retries=3, delay=2):
    """Download InterPro data with retry mechanism"""
    url = f"https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/pfam/{ipr_id}/?page_size=200"
    
    session = create_session_with_retries(retries=max_retries)
    attempt = 0
    
    while attempt < max_retries:
        try:
            logger.info(f"Fetching InterPro data for {ipr_id}")
            response = session.get(url, headers={"Accept": "application/json"}, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse data (keep original logic)
                extracted_data = []
                for result in data['results']:
                    protein_info = {
                        "accession": result['metadata']['accession'],
                        "protein_name": result['metadata']['name'],
                        "organism_name": result['metadata']['source_organism']['scientificName'],
                        "length": result['metadata']['length'],
                        "domains": []
                    }
                    
                    for entry in result['entries']:
                        for location in entry.get('entry_protein_locations', []):
                            for fragment in location['fragments']:
                                domain_info = {
                                    "domain_accession": entry['accession'],
                                    "start": fragment['start'],
                                    "end": fragment['end'],
                                    "model": location.get('model'),
                                    "score": location.get('score')
                                }
                                protein_info["domains"].append(domain_info)
                    
                    extracted_data.append(protein_info)
                
                # Save to TSV file
                with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
                    writer = csv.writer(tsvfile, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
                    writer.writerow(['Accession', 'Protein Name', 'Organism', 'Length', 'Domain Accession', 'Start', 'End', 'Model', 'Score'])
                    
                    for protein in extracted_data:
                        for domain in protein['domains']:
                            writer.writerow([
                                protein['accession'], 
                                protein['protein_name'], 
                                protein['organism_name'], 
                                protein['length'], 
                                domain['domain_accession'], 
                                domain['start'], 
                                domain['end'], 
                                domain['model'], 
                                domain['score']
                            ])
                
                logger.info(f"Data for {ipr_id} saved to {output_file}")
                return True
                
            else:
                logger.warning(f"Failed to retrieve data. Status code: {response.status_code}")
                attempt += 1
                if attempt < max_retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"All {max_retries} attempts failed for {ipr_id}")
                    return False
                    
        except requests.exceptions.RequestException as e:
            attempt += 1
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                logger.error(f"All {max_retries} attempts failed for {ipr_id}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False
    
    return False


def get_true_label_pdb_and_domain(ipr_ids, ipr_names, output_dir, keyword):
    """优化版本的主函数"""
    domain_names = ["REC", "NUC", "Bridge", "RuvC", "HNH", "PI", "BH"] 
    directory = os.path.join(output_dir, "Pfam_domain_reviewed/")
    ensure_dir(directory)

    for i in range(len(ipr_ids)):
        output_file_un = os.path.join(directory, f"uniprot_{ipr_names[i]}_{ipr_ids[i]}.tsv")
        output_file_in = os.path.join(directory, f"interpro_{ipr_names[i]}_{ipr_ids[i]}.tsv")
        
        # Download UniProt data (with retry)
        success_uniprot = download_uniprot_data_pages(ipr_ids[i], output_file_un, max_retries=3, delay=2)
        if not success_uniprot:
            logger.error(f"Failed to download UniProt data for {ipr_ids[i]}")
            continue
        
        # Download InterPro data (with retry)
        success_interpro = download_interpro_data(ipr_ids[i], output_file_in, max_retries=3, delay=2)
        if not success_interpro:
            logger.error(f"Failed to download InterPro data for {ipr_ids[i]}")
            continue
        
        # Brief pause to avoid too frequent requests
        time.sleep(1)
    
    # Parse and merge all UniProt data
    output_file = os.path.join(directory, "uniprot_all_domains.tsv")
    unique_file = os.path.join(directory, "uniprot_unique_domains.tsv")
    # If uniprot and interpro information and domain information are already obtained, no need to parse again
    parse_uniprot_tsv_files(directory, output_file)
    parse_interpro_tsv_files(directory, output_file)
    uniprot_domain_data(output_file, unique_file, keyword,domain_names)
    print("Note: Domain range requires human adjustment!!!")
    print("Get true labels completed.")
    
    
    
def extract_cc_domain_ranges(text):
    domain_names = ["REC", "NUC", "Bridge", "RuvC", "HNH", "PI", "BH"]
    cc_range_pattern = re.compile(r"(\d+-\d+)")
    # Update regex pattern to support extracting multiple ranges
    cc_domain_pattern = re.compile(
        r"(?P<domain>{})(.*?)(?=\b(?:{}|$))".format("|".join(domain_names), "|".join(domain_names))
    )
    if not pd.isna(text):
        extracted_domains = []
        
        # Match domains and their range segments
        domain_matches = cc_domain_pattern.finditer(text)
        
        for domain_match in domain_matches:
            domain_name = domain_match.group('domain')
            domain_content = domain_match.group(0)
            
            # Match all number-number combinations
            ranges = cc_range_pattern.findall(domain_content)
            
            if ranges:
                # Combine number ranges as string and associate with domain name
                range_str = "_".join(ranges)
                extracted_domains.append(f"{domain_name}:{range_str}")
        
        return ";".join(extracted_domains)
    else:
        return ";"
def get_cif_pdb(id_info_file, key,output_dir):
    id_info_df = pd.read_csv(id_info_file)
    pdb_ids = id_info_df['PDB ID'].tolist() 
    protein_ids = id_info_df['UniProt ID'].tolist() 
    
    pdb_dir = os.path.join(output_dir, "pdb_data")
    cif_dir = os.path.join(output_dir, "cif_data")
    header = ["UniProt ID", "PDB ID", "PDB Source"]

    ensure_dir(pdb_dir)
    ensure_dir(cif_dir)

    results = []
    id_i=1
    for id_index in range(len(protein_ids)):
        protein_id = protein_ids[id_index]
        pdb_id = pdb_ids[id_index]
        print(f"Processing {str(id_i)}: {protein_id}...")
        id_i += 1
        
        
        if pd.notna(pdb_id):
            source = download_pdb_structure(pdb_id, protein_id, pdb_dir, cif_dir)
        else:
            source = None  

        
        if not source:
            source = download_alphafold_structure(protein_id, pdb_dir, cif_dir)
        
       
        if not source:
            source = "Not Found"
            print(f"!!!!!!!!Not Found {str(id_i)}: {protein_id}...")
        
       
        results.append({
            "UniProt ID": protein_id,
            "PDB ID": pdb_id,
            "PDB Source": source
        })
        
        # Convert current results to DataFrame
        
def parse_uniprot_tsv_files(directory, output_file):
    all_data = []

    
    for filename in os.listdir(directory):
        if filename.endswith(".tsv") and filename.startswith("uniprot"):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {filename}")

            df = pd.read_csv(file_path, sep='\t')
            df['Domain [CC] Extracted'] = df['Domain [CC]'].apply(extract_cc_domain_ranges)
            df['Domain [FT] Extracted'] = df['Domain [FT]'].apply(extract_ft_residues)
    
            df['File Name'] = filename
       
            all_data.append(df)

    result_df = pd.concat(all_data, ignore_index=True)
    
    result_df.to_csv(output_file, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    

def extract_ft_residues(text):
    residue_pattern_ft = re.compile(r"DOMAIN\s+(\d+)\.\.(\d+);\s*/note=\"(.*?)\"", re.ASCII)
    if pd.isnull(text):
        return ""

    matches = residue_pattern_ft.findall(text)
    if matches:
        extracted_domains = []
        for match in matches:
            start, end, domain = match
            domain = domain.split()[0]  
            extracted_domains.append(f"{domain}:{start}-{end}")
        return ";".join(extracted_domains)
    return ""
   
def uniprot_domain_data(input_file, output_file,keyword,domain_names):
    """
    Process protein ID data, merge domain information under same protein ID, and generate new TSV file.

    Parameters:
    input_file: Input TSV file path
    output_file: Output TSV file path
    """
    
    df = pd.read_csv(input_file, sep='\t')

    
    df['Domain_CC_Combined'] = ''
    df['Domain_FT_Combined'] = ''
    df['Domain_CC_Different'] = ''
    df['Domain_FT_Different'] = ''

    grouped = df.groupby('Entry')
    unique_entries = []
    
    for name, group in grouped:
        entry_name = group.iloc[0]["Entry Name"]
        protein_name = group.iloc[0]["Protein names"]
        if keyword.lower() in str(entry_name).lower() or keyword.lower() in str(protein_name).lower():
            combined_cc, different_cc = combine_domains(group, 'Domain [CC] Extracted', 'File Name',domain_names)
            combined_ft, different_ft = combine_domains(group, 'Domain [FT] Extracted', 'File Name', domain_names)

            
            first_row = group.iloc[0].copy()
            first_row['Domain_CC_Combined'] = combined_cc
            first_row['Domain_FT_Combined'] = combined_ft
            first_row['Domain_CC_Different'] = different_cc
            first_row['Domain_FT_Different'] = different_ft

            unique_entries.append(first_row)

    result_df = pd.DataFrame(unique_entries)
    result_df.to_csv(output_file, sep='\t', index=False)

    print(f"Processing complete. Output saved to {output_file}")
    
def combine_domains(group, domain_col, file_col,domain_names):
    domain_dict = {}
    if group['Entry'].iloc[0] == 'Q99ZW2':
        print("here")
    for idx, row in group.iterrows():
        if pd.isnull(row[domain_col]):
            continue
        else:
            domains = row[domain_col].split(';')
            for domain in domains:
                if ':' in domain:  
                    domain_name, domain_range = domain.split(':')
                    if domain_name in domain_names:
                        if domain_name not in domain_dict:
                            domain_dict[domain_name] = {domain_range: row[file_col]}
                        else:
                            if domain_range not in domain_dict[domain_name]:
                                domain_dict[domain_name][domain_range] = row[file_col]

    combined = []
    different = []
    for domain_name, ranges in domain_dict.items():
        if len(ranges) == 1:
            combined.append(f"{domain_name}:{list(ranges.keys())[0]}")
        else:
            for domain_range, file_name in ranges.items():
                different.append(f"{domain_name}:{domain_range}({file_name})")

    return ';'.join(combined), ';'.join(different)    

def get_uniprot_details(uniprot_ids,save_dir):
    
    os.makedirs(save_dir, exist_ok=True) 
    base_url_unisave = "https://rest.uniprot.org/unisave/"
    for uniprot_id in uniprot_ids:
        txt_url = f"{base_url_unisave}{uniprot_id}?format=txt"
        response = requests.get( txt_url)

        if response.status_code == 200:
            lines = response.text.splitlines()
            file_path = os.path.join(save_dir, f"{uniprot_id}_uniprot_details.txt") 
            with open(file_path, 'w') as file:
                file.write("\n".join(lines))  
            print(f"Saved data for {uniprot_id} to {file_path}")
        else:
            print(f"Error fetching data for UniProt ID {uniprot_id}: {response.status_code}")

def get_pdb_info_from_uniprot(uniprot_ids, uniprot_details_dir,csv_file_path):
    results = [] 

    for uniprot_id in uniprot_ids:
        file_path = os.path.join(uniprot_details_dir, f"{uniprot_id}_uniprot_details.txt")
                
        if not os.path.exists(file_path):
            print(f"File not found for UniProt ID {uniprot_id}: {file_path}")
            results.append((uniprot_id, None, "File not found"))
            continue

        with open(file_path, 'r') as file:
            lines = file.readlines()
            pdb_id = None
            description = None

            for line in lines:
                if "PDB" in line:
                    if "X-ray" in line:
                        pdb_id = line.split(';')[1].strip()  
                        description = line.strip()  
                        breakpoint
                    elif "NMR" in line and not pdb_id:
                        pdb_id = line.split(';')[1].strip()   
                        description = line.strip()  

            if pdb_id:
                results.append((uniprot_id, pdb_id, description))
            else:
                results.append((uniprot_id, None, None))

            df = pd.DataFrame([results[-1]], columns=["UniProt ID", "PDB ID", "Description"])
            df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)

    return results

def parse_interpro_tsv_files(directory, output_file):
    write_header = not os.path.exists(output_file)  # If file exists, don't write header

    # Traverse each file in the folder
    for filename in os.listdir(directory):
        # Check if file starts with "interpro" and ends with ".tsv"
        if filename.endswith(".tsv") and filename.startswith("interpro"):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {filename}")
            # If file is empty, skip and delete file
            if os.path.getsize(file_path) == 0:
                print(f"Skipping and deleting empty file: {filename}")
                os.remove(file_path)
                continue
            # Read TSV file
            df = pd.read_csv(file_path, sep='\t')
            domain_str = file_path.split('/')[-1].split('_')[2]
            # Generate 'Domain [CC] Extracted' column, format as 'Domain Name':'Domain Start-End'
            df['Domain [CC] Extracted'] = df.apply(lambda row: f"{domain_str}:{row['Start']}-{row['End']}", axis=1)
            
            # Create new DataFrame matching required format
            result_df = pd.DataFrame({
                'Entry': df['Accession'],  # Corresponds to Entry
                'Entry Name': df['Protein Name'],  # Corresponds to Entry Name
                'Protein names': '',  # Empty column
                'Domain [CC]': '',  # Empty column
                'Domain [FT]': '',  # Empty column
                'Organism': '',  # Empty column
                'Length': '',  # Empty column
                'Domain [CC] Extracted': df['Domain [CC] Extracted'],  # Extracted domain information
                'Domain [FT] Extracted': '',  # Empty column
                'File Name': filename  # Filename
            })
            
            # Append data to output file, set mode='a' and header=write_header
            result_df.to_csv(output_file, sep='\t', index=False, mode='a', header=write_header, quoting=csv.QUOTE_NONE, escapechar='\\')
            
            # Don't write header afterwards
            write_header = False
def ensure_file(file_path):
    """Ensure file exists, create empty file if it doesn't exist"""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass  # Create an empty file
        print(f"File created: {file_path}")
    else:
        print(f"File already exists: {file_path}")
        
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
           

# step1 function


def fetch_interpro_description(interpro_id):
    """
    Get description information based on InterPro ID
    """
    try:
        url = f"https://www.ebi.ac.uk/interpro/api/entry/interpro/{interpro_id}"
        headers = {"Accept": "application/json"}
        r = session.get(url, headers=headers, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            meta = data.get('metadata', {})
            description = data.get('metadata', {}).get('name', '')
            return {
                'name': meta.get('name', ''),
                'description': meta.get('description', ''),
            }
    except Exception:
        return ""


def get_latest_uniprot_version_by_scraping(uniprot_id):
    """
    Scrape UniProt history page, parse all version numbers, return the largest one.
    
    """
    url = f"https://rest.uniprot.org/unisave/{uniprot_id}?format=json"
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()  # If request returns error status code, will throw HTTPError
        data = r.json()
        # Extract latest version number
        latest_version = max(entry['entryVersion'] for entry in data['results'])
        return latest_version

    except (requests.RequestException, ValueError, KeyError) as e:
        # Catch request exceptions, JSON parsing errors, or missing 'results' field issues
        print(f"Error fetching or parsing data for {uniprot_id}: {e}")
        return None  # If error occurs, return None
def fetch_details_for_uniprot_id(uniprot_id, pdb_id=None, max_versions=20):
    """
    Only responsible for getting UniProt original TXT, no subsequent processing
    """

    # Method 1: REST TXT
    txt_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}?format=txt"
    try:
        r = session.get(txt_url, timeout=10)
        if r.status_code == 200 and r.text.strip():
            txt= r.text
            txt = txt_url + "\n"+ txt
            return txt
    except requests.RequestException:
        pass
    time.sleep(1)
    # Method 2: First call method 1 to get latest version
    latest_version = get_latest_uniprot_version_by_scraping(uniprot_id)
    time.sleep(1)
    if latest_version:
        # Try to download latest version
        url = f"https://rest.uniprot.org/unisave/{uniprot_id}?format=txt&versions={latest_version}"
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200 and r.text.strip():
                txt= r.text
                txt = str(latest_version)+"\n"+url + "\n"+ txt
                return txt
            else:
                print(f"[Info] Version {latest_version} download failed, status code: {r.status_code}")
        except requests.RequestException as e:
            print(f"[Warning] {uniprot_id} error downloading version {latest_version}: {e}")

    # Method 2: Historical version fallback
    for version in range(latest_version, 0, -1):
        time.sleep(1)
        url = f"https://rest.uniprot.org/unisave/{uniprot_id}?format=txt&versions={version}"
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200 and r.text.strip():
                txt= r.text
                txt = str(version)+"\n"+url + "\n"+ txt
                return txt
        except requests.RequestException:
            continue

    raise RuntimeError(f"failed Unable to get any data for UniProt ID {uniprot_id}")
import glob     

def merge_predict_details(csv_path: str) -> str:
    """
    For specified main CSV file, find other CSVs in same directory with same prefix but numbered suffix,
    fill empty details in main table with non-empty details from them, and save as new file.

    Parameters:
        csv_path: Complete path of main CSV file

    Returns:
        Path of new CSV file
    """
    # 1. Read main table
    df_main = pd.read_csv(csv_path, dtype=str)
    if 'pdb_id' not in df_main.columns or 'details' not in df_main.columns:
        raise ValueError("Input CSV must contain 'pdb_id' and 'details' columns")
    df_main.set_index('pdb_id', inplace=True)

    # 2. Construct wildcard pattern in same directory, exclude self
    dirpath, filename = os.path.split(csv_path)
    stem, ext = os.path.splitext(filename)
    # Match files starting with main file prefix, followed by any characters then ending with .csv
    pattern = os.path.join(dirpath, f"{stem}*.csv")
    candidates = glob.glob(pattern)
    # Don't include main file itself as candidate
    other_files = [f for f in candidates if os.path.abspath(f) != os.path.abspath(csv_path)]

    # 3. Read other files one by one, merge details
    for other in sorted(other_files):
        df_i = pd.read_csv(other, dtype=str)
        if 'pdb_id' not in df_i.columns or 'details' not in df_i.columns:
            continue  # If columns missing, skip
        df_i.set_index('pdb_id', inplace=True)
        # Use non-empty details in df_i to fill main table
        # fillna will automatically align by index
        df_main['details'] = df_main['details'].fillna(df_i['details'])

    # 4. Save as new CSV, filename with "_all"
    new_stem = stem + '_all'
    new_filename = new_stem + ext
    new_path = os.path.join(dirpath, new_filename)
    # Write index pdb_id back to column
    df_main.reset_index().to_csv(new_path, index=False)
    print(f"Merged file saved to: {new_path}")
    return new_path

def fetch_details_for_uniprot(csv2_path, output_csv3,
                              pdb_id=None, max_workers=50,
                              batch_size=500, checkpoint_interval=1000):
    """
    High concurrency version: Read CSV, parallel fetch for each row with empty details UniProt ID,
    and extract InterPro information and concatenate after getting TXT.
    """
    # Global cache to avoid repeated InterPro requests
    interpro_cache = {}

    def fetch_interpro_description_cached(ipr):
        if ipr not in interpro_cache:
            interpro_cache[ipr] = fetch_interpro_description(ipr)
        return interpro_cache[ipr]

    # Load data, only process rows with empty details
    df0 = pd.read_csv(csv2_path)
    # If no details column, select all rows; otherwise only select rows with empty details
    if 'details' in df0.columns:
        mask = df0['details'].isna() | (df0['details'] == '')
    else:
        # All True, meaning all rows need processing
        mask = pd.Series(True, index=df0.index)
    df = df0[mask].copy()
    total = len(df)

    # checkpoint file
    checkpoint_file = f"{output_csv3}.checkpoint.json"
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file) as f:
                processed = json.load(f)
            print(f"Loaded checkpoint, {len(processed)} entries done")
        except:
            processed = {}
    else:
        processed = {}

    # Prepare task list
    tasks = []
    for idx, row in df.iterrows():
        if str(idx) not in processed:
            try:
                # target format "xxx-<UniProtID>"
                up = str(row['target']).split('-')[1]
                tasks.append((idx, up))
            except:
                processed[str(idx)] = ""  # Exception leaves empty

    # Adjust concurrency based on task count
    workers = min(max_workers, max(1, len(tasks)//2))
    print(f"To process {len(tasks)} rows, using {workers} workers")

    lock = threading.Lock()
    processed_count = 0
    error_count = 0
    start = time.time()

    def save_ckpt():
        with open(checkpoint_file, 'w') as f:
            json.dump(processed, f)

    def process_task(item):
        nonlocal processed_count, error_count
        idx, upid = item
        try:
            txt = fetch_details_for_uniprot_id(upid, pdb_id)

            # Extract InterPro ID list, deduplicate
            iprs = set(re.findall(r"IPR\d{6}", txt))
            infos = []
            for ipr in iprs:
                name = fetch_interpro_description_cached(ipr)
                if name:
                    infos.append(f"{ipr}: {name}")

            # Concatenate result
            if infos:
                txt += "\n\n=== InterPro Information ===\n" + "\n".join(infos)

            with lock:
                processed[str(idx)] = txt
                processed_count += 1
                if processed_count % checkpoint_interval == 0:
                    save_ckpt()
        except Exception as e:
            with lock:
                processed[str(idx)] = ""
                processed_count += 1
                error_count += 1
        return idx


    # Initialize progress bar, total is all rows to process
    pbar = tqdm(total=len(tasks), desc="Overall progress")
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        # Parallel submit this batch
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_task, t) for t in batch]
            # Wait for this batch to complete, but don't update progress bar here
            for fut in as_completed(futures):
                fut.result()  # Can capture exception or get return value if needed

        # After entire batch done, update pbar once
        pbar.update(len(batch))
        # Save checkpoint after each batch
        save_ckpt()

    pbar.close()
            
    """       
    # Run batches in parallel
    pbar = tqdm(total=len(tasks), desc="Overall progress")
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(process_task, t) for t in batch]):
                pbar.update(1)
        save_ckpt()
    pbar.close()
    """ 
    # Write back to CSV
    df['details'] = [processed.get(str(idx), "") for idx in df.index]
    df0.loc[mask, 'details'] = df['details']
    df0.to_csv(output_csv3, index=False)

    # Clean up checkpoint
    if error_count == 0 and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    elapsed = time.time() - start
    print(f"Completed: {processed_count} entries, failed {error_count} entries, time taken {elapsed:.1f} seconds, average {(elapsed/processed_count):.2f} seconds/entry")


def append_ref_ted_predition(input_dir: str, csv_ref_path: str, output_dir: str = None, suffix: str = '_with_ref') -> None:
    """
    Traverse all Excel and CSV files in specified folder, merge each file's DataFrame with reference CSV by specified key (pdb_id vs unique_ID),
    and append selected reference columns to original DataFrame, then save as new file.

    Parameters:
    - input_dir: Folder path containing files to process.
    - csv_ref_path: Reference CSV file path, containing unique_ID column and fields to append.
    - output_dir: Optional, folder to save result files. If None, create subfolder 'output' under input_dir.
    - suffix: Optional, suffix to append to filename when saving (default '_with_ref').

    Dependencies:
    pandas
    """
    # Read reference data
    ref_df = pd.read_csv(csv_ref_path)
    print(csv_ref_path)
    print(ref_df)
    # Only keep columns to append (except unique_ID)
    join_key = 'unique_ID'

    print('1')
    ref_cols = [c for c in ref_df.columns if c != join_key]
    print(ref_cols)
    print('2')   
    # Prepare output directory
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    print('3')
    # Traverse all files in folder
    for fname in os.listdir(input_dir):
        file_path = os.path.join(input_dir, fname)
        base, ext = os.path.splitext(fname)
        ext = ext.lower()
        print('4')
        if ext in ['.xlsx', '.xls']:
            # Process Excel file
            excel = pd.ExcelFile(file_path)
            
            
            writer = pd.ExcelWriter(
                os.path.join(output_dir, f"{base}{suffix}.xlsx"),
                engine='openpyxl'
            )
            for sheet in excel.sheet_names:
                df = pd.read_excel(excel, sheet_name=sheet, dtype=str)
                # Rename df key column for merging
                print('5')
                df = df.rename(columns={'pdb_id': join_key})
                print(sheet)
                print(df)
                # Merge
                merged = df.merge(
                    ref_df[[join_key] + ref_cols],
                    on=join_key,
                    how='left'
                )
                print('6')
                # Restore column names
                merged = merged.rename(columns={join_key: 'pdb_id'})
                # Write
                merged.to_excel(writer, sheet_name=sheet, index=False)
            writer.close()

        elif ext == '.csv':
            # Process CSV file
            df = pd.read_csv(file_path, dtype=str)
            df = df.rename(columns={'pdb_id': join_key})
            merged = df.merge(
                ref_df[[join_key] + ref_cols],
                on=join_key,
                how='left'
            )
            merged = merged.rename(columns={join_key: 'pdb_id'})
            # Save CSV
            merged.to_csv(
                os.path.join(output_dir, f"{base}{suffix}.csv"),
                index=False
            )
        else:
            # Skip other file types
            continue
def sequence_to_vector(seq, k=3):
    """Convert protein sequence to k-mer frequency vector"""
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    seq_counter = Counter(kmers)
    vector = np.zeros(len(amino_acids) ** k)
    kmers_set = [''.join(x) for x in itertools.product(amino_acids, repeat=k)]
    for i, kmer in enumerate(kmers_set):
        vector[i] = seq_counter.get(kmer, 0)
    return vector

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

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
        # Setting gap_open=-10, gap_extend=-0.5 are common parameters
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
        
        # Normalized score (take smaller of two self-alignment scores for normalization)
        normalized_score = alignment_score / min(self_score1, self_score2)
        
        # Ensure score in 0-1 range
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        return round(normalized_score, 4)
        
    except Exception as e:
        print(f"Error calculating sequence similarity: {e}")
        return 0.0

def convert_cif_to_dssp_parallel(cif_dir, dssp_dir, max_workers=None):
    """
    Convert all CIF files in directory to DSSP files using multi-process parallel processing.

    Args:
        cif_dir (str): CIF file directory path.
        dssp_dir (str): DSSP output directory path.
        max_workers (int, optional): Maximum parallel processes, default None means automatically select appropriate number.
    """
    cif_path = Path(cif_dir)
    dssp_path = Path(dssp_dir)

    # Check if paths exist
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF directory does not exist: {cif_path}")
    if not dssp_path.exists():
        dssp_path.mkdir(parents=True, exist_ok=True)  # Create output directory

    # Get all CIF files
    cif_files = list(cif_path.glob("*.pdb"))
    print(max_workers)
    # Use ProcessPoolExecutor for multi-process parallel processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all conversion tasks
        futures = []
        for cif_file in cif_files:
            base_name = cif_file.stem  # Extract filename (without extension)
            output_dssp = dssp_path / f"{base_name}.dssp"
            futures.append(executor.submit(dssp_single, cif_file, output_dssp))

        # Wait for tasks to complete and get results
        for future in as_completed(futures):
            future.result()  # Get each task's execution result, throw exception if failed

def dssp_single(cif_file, output_dssp):
    """
    Process single file conversion, run mkdssp command.

    Args:
        cif_file (Path): Input CIF file path.
        output_dssp (Path): Output DSSP file path.
    """
    try:
        subprocess.run(
            ["mkdssp", "-i", str(cif_file), "-o", str(output_dssp)],
            check=True
        )
        #print(f"Converted: {cif_file} -> {output_dssp}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to convert {cif_file}: {e}")


def convert_cif_to_dssp(cif_dir, dssp_dir):
    """
    Convert all CIF files in directory to DSSP files.
    
    Args:
        cif_dir (str): CIF file directory path.
        dssp_dir (str): DSSP output directory path.
    """
    cif_path = Path(cif_dir)
    dssp_path = Path(dssp_dir)

    # Check if paths exist
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF directory does not exist: {cif_path}")
    if not dssp_path.exists():
        dssp_path.mkdir(parents=True, exist_ok=True)  # Create output directory

    # Traverse CIF files and convert
    for cif_file in cif_path.glob("*.pdb"):
        base_name = cif_file.stem  # Extract filename (without extension)
        output_dssp = dssp_path / f"{base_name}.dssp"

        # Run mkdssp command
        try:
            subprocess.run(
                ["mkdssp", "-i", str(cif_file), "-o", str(output_dssp)],
                check=True
            )
            #print(f"Converted: {cif_file} -> {output_dssp}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to convert {cif_file}: {e}")
            

def copy_file_to_OSS(source_path, destination_path):
    """
    Copy a file or folder from source to destination.

    Args:
        source_path (str): The source file or folder path.
        destination_path (str): The destination path (can be a directory or file path).
    
    Returns:
        None
    """
    # Check if the source path exists
    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' does not exist.")
        return

    # If destination is a directory, move source to that directory
    if os.path.isdir(destination_path):
        # If it's a directory, append the source filename/folder to the destination
        destination_path = os.path.join(destination_path, os.path.basename(source_path))

    try:
        # Check if the source path is a file or directory
        if os.path.isdir(source_path):
            # If it's a directory, copy the entire directory
            shutil.copytree(source_path, destination_path)
            print(f"Successfully copied directory '{source_path}' to '{destination_path}'.")
        else:
            # If it's a file, copy the file
            shutil.copy2(source_path, destination_path)  # copy2 to preserve metadata
            print(f"Successfully copied file '{source_path}' to '{destination_path}'.")
    except Exception as e:
        print(f"Error copying '{source_path}' to '{destination_path}': {e}")

def move_file_to_OSS(source_path, destination_path):
    """
    Move a file or folder from source to destination.

    Args:
        source_path (str): The source file or folder path.
        destination_path (str): The destination path (can be a directory or file path).
    
    Returns:
        None
    """
    # Check if the source path exists
    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' does not exist.")
        return

    # If destination is a directory, move source to that directory
    if os.path.isdir(destination_path):
        # If it's a directory, append the source filename/folder to the destination
        destination_path = os.path.join(destination_path, os.path.basename(source_path))
    
    try:
        # Move the file or folder
        shutil.move(source_path, destination_path)
        print(f"Successfully moved '{source_path}' to '{destination_path}'.")
    except Exception as e:
        print(f"Error moving '{source_path}' to '{destination_path}': {e}")
        
def replace_subsequence(A, start, end, a):
    # If A or a is empty or None, return empty string
    if not A or not a:
        print("A or a is empty, return empty")
        return ""
    
    # Ensure start and end ranges are valid
    if start < 0 or end > len(A) or start >= end:
        raise ValueError("Invalid start or end position.")
    
    # Split A into three parts: part before replacement, part to replace, and part after replacement
    before = A[:start]
    after = A[end:]
    
    # Generate new sequence B, start to end range replaced by a
    B = before + a + after
    
    return B


def add_domain_info_to_target_cath_teddb_check(csv1_path, csv2_path, fasta_dict, output_path):
    # 1. Read csv1 and csv2 files
    csv1 = pd.read_csv(csv1_path)
    csv2  = pd.read_excel(csv2_path)
    csv2  = csv2.dropna(subset=['chopping_check'])
    csv2_unique = csv2.drop_duplicates(subset='target', keep='first')
    csv2_dict = csv2_unique.set_index('target')[['chopping', 'chopping_check', 'plddt', 'cath_label']].to_dict(orient='index')
    
    
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
        # Extract uniprot_id from target (assume format target-xxxx, take part after '-')
        target = row['target']
        source = row['source']
        target_id = target.split('-')[1]
        source_id = source.split('_')[0]
        # 1) Find corresponding ted_id information
        ted_id = "_".join(target.split('_')[:3])
        # Find rows with related ted_id in csv2
        if ted_id in csv2_dict:
            chopping = csv2_dict[ted_id]['chopping']
            chopping_check = csv2_dict[ted_id]['chopping_check']
            plddt = csv2_dict[ted_id]['plddt']
            cath_label = csv2_dict[ted_id]['cath_label']


            
        else:
            chopping = plddt = cath_label = chopping_check = None  # If ted_id not found, set to None

        # 2) Get sequence length for UniProt ID
        target_seq = uniprot_seq.get(target_id, None)
        if len(target_seq)>0:
            start_res, end_res = int(chopping_check.split('-')[0]),int(chopping_check.split('-')[1])
            target_domain_seq  = target_seq[start_res-1:end_res]
        else:
            target_domain_seq = []
            
        # 2) Get sequence length for UniProt ID
        source_seq = uniprot_seq.get(source_id, None)
        if len(source_seq)>0:
            start_res, end_res =  int(source.split('_')[-2].split('-')[0]),int(source.split('_')[-2].split('-')[1])
            source_domain_seq  = source_seq[start_res-1:end_res]
        else:
            source_domain_seq = []
        target_len = uniprot_to_len.get(target_id, None)  # If ID not found, return None
        target_info = uniprot_info.get(target_id, None)
        #if target_len is None:
        #    print(uniprot_id)
        protein_seq_sim  = calculate_protein_sequence_similarity(target_seq, source_seq)
        domain_seq_sim =  calculate_protein_sequence_similarity(target_domain_seq, source_domain_seq)
        assemble_seq =  replace_subsequence(source_seq, start_res, end_res, target_domain_seq)
        assemle_protein_sim = calculate_protein_sequence_similarity(assemble_seq, source_seq)
        return pd.Series([chopping,chopping_check, plddt, cath_label, target_len,target_info,source_domain_seq,target_domain_seq,protein_seq_sim, domain_seq_sim,assemble_seq,assemle_protein_sim,target_seq,source_seq ], index=['chopping', 'chopping_check','plddt', 'cath_label', 'target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq'])
    
    # 5. Apply get_additional_info function to each row in csv1

    csv1[['chopping', 'chopping_check','plddt', 'cath_label', 'target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq']] = csv1.apply(get_additional_info, axis=1)
    
    
    columns_order = ['source','target', 'FS_weight','TM_weight','Dali_weight','target_info','chopping_check', 'target_len', 'plddt', 'cath_label','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim'] + [col for col in csv1.columns if col not in['source','target', 'FS_weight','TM_weight','Dali_weight','target_info','chopping_check', 'target_len', 'plddt', 'cath_label','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim']]

    # Adjust column order
    csv1 = csv1[columns_order]
    
    # 6. Save merged csv1
    csv1.to_csv(output_path, index=False)
    print(f"Output file saved to: {output_path}")


def add_chopping_check_to_csv(csv_file, excel_file, sheet_name, output_csv):
    # Read CSV file
    df_csv = pd.read_csv(csv_file)
    
    # Read specified Sheet in Excel file
    df_excel = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # Process target column in Excel, split by "_" remove last part
    df_excel['target'] = df_excel['target'].str.split('_').str[:-1].str.join('_')

    # Before merging, directly rename columns, add _check suffix to FS_weight, TM_weight, Dali_weight
    df_excel = df_excel.rename(columns={
        'FS_weight': 'FS_weight_check',
        'TM_weight': 'TM_weight_check',
        'Dali_weight': 'Dali_weight_check'
    })

    # Merge by source and target columns, ensure only fill corresponding columns when both match
    df_merged = pd.merge(df_csv, df_excel[['source', 'target', 'chopping_check', 'FS_weight_check', 'TM_weight_check', 'Dali_weight_check']],
                         on=['source', 'target'], how='left')

    # Move needed columns to front
    cols_order = ['source', 'target', 'chopping', 'chopping_check', 'FS_weight_check', 'TM_weight_check', 'Dali_weight_check'] + [col for col in df_merged.columns if col not in ['source', 'target', 'chopping', 'chopping_check', 'FS_weight_check', 'TM_weight_check', 'Dali_weight_check']]
    df_merged = df_merged[cols_order]

    # Save merged data as new CSV file
    df_merged.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"Data successfully updated and saved as {output_csv}")
from tqdm.auto import tqdm
tqdm.pandas()  # Register progress bar for pandas

def process_fs_reslult_new(input_csv, output_csv):
    # Read CSV file
    df = pd.read_csv(input_csv)
    
    # Rename columns
    df.rename(columns={'weight': 'FS_weight', 'alntmscore': 'TM_weight'}, inplace=True)
    
    # Convert FS_weight and TM_weight columns to numeric, keep four decimal places
    df['FS_weight'] = pd.to_numeric(df['FS_weight'], errors='coerce').round(4)
    df['TM_weight'] = pd.to_numeric(df['TM_weight'], errors='coerce').round(4)
    
    
    # Rearrange column order, move source, target, FS_weight and TM_weight to front
    cols = ['source', 'target', 'FS_weight', 'TM_weight'] + [col for col in df.columns if col not in ['source', 'target', 'FS_weight', 'TM_weight']]
    df = df[cols]
    
    # Filter rows with FS_weight >= 0.7 and TM_weight >= 0.5
    #df_filtered = df[(df['FS_weight'] >= 0.7) & (df['TM_weight'] >= 0.5)]
    df =  df[df['source'].str.contains('Q99ZW2', na=False)]
    df_filtered = df[(df['FS_weight'] >= 0) & (df['TM_weight'] >= 0)]
    
    # Save filtered data as new CSV file
    df_filtered.to_csv(output_csv, index=False)


def add_domain_info_to_target_cath_teddb(csv1_path, csv2_path, fasta_dict, output_path):
    # 1. Read csv1 and csv2 files
    csv1 = pd.read_csv(csv1_path)
    csv2 = pd.read_csv(csv2_path)
    
    # 2. Create a dictionary, index csv2's ted_id as key, chopping, plddt, cath_label as values
    columns_to_extract = ['chopping', 'plddt', 'cath_label', 'Cluster_representative']
    csv2_dict = {
        row['ted_id']: {
            col: row[col] if col in row else None 
            for col in columns_to_extract
        }
        for _, row in csv2.iterrows()
    }
    
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
        # Extract uniprot_id from target (assume format target-xxxx, take part after '-')
        target = row['target']
        source = row['source']
        target_id = target.split('-')[1]
        source_id = source.split('_')[0]
        
        # 1) Find corresponding ted_id information
        ted_id = "_".join(target.split('_')[:3])
        #print(f'cathdb:{ted_id}')
        # Find rows with related ted_id in csv2
        if ted_id in csv2_dict:
            chopping = csv2_dict[ted_id]['chopping']
            plddt = csv2_dict[ted_id]['plddt']
            cath_label = csv2_dict[ted_id]['cath_label']
            Cluster_representative = csv2_dict[ted_id]['Cluster_representative']
        else:
            Cluster_representative = chopping = plddt = cath_label  = None  # If ted_id not found, set to None

        
        # 2) Get sequence length for UniProt ID
        target_seq = uniprot_seq.get(target_id, None)
        if len(target_seq)>0:
            target_domain_seq = []
            # Parse chopping information
            ranges = []
            for segment in chopping.split('_'):
                start, end = map(int, segment.split('-'))
                ranges.append((start, end))
            for start_res, end_res in ranges:
                target_domain_seq +=  target_seq[start_res-1:end_res]
        else:
            target_domain_seq = []
        target_domain_seq = ''.join(target_domain_seq)

        
        # 2) Get sequence length for UniProt ID
        source_seq = uniprot_seq.get(source_id, None)
        if len(source_seq)>0:
            start_res, end_res =  int(source.split('_')[-2].split('-')[0]),int(source.split('_')[-2].split('-')[1])
            source_domain_seq  = source_seq[start_res-1:end_res]
        else:
            source_domain_seq = []
        target_len = uniprot_to_len.get(target_id, None)  # If ID not found, return None
        target_info = uniprot_info.get(target_id, None)
        #if target_len is None:
        #    print(uniprot_id)
        protein_seq_sim  = calculate_protein_sequence_similarity(target_seq, source_seq)
        domain_seq_sim =  calculate_protein_sequence_similarity(target_domain_seq, source_domain_seq)
        assemble_seq =  replace_subsequence(source_seq, start_res, end_res, target_domain_seq)
        assemle_protein_sim = calculate_protein_sequence_similarity(assemble_seq, source_seq)
        return pd.Series([chopping,plddt, cath_label, Cluster_representative, target_len,target_info,source_domain_seq,target_domain_seq,protein_seq_sim, domain_seq_sim,assemble_seq,assemle_protein_sim,target_seq,source_seq ], index=['chopping', 'plddt', 'cath_label','Cluster_representative', 'target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq'])
        



        
    # 5. Apply get_additional_info function to each row in csv1
    csv1[['chopping', 'plddt', 'cath_label', 'Cluster_representative','target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq']] = csv1.progress_apply(get_additional_info, axis=1)
    #csv1.apply(get_additional_info, axis=1)
    
    
    # Define first few column names
    first_columns=['source','target', 'FS_weight','TM_weight','target_info','chopping', 'source_domain_seq','target_domain_seq','target_len', 'plddt', 'cath_label','Cluster_representative','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq']
    # Get remaining column names
    remaining_columns = [col for col in csv1.columns if col not in first_columns]

    # Arrange columns in required order
    final_columns = first_columns + remaining_columns
    

    # Adjust column order
    csv1 = csv1[final_columns]
    
    # 6. Save merged csv1
    csv1.to_csv(output_path, index=False)
    print(f"Output file saved to: {output_path}")
    

def add_domain_info_to_target_cluster_teddb(csv1_path, csv2_path, fasta_dict, output_path):
    # 1. Read csv1 and csv2 files
    csv1 = pd.read_csv(csv1_path)
    csv2 = pd.read_csv(csv2_path)

    # 2. Create a dictionary, index csv2's ted_id as key, chopping, plddt, cath_label as values
    columns_to_extract = ['chopping', 'plddt', 'cath_label', 'Cluster_representative']
    csv2_dict = {
        row['ted_id']: {
            col: row[col] if col in row else None 
            for col in columns_to_extract
        }
        for _, row in csv2.iterrows()
    }
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
        # Extract uniprot_id from target (assume format target-xxxx, take part after '-')
        target = row['target']
        source = row['source']
        target_id = target.split('-')[1]
        source_id = source.split('_')[0]
        
        # 1) Find corresponding ted_id information
        ted_id = "_".join(target.split('_')[:3])
        # Find rows with related ted_id in csv2
        if ted_id in csv2_dict:
            chopping = csv2_dict[ted_id]['chopping']
            plddt = csv2_dict[ted_id]['plddt']
            cath_label = csv2_dict[ted_id]['cath_label']
            Cluster_representative = csv2_dict[ted_id]['Cluster_representative']
        else:
            Cluster_representative=chopping = plddt = cath_label = None  # If ted_id not found, set to None

        # 2) Get sequence length for UniProt ID
        target_seq = uniprot_seq.get(target_id, None)
        if target_seq is None or len(target_seq) == 0:  # First check if None
            print(ted_id)
        
        if len(target_seq)>0:
            target_domain_seq = []
            # Parse chopping information
            ranges = []
            for segment in chopping.split('_'):
                start, end = map(int, segment.split('-'))
                ranges.append((start, end))
            for start_res, end_res in ranges:
                target_domain_seq +=  target_seq[start_res-1:end_res]
        else:
            target_domain_seq = []
        target_domain_seq = ''.join(target_domain_seq)

        
        # 2) Get sequence length for UniProt ID
        source_seq = uniprot_seq.get(source_id, None)
        if len(source_seq)>0:
            start_res, end_res =  int(source.split('_')[-2].split('-')[0]),int(source.split('_')[-2].split('-')[1])
            source_domain_seq  = source_seq[start_res-1:end_res]
        else:
            source_domain_seq = []
        target_len = uniprot_to_len.get(target_id, None)  # If ID not found, return None
        target_info = uniprot_info.get(target_id, None)
        #if target_len is None:
        #    print(uniprot_id)
        protein_seq_sim  = 0
        domain_seq_sim =  0
        assemble_seq =  0
        assemle_protein_sim = 0
        """
        protein_seq_sim  = calculate_protein_sequence_similarity(target_seq, source_seq)
        domain_seq_sim =  calculate_protein_sequence_similarity(target_domain_seq, source_domain_seq)
        assemble_seq =  replace_subsequence(source_seq, start_res, end_res, target_domain_seq)
        assemle_protein_sim = calculate_protein_sequence_similarity(assemble_seq, source_seq)
        """
        return pd.Series([chopping,plddt,cath_label, Cluster_representative, target_len,target_info,source_domain_seq,target_domain_seq,protein_seq_sim, domain_seq_sim,assemble_seq,assemle_protein_sim,target_seq,source_seq ], index=['chopping', 'plddt', 'cath_label','Cluster_representative', 'target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq'])
    
        
    #csv1.apply(get_additional_info, axis=1)
    
    # 5. Apply get_additional_info function to each row in csv1
    csv1[['chopping', 'plddt', 'cath_label','Cluster_representative', 'target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq']] = csv1.apply(get_additional_info, axis=1)

    # Define first few column names
    first_columns=['source','target', 'FS_weight','TM_weight','target_info','chopping', 'source_domain_seq','target_domain_seq','target_len', 'plddt', 'cath_label','Cluster_representative','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq']
    # Get remaining column names
    remaining_columns = [col for col in csv1.columns if col not in first_columns]
    # Arrange columns in required order
    final_columns = first_columns + remaining_columns
    # Adjust column order
    csv1 = csv1[final_columns]
    
    # 6. Save merged csv1
    csv1.to_csv(output_path, index=False)
    print(f"Output file saved to: {output_path}")
    
    
    
def add_domain_info_to_target_cluster_teddb_check(csv1_path, csv2_path, fasta_dict, output_path):
    # 1. Read csv1 and csv2 files
    csv1 = pd.read_csv(csv1_path)
    csv2  = pd.read_excel(csv2_path)
    csv2  = csv2.dropna(subset=['chopping_check'])
    csv2_unique = csv2.drop_duplicates(subset='target', keep='first')
    csv2_dict = csv2_unique.set_index('target')[['chopping', 'chopping_check', 'plddt', 'cath_label','Cluster_representative']].to_dict(orient='index')
    
    
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
        # Extract uniprot_id from target (assume format target-xxxx, take part after '-')
        target = row['target']
        source = row['source']
        target_id = target.split('-')[1]
        source_id = source.split('_')[0]
        
        # 1) Find corresponding ted_id information
        ted_id = "_".join(target.split('_')[:3])
        print(ted_id)
        # Find rows with related ted_id in csv2
        if ted_id in csv2_dict:
            chopping = csv2_dict[ted_id]['chopping']
            chopping_check = csv2_dict[ted_id]['chopping_check']
            plddt = csv2_dict[ted_id]['plddt']
            cath_label = csv2_dict[ted_id]['cath_label']
            Cluster_representative = csv2_dict[ted_id]['Cluster_representative']
        else:
            chopping = plddt = cath_label = chopping_check = None  # If ted_id not found, set to None

        # 2) Get sequence length for UniProt ID
        target_seq = uniprot_seq.get(target_id, None)
        if len(target_seq)>0:
            start_res, end_res = int(chopping_check.split('-')[0]),int(chopping_check.split('-')[1])
            target_domain_seq  = target_seq[start_res-1:end_res]
        else:
            target_domain_seq = []
            
        # 2) Get sequence length for UniProt ID
        source_seq = uniprot_seq.get(source_id, None)
        if len(source_seq)>0:
            start_res, end_res =  int(source.split('_')[-2].split('-')[0]),int(source.split('_')[-2].split('-')[1])
            source_domain_seq  = source_seq[start_res-1:end_res]
        else:
            source_domain_seq = []
        target_len = uniprot_to_len.get(target_id, None)  # If ID not found, return None
        target_info = uniprot_info.get(target_id, None)
        #if target_len is None:
        #    print(uniprot_id)
        protein_seq_sim  = calculate_protein_sequence_similarity(target_seq, source_seq)
        domain_seq_sim =  calculate_protein_sequence_similarity(target_domain_seq, source_domain_seq)
        assemble_seq =  replace_subsequence(source_seq, start_res, end_res, target_domain_seq)
        assemle_protein_sim = calculate_protein_sequence_similarity(assemble_seq, source_seq)
        
        return pd.Series([chopping, chopping_check,plddt, cath_label, Cluster_representative,target_len,target_info,source_domain_seq,target_domain_seq,protein_seq_sim, domain_seq_sim,assemble_seq,assemle_protein_sim,target_seq,source_seq], index=['chopping', 'chopping_check','plddt', 'cath_label', 'Cluster_representative', 'target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq'])
    
    # 5. Apply get_additional_info function to each row in csv1
    csv1[['chopping', 'chopping_check','plddt', 'cath_label', 'Cluster_representative','target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq']] =  csv1.apply(get_additional_info, axis=1)
    
    
    columns_order = ['source','target', 'FS_weight','TM_weight','Dali_weight','target_info','chopping_check', 'target_len', 'plddt', 'cath_label','Cluster_representative','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim'] + [col for col in csv1.columns if col not in['source','target', 'FS_weight','TM_weight','Dali_weight','target_info','chopping_check', 'target_len', 'plddt', 'cath_label','Cluster_representative','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim']]

    # Adjust column order
    csv1 = csv1[columns_order]
    
    # 6. Save merged csv1
    csv1.to_csv(output_path, index=False)
    print(f"Output file saved to: {output_path}")

# Preload FASTA file and store in dictionary
def load_fasta_to_dict(fasta_file):
    fasta_dict = {}  # Store first occurring sequence
    with open(fasta_file, "r") as fasta_in:
        for record in SeqIO.parse(fasta_in, "fasta"):
            record_id = record.id.split('|')[1]  # Extract second part ID
            if record_id not in fasta_dict:  # Only keep first occurrence
                fasta_dict[record_id] = record
    return fasta_dict  # Return {id: fasta_record} dictionary

def add_domain_info_to_target_cathdb(csv1_path, csv2_path, fasta_dict, output_path):
    # 1. Read csv1 and csv2 files
    csv1 = pd.read_csv(csv1_path)
    csv2 = pd.read_csv(csv2_path)
    csv2 = csv2.drop_duplicates(subset='domain_id')
    # 2. Create a dictionary, index csv2's ted_id as key, chopping, plddt, cath_label as values
    csv2_dict = csv2.set_index('domain_id')[['uniprot_id',  'cath_label' ]].to_dict(orient='index')
    
    # 3. Read fasta file once and extract UniProt ID and sequence length
    #with open(fasta_path, "r") as fasta_in:
    #    fasta_records = list(SeqIO.parse(fasta_in, "fasta"))
    fasta_records = fasta_dict
    # Convert FASTA records to dictionary, key as UniProt ID, value as sequence length
    uniprot_to_len = {}
    uniprot_info = {}
    for key in fasta_records.keys():
        # Extract UniProt ID: assume FASTA ID format as "tr|X6BLS5|X6BLS5_9HYPH"
        record = fasta_records[key]
        uniprot_id = record.id.split('|')[1]  # Get UniProt ID part by splitting
        info = record.description
        uniprot_to_len[uniprot_id] = len(record.seq)
        uniprot_info[uniprot_id] = info
    
    # 4. Merge data to csv1
    def get_additional_info(target):
        # Extract uniprot_id from target (assume format target-xxxx, take part after '-')
        uniprot_id = target
        
        # 1) Find corresponding ted_id information
        ted_id = target
        
        # Find rows with related ted_id in csv2
        if ted_id in csv2_dict:
            uniprot = csv2_dict[ted_id]['uniprot_id']
            cath_label = csv2_dict[ted_id]['cath_label']
        else:
            uniprot = cath_label = None  # If ted_id not found, set to None

        # 2) Get sequence length for UniProt ID
        target_len = uniprot_to_len.get(uniprot, None)  # If ID not found, return None
        target_info = uniprot_info.get(uniprot, None) 
        if target_len is None:
            print(uniprot_id)
        return pd.Series([uniprot, cath_label, target_len,target_info], index=['uniprot_id', 'cath_label', 'target_len','target_info'])
    
    # 5. Apply get_additional_info function to each row in csv1
    csv1[['uniprot_id', 'cath_label', 'target_len','target_info']] = csv1['target'].apply(get_additional_info)
    
    
    columns_order = ['source','target', 'FS_weight','TM_weight','Dali_weight','uniprot_id', 'target_info','target_len', 'cath_label'] + [col for col in csv1.columns if col not in ['source','target', 'FS_weight','TM_weight','Dali_weight','uniprot_id', 'target_info','target_len', 'cath_label'] ]

    # Adjust column order
    csv1 = csv1[columns_order]
    
    # 6. Save merged csv1
    csv1.to_csv(output_path, index=False)
    print(f"Output file saved to: {output_path}")



def add_domain_chop_to_target_cathdb(csv1_path, domain_boundaries_data,  output_path):
    # 1. Read csv1 and csv2 files
    csv1 = pd.read_csv(csv1_path)
    with open(domain_boundaries_data, "rb") as f:
        domain_boundaries = pickle.load(f)
    
    
    
    # 4. Merge data to csv1
    def get_additional_info(target):
        # Extract uniprot_id from target (assume format target-xxxx, take part after '-')

        if target[-2:] == "00":
            key = target[:-2] + "01"
        else:
            key = target
        
        chain_id = domain_boundaries[key]["chain_id"]
        ranges = domain_boundaries[key]["ranges"]
        chopping = str(ranges[0][0])+'-'+str(ranges[0][1])
        
        return pd.Series([ chain_id,chopping], index=['chain_id','chopping'])
    
    # 5. Apply get_additional_info function to each row in csv1
    csv1[['chain_id','chopping']] = csv1['target'].apply(get_additional_info)
    
    
    columns_order = ['source','target', 'FS_weight','TM_weight','Dali_weight','uniprot_id', 'target_info','target_len', 'cath_label','chopping','chain_id'] + [col for col in csv1.columns if col not in ['source','target', 'FS_weight','TM_weight','Dali_weight','uniprot_id', 'target_info','target_len', 'cath_label','chopping','chain_id'] ]

    # Adjust column order
    csv1 = csv1[columns_order]
    
    # 6. Save merged csv1
    csv1.to_csv(output_path, index=False)
    print(f"Output file saved to: {output_path}")



def get_protein_ted_info(protein_id):
    """
    Simulate getting protein detailed information from database or API.
    Return value is simulated data in JSON format (can replace with actual API query in real use).
    """
    url = f"{TED_info_URL}/uniprot/summary/{protein_id}"  # Replace with actual API address
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Return JSON data
    else:
        print(f"Error fetching data for {protein_id}, status code: {response.status_code}")
        return None


def process_protein_ted_info(protein_data):
    """
    Process protein data and save as DataFrame.

    Args:
        protein_data (dict): Dictionary containing protein fragment information.

    Returns:
        pd.DataFrame: Processed DataFrame containing all fragment related information.
    """
    # If "data" key doesn't exist, return empty DataFrame
    if "data" not in protein_data:
        print("Warning: 'data' field not found in protein_data.")
        return pd.DataFrame()

    # Get fragment list
    fragments = protein_data["data"]

    # Create a list to store table data
    processed_data = []

    # Traverse each fragment, extract related information
    for fragment in fragments:
        # Basic information
        ted_id = fragment.get("ted_id", "-")
        uniprot_acc = fragment.get("uniprot_acc", "-")
        chopping = fragment.get("chopping", "-")
        nres_domain = fragment.get("nres_domain", "-")
        num_segments = fragment.get("num_segments", "-")
        plddt = fragment.get("plddt", "-")
        num_helix = fragment.get("num_helix", "-")
        num_strand = fragment.get("num_strand", "-")
        num_turn = fragment.get("num_turn", "-")
        cath_label = fragment.get("cath_label", "-")
        cath_assignment_level = fragment.get("cath_assignment_level", "-")
        cath_assignment_method = fragment.get("cath_assignment_method", "-")
        packing_density = fragment.get("packing_density", "-")
        norm_rg = fragment.get("norm_rg", "-")
        tax_common_name = fragment.get("tax_common_name", "-")
        tax_scientific_name = fragment.get("tax_scientific_name", "-")
        tax_lineage = fragment.get("tax_lineage", "-")

        # Interaction information (if exists)
        interactions = fragment.get("interactions", [])
        interaction_count = len(interactions)

        # Store extracted data as dictionary
        processed_data.append({
            "ted_id": ted_id,
            "uniprot_acc": uniprot_acc,
            "chopping": chopping,
            "nres_domain": nres_domain,
            "num_segments": num_segments,
            "plddt": plddt,
            "num_helix": num_helix,
            "num_strand": num_strand,
            "num_turn": num_turn,
            "cath_label": cath_label,
            "cath_assignment_level": cath_assignment_level,
            "cath_assignment_method": cath_assignment_method,
            "packing_density": packing_density,
            "norm_rg": norm_rg,
            "tax_common_name": tax_common_name,
            "tax_scientific_name": tax_scientific_name,
            "tax_lineage": tax_lineage,
            "interaction_count": interaction_count,
            "interactions": interactions  # Keep complete interaction information (if needed)
        })

    # Convert processed data to DataFrame
    df = pd.DataFrame(processed_data)
    return df


def process_fs_reslult_new(input_csv, output_csv):
    # Read CSV file
    df = pd.read_csv(input_csv)
    
    # Rename columns
    df.rename(columns={'weight': 'FS_weight', 'alntmscore': 'TM_weight'}, inplace=True)
    
    # Convert FS_weight and TM_weight columns to numeric, keep four decimal places
    df['FS_weight'] = pd.to_numeric(df['FS_weight'], errors='coerce').round(4)
    df['TM_weight'] = pd.to_numeric(df['TM_weight'], errors='coerce').round(4)
    
    
    # Rearrange column order, move source, target, FS_weight and TM_weight to front
    cols = ['source', 'target', 'FS_weight', 'TM_weight'] + [col for col in df.columns if col not in ['source', 'target', 'FS_weight', 'TM_weight']]
    df = df[cols]
    
    # Filter rows with FS_weight >= 0.7 and TM_weight >= 0.5
    #df_filtered = df[(df['FS_weight'] >= 0.7) & (df['TM_weight'] >= 0.5)]
    df =  df[df['source'].str.contains('Q99ZW2', na=False)]
    df_filtered = df[(df['FS_weight'] >= 0) & (df['TM_weight'] >= 0)]
    
    # Save filtered data as new CSV file
    df_filtered.to_csv(output_csv, index=False)


def get_protein_ted_info_to_csv(input_csv, output_csv):
    """
    Read ID list from input protein ID file, get related protein information,
    and save processed data to output CSV file.

    Args:
        input_csv (str): Input file path containing protein ID list.
        output_csv (str): Output file path to save processed protein information.
    """
    # Determine if file is TSV or CSV format
    try:
        # Try to read as TSV format
        protein_ids = pd.read_csv(input_csv, sep="\t")["Entry"].tolist()
    except Exception as e:
        # If fails, try to read as CSV format
        protein_ids = pd.read_csv(input_csv)["Entry"].tolist()

    # Initialize empty DataFrame to store all protein data
    all_protein_data = pd.DataFrame()

    # Traverse each protein ID, get data and process
    for idx, protein_id in enumerate(protein_ids):
        print(f"Processing protein {idx + 1}/{len(protein_ids)}: {protein_id}")

        # Call API to get protein data
        protein_data = get_protein_ted_info(protein_id)  # Ensure fetch_protein_data function is defined

        # If getting data empty, skip
        if not protein_data:
            print(f"Warning: No data found for protein ID {protein_id}. Skipping...")
            continue

        # Process protein data
        protein_df = process_protein_ted_info(protein_data)  # Use updated process_protein_data function

        # If processed data empty, skip
        if protein_df.empty:
            print(f"Warning: No fragments found for protein ID {protein_id}. Skipping...")
            continue

        # Merge current protein data to total data
        all_protein_data = pd.concat([all_protein_data, protein_df], ignore_index=True)

        # Add delay to avoid too frequent requests (optional)
        time.sleep(1)

    # Save all processed data to output CSV file
    all_protein_data.to_csv(output_csv, index=False)
    print(f"Protein data successfully saved to {output_csv}")


def get_protein_domains(protein_id, api_url="https://api.ted-database.org/protein"):
    """
    Get domain data for specified protein from TED interface
    Args:
        protein_id (str): Target protein's UniProt ID (e.g. 'Q99ZW2')
        api_url (str): TED API base URL
    Returns:
        list: Dictionary list containing domain information
    """
    url = f"{api_url}/{protein_id}/domains"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()  # Assume return is JSON format
        return data.get("data", [])
    else:
        print(f"Failed to retrieve data for {protein_id}. Status code: {response.status_code}")
        return []

def process_protein_domains(domain_data):
    """
    Process and format domain data
    Args:
        domain_data (list): Domain data obtained from TED (JSON format)
    Returns:
        pd.DataFrame: Formatted as Pandas DataFrame
    """
    # Extract fields of interest
    domain_list = []
    for domain in domain_data:
        domain_list.append({
            "ted_id": domain.get("ted_id", "-"),
            "uniprot_acc": domain.get("uniprot_acc", "-"),
            "chopping": domain.get("chopping", "-"),
            "nres_domain": domain.get("nres_domain", "-"),
            "plddt": domain.get("plddt", "-"),
            "cath_label": domain.get("cath_label", "-"),
            "cath_assignment_level": domain.get("cath_assignment_level", "-"),
            "cath_assignment_method": domain.get("cath_assignment_method", "-"),
            "packing_density": domain.get("packing_density", "-"),
            "norm_rg": domain.get("norm_rg", "-")
        })
    return pd.DataFrame(domain_list)

def save_protein_domains_to_csv(protein_id, output_file):
    """
    Get domain data for specified protein and save to CSV file
    Args:
        protein_id (str): Target protein's UniProt ID
        output_file (str): Output CSV file path
    """
    domain_data = get_protein_domains(protein_id)
    if domain_data:
        df = process_protein_domains(domain_data)
        df.to_csv(output_file, index=False)
        print(f"Domain data for {protein_id} saved to {output_file}")
    else:
        print(f"No domain data found for {protein_id}")


def get_cath_family_info(cath_label):
    """
    Fetch domain information for a given CATH label using CATH API.

    Args:
        cath_label (str): CATH label to query.

    Returns:
        list: A list of dictionaries containing domain information.
    """
    base_url = "https://www.cathdb.info/version/v4_3_0/api/rest"
    depth_url = f"{base_url}/cathtree/from_cath_id_to_depth/{cath_label}/9"  # Depth 9 includes H level
    response = requests.get(depth_url, timeout=10)  # Timeout for the request

    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            print(f"Invalid JSON response for CATH label {cath_label}")
            return []
    else:
        print(f"Error fetching data for CATH label {cath_label}: HTTP {response.status_code}")
        return []

def get_domains_from_cath(input_file, output_file):
    """
    Extract domain information for CATH labels from the input Excel file and save it to a CSV file.

    Args:
        input_file (str): Path to the input Excel file.
        output_file (str): Path to the output CSV file.
    """
    try:
        import pandas as pd

        # Read the input Excel file
        df = pd.read_excel(input_file)

        # Ensure 'cath_label' column exists
        if 'cath_label' not in df.columns:
            raise ValueError("Input file must contain a 'cath_label' column.")

        # Extract unique non-null CATH labels
        cath_labels = df['cath_label'].dropna().unique()
        output_data = []
        fetched_labels = {}

        for cath_label in cath_labels:
            if cath_label == "-":
                # Skip unassigned labels
                continue

            if cath_label in fetched_labels:
                domain_info = fetched_labels[cath_label]
            else:
                print(f"Fetching data for CATH label: {cath_label}")
                try:
                    domain_info = get_cath_family_info(cath_label)
                    fetched_labels[cath_label] = domain_info
                except Exception as e:
                    print(f"Error processing CATH label {cath_label}: {e}")
                    continue

            # Validate domain_info type
            if isinstance(domain_info, dict):
                # Start with the root node and process children
                stack = [domain_info]
            elif isinstance(domain_info, list):
                # Directly process as a list of domains
                stack = domain_info.copy()
            else:
                print(f"Warning: Unexpected domain_info format for {cath_label}: {type(domain_info)}")
                continue

            # Process the domain information, including nested children
            while stack:
                domain = stack.pop()

                # Ensure domain is a dictionary
                if not isinstance(domain, dict):
                    print(f"Warning: Unexpected domain format in stack for {cath_label}: {domain}")
                    continue

                # Add current domain info to the output data
                output_data.append({
                    "cath_label": cath_label,
                    "domain_id": domain.get("example_domain_id", "-"),
                    "cath_id": domain.get("cath_id", "-"),
                    "cath_id_padded": domain.get("cath_id_padded", "-"),
                    "name": domain.get("name", "-"),
                    "description": domain.get("description", "-"),
                    "children_count": domain.get("children_count", 0),
                    "descendants_count": domain.get("descendants_count", 0),
                })

                # Add children to the stack if present
                if "children" in domain and isinstance(domain["children"], list):
                    stack.extend(domain["children"])

        # Save output to CSV
        if output_data:
            output_df = pd.DataFrame(output_data)
            output_df.to_csv(output_file, index=False)
            print(f"Data successfully saved to {output_file}")
        else:
            print("No data to save.")

    except Exception as e:
        print(f"An error occurred: {e}")

            
def add_uniprot_to_output(output_file, output_with_uniprot_file):
    """
    Add a UniProt ID column to the output file based on the PDB ID (first 4 characters of domain_id).

    Args:
        output_file (str): Path to the CSV file containing domain information.
        output_with_uniprot_file (str): Path to save the updated CSV file with UniProt IDs.

    Returns:
        None
    """
    def get_uniprot_id(pdb_id):
        """Fetch UniProt ID for a given PDB ID using an API."""
        url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return list(data[pdb_id]['UniProt'].keys())[0] if pdb_id in data and 'UniProt' in data[pdb_id] else None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching UniProt ID for PDB ID {pdb_id}: {e}")
            return None

    try:
        # Load the CSV file
        df = pd.read_csv(output_file)

        # Extract PDB ID from domain_id (first 4 characters)
        df['pdb_id'] = df['domain_id'].str[:4]

        # Map PDB IDs to UniProt IDs
        df['uniprot_id'] = df['pdb_id'].apply(get_uniprot_id)

        # Save the updated DataFrame to a new file
        df.to_csv(output_with_uniprot_file, index=False)
        print(f"Updated file with UniProt IDs saved to {output_with_uniprot_file}")

    except Exception as e:
        print(f"An error occurred while adding UniProt IDs: {e}")   
        
        

     
def download_pdb(pdb_id, pdb_dir, pdb_url_template="https://files.rcsb.org/download/{pdb_id}.pdb"):
    """Download PDB file with retry mechanism"""
    pdb_url = pdb_url_template.format(pdb_id=pdb_id)
    save_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")

    if os.path.exists(save_path):
        print(f"PDB file {pdb_id} already exists. Skipping download.")
    else:
        #print(pdb_url)
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


def download_fasta(uniprot_id, pdb_id, fasta_url_template="https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta", max_versions=20, max_retries=5):
    """Download FASTA sequence and return FASTA data"""
    
    if pd.isna(uniprot_id) or not uniprot_id:
        print(f"Invalid UniProt ID for PDB {pdb_id}: {uniprot_id}. Skipping FASTA download.")
        return None

    fasta_url = fasta_url_template.format(uniprot_id=uniprot_id)
    
    # Try to download FASTA data from standard URL with retry mechanism
    for attempt in range(max_retries):
        try:
            response = requests.get(fasta_url)
            if response.status_code == 200 and response.text.strip():  # Check if response valid and not empty
                # Modify FASTA header, add PDB ID
                fasta_data = response.text
                lines = fasta_data.splitlines()

                if lines and lines[0].startswith(">"):
                    lines[0] = lines[0] + f"__{pdb_id}"
                return "\n".join(lines)
            else:
                print(f"Attempt {attempt+1}: FASTA content is empty or invalid from standard URL.")
        except requests.RequestException as e:
            print(f"Attempt {attempt+1}: Exception occurred: {e}")
        # Can set retry interval as needed, e.g. wait 1 second
        time.sleep(5)

    # If still not successful after multiple retries, enter except block, try to download from UniSave
    print("Standard FASTA URL failed after retries. Trying UniSave for multiple versions...")
    for version in range(max_versions, 0, -1):  # From max_versions to 1
        unisave_url = f"https://rest.uniprot.org/unisave/{uniprot_id}?format=fasta&versions={version}"
        try:
            response = requests.get(unisave_url)
            if response.status_code == 200 and response.text.strip():
                fasta_data = response.text
                lines = fasta_data.splitlines()
                if lines and lines[0].startswith(">"):
                    lines[0] = lines[0] + f"__{pdb_id}"
                return "\n".join(lines)
            else:
                continue  # Current version failed, try next version
        except requests.RequestException:
            continue  # Current version request failed, try next version

    print(f"Unable to download FASTA sequence for {uniprot_id} after trying all versions.")
    return None


def download_fasta33(uniprot_id, pdb_id, fasta_url_template="https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta", max_versions=20):
    """Download FASTA sequence and return FASTA data"""
    
    if pd.isna(uniprot_id) or not uniprot_id:
        print(f"Invalid UniProt ID for PDB {pdb_id}: {uniprot_id}. Skipping FASTA download.")
        return None

    fasta_url = fasta_url_template.format(uniprot_id=uniprot_id)
    
    # Try downloading the FASTA data from the standard URL first
    try:
        response = requests.get(fasta_url)
        if response.status_code == 200 and response.text.strip():  # Check if response is valid and non-empty
            # Modify the FASTA header by adding the PDB ID
            fasta_data = response.text
            lines = fasta_data.splitlines()

            # Process the header line and add pdb_id to it
            if lines and lines[0].startswith(">"):
                lines[0] = lines[0] + f"__{pdb_id}"
            return "\n".join(lines)  # Return the FASTA data as a string

        else:
            raise requests.RequestException("FASTA content is empty or invalid from standard URL.")

    except requests.RequestException as e:
        # If the standard FASTA URL fails, attempt to fetch from UniSave for multiple versions
        fasta_downloaded = False
        for version in range(max_versions, 0, -1):  # Try versions from max_versions down to 1
            unisave_url = f"https://rest.uniprot.org/unisave/{uniprot_id}?format=fasta&versions={version}"
            try:
                response = requests.get(unisave_url)
                if response.status_code == 200 and response.text.strip():  # Check if response is valid and non-empty
                    # Modify the FASTA header by adding the PDB ID
                    fasta_data = response.text
                    lines = fasta_data.splitlines()

                    # Process the header line and add pdb_id to it
                    if lines and lines[0].startswith(">"):
                        lines[0] = lines[0] + f"__{pdb_id}"
                    return "\n".join(lines)  # Return the FASTA data as a string
                else:
                    continue  # Try the next version if the current one fails
            except requests.RequestException as e:
                continue  # Try the next version if the current one fails

        print(f"Unable to download FASTA sequence for {uniprot_id} after trying all versions.")
        return None
def download_fasta_write_to_file(uniprot_id, pdb_id,  fasta_output_dir, fasta_url_template="https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta",max_versions=20):
    """Download FASTA sequence with retry mechanism"""
    if pd.isna(uniprot_id) or not uniprot_id:
        print(f"Invalid UniProt ID for PDB {pdb_id}: {uniprot_id}. Skipping FASTA download.")
        return None

    fasta_url = fasta_url_template.format(uniprot_id=uniprot_id)
    #print(f"Downloading FASTA sequence for {uniprot_id} (PDB {pdb_id})...")
    
    
    # Try downloading the FASTA data from the standard URL first
    try:
        response = requests.get(fasta_url)
        if response.status_code == 200 and response.text.strip():  # Check if response is valid and non-empty
            # Modify the FASTA header by adding the PDB ID
            fasta_data = response.text
            lines = fasta_data.splitlines()

            # Process the header line and add pdb_id to it
            if lines and lines[0].startswith(">"):
                lines[0] = lines[0] + f"__{pdb_id}"
                
            # Save the FASTA sequence to a file named after the UniProt ID
            fasta_file_path = os.path.join(fasta_output_dir, f"{uniprot_id}.fasta")
            with open(fasta_file_path, "w") as fasta_out:
                fasta_out.writelines("\n".join(lines) + "\n")
            print(f"Downloaded FASTA sequence for {uniprot_id} (PDB {pdb_id})...")
            return
        else:
            raise requests.RequestException("FASTA content is empty or invalid from standard URL.")

    except requests.RequestException as e:
        #print(f"Failed to download FASTA sequence for {uniprot_id} from {fasta_url}: {e}")
        #print("Attempting to get FASTA from UniSave...")

        # If the standard FASTA URL fails, attempt to fetch from UniSave for multiple versions
        fasta_downloaded = False
        for version in range(max_versions, 0, -1):  # Try versions from max_versions down to 1
            unisave_url = f"https://rest.uniprot.org/unisave/{uniprot_id}?format=fasta&versions={version}"
            try:
                response = requests.get(unisave_url)
                if response.status_code == 200 and response.text.strip():  # Check if response is valid and non-empty
                    # Modify the FASTA header by adding the PDB ID
                    fasta_data = response.text
                    lines = fasta_data.splitlines()

                    # Process the header line and add pdb_id to it
                    if lines and lines[0].startswith(">"):
                        lines[0] = lines[0] + f"__{pdb_id}"
                    # Write the modified FASTA sequence to the output file
                    # Save the FASTA sequence to a file named after the UniProt ID
                    fasta_file_path = os.path.join(fasta_output_dir, f"{uniprot_id}.fasta")
                    with open(fasta_file_path, "w") as fasta_out:
                        fasta_out.writelines("\n".join(lines) + "\n")
                    print(f"Downloaded and appended FASTA sequence for {uniprot_id} from UniSave version {version}")
                    fasta_downloaded = True
                    break  # Successfully downloaded, no need to try further versions
                else:
                    #print(f"Version {version} did not return valid FASTA data for {uniprot_id}, trying next version.")
                    temp = fasta_downloaded
            except requests.RequestException as e:
                #print(f"Failed to download FASTA sequence from UniSave version {version} for {uniprot_id}: {e}")
                continue  # Try the next version if the current one fails

        if not fasta_downloaded:
            print(f"Unable to download FASTA sequence for {uniprot_id} after trying all versions.")
            return
    return

def download_pdb_and_fasta_by_cathid_parallel(output_file, pdb_dir, fasta_file, pdb_url_template="https://files.rcsb.org/download/{pdb_id}.pdb", fasta_url_template="https://www.uniprot.org/uniprot/{uniprot_id}.fasta"):
    """
    Download full PDB files and FASTA sequences based on unique PDB IDs from the output CSV file.

    Args:
        output_file (str): Path to the CSV file containing domain information.
        pdb_dir (str): Directory to save the downloaded PDB files.
        fasta_file (str): Path to save the downloaded FASTA sequences file.
        pdb_url_template (str): URL template for downloading PDB files, with '{pdb_id}' as a placeholder.
        fasta_url_template (str): URL template for downloading FASTA sequences, with '{uniprot_id}' as a placeholder.

    Returns:
        None
    """
    # Ensure the output directories exist
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(fasta_file), exist_ok=True)  # Ensure fasta file directory exists

    # Load the CSV file
    try:
        df = pd.read_csv(output_file)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Extract unique PDB IDs from domain_id
    pdb_ids = df['domain_id'].str[:4].dropna().unique()

    # Initialize a list to store all FASTA sequences
    fasta_sequences = []
    # Use thread pool to download PDB files in parallel
    with ThreadPoolExecutor() as pdb_executor:
        # Submit each PDB download task
        pdb_futures = [pdb_executor.submit(download_pdb, pdb_id, pdb_dir) for pdb_id in pdb_ids]
        
        # Wait for all PDB download tasks to complete
        for future in as_completed(pdb_futures):
            future.result()  # Get each thread's return result, although we don't need it now

    with ThreadPoolExecutor() as fasta_executor:
        # Submit each FASTA download task
        fasta_futures = []
        for pdb_id in pdb_ids:
            uniprot_id = df.loc[df['domain_id'].str[:4] == pdb_id, 'uniprot_id'].iloc[0]# Retrieve UniProt ID from the CSV file
            print(uniprot_id)
            if pd.notna(uniprot_id):  # Ensure uniprot_id is not empty
                fasta_futures.append(fasta_executor.submit(download_fasta, uniprot_id, pdb_id, fasta_url_template))
            else:
                print(f"Skipping FASTA download for PDB {pdb_id} as UniProt ID is NaN")


        # Collect all returned FASTA data
        for future in as_completed(fasta_futures):
            fasta_data = future.result()
            if fasta_data:
                fasta_sequences.append(fasta_data)
                
    # Write all FASTA sequences to file
    with open(fasta_file, "w") as fasta_out:
        for fasta in fasta_sequences:
                fasta_out.writelines(fasta + "\n")
        print(f"FASTA sequences saved to {fasta_file}.")
           
                    

from Bio import SeqIO


def process_pdb_for_domain(domain_id, pdb_dir, domain_pdb_dir, domain_boundaries):
    """Process PDB files and extract domain-level PDB"""
    pdb_id = domain_id[:4]
    if domain_id[-2:] == "00":
        key = domain_id[:-2] + "01"
    else:
        key = domain_id

    if key not in domain_boundaries:
        print(f"No boundary information found for {key}. Skipping.")
        return None

    chain_id = domain_boundaries[key]["chain_id"]
    ranges = domain_boundaries[key]["ranges"]
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    domain_pdb_path = os.path.join(domain_pdb_dir, f"{domain_id}.pdb")

    # Skip if PDB file doesn't exist
    if not os.path.exists(pdb_path):
        print(f"PDB file {pdb_id} not found. Skipping.")
        return None

    try:
        with open(pdb_path, "r") as pdb_file:
            pdb_lines = pdb_file.readlines()

        with open(domain_pdb_path, "w") as domain_pdb_file:
            for line in pdb_lines:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    res_seq = int(line[22:26].strip())
                    chain = line[21]
                    for start_res, end_res in ranges:
                        if chain == chain_id and start_res <= res_seq <= end_res:
                            domain_pdb_file.write(line)

        print(f"Saved domain {domain_id} to {domain_pdb_path}.")
        return domain_id, pdb_id, ranges  # Return domain info for FASTA processing

    except Exception as e:
        print(f"Error processing PDB file {pdb_id} for domain {domain_id}: {e}")
        return None

def process_fasta_for_domain_check(domain_id, pdb_id, ranges, fasta_records):
    """Process FASTA files and extract domain-level FASTA"""
    try:
        domain_sequence = ""
        id = pdb_id.split('-')[1]
        record = fasta_records[id]
        seq = str(record.seq)
        for start_res, end_res in ranges:
            domain_sequence += seq[start_res-1:end_res]
        range_str = "_".join([f"{start}-{end}" for start, end in ranges])
        return f">{pdb_id}_{domain_id}_{range_str}\n{domain_sequence}\n"
    except Exception as e:
        print(f"Error processing FASTA sequence for {domain_id}: {e}")
        return None


def process_fasta_for_domain(domain_id, pdb_id, ranges, fasta_records):
    """Process FASTA files and extract domain-level FASTA"""
    try:
        domain_sequence = ""
        # Find corresponding FASTA record
        for record in fasta_records:
            if record.description.split("__")[1] == pdb_id:
                seq = str(record.seq)
                for start_res, end_res in ranges:
                    domain_sequence += seq[start_res-1:end_res]
                range_str = "_".join([f"{start}-{end}" for start, end in ranges])
                return f">{pdb_id}_{domain_id}_{range_str}\n{domain_sequence}\n"
    except Exception as e:
        print(f"Error processing FASTA sequence for {domain_id}: {e}")
        return None

def split_domain_by_cath_boundary_and_save_fasta_parallel(pdb_dir, boundary_file, domain_pdb_dir, domain_fasta_file, csv_file, fasta_file):
    """
    Split full PDB files into domain-level PDB files and save domain-level FASTA sequences into a single FASTA file.
    """
    os.makedirs(domain_pdb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(domain_fasta_file), exist_ok=True)

    # Load CSV file with domain IDs
    try:
        csv_df = pd.read_csv(csv_file)
        valid_domain_ids = set(csv_df['domain_id'].dropna().unique())
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Load domain boundaries (using pickle if available
    save_path =  "/mnt/sdb4/protein_gen/Cas9_domain_work/data/TED/Cas9/cath_domain_boundaries.pkl"
    # Load the domain boundaries (or process the boundary file)
    # Load the domain boundaries (or process the boundary file)
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            domain_boundaries = pickle.load(f)
    else:
        # Parse the boundary file into a dictionary
        domain_boundaries = {}
        try:
            with open(boundary_file, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue  # Skip comment lines
                    
                    parts = line.strip().split()
                    
                    # Ensure the line has enough parts to process
                    if len(parts) < 6:
                        print(f"Skipping invalid line: {line}")
                        continue
                    
                    pdb_id = parts[0][:4]  # PDB code (e.g., '1bcm')
                    if pdb_id == "10gs":
                        print("dd")
                    if pdb_id == "12e8":
                        print("dd")
                    if pdb_id == "1chm":
                        print("d")
                    chain_id = parts[0][4]  # Chain identifier (e.g., 'A')
                    num_domains = int(parts[1][1:])  # Number of domains (e.g., 'D02' -> 2)
                    num_fragments = int(parts[2][1:])  # Number of fragments (e.g., 'F02' -> 2)

                    domain_id_start_index = 5  # Start index of domain/fragment information

                    # Extract domain data
                    i = domain_id_start_index
                    domain_index = 1  # Start with domain 1
                    while domain_index <= num_domains:
                        key = f"{pdb_id}{chain_id}{domain_index:02d}"  # Domain key
                        ranges = []

                        # Number of segments in the current domain
                        num_segments = int(parts[i-2])  # Get number of segments from part[3]

                        # Read the segments for the current domain
                        for segment_index in range(num_segments):
                            # Start reading each segment
                            if i + 3 < len(parts):
                                start_res = int(parts[i].strip("A-Z"))  # Start residue
                                end_res = int(parts[i + 3].strip("A-Z"))  # End residue

                                # Only append valid ranges (start <= end)
                                if start_res <= end_res:
                                    ranges.append((start_res, end_res))  # Store range
                            i +=6

                        # Add the domain boundary information to the dictionary
                        if ranges:
                            domain_boundaries[key] = {"chain_id": chain_id, "ranges": ranges}

                        domain_index += 1
                        i+=1

            # Ensure the save directory exists
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save the resulting dictionary as a pickle file
            with open(save_path, "wb") as f:
                pickle.dump(domain_boundaries, f)

            print(f"Domain boundaries have been successfully saved to '{save_path}'.")
        
        except Exception as e:
            print(f"Error reading the boundary file: {e}")

    # Read complete FASTA file data
    fasta_records = []
    try:
        with open(fasta_file, "r") as fasta_in:
            fasta_records = list(SeqIO.parse(fasta_in, "fasta"))
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return

    # Process PDB files and extract domains
    # First process all PDB files, then process FASTA files
    pdb_futures = []
    with ThreadPoolExecutor() as pdb_executor:
        # Submit PDB processing tasks
        pdb_futures = [
            pdb_executor.submit(process_pdb_for_domain, domain_id, pdb_dir, domain_pdb_dir, domain_boundaries)
            for domain_id in valid_domain_ids
        ]

    # Collect PDB processing results, and process FASTA files
    domain_fasta_sequences = []
    with ThreadPoolExecutor() as fasta_executor:
        # Submit FASTA processing tasks
        for future in as_completed(pdb_futures):
            result = future.result()
            if result:
                domain_id, pdb_id, ranges = result
                # Get domain FASTA sequence
                fasta_sequence = process_fasta_for_domain(domain_id, pdb_id, ranges, fasta_records)
                if fasta_sequence:
                    domain_fasta_sequences.append(fasta_sequence)

    # Write all FASTA sequences to file
    with open(domain_fasta_file, "w") as fasta_out:
        fasta_out.writelines(domain_fasta_sequences)

    print("Domain-level PDB and FASTA files have been processed and saved.")

def split_domain_by_cath_boundary_and_save_fasta(pdb_dir, boundary_file, domain_pdb_dir, domain_fasta_file, csv_file, fasta_file):
    """
    Split full PDB files into domain-level PDB files and save domain-level FASTA sequences into a single FASTA file.

    Args:
        pdb_dir (str): Directory containing full PDB files.
        boundary_file (str): Path to the CATH domain boundaries file.
        domain_pdb_dir (str): Directory to save the domain-level PDB files.
        domain_fasta_file (str): File to save all domain sequences in FASTA format.
        csv_file (str): Path to the CSV file containing domain IDs to be processed.
        fasta_file (str): Path to the full protein FASTA file.

    Returns:
        None
    """
    # Ensure the domain PDB and FASTA directories exist
    os.makedirs(domain_pdb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(domain_fasta_file), exist_ok=True)  # Ensure domain FASTA file directory exists

    # Load the CSV file with domain IDs and valid ranges
    try:
        csv_df = pd.read_csv(csv_file)
        valid_domain_ids = set(csv_df['domain_id'].dropna().unique())
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return
    save_path =  "/mnt/sdb4/protein_gen/Cas9_domain_work/data/TED/Cas9/cath_domain_boundaries.pkl"  
    # Load the domain boundaries (or process the boundary file)
    # Load the domain boundaries (or process the boundary file)
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            domain_boundaries = pickle.load(f)
    else:
        # Parse the boundary file into a dictionary
        domain_boundaries = {}
        try:
            with open(boundary_file, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue  # Skip comment lines
                    
                    parts = line.strip().split()
                    
                    # Ensure the line has enough parts to process
                    if len(parts) < 6:
                        print(f"Skipping invalid line: {line}")
                        continue
                    
                    pdb_id = parts[0][:4]  # PDB code (e.g., '1bcm')
                    if pdb_id == "10gs":
                        print("dd")
                    if pdb_id == "12e8":
                        print("dd")
                    if pdb_id == "1chm":
                        print("d")
                    chain_id = parts[0][4]  # Chain identifier (e.g., 'A')
                    num_domains = int(parts[1][1:])  # Number of domains (e.g., 'D02' -> 2)
                    num_fragments = int(parts[2][1:])  # Number of fragments (e.g., 'F02' -> 2)

                    domain_id_start_index = 5  # Start index of domain/fragment information

                    # Extract domain data
                    i = domain_id_start_index
                    domain_index = 1  # Start with domain 1
                    while domain_index <= num_domains:
                        key = f"{pdb_id}{chain_id}{domain_index:02d}"  # Domain key
                        ranges = []

                        # Number of segments in the current domain
                        num_segments = int(parts[i-2])  # Get number of segments from part[3]

                        # Read the segments for the current domain
                        for segment_index in range(num_segments):
                            # Start reading each segment
                            if i + 3 < len(parts):
                                start_res = int(parts[i].strip("A-Z"))  # Start residue
                                end_res = int(parts[i + 3].strip("A-Z"))  # End residue

                                # Only append valid ranges (start <= end)
                                if start_res <= end_res:
                                    ranges.append((start_res, end_res))  # Store range
                            i +=6

                        # Add the domain boundary information to the dictionary
                        if ranges:
                            domain_boundaries[key] = {"chain_id": chain_id, "ranges": ranges}

                        domain_index += 1
                        i+=1

            # Ensure the save directory exists
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Save the resulting dictionary as a pickle file
            with open(save_path, "wb") as f:
                pickle.dump(domain_boundaries, f)

            print(f"Domain boundaries have been successfully saved to '{save_path}'.")
        
        except Exception as e:
            print(f"Error reading the boundary file: {e}")
    
    # Process each valid domain ID

    # Open the domain FASTA file for appending the sequences
    with open(domain_fasta_file, "w") as fasta_out:
        # Process each valid domain ID
        for domain_id in valid_domain_ids:
            pdb_id = domain_id[:4]
            if domain_id[-2:] == "00":
                key = domain_id[:-2] + "01" 
            else:
                key = domain_id
            if key not in domain_boundaries:
                print(f"No boundary information found for {key}. Skipping.")
                continue

            chain_id = domain_boundaries[key]["chain_id"]
            ranges = domain_boundaries[key]["ranges"]
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
            domain_pdb_path = os.path.join(domain_pdb_dir, f"{domain_id}.pdb")

            # Skip if PDB file doesn't exist
            if not os.path.exists(pdb_path):
                print(f"PDB file {pdb_id} not found. Skipping.")
                continue

            # Extract and save domain-level PDB and FASTA
            try:
                # Extract relevant PDB data
                with open(pdb_path, "r") as pdb_file:
                    pdb_lines = pdb_file.readlines()

                # Extract and save domain-level PDB
                with open(domain_pdb_path, "w") as domain_pdb_file:
                    for line in pdb_lines:
                        if line.startswith("ATOM") or line.startswith("HETATM"):
                            res_seq = int(line[22:26].strip())
                            chain = line[21]
                            for start_res, end_res in ranges:
                                if chain == chain_id and start_res <= res_seq <= end_res:
                                    domain_pdb_file.write(line)

                print(f"Saved domain {domain_id} to {domain_pdb_path}.")

                # Extract domain sequence from the corresponding full protein FASTA file
                with open(fasta_file, "r") as fasta_in:
                    fasta_records = SeqIO.parse(fasta_in, "fasta")
                    for record in fasta_records:
                        if record.description.split("__")[1] == pdb_id:
                            seq = str(record.seq)
                            # Extract domain sequence based on the residue ranges
                            domain_sequence = ""
                            if len(ranges)>1:
                                print('dd')
                            for start_res, end_res in ranges:
                                domain_sequence += seq[start_res-1:end_res]
                            
                            # Write domain FASTA sequence
                            fasta_out.write(f">{pdb_id}_{domain_id}_{ranges[0][0]}-{ranges[-1][1]}\n{domain_sequence}\n")
                            print(f"Saved domain {domain_id} FASTA to {domain_fasta_file}.")
                            break  # Only process the first matching sequence (since one pdb_id has one FASTA record)

            except Exception as e:
                print(f"Error processing PDB file {pdb_id} for domain {domain_id}: {e}")


 
def split_domain_by_cath_boundary(pdb_dir, boundary_file, domain_pdb_dir,csv_file):
    """
    Split full PDB files into domain-level PDB files based on CATH domain boundaries.

    Args:
        pdb_dir (str): Directory containing full PDB files.
        boundary_file (str): Path to the CATH domain boundaries file.
        domain_pdb_dir (str): Directory to save the domain-level PDB files.
        csv_file (str): Path to the CSV file containing domain IDs to be processed.

    Returns:
        None
    """
    # Ensure the domain PDB directory exists
    os.makedirs(domain_pdb_dir, exist_ok=True)

    # Load the CSV file
    try:
        csv_df = pd.read_csv(csv_file)
        valid_domain_ids = set(csv_df['domain_id'].dropna().unique())
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return
    
    if os.path.exists("/mnt/sdb4/protein_gen/Cas9_domain_work/data/TED/Cas9/cath_domain_boundaries.pkl"):
        with open("/mnt/sdb4/protein_gen/Cas9_domain_work/data/TED/Cas9/cath_domain_boundaries.pkl","rb") as f:
            domain_boundaries = pickle.load(f)
    else:
        # Parse the boundary file into a dictionary
        domain_boundaries = {}
        try:
            with open(boundary_file, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split()
                    pdb_id = parts[0][:4]
                    chain_id = parts[0][4]
                    domain_id = parts[2]
                    key = f"{parts[0][:5]}{domain_id[1:]}"

                    # Parse residue ranges
                    ranges = []
                    i = 5
                    while i < len(parts):
                        if parts[i] == "-":
                            if i+1<len(parts) and parts[i+1] == "1":
                                break
                            i += 2
                            continue
                        try:
                            start_res = int(parts[i].strip("A-Z"))
                            end_res = int(parts[i + 3].strip("A-Z"))
                            ranges.append((start_res, end_res))
                            i += 4
                        except (ValueError, IndexError):
                            break

                    domain_boundaries[key] = {"chain_id": chain_id, "ranges": ranges}
        except Exception as e:
            print(f"Error reading the boundary file: {e}")
            return
    # First run
    #with open("/mnt/sdb4/protein_gen/Cas9_domain_work/data/TED/Cas9/cath_domain_boundaries.pkl", "wb") as f:
    #    pickle.dump(domain_boundaries, f)
    
    # Process each valid domain ID
    for domain_id in valid_domain_ids:
        pdb_id = domain_id[:4]
        key = domain_id
        if key not in domain_boundaries:
            print(f"No boundary information found for {key}. Skipping.")
            continue

        chain_id = domain_boundaries[key]["chain_id"]
        ranges = domain_boundaries[key]["ranges"]
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        domain_pdb_path = os.path.join(domain_pdb_dir, f"{domain_id}.pdb")

        # Skip if PDB file doesn't exist
        if not os.path.exists(pdb_path):
            print(f"PDB file {pdb_id} not found. Skipping.")
            continue

        # Extract and save domain-level PDB
        try:
            with open(pdb_path, "r") as pdb_file:
                pdb_lines = pdb_file.readlines()

            # Extract relevant lines based on residue ranges
            with open(domain_pdb_path, "w") as domain_pdb_file:
                for line in pdb_lines:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        res_seq = int(line[22:26].strip())
                        chain = line[21]
                        for start_res, end_res in ranges:
                            if chain == chain_id and start_res <= res_seq <= end_res:
                                domain_pdb_file.write(line)
                                break

            print(f"Saved domain {domain_id} to {domain_pdb_path}.")

        except Exception as e:
            print(f"Error processing PDB file {pdb_id} for domain {domain_id}: {e}")


import pandas as pd

def copy_files(src_folder, dest_folder):
    # Ensure target folder exists, create if not exists
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # Get all files in source folder
    for item in os.listdir(src_folder):
        # Build complete paths for source and target files
        src_path = os.path.join(src_folder, item)
        dest_path = os.path.join(dest_folder, item)
        
        # If it's a file, copy it
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)
        # If it's a subfolder, can also copy recursively
        elif os.path.isdir(src_path):
            shutil.copytree(src_path, dest_path)

def get_ted_domains_by_cath_label(true_hnh_info, domain_summary_file, output_file, chunksize=100000):
    """
    Filter domains from domain summary file based on CATH labels from True_HNH_ted_info, using chunked reading.

    Args:
        true_hnh_info (str): Path to the True_HNH_ted_info Excel file.
        domain_summary_file (str): Path to the domain summary TSV file.
        output_file (str): Path to save the filtered results.
        chunksize (int): Number of rows to read per chunk for large files.

    Returns:
        None
    """
    try:
        # Load the True_HNH_ted_info file and extract unique CATH labels
        hnh_df = pd.read_excel(true_hnh_info)
        if 'cath_label' not in hnh_df.columns:
            raise ValueError("Column 'cath_label' not found in True_HNH_ted_info.")

        unique_cath_labels = hnh_df['cath_label'].dropna().unique()
        unique_cath_labels = [label for label in unique_cath_labels if label != '-']  # Remove '-' entries
        print(f"Unique CATH labels extracted: {unique_cath_labels}")

        # Define the column names for the domain summary file
        column_names = [
            "ted_id", "md5_domain", "consensus_level", "chopping", "nres_domain",
            "num_segments", "plddt", "num_helix_strand_turn", "num_helix", "num_strand",
            "num_helix_strand", "num_turn", "proteome_id", "cath_label", "cath_assignment_level",
            "cath_assignment_method", "packing_density", "norm_rg", "tax_common_name",
            "tax_scientific_name", "tax_lineage"
        ]

        # Separate T-level and H-level labels
        t_labels = {label for label in unique_cath_labels if label.count('.') == 2}  # T-level has 2 dots
        h_labels = set(unique_cath_labels) - t_labels
        print(f"Getting ted domains with same Cath labels.")
        # Open output file for appending filtered results
        with open(output_file, 'w') as output:
            header_written = False

            # Read the domain summary file in chunks with progress bar
            for chunk in tqdm(pd.read_csv(domain_summary_file, sep='\t', names=column_names, header=0, chunksize=chunksize)):
                if 'cath_label' not in chunk.columns or 'cath_assignment_level' not in chunk.columns:
                    raise ValueError("Required columns 'cath_label' or 'cath_assignment_level' not found in domain summary file.")

                # Filter T-level labels (include sublevels)
                t_filtered = chunk[chunk['cath_label'].apply(
                    lambda x: any(x.startswith(label + ".") or x == label for label in t_labels)
                )]

                # Filter H-level labels (exact match only)
                h_filtered = chunk[chunk['cath_label'].isin(h_labels)]

                # Combine results
                filtered_chunk = pd.concat([t_filtered, h_filtered])

                # Write filtered results to the output file
                if not filtered_chunk.empty:
                    filtered_chunk.to_csv(output, index=False, header=not header_written)
                    header_written = True

        print(f"Filtered domains saved to {output_file}.")
        print(f"Completed.")
    except Exception as e:
        print(f"An error occurred: {e}")



def add_cluster_representative(true_hnh_info, clustering_file, output_file,chunksize=100000):
    """
    Add Cluster_representative column to True_HNH_ted_info based on the clustering file, using chunked reading.

    Args:
        true_hnh_info (str): Path to the True_HNH_ted_info file (Excel format).
        clustering_file (str): Path to the clustering TSV file.
        output_file (str): Path to save the updated True_HNH_ted_info file.
        chunksize (int): Number of rows to read per chunk for large files.

    Returns:
        None
    """
    try:
        # Load True_HNH_ted_info
        hnh_df = pd.read_excel(true_hnh_info)
        if 'ted_id' not in hnh_df.columns:
            raise ValueError("Column 'ted_id' not found in True_HNH_ted_info.")

        # Initialize a dictionary to map ted_id to Cluster_representative
        cluster_map = {}
        found_ids = set()

        # Read the clustering file in chunks
        for chunk in tqdm(pd.read_csv(clustering_file, sep='\t', names=[
            "Cluster_representative", "Cluster_member", "CATH_code", "Assignment_type"
        ], header=0, chunksize=chunksize), desc="Processing clustering file"):

            # Filter chunk for ted_ids in hnh_df that haven't been mapped yet
            chunk_filtered = chunk[chunk['Cluster_member'].isin(hnh_df['ted_id'])]

            # Update cluster_map with the mappings from the chunk
            for _, row in chunk_filtered.iterrows():
                cluster_map[row['Cluster_member']] = row['Cluster_representative']
                found_ids.add(row['Cluster_member'])

            # Stop reading if all ted_ids have been found
            if len(found_ids) == len(hnh_df['ted_id']):
                break

        # Add Cluster_representative column to True_HNH_ted_info
        hnh_df['Cluster_representative'] = hnh_df['ted_id'].map(cluster_map)

        # Save updated file
        hnh_df.to_excel(output_file, index=False)
        print(f"Updated True_HNH_ted_info saved to {output_file}.")

    except Exception as e:
        print(f"An error occurred: {e}")


def get_ted_domains_by_cluster(true_hnh_info_with_cluster, clustering_file, output_file,chunksize=100000):
    """
    Extract all domains belonging to unique Cluster_representative IDs from clustering file.

    Args:
        true_hnh_info_with_cluster (str): Path to the updated True_HNH_ted_info file with Cluster_representative column.
        clustering_file (str): Path to the clustering TSV file.
        output_file (str): Path to save the extracted domains.

    Returns:
        None
    """
    try:
        # Load True_HNH_ted_info with Cluster_representative
        hnh_df = pd.read_excel(true_hnh_info_with_cluster)
        if 'Cluster_representative' not in hnh_df.columns:
            raise ValueError("Column 'Cluster_representative' not found in True_HNH_ted_info.")

        # Get unique Cluster_representative IDs
        cluster_rep_ids = set(hnh_df['Cluster_representative'].dropna().unique())

        # Initialize an empty DataFrame to store filtered data
        filtered_data = pd.DataFrame()

        # Read the clustering file in chunks
        for chunk in tqdm(pd.read_csv(clustering_file, sep='\t', names=[
            "Cluster_representative", "Cluster_member", "CATH_code", "Assignment_type"
        ], header=0, chunksize=chunksize), desc="Processing clustering file"):

            # Filter chunk for matching Cluster_representative IDs
            filtered_chunk = chunk[chunk['Cluster_representative'].isin(cluster_rep_ids)]

            # Append filtered data to the DataFrame
            if not filtered_chunk.empty:
                filtered_data = pd.concat([filtered_data, filtered_chunk], ignore_index=True)

        # Save the filtered data to the output file
        if not filtered_data.empty:
            filtered_data.to_csv(output_file, index=False)
            print(f"Extracted domains saved to {output_file}.")
        else:
            print("No matching domains found.")

    except Exception as e:
        print(f"An error occurred: {e}")
        

# Unified download, then write only once, may lose data, so save a file after each FASTA download
def download_pdb_and_fasta_from_AFDB_parallel(hnh_cath_in_ted_file, output_dir,  all_fasta_output_file):
    """
    Download PDB data and FASTA sequences from AlphaFold database based on TED IDs.

    Args:
        hnh_cath_in_ted_file (str): Path to the CSV file containing TED ID and chopping information.
        output_dir (str): Directory to save the downloaded PDB files.
        

    Returns:
        None
    """
    try:
        
        # Load the HNH_cath_in_ted file
        df = pd.read_csv(hnh_cath_in_ted_file)
        if 'ted_id' not in df.columns:
            raise ValueError("Column 'ted_id' not found in HNH_cath_in_ted file.")

        # Extract unique PDB IDs from the ted_id column
        pdb_ids = (
            df['ted_id']
            .dropna()  # Drop NaN values
            .astype(str)  # Convert to string
            .apply(lambda x: x.split('_')[0] + '_' + x.split('_')[1])  # Split and extract the first part
            .unique()  # Get unique values
        )
        # Initialize a list to store all FASTA sequences
        fasta_sequences = []
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Base URL for downloading PDB data from AlphaFold
        
        # Initialize a list to store all FASTA sequences
        fasta_sequences = []
        
        # Use thread pool to download PDB files in parallel
        
        print("Check Alphafold pdb template should update or not")
        with ThreadPoolExecutor() as pdb_executor:
            # Submit each PDB download task
            pdb_futures = [pdb_executor.submit(download_pdb, pdb_id, output_dir, pdb_url_template = "https://alphafold.ebi.ac.uk/files/{pdb_id}.pdb" ) for pdb_id in pdb_ids]
            
            # Wait for all PDB download tasks to complete
            for future in as_completed(pdb_futures):
                future.result()  # Get each thread's return result, although we don't need it now
        
        
        # Use thread pool to download FASTA sequences in parallel
        with ThreadPoolExecutor() as fasta_executor:
            # Submit each FASTA download task
            fasta_futures = []
            # Use tqdm progress bar to wrap tasks

            for pdb_id in pdb_ids:
                uniprot_id = pdb_id.split('-')[1]# Retrieve UniProt ID from the CSV file
                if pd.notna(uniprot_id):  # Ensure uniprot_id is not empty
                    print(uniprot_id)
                    fasta_futures.append(fasta_executor.submit(download_fasta, uniprot_id, pdb_id,fasta_url_template="https://www.uniprot.org/uniprot/{uniprot_id}.fasta",max_versions=20))
                else:
                    print(f"Skipping FASTA download for PDB {pdb_id} as UniProt ID is NaN")
                    
            
        # Collect all returned FASTA data
            for future in as_completed(fasta_futures):
                fasta_data = future.result()
                if fasta_data:
                    fasta_sequences.append(fasta_data)
        # Write all FASTA sequences to file
        with open(all_fasta_output_file, "w") as fasta_out:
            for fasta in fasta_sequences:
                 fasta_out.writelines(fasta + "\n")
            print(f"FASTA sequences saved to {all_fasta_output_file}.")


    except Exception as e:
        print(f"An error occurred: {e}")
        


def extract_fasta_sequence(fasta_file, pdb_id, domain_id, ranges, output_fasta_path):
    """
    Extract the domain sequence from the full FASTA file and write it to the output FASTA file.

    Args:
        fasta_file (str): Path to the full protein FASTA file.
        pdb_id (str): PDB ID.
        domain_id (str): Domain ID (e.g., '1cnsA01').
        ranges (list): List of residue ranges for the domain.
        output_fasta_path (str): Path to the output FASTA file where domain sequences will be saved.
        
    Returns:
        None
    """
    # Read all records from the fasta file into memory
    with open(fasta_file, "r") as fasta_in:
        fasta_records = list(SeqIO.parse(fasta_in, "fasta"))
    
    # Iterate over all records in memory
    for record in fasta_records:
        # Check if this record matches the given pdb_id
        if record.description.split("__")[1] == pdb_id:
            seq = str(record.seq)  # Convert sequence to string
            domain_sequence = ""
            
            # Extract the domain sequence based on the ranges
            for start_res, end_res in ranges:
                domain_sequence += seq[start_res-1:end_res]
            range_str = "_".join([f"{start}-{end}" for start, end in ranges])
            # Write the domain FASTA sequence to the output file
            with open(output_fasta_path, "a") as fasta_out:  # Open the output file in append mode
                fasta_out.write(f">{domain_id}_range_{range_str}\n{domain_sequence}\n")
                print(f"Saved domain {domain_id} FASTA to {output_fasta_path}")
            break  # We stop after finding the first matching record
def process_pdb_for_domain_AFDB(row, pdb_dir, domain_pdb_dir):
    """Process PDB files and extract domain-level PDB"""
    ted_id = row['ted_id']
    pdb_id = "_".join(ted_id.split('_')[:2])
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    domain_pdb_path = os.path.join(domain_pdb_dir, f"{ted_id}.pdb")
    
    # Skip if PDB file doesn't exist
    
    chopping = row['chopping']
    # Parse chopping information
    ranges = []
    for segment in chopping.split('_'):
        start, end = map(int, segment.split('-'))
        ranges.append((start, end))
    
    # Skip if PDB file doesn't exist
    if not os.path.exists(pdb_path):
        print(f"PDB file {pdb_id} not found. Skipping.")
        return None


    if os.path.exists(domain_pdb_path):
        print(f"domain already exists {domain_pdb_path}. Skipping.")
        return  ted_id, pdb_id, ranges
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    
    # This function assumes you already have ranges in hand
    domain_structure = PDB.Structure.Structure(ted_id)
    for model in structure:
        domain_model = PDB.Model.Model(model.id)
        for chain in model:
            domain_chain = PDB.Chain.Chain(chain.id)
            for res in chain:
                res_id = res.id[1]
                if any(start <= res_id <= end for start, end in ranges):  # Assuming ranges is available
                    domain_chain.add(res.copy())
            if len(domain_chain):
                domain_model.add(domain_chain)

        if len(domain_model):
            domain_structure.add(domain_model)

    # Save the extracted domain to a PDB file
    io = PDB.PDBIO()
    io.set_structure(domain_structure)
    io.save(domain_pdb_path)
    print(f"Saved domain {ted_id} to {domain_pdb_path}.")
    
    return  ted_id, pdb_id, ranges  # Return domain info for FASTA processing



def process_pdb_for_domain_check(row, pdb_dir, domain_pdb_dir):
    """Process PDB files and extract domain-level PDB"""
    ted_id = row['target']
    pdb_id = "_".join(ted_id.split('_')[:2])
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    domain_pdb_path = os.path.join(domain_pdb_dir, f"{ted_id}_check.pdb")
    ted_id_check = ted_id + "_check"
    # Skip if PDB file doesn't exist
    
    chopping = row['chopping_check']
    # Parse chopping information
    ranges = []
    for segment in chopping.split('_'):
        start, end = map(int, segment.split('-'))
        ranges.append((start, end))
    
    # Skip if PDB file doesn't exist
    if not os.path.exists(pdb_path):
        print(f"PDB file {pdb_id} not found. Skipping.")
        return None


    if os.path.exists(domain_pdb_path):
        print(f"domain already exists {domain_pdb_path}. Skipping.")
        return  ted_id_check, pdb_id, ranges
    
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    
    # This function assumes you already have ranges in hand
    domain_structure = PDB.Structure.Structure(ted_id_check)
    for model in structure:
        domain_model = PDB.Model.Model(model.id)
        for chain in model:
            domain_chain = PDB.Chain.Chain(chain.id)
            for res in chain:
                res_id = res.id[1]
                if any(start <= res_id <= end for start, end in ranges):  # Assuming ranges is available
                    domain_chain.add(res.copy())
            if len(domain_chain):
                domain_model.add(domain_chain)

        if len(domain_model):
            domain_structure.add(domain_model)

    # Save the extracted domain to a PDB file
    io = PDB.PDBIO()
    io.set_structure(domain_structure)
    io.save(domain_pdb_path)
    print(f"Saved domain {ted_id_check} to {domain_pdb_path}.")
    
    return  ted_id_check, pdb_id, ranges  # Return domain info for FASTA processing




def get_figure2f_data(input_file, output_file):
    """
    Convert csv1 to csv2 format
    
    Parameters:
    input_file: Input CSV file path
    output_file: Output CSV file path
    """
    # Read input CSV file
    df1 = pd.read_csv(input_file)
    
    # List to store results
    result_data = []
    
    # Weight column names mapped to weight_type
    weight_columns = {
        'FS_weight_Dali': 'FS_weight',
        'FS_weight': 'FS_weight', 
        'TM_weight_Dali': 'TM_weight',
        'TM_weight': 'TM_weight'
    }
    
    # Process each row
    for index, row in df1.iterrows():
        current_id = index + 1  # id starts from 1
        cath_label = row['cath_label']
        
        # Determine first part of source_file
        if '.' in str(cath_label):
            first_part = 'TedCath_'
        else:
            first_part = 'TedCluster_'
            
        # Process four weight values
        for weight_col in weight_columns:
            value = row[weight_col]
            weight_type = weight_columns[weight_col]
            
            # Determine second part of source_file
            if weight_col in ['FS_weight_Dali', 'TM_weight_Dali']:
                second_part = 'Sim_Check'
            else:
                second_part = 'Sim'
                
            source_file = first_part + second_part
            
            # Add to result
            result_data.append({
                'id': current_id,
                'value': value,
                'weight_type': weight_type,
                'source_file': source_file
            })
    
    # Create output DataFrame
    df2 = pd.DataFrame(result_data)
    
    # Save to CSV file
    df2.to_csv(output_file, index=False)
    
    print(f"Conversion complete! Output file: {output_file}")
    print(f"Input rows: {len(df1)}")
    print(f"Output rows: {len(df2)}")
    
    return df2

def split_domain_by_ted_boundary_and_save_fasta_parallel(hnh_cath_in_ted_file, pdb_dir, domain_pdb_dir, domain_fasta_file, fasta_file):
    """
    Split full PDB files into domain-level PDB files and save domain-level FASTA sequences into a single FASTA file.

    Args:
        hnh_cath_in_ted_file (str): Path to the CSV file containing TED IDs and chopping information.
        pdb_dir (str): Directory containing the full PDB files.
        domain_pdb_dir (str): Directory to save domain-level PDB files.
        domain_fasta_file (str): File to save all domain sequences in FASTA format.
        fasta_file (str): Path to the full protein FASTA file.

    Returns:
        None
    """
    # Ensure the domain PDB and FASTA directories exist
    os.makedirs(domain_pdb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(domain_fasta_file), exist_ok=True)  # Ensure domain FASTA file directory exists

    try:
        # Load the HNH_cath_in_ted file
        hnh_cath_df = pd.read_csv(hnh_cath_in_ted_file)
        if 'ted_id' not in hnh_cath_df.columns or 'chopping' not in hnh_cath_df.columns:
            raise ValueError("'ted_id' or 'chopping' column not found in HNH_cath_in_ted file.")
        
                # Read complete FASTA file data
        fasta_records = []
        try:
            with open(fasta_file, "r") as fasta_in:
                fasta_records = list(SeqIO.parse(fasta_in, "fasta"))
        except Exception as e:
            print(f"Error reading FASTA file: {e}")
            return
    
        # Process PDB files and extract domains
        pdb_futures = []
        with ThreadPoolExecutor() as pdb_executor:
            # Submit PDB processing tasks
            pdb_futures = [
                pdb_executor.submit(process_pdb_for_domain_AFDB, row, pdb_dir, domain_pdb_dir)
                for _, row in hnh_cath_df.iterrows()
            ]

        # Collect PDB processing results, and process FASTA files
        domain_fasta_sequences = []
        with ThreadPoolExecutor() as fasta_executor:
            # Submit FASTA processing tasks
            for future in as_completed(pdb_futures):
                result = future.result()
                if result:
                    domain_id, pdb_id, ranges = result
                    # Get domain FASTA sequence
                    fasta_sequence = process_fasta_for_domain(domain_id,  pdb_id.replace('_v6', '_v4'), ranges, fasta_records)
                    if fasta_sequence:
                        domain_fasta_sequences.append(fasta_sequence)

        # Write all FASTA sequences to file
        with open(domain_fasta_file, "w") as fasta_out:
            fasta_out.writelines(domain_fasta_sequences)

        print("Domain-level PDB and FASTA files have been processed and saved.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        

def split_domain_after_check_and_save_fasta_parallel(domain_check_info, pdb_dir, domain_pdb_dir, domain_fasta_file, fasta_file):
    """
    Split full PDB files into domain-level PDB files and save domain-level FASTA sequences into a single FASTA file.

    Args:
        hnh_cath_in_ted_file (str): Path to the CSV file containing TED IDs and chopping information.
        pdb_dir (str): Directory containing the full PDB files.
        domain_pdb_dir (str): Directory to save domain-level PDB files.
        domain_fasta_file (str): File to save all domain sequences in FASTA format.
        fasta_file (str): Path to the full protein FASTA file.

    Returns:
        None
    """
    # Ensure the domain PDB and FASTA directories exist
    os.makedirs(domain_pdb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(domain_fasta_file), exist_ok=True)  # Ensure domain FASTA file directory exists

    try:
        # Load the HNH_cath_in_ted file
        hnh_cath_df = pd.read_excel(domain_check_info)
        hnh_cath_df = hnh_cath_df.dropna(subset=['chopping_check'])

        fasta_records = fasta_dict


        # Process PDB files and extract domains
        pdb_futures = []
        with ThreadPoolExecutor() as pdb_executor:
            # Submit PDB processing tasks
            pdb_futures = [
                pdb_executor.submit(process_pdb_for_domain_check, row, pdb_dir, domain_pdb_dir)
                for _, row in hnh_cath_df.iterrows()
            ]

        # Collect PDB processing results, and process FASTA files
        domain_fasta_sequences = []
        with ThreadPoolExecutor() as fasta_executor:
            # Submit FASTA processing tasks
            for future in as_completed(pdb_futures):
                result = future.result()
                if result:
                    domain_id, pdb_id, ranges = result
                    # Get domain FASTA sequence
                    fasta_sequence = process_fasta_for_domain_check(domain_id, pdb_id, ranges, fasta_records)
                    if fasta_sequence:
                        domain_fasta_sequences.append(fasta_sequence)

        # Write all FASTA sequences to file
        with open(domain_fasta_file, "w") as fasta_out:
            fasta_out.writelines(domain_fasta_sequences)

        print("Domain-level PDB and FASTA files have been processed and saved.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
      
   

def split_domain_by_ted_boundary_and_save_fasta(hnh_cath_in_ted_file, pdb_dir, domain_pdb_dir, domain_fasta_file, fasta_file):
    """
    Split full PDB files into domain-level PDB files and save domain-level FASTA sequences into a single FASTA file.

    Args:
        hnh_cath_in_ted_file (str): Path to the CSV file containing TED IDs and chopping information.
        pdb_dir (str): Directory containing the full PDB files.
        domain_pdb_dir (str): Directory to save domain-level PDB files.
        domain_fasta_file (str): File to save all domain sequences in FASTA format.
        fasta_file (str): Path to the full protein FASTA file.

    Returns:
        None
    """
    # Ensure the domain PDB and FASTA directories exist
    os.makedirs(domain_pdb_dir, exist_ok=True)
    os.makedirs(os.path.dirname(domain_fasta_file), exist_ok=True)  # Ensure domain FASTA file directory exists

    try:
        # Load the HNH_cath_in_ted file
        hnh_cath_df = pd.read_csv(hnh_cath_in_ted_file)
        if 'ted_id' not in hnh_cath_df.columns or 'chopping' not in hnh_cath_df.columns:
            raise ValueError("'ted_id' or 'chopping' column not found in HNH_cath_in_ted file.")

        # Initialize PDB parser and I/O
        parser = PDB.PDBParser(QUIET=True)
        io = PDB.PDBIO()

        # Open the domain FASTA file for writing all sequences
        # Iterate through the data to process each domain
        for _, row in tqdm(hnh_cath_df.iterrows(), desc="Processing TED domains", total=hnh_cath_df.shape[0]):
            ted_id = row['ted_id']
            chopping = row['chopping']

            # Extract the full PDB file ID
            pdb_id = "_".join(ted_id.split('_')[:2])
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")

            if not os.path.exists(pdb_path):
                print(f"PDB file {pdb_path} does not exist, skipping.")
                continue

            # Parse the chopping information (start and end residues)
            ranges = []
            for segment in chopping.split('_'):
                start, end = map(int, segment.split('-'))
                ranges.append((start, end))

            # Parse the PDB file and extract the corresponding domain
            structure = parser.get_structure(pdb_id, pdb_path)
            domain_structure = PDB.Structure.Structure(ted_id)
            for model in structure:
                domain_model = PDB.Model.Model(model.id)
                for chain in model:
                    domain_chain = PDB.Chain.Chain(chain.id)
                    for res in chain:
                        res_id = res.id[1]  # Residue ID number
                        if any(start <= res_id <= end for start, end in ranges):
                            domain_chain.add(res.copy())
                    if len(domain_chain):
                        domain_model.add(domain_chain)

                if len(domain_model):
                    domain_structure.add(domain_model)

            # Save the extracted domain to a PDB file
            domain_pdb_path = os.path.join(domain_pdb_dir, f"{ted_id}.pdb")
            io.set_structure(domain_structure)
            io.save(domain_pdb_path)
            print(f"Saved domain to {domain_pdb_path}")

            # Extract and save the corresponding FASTA sequence for the domain
            extract_fasta_sequence(fasta_file, pdb_id, ted_id, ranges,  domain_fasta_file)

    except Exception as e:
        print(f"An error occurred: {e}")
        
        
        
def split_domain_by_ted_boundary(hnh_cath_in_ted_file, pdb_dir, domain_dir):
    """
    Extract domains from complete PDB data based on chopping information in HNH_cath_in_ted file.

    Parameters:
        hnh_cath_in_ted_file (str): CSV file path containing ted_id and chopping columns.
        pdb_dir (str): Directory storing complete PDB files.
        domain_dir (str): Directory to save extracted domain PDB files.

    Returns:
        None
    """
    try:
        # Read HNH_cath_in_ted file
        hnh_cath_df = pd.read_csv(hnh_cath_in_ted_file)
        if 'ted_id' not in hnh_cath_df.columns or 'chopping' not in hnh_cath_df.columns:
            raise ValueError("'ted_id' or 'chopping' column not found in HNH_cath_in_ted file.")

        # Ensure output directory exists
        os.makedirs(domain_dir, exist_ok=True)

        # Initialize PDB parser
        parser = PDB.PDBParser(QUIET=True)
        io = PDB.PDBIO()

        # Traverse each row to extract domain
        for _, row in tqdm(hnh_cath_df.iterrows(), desc="Extracting domains", total=hnh_cath_df.shape[0]):
            ted_id = row['ted_id']
            chopping = row['chopping']

            # Extract complete PDB file ID
            pdb_id = "_".join(ted_id.split('_')[:2])
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")

            if not os.path.exists(pdb_path):
                print(f"PDB file {pdb_path} does not exist, skipping.")
                continue

            # Parse PDB file
            structure = parser.get_structure(pdb_id, pdb_path)

            # Parse chopping information
            ranges = []
            for segment in chopping.split('_'):
                start, end = map(int, segment.split('-'))
                ranges.append((start, end))

            # Extract residues in specified ranges
            domain_structure = PDB.Structure.Structure(ted_id)
            for model in structure:
                domain_model = PDB.Model.Model(model.id)
                for chain in model:
                    domain_chain = PDB.Chain.Chain(chain.id)
                    for res in chain:
                        res_id = res.id[1]  # residue id number
                        if any(start <= res_id <= end for start, end in ranges):
                            domain_chain.add(res.copy())
                    if len(domain_chain):
                        domain_model.add(domain_chain)
                if len(domain_model):
                    domain_structure.add(domain_model)

            # Save extracted domain to file
            output_path = os.path.join(domain_dir, f"{ted_id}.pdb")
            io.set_structure(domain_structure)
            io.save(output_path)
            print(f"Saved domain to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


def add_domain_range_with_ted_info(cluster_file, domain_summary_file, output_file, chunksize=100000):
    """
    Expand each row of data in cluster_file, find rows corresponding to Cluster_member in domain_summary_file,
    and concatenate these rows' data to cluster_file, output as new file.

    Parameters:
        cluster_file (str): CSV file path containing Cluster_representative and Cluster_member columns.
        domain_summary_file (str): TED domain data file path.
        output_file (str): File path to save expanded data.
        chunksize (int): Number of rows to read each time from domain_summary_file.

    Returns:
        None
    """
    try:
        # Read Cluster file
        cluster_df = pd.read_csv(cluster_file)
        if 'Cluster_member' not in cluster_df.columns:
            raise ValueError("'Cluster_member' column not found in cluster_file.")

        # Get all unique Cluster_member values
        cluster_members = set(cluster_df['Cluster_member'].dropna().unique())

        # Initialize expanded data storage
        expanded_data = []

        # Define column names for domain_summary_file
        column_names = [
            "ted_id", "md5_domain", "consensus_level", "chopping", "nres_domain",
            "num_segments", "plddt", "num_helix_strand_turn", "num_helix", "num_strand",
            "num_helix_strand", "num_turn", "proteome_id", "cath_label", "cath_assignment_level",
            "cath_assignment_method", "packing_density", "norm_rg", "tax_common_name",
            "tax_scientific_name", "tax_lineage"
        ]

        # Read domain_summary_file in chunks and match data
        for chunk in tqdm(pd.read_csv(domain_summary_file, sep='\t', names=column_names, header=0, chunksize=chunksize),
                          desc="Processing domain summary file"):
            # Filter rows matching cluster_members
            filtered_chunk = chunk[chunk['ted_id'].isin(cluster_members)]

            # Concatenate to result data
            if not filtered_chunk.empty:
                expanded_data.append(filtered_chunk)

        # Merge all matching rows
        if expanded_data:
            expanded_df = pd.concat(expanded_data)

            # Merge expanded data with cluster_file by Cluster_member
            merged_df = cluster_df.merge(expanded_df, left_on='Cluster_member', right_on='ted_id', how='left')

            # Save merged data to output file
            merged_df.to_csv(output_file, index=False)
            print(f"Expanded data saved to {output_file}.")
        else:
            print("No matching TED information found.")

    except Exception as e:
        print(f"An error occurred: {e}")
        
# Function to parse a single query_id.txt file
def parse_dali_result_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize variables
    query_id = ""
    data = []
    equivalences_query = []
    equivalences_target = []
    
    # Process each line in the file
    for i in range(len(lines)):
        # Check for query_id in the file
        line = lines[i]
        if line.startswith("# Query:"):
            query_id = line.strip().split(":")[1].strip()
        
        # Check for the result information
        if line.startswith("# No:"):
            # Extract the result data lines that follow the # No: section
            result_lines = lines[lines.index(line) + 1:]
            for result in result_lines:
                if result.startswith("# Structural equivalences"):
                    break
                else:
                    result_parts = result.split()
                    target_id = result_parts[1]
                    Z = result_parts[2]
                    RMSD = result_parts[3]
                    LALI = result_parts[4]
                    NRES = result_parts[5]
                    ID = result_parts[6]
                    i+=1
                    
                    # Append the extracted data to the list
                    data.append([query_id, target_id, Z, RMSD, LALI, NRES, ID])
                    
            break
    equivalences_query = [[] for _ in range(len(data))]
    equivalences_target = [[] for _ in range(len(data))]
    for j in range(i,len(lines)):
        # Process structural equivalences
        line = lines[j]
        if line.startswith("# Structural equivalences"):
            equivalence_lines = lines[lines.index(line) + 1:]
            eq_id = '1'
            for eq_line in equivalence_lines:
                if eq_line.strip():
                    if "# Translation-rotation matrices" in eq_line:
                        break
                    else:
                        eq_id =  eq_line.split(':')[0]
                        query_equiv_match =  re.findall(r'(\d+)\s* - \s*(\d+)', eq_line.split("<=>")[0])
                        query_equiv_match = '-'.join(query_equiv_match[0])
                        target_equiv_match = re.findall(r'(\d+)\s* - \s*(\d+)', eq_line.split("<=>")[1])
                        target_equiv_match = '-'.join(target_equiv_match[0])
                        if query_equiv_match and target_equiv_match:
                            equivalences_query[int(eq_id)-1].append(query_equiv_match)
                            equivalences_target[int(eq_id)-1].append(target_equiv_match)
            break
    
    # Add equivalences to the data
    for i in range(len(data)):
        data[i].append(equivalences_query[i])
        data[i].append(equivalences_target[i])
    
    return data


# Function to process all the .txt files in the given directory and save to CSV
def dali_results_to_csv(directory_path, output_csv_path):
    all_data = []
    
    # Iterate through all .txt files in the specified directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            file_data = parse_dali_result_txt(file_path)
            all_data.extend(file_data)
    
    # Create a DataFrame from the collected data
    #columns = ["query_id", "target_id", "Z", "RMSD", "LALI", "NRES", "%ID", "query_Equivalences", "target_Equivalences"]
    columns = ["source", "target", "weight", "RMSD", "LALI", "NRES", "%ID", "query_Equivalences", "target_Equivalences"]

    df = pd.DataFrame(all_data, columns=columns)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)        


def merge_results_by_source_target_new(FS_cath_teddb, TM_cath_teddb, Dali_cath_teddb, output_csv):
    # Initialize df_merged as empty DataFrame
    df_merged = pd.DataFrame()

    # Read FS_cath_teddb, if path is not empty and file exists and is not empty
    if FS_cath_teddb and os.path.exists(FS_cath_teddb):
        df1 = pd.read_csv(FS_cath_teddb)
        if not df1.empty:
            df1['source'] = df1['source'].str.strip()
            df1['target'] = df1['target'].str.strip()
            df1_rename = df1.rename(columns={col: f"FS_{col}" for col in df1.columns if col not in ['source', 'target']})
            if df_merged.empty:
                df_merged = df1_rename
            else:
                df_merged = df_merged.merge(df1_rename, on=['source', 'target'], how='outer')

    # Read TM_cath_teddb, if path is not empty and file exists and is not empty
    if TM_cath_teddb and os.path.exists(TM_cath_teddb):
        df2 = pd.read_csv(TM_cath_teddb)
        if not df2.empty:
            df2['source'] = df2['source'].str.strip()
            df2['target'] = df2['target'].str.strip()
            df2_rename = df2.rename(columns={col: f"TM_{col}" for col in df2.columns if col not in ['source', 'target']})
            if df_merged.empty:
                df_merged = df2_rename
            else:
                df_merged = df_merged.merge(df2_rename, on=['source', 'target'], how='outer')

    # Read Dali_cath_teddb, if path is not empty and file exists and is not empty
    if Dali_cath_teddb and os.path.exists(Dali_cath_teddb):
        df3 = pd.read_csv(Dali_cath_teddb)
        if not df3.empty:
            df3['source'] = df3['source'].str[:-1].str.strip()
            df3['target'] = df3['target'].str[:-2].str.strip()
            df3_rename = df3.rename(columns={col: f"Dali_{col}" for col in df3.columns if col not in ['source', 'target']})
            if df_merged.empty:
                df_merged = df3_rename
            else:
                df_merged = df_merged.merge(df3_rename, on=['source', 'target'], how='outer')

    # If df_merged is not empty, perform numerical processing and save
    if not df_merged.empty:
        df_merged['FS_weight'] = df_merged.get('FS_weight', pd.Series()).astype(float).round(5)
        df_merged['TM_weight'] = df_merged.get('TM_weight', pd.Series()).astype(float).round(5)
        df_merged['Dali_weight'] = df_merged.get('Dali_weight', pd.Series()).astype(float).round(5)
        
        # Save merged data as CSV file
        df_merged.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Data successfully merged and saved as {output_csv}")
    else:
        print("No data to merge, output file is empty.")


def merge_results_by_source_target(FS_cath_teddb, TM_cath_teddb, Dali_cath_teddb, output_csv):
    # Initialize df_merged as empty DataFrame
    df_merged = pd.DataFrame()
    
    # Read three CSV files
    df1 = pd.read_csv(FS_cath_teddb)
    df2 = pd.read_csv(TM_cath_teddb)
    df3 = pd.read_csv(Dali_cath_teddb)
    df3['source'] = df3['source'].str[:-1]
    df3['target'] = df3['target'].str[:-2]
    
    # Keep source and target columns and rename other columns, add FS_, TM_, Dali_ prefixes respectively
    df1_rename = df1.rename(columns={col: f"FS_{col}" for col in df1.columns if col not in ['source', 'target']})
    df2_rename = df2.rename(columns={col: f"TM_{col}" for col in df2.columns if col not in ['source', 'target']})
    df3_rename = df3.rename(columns={col: f"Dali_{col}" for col in df3.columns if col not in ['source', 'target']})
    
    # Merge datasets, prioritize merging by source and target columns
    df_merged = df1_rename.merge(df2_rename, on=['source', 'target'], how='outer')
    df_merged = df_merged.merge(df3_rename, on=['source', 'target'], how='outer')
    
    df_merged['FS_weight'] = df_merged['FS_weight'].astype(float)
    df_merged['TM_weight'] = df_merged['TM_weight'].astype(float)
    df_merged['Dali_weight'] = df_merged['Dali_weight'].astype(float)
    
    df_merged['FS_weight'] = df_merged['FS_weight'].round(5)
    df_merged['TM_weight'] = df_merged['TM_weight'].round(5)
    df_merged['Dali_weight'] = df_merged['Dali_weight'].round(5)
    
    
    # Save merged data as new CSV file
    df_merged.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"Data successfully merged and saved as {output_csv}")


def filter_sequence_len_for_csv(csv1, column_name, value1, value2, csv2):
    """
    Read specified column from csv1 file, filter rows greater than value1 and less than value2, save to csv2 file.
    
    :param csv1: Input CSV file path
    :param column_name: Column name to filter
    :param value1: Lower limit value (greater than this value)
    :param value2: Upper limit value (less than this value)
    :param csv2: Output CSV file path
    """
    # Read csv1 file
    df = pd.read_csv(csv1)
    df['FS_weight'] = df['FS_weight'].astype(float)
    df['TM_weight'] = df['TM_weight'].astype(float)
    df['Dali_weight'] = df['Dali_weight'].astype(float)
    
    df['FS_weight'] = df['FS_weight'].round(5)
    df['TM_weight'] = df['TM_weight'].round(5)
    df['Dali_weight'] = df['Dali_weight'].round(5)
    
    # Ensure target column is numeric
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')  # Set values that cannot be converted to numbers as NaNdf['TM_weight'] = df['TM_weight'].astype(float)

    # Filter rows greater than value1 and less than value2
    filtered_df = df[(df[column_name] >= value1) & (df[column_name] <= value2)]
    
    # Save result to csv2 file
    filtered_df.to_csv(csv2, index=False,float_format='%.5f')
    
    print(f"Filtered data has been saved to {csv2}")


import requests
import pandas as pd

def fetch_uniprot_batch(ids, fields, out_path, batch_size=10):
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    header_written = False

    with open(out_path, "w") as out:
        for i in range(0, len(ids), batch_size):
            chunk = ids[i:i+batch_size]
            query = " OR ".join(f"accession:{uid}" for uid in chunk)
            params = {
                "format": "tsv",
                "fields": ",".join(fields),
                "query": f"({query})"
            }

            # Print current query ID count
            print(f"Requesting batch {i//batch_size + 1} with {len(chunk)} IDs.")

            r = requests.get(base_url, params=params)

            # Check if request was successful
            if r.status_code != 200:
                print(f"Error: Received status code {r.status_code} for batch {i//batch_size + 1}")
                continue  # If request fails, skip this batch

            r_text = r.text.strip()
            if not r_text:  # If return is empty, means no results
                print(f"Warning: No results found for batch {i//batch_size + 1}")
                continue

            lines = r_text.split("\n")

            if not header_written:
                out.write("\n".join(lines) + "\n")
                header_written = True
            else:
                out.write("\n".join(lines[1:]) + "\n")

    # Read saved results
    df = pd.read_csv(out_path, sep="\t")
    return df





    
def filter_weight_for_csv(csv1,  value1, value2,  value3, csv2):
    """
    Filter rows meeting conditions from csv1 file, save to csv2 file.
    
    :param csv1: Input CSV file path
    :param value1: FS_weight column lower limit value (greater than or equal to this value)
    :param value2: TM_weight column lower limit value (greater than or equal to this value)
    :param value3: Dali_weight column lower limit value (greater than or equal to this value)
    :param csv2: Output CSV file path
    """
    # Read csv1 file
    df = pd.read_csv(csv1)
    
    """ View distribution of a certain value
    df['FS_weight'] = pd.to_numeric(df['FS_weight'], errors='coerce')
    bins = pd.cut(df['FS_weight'], bins=10)
    value_counts = bins.value_counts(normalize=True).sort_index()
    print(value_counts)
    """
    
    # Replace None or NaN values with 0
    df['FS_weight'] = df['FS_weight'].fillna(0)
    df['TM_weight'] = df['TM_weight'].fillna(0)
    df['Dali_weight'] = df['Dali_weight'].fillna(0)
    
    # Filter rows meeting conditions
    filtered_df = df[
        (df['FS_weight'] >= value1) &
        (df['TM_weight'] >= value2) &
        (df['Dali_weight'] >= value3)
    ]
    
    # Save qualified rows as csv2 file
    filtered_df.to_csv(csv2, index=False)
    
    print(f"Qualified data has been saved to {csv2}")

def update_data_with_wet_id_batch2(data1_path, data2_path):
    # Read data2 file (assume Excel)
    data2 = pd.read_excel(data2_path)
    # Extract unique markers from data2's source and target columns and wet_ID
    data2['unique_key'] = data2['source'].str.split('_').str[0] + '_'+ data2['target'].str.split('-').str[1]
    #wet_id_dict = dict(zip(data2['unique_key'], data2['wet_ID']))
    wet_id_dict = (
    data2
    .groupby('unique_key')['wet_ID']
    .agg(lambda ids: '+'.join(ids.astype(str)))
    .to_dict()
    )
    # Determine if data1 is Excel or CSV file
    if data1_path.endswith('.xlsx'):  # If Excel file

        # Read all sheets in Excel file
        excel_file = pd.ExcelFile(data1_path)
        os.remove( data1_path)
        with pd.ExcelWriter(data1_path, engine='openpyxl') as writer:
            for sheet_name in excel_file.sheet_names:
                df = excel_file.parse(sheet_name)
                # Add wet_ID column for each row
                df['unique_key'] = df['source'].str.split('_').str[0] + '_'+ df['target'].str.split('-').str[1]
                df['wet_ID'] = df['unique_key'].map(wet_id_dict)  # Map wet_ID based on unique_key
                
                # Delete intermediate column unique_key
                df.drop(columns=['unique_key'], inplace=True)
                df = df[['wet_ID'] + [col for col in df.columns if col != 'wet_ID']]
                # Save updated data to original Excel file
                df.to_excel(writer, sheet_name=sheet_name+'_wet_ID', index=False)
            
    elif data1_path.endswith('.csv'):  # If CSV file
        # Read CSV file
        df = pd.read_csv(data1_path)
        # Add wet_ID column for each row
        df['unique_key'] = df['source'].str.split('_').str[0] + '_'+df['target'].str.split('-').str[1]
        df['wet_ID'] = df['unique_key'].map(wet_id_dict)  # Map wet_ID based on unique_key
        
        # Delete intermediate column unique_key
        df.drop(columns=['unique_key'], inplace=True)
        df = df[['wet_ID'] + [col for col in df.columns if col != 'wet_ID']]
        # Save updated CSV file
        df.to_csv(data1_path, index=False)
    
    print(f"File {data1_path} has been updated.")
    
import requests
import time
import pandas as pd
from functools import lru_cache








def filter_for_nonCas9(csv_path, output_csv2, col_name,
                       filter_rules=None, 
                       dali_range=(0, float('inf'))):
    """
    Read csv_path file, filter according to rules:
    1) target_info column: Filter out rows containing keywords based on filter_rules.
       filter_rules is a list, elements are OR relationship; each element is string list, internal is AND relationship.
       Example: [["crispr",], ["HNH","domain-containing"]]
    2) domain_seq_sim_Dali column: Keep rows with values in dali_range [A1, A2] range.
    Save filtered result to output_csv2.
    """
    #df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path, encoding='latin1')

    # Text filtering
    if filter_rules is None:
        filter_rules = [["crispr"], ["csn1"], ["cas9"], ["HNH", "domain-containing"]]
        
        
    # If no pdb_id column, first add an empty column
    if 'pdb_id' not in df.columns:
        df['pdb_id'] = ''

    # Mark exempt rows: pdb_id contains '_D'
    mask_exempt = df['pdb_id'].astype(str).str.contains('_D')

    # Define text filtering function
    def to_exclude(text):
        txt = str(text).lower()
        for rule in filter_rules:
            if all(keyword.lower() in txt for keyword in rule):
                return True
        return False

    # Perform text filtering on non-exempt rows (True means keep)
    mask_text = (~mask_exempt) & df[col_name].apply(lambda x: not to_exclude(x))

    # Dali range filtering (True means keep)
    a1, a2 = dali_range
    mask_dali = (~mask_exempt) & df['domain_seq_sim_Dali'].between(a1, a2, inclusive='both')

    # Final keep: exempt rows OR (text & Dali)
    final_mask = mask_exempt | (mask_text & mask_dali)
    df_filtered = df[final_mask].copy()

    # Write out
    df_filtered.to_csv(output_csv2, index=False)
    print(f"Filtered data saved to {output_csv2}: {len(df_filtered)} rows.")

from Bio.PDB import MMCIFParser, PDBIO

def convert_cif_to_pdb(input_dir, output_dir):
    # Ensure output directory exists, create if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create parser and output objects
    parser = MMCIFParser()
    io = PDBIO()

    # Traverse all .cif files in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".cif"):
            cif_path = os.path.join(input_dir, filename)
            pdb_filename = filename.replace(".cif", ".pdb")
            pdb_path = os.path.join(output_dir, pdb_filename)

            # Parse .cif file and save as .pdb file
            try:
                structure = parser.get_structure(filename, cif_path)
                io.set_structure(structure)
                io.save(pdb_path)
                print(f"Converted {filename} to {pdb_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")


def create_fs_cytoscape_network(fs_file, output_file):
    """
    Create Cytoscape network file from Foldseek result .m8 file.

    :param fs_file: Foldseek result .m8 file path
    :param output_file: Output Cytoscape network file path
    """
    try:
        # Check if file exists and is not empty
        if os.path.exists(fs_file) and os.path.getsize(fs_file) > 0:
            # Read Foldseek result file
            # Read Foldseek result file
            fs_results = pd.read_csv(fs_file, sep="\t", header=None)
            #fs_results.columns = ['query', 'target', 'qstart', 'qend', 'qlen', 
            #                      'tstart', 'tend', 'tlen', 'qcov', 'tcov', 
            #                      'bits', 'evalue']
            print("processing fs network")
            fs_results.columns = ['query','target','fident','alnlen','mismatch','gapopen','qstart','qend','qlen','tstart',
                                'tend','tlen','qcov','tcov','bits','evalue','qca','tca','alntmscore','qtmscore',
                                'ttmscore','u','t','lddt','lddtfull','prob']

            fs_results = fs_results.drop(columns=['qca', 'tca','lddt','lddtfull'])
            # Remove '_' and characters after it in query and target columns
            #fs_results.loc[:, 'query'] = fs_results['query'].str.split('_').str[0] # HNH process doesn't need to remove '_' and characters after
            #fs_results.loc[:, 'target'] = fs_results['target'].str.split('_').str[0]# HNH process doesn't need to remove '_' and characters after
            
            # Remove rows where target=source
            print(f"len of original fs{len(fs_results)}")
            fs_results = fs_results[fs_results['query'] != fs_results['target']]
            print(f"len of target!=source fs{len(fs_results)}")
            # Create temporary sort key (not saved as new column)
            """
            fs_results['_key'] = fs_results.apply(
                lambda x: tuple(sorted([x['query'], x['target']])), 
                axis=1
            )

            # Remove duplicate rows
            fs_results = fs_results.drop_duplicates(subset=['_key'])

            # Remove temporary column
            fs_results = fs_results.drop('_key', axis=1)
            print(f"len of final fs{len(fs_results)}")
            """
            fs_results['weight'] = (fs_results['qcov'] + fs_results['tcov']) / 2
            # Adjust column order
            cols_order = ['query', 'target', 'weight', 'evalue', 'bits', 'qcov', 'tcov']
            other_cols = [col for col in fs_results.columns if col not in cols_order]
            fs_results = fs_results[cols_order + other_cols]

            # Rename 'query' column to 'source'
            fs_results = fs_results.rename(columns={'query': 'source'})

            fs_results.to_csv(output_file, index=False)
        else:
            print(f"Warning: Foldseek result file {fs_file} does not exist or is empty, creating empty result file")
            # Create empty DataFrame and save
            empty_df = pd.DataFrame(columns=['source', 'target', 'weight', 'alntmscore'])
            empty_df.to_csv(output_file, index=False)
            return
    
    except pd.errors.EmptyDataError:
        print(f"Warning: Foldseek result file {fs_file} is empty, creating empty result file")
        # Create empty DataFrame and save
        empty_df = pd.DataFrame(columns=['source', 'target', 'weight', 'alntmscore'])
        empty_df.to_csv(output_file, index=False)
        return

def convert_pdb_to_foldseek_db(fs_bin,pdb_dir, fs_db_dir, fs_db_name="fs_db"):
    fs_querydb_path = os.path.join(fs_db_dir, fs_db_name+".db")
    ensure_dir(fs_db_dir)
    subprocess.run(
        [
            fs_bin,
            "createdb",
            "--input-format",
            "1",
            pdb_dir,
            fs_querydb_path,
        ],
        stderr=subprocess.DEVNULL,
        check=True,
    )


from cath_alphaflow.settings import get_default_settings

config = get_default_settings()
logging.basicConfig(level=logging.DEBUG)
FS_BINARY_PATH = config.FS_BINARY_PATH
FS_TMP_PATH = config.FS_TMP_PATH
FS_OVERLAP = config.FS_OVERLAP
DEFAULT_FS_COV_MODE = "0" # overlap over query and target
DEFAULT_FS_ALIGNER = "2" # 3di+AA (fast, accurate)
DEFAULT_FS_FORMAT_OUTPUT = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,qlen,tstart,tend,tlen,qcov,tcov,bits,evalue,qca,tca,alntmscore,qtmscore,ttmscore,u,t,lddt,lddtfull,prob"


  
def run_foldseek(fs_bin,fs_querydb, fs_targetdb, fs_rawdata="./fs_query_structures.raw", fs_results="./fs_query_results.m8", tmp_dir=FS_TMP_PATH, cov_mode=DEFAULT_FS_COV_MODE, coverage=FS_OVERLAP, alignment_type=DEFAULT_FS_ALIGNER):
    "Run Foldseek Query DB against Target DB"
    #alignment_type = 1  0：3Di Gotoh-Smith-Waterman（局部，不推荐），1：TMalign（全局，慢），2：3Di+AA Gotoh-Smith-Waterman（局部，默认）
    #ensure_dir(fs_results)
    assert str(fs_rawdata) != ''
    subprocess.run(
        [ "nice", "-n", "0", "ionice", "-c", "2", "-n", "0",
            fs_bin,
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
            fs_bin,
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
    

def merge_fasta_files(fasta_files, output_file):
    """Merge multiple FASTA files into one"""
    logger.info(f"Merging FASTA files: {fasta_files} -> {output_file}")
    
    with open(output_file, 'w') as outfile:
        for fasta_file in fasta_files:
            if os.path.exists(fasta_file):
                logger.debug(f"Adding FASTA file: {fasta_file}")
                with open(fasta_file, 'r') as infile:
                    outfile.write(infile.read())
                # Add a newline between files to ensure proper separation
                outfile.write('\n')
            else:
                logger.warning(f"FASTA file not found: {fasta_file}")
    
    logger.info(f"Merged FASTA file created: {output_file}")
    return output_file
