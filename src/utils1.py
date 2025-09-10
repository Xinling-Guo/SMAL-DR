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
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MAX_RETRIES = 5
RETRY_DELAY = 5  # 延迟时间，单位：秒
TED_info_URL =  "https://ted.cathdb.info/api/v1"

def create_session_with_retries(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 503, 504)):
    """创建带有重试机制的会话"""
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
    """下载UniProt数据，带有重试机制"""
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
                        
                        # 检查是否有下一页
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
                delay *= 2  # 指数退避
            else:
                logger.error(f"All {max_retries} attempts failed for {ipr_id}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False
    
    return False

def download_interpro_data(ipr_id, output_file, max_retries=3, delay=2):
    """下载InterPro数据，带有重试机制"""
    url = f"https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/pfam/{ipr_id}/?page_size=200"
    
    session = create_session_with_retries(retries=max_retries)
    attempt = 0
    
    while attempt < max_retries:
        try:
            logger.info(f"Fetching InterPro data for {ipr_id}")
            response = session.get(url, headers={"Accept": "application/json"}, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # 解析数据（保持原有逻辑）
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
                
                # 保存到TSV文件
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

# 在你的主函数中调用时，可以这样使用：
def get_true_label_pdb_and_domain(ipr_ids, ipr_names, output_dir, keyword):
    """优化版本的主函数"""
    domain_names = ["REC", "NUC", "Bridge", "RuvC", "HNH", "PI", "BH"] 
    directory = os.path.join(output_dir, "Pfam_domain_reviewed/")
    ensure_dir(directory)

    for i in range(len(ipr_ids)):
        output_file_un = os.path.join(directory, f"uniprot_{ipr_names[i]}_{ipr_ids[i]}.tsv")
        output_file_in = os.path.join(directory, f"interpro_{ipr_names[i]}_{ipr_ids[i]}.tsv")
        
        # 下载UniProt数据（带重试）
        success_uniprot = download_uniprot_data_pages(ipr_ids[i], output_file_un, max_retries=3, delay=2)
        if not success_uniprot:
            logger.error(f"Failed to download UniProt data for {ipr_ids[i]}")
            continue
        
        # 下载InterPro数据（带重试）
        success_interpro = download_interpro_data(ipr_ids[i], output_file_in, max_retries=3, delay=2)
        if not success_interpro:
            logger.error(f"Failed to download InterPro data for {ipr_ids[i]}")
            continue
        
        # 短暂暂停，避免请求过于频繁
        time.sleep(1)
    
    # 解析并合并所有 UniProt 数据
    output_file = os.path.join(directory, "uniprot_all_domains.tsv")
    unique_file = os.path.join(directory, "uniprot_unique_domains.tsv")
    # 如果已经得到了uniprot和interprt的信息和domain信息，则不需要再解析
    parse_uniprot_tsv_files(directory, output_file)
    parse_interpro_tsv_files(directory, output_file)
    uniprot_domain_data(output_file, unique_file, keyword,domain_names)
    print("Note: Domain range requires human adjustment!!!")
    print("Get true labels completed.")
    
    
    
def extract_cc_domain_ranges(text):
    domain_names = ["REC", "NUC", "Bridge", "RuvC", "HNH", "PI", "BH"]
    cc_range_pattern = re.compile(r"(\d+-\d+)")
    # 更新正则表达式，支持提取多个区间
    cc_domain_pattern = re.compile(
        r"(?P<domain>{})(.*?)(?=\b(?:{}|$))".format("|".join(domain_names), "|".join(domain_names))
    )
    if not pd.isna(text):
        extracted_domains = []
        
        # 匹配功能域和它们的区间段
        domain_matches = cc_domain_pattern.finditer(text)
        
        for domain_match in domain_matches:
            domain_name = domain_match.group('domain')
            domain_content = domain_match.group(0)
            
            # 匹配所有数字-数字的组合
            ranges = cc_range_pattern.findall(domain_content)
            
            if ranges:
                # 将数字区间组合为字符串并与功能域名称关联
                range_str = "_".join(ranges)
                extracted_domains.append(f"{domain_name}:{range_str}")
        
        return ";".join(extracted_domains)
    else:
        return ";"
def get_cif_pdb(id_info_file, key,output_dir):
    id_info_df = pd.read_csv(id_info_file)
    pdb_ids = id_info_df['PDB ID'].tolist()  # 获取所有的 PDB ID
    protein_ids = id_info_df['UniProt ID'].tolist()  # 获取所有的 UniProt ID
    
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
        
        # 如果对应的 PDB ID 不是 None，尝试从 PDB 下载
        if pd.notna(pdb_id):
            source = download_pdb_structure(pdb_id, protein_id, pdb_dir, cif_dir)
        else:
            source = None  # 先将 source 初始化为 None，以便后续检查

        # 如果 PDB 没有数据，尝试从 AlphaFold 下载
        if not source:
            source = download_alphafold_structure(protein_id, pdb_dir, cif_dir)
        
        # 如果两个地方都没有数据，标记为"Not Found"
        if not source:
            source = "Not Found"
            print(f"!!!!!!!!Not Found {str(id_i)}: {protein_id}...")
        
        # 保存结果
        results.append({
            "UniProt ID": protein_id,
            "PDB ID": pdb_id,
            "PDB Source": source
        })
        
        # Convert current results to DataFrame
        
def parse_uniprot_tsv_files(directory, output_file):
    all_data = []

    # 遍历文件夹中的每个文件
    for filename in os.listdir(directory):
        # 检查文件是否以 "uniprot" 开头并且以 ".tsv" 结尾
        if filename.endswith(".tsv") and filename.startswith("uniprot"):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {filename}")
            
            # 读取TSV文件
            df = pd.read_csv(file_path, sep='\t')
            
            # 提取Domain [CC] 和 Domain [FT]的功能域及残基编号
            df['Domain [CC] Extracted'] = df['Domain [CC]'].apply(extract_cc_domain_ranges)
            df['Domain [FT] Extracted'] = df['Domain [FT]'].apply(extract_ft_residues)
            
            # 增加文件名列
            df['File Name'] = filename
            
            # 将结果添加到总数据中
            all_data.append(df)
    
    # 合并所有数据并保存为新的文件
    result_df = pd.concat(all_data, ignore_index=True)
    
    # 只保留需要的列并保存到输出文件
    result_df.to_csv(output_file, sep='\t', index=False, quoting=csv.QUOTE_NONE, escapechar='\\')
    

# 提取Domain [FT]的功能域和残基区间
def extract_ft_residues(text):
    residue_pattern_ft = re.compile(r"DOMAIN\s+(\d+)\.\.(\d+);\s*/note=\"(.*?)\"", re.ASCII)
    if pd.isnull(text):
        return ""
    
    # 查找符合 "DOMAIN x..y" 和功能域的内容
    matches = residue_pattern_ft.findall(text)
    if matches:
        extracted_domains = []
        for match in matches:
            start, end, domain = match
            domain = domain.split()[0]  # 提取功能域名称
            extracted_domains.append(f"{domain}:{start}-{end}")
        return ";".join(extracted_domains)
    return ""
   
def uniprot_domain_data(input_file, output_file,keyword,domain_names):
    """
    处理蛋白质ID数据，合并相同蛋白质ID下的功能域信息，并生成新的TSV文件。

    参数：
    input_file: 输入的TSV文件路径
    output_file: 输出的TSV文件路径
    """
    
    # 读取数据文件
    df = pd.read_csv(input_file, sep='\t')

    # 新增存储列
    df['Domain_CC_Combined'] = ''
    df['Domain_FT_Combined'] = ''
    df['Domain_CC_Different'] = ''
    df['Domain_FT_Different'] = ''

    # 按蛋白质ID分组
    grouped = df.groupby('Entry')
    unique_entries = []
    # 分别处理Domain [CC] Extracted和Domain [FT] Extracted
    for name, group in grouped:
        entry_name = group.iloc[0]["Entry Name"]
        protein_name = group.iloc[0]["Protein names"]
        if keyword.lower() in str(entry_name).lower() or keyword.lower() in str(protein_name).lower():
            combined_cc, different_cc = combine_domains(group, 'Domain [CC] Extracted', 'File Name',domain_names)
            combined_ft, different_ft = combine_domains(group, 'Domain [FT] Extracted', 'File Name', domain_names)

            # 保留唯一的 entry 数据
            first_row = group.iloc[0].copy()
            first_row['Domain_CC_Combined'] = combined_cc
            first_row['Domain_FT_Combined'] = combined_ft
            first_row['Domain_CC_Different'] = different_cc
            first_row['Domain_FT_Different'] = different_ft

            # 添加到结果列表
            unique_entries.append(first_row)
    # 创建新的DataFrame，保存唯一的Entry
    result_df = pd.DataFrame(unique_entries)
    # 保存为新的TSV文件
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
                if ':' in domain:  # 过滤不符合格式的域
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
    
    os.makedirs(save_dir, exist_ok=True)  # 创建保存目录（如果不存在）
    base_url_unisave = "https://rest.uniprot.org/unisave/"
    for uniprot_id in uniprot_ids:
        txt_url = f"{base_url_unisave}{uniprot_id}?format=txt"
        response = requests.get( txt_url)

        if response.status_code == 200:
            lines = response.text.splitlines()
            file_path = os.path.join(save_dir, f"{uniprot_id}_uniprot_details.txt")  # 文件路径
            with open(file_path, 'w') as file:
                file.write("\n".join(lines))  # 保存所有行到文件
            print(f"Saved data for {uniprot_id} to {file_path}")
        else:
            print(f"Error fetching data for UniProt ID {uniprot_id}: {response.status_code}")

def get_pdb_info_from_uniprot(uniprot_ids, uniprot_details_dir,csv_file_path):
    results = []  # 用于存储每个 UniProt ID 的结果

    for uniprot_id in uniprot_ids:
        file_path = os.path.join(uniprot_details_dir, f"{uniprot_id}_uniprot_details.txt")
                
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"File not found for UniProt ID {uniprot_id}: {file_path}")
            results.append((uniprot_id, None, "File not found"))
            continue

    # 将结果写入 CSV 文件

        with open(file_path, 'r') as file:
            lines = file.readlines()
            pdb_id = None
            description = None

            for line in lines:
                if "PDB" in line:
                    if "X-ray" in line:
                        # 提取 PDB ID 和描述
                        pdb_id = line.split(';')[1].strip()  # 获取 PDB ID
                        description = line.strip()  # 获取整行作为描述
                        break  # 找到后退出循环
                    elif "NMR" in line and not pdb_id:
                        # 如果没有找到 X-ray，检查 NMR
                        pdb_id = line.split(';')[1].strip()   # 获取 PDB ID
                        description = line.strip()  # 获取整行作为描述

            # 添加结果到列表
            if pdb_id:
                results.append((uniprot_id, pdb_id, description))
            else:
                results.append((uniprot_id, None, None))
            
        # 将当前结果追加到 CSV 文件
            df = pd.DataFrame([results[-1]], columns=["UniProt ID", "PDB ID", "Description"])
            df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)

    return results

def parse_interpro_tsv_files(directory, output_file):
    write_header = not os.path.exists(output_file)  # 如果文件已存在，则不写入表头

    # 遍历文件夹中的每个文件
    for filename in os.listdir(directory):
        # 检查文件是否以 "interpro" 开头并且以 ".tsv" 结尾
        if filename.endswith(".tsv") and filename.startswith("interpro"):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {filename}")
            # 如果文件为空，跳过并删除文件
            if os.path.getsize(file_path) == 0:
                print(f"Skipping and deleting empty file: {filename}")
                os.remove(file_path)
                continue
            # 读取TSV文件
            df = pd.read_csv(file_path, sep='\t')
            domain_str = file_path.split('/')[-1].split('_')[2]
            # 生成 'Domain [CC] Extracted' 列，格式为 'Domain Name':'Domain Start-End'
            df['Domain [CC] Extracted'] = df.apply(lambda row: f"{domain_str}:{row['Start']}-{row['End']}", axis=1)
            
            # 创建一个新的DataFrame，匹配到所需格式
            result_df = pd.DataFrame({
                'Entry': df['Accession'],  # 对应 Entry
                'Entry Name': df['Protein Name'],  # 对应 Entry Name
                'Protein names': '',  # 空列
                'Domain [CC]': '',  # 空列
                'Domain [FT]': '',  # 空列
                'Organism': '',  # 空列
                'Length': '',  # 空列
                'Domain [CC] Extracted': df['Domain [CC] Extracted'],  # 提取的功能域信息
                'Domain [FT] Extracted': '',  # 空列
                'File Name': filename  # 文件名
            })
            
            # 追加数据到输出文件，设置 mode='a' 和 header=write_header
            result_df.to_csv(output_file, sep='\t', index=False, mode='a', header=write_header, quoting=csv.QUOTE_NONE, escapechar='\\')
            
            # 之后不再写入表头
            write_header = False
def ensure_file(file_path):
    """确保文件存在，如果文件不存在则创建一个空文件"""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            pass  # 创建一个空文件
        print(f"File created: {file_path}")
    else:
        print(f"File already exists: {file_path}")
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
        
        

# step1 function


def fetch_interpro_description(interpro_id):
    """
    根据 InterPro ID 获取描述信息
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
    抓取 UniProt history 页面，解析出所有版本号，返回最大的那个。
    
    """
    url = f"https://rest.uniprot.org/unisave/{uniprot_id}?format=json"
    try:
        r = session.get(url, timeout=10)
        r.raise_for_status()  # 如果请求返回错误状态码，会抛出 HTTPError
        data = r.json()
        # 提取最新版本号
        latest_version = max(entry['entryVersion'] for entry in data['results'])
        return latest_version

    except (requests.RequestException, ValueError, KeyError) as e:
        # 捕获请求异常、JSON 解析错误或缺少 'results' 字段等问题
        print(f"Error fetching or parsing data for {uniprot_id}: {e}")
        return None  # 如果出错，返回 None
def fetch_details_for_uniprot_id(uniprot_id, pdb_id=None, max_versions=20):
    """
    只负责获取 UniProt 原始 TXT，不做任何后续处理
    """

    # 方法1：REST TXT
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
    # 方法2 先调用方法1获取最新版本
    latest_version = get_latest_uniprot_version_by_scraping(uniprot_id)
    time.sleep(1)
    if latest_version:
        # 尝试下载最新版本
        url = f"https://rest.uniprot.org/unisave/{uniprot_id}?format=txt&versions={latest_version}"
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200 and r.text.strip():
                txt= r.text
                txt = str(latest_version)+"\n"+url + "\n"+ txt
                return txt
            else:
                print(f"[Info] 版本 {latest_version} 下载失败，状态码：{r.status_code}")
        except requests.RequestException as e:
            print(f"[Warning] {uniprot_id} 下载版本 {latest_version} 时出错：{e}")

    # 方法2：历史版本回退
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

    raise RuntimeError(f"failed 无法获取 UniProt ID {uniprot_id} 的任何数据")
import glob     
def merge_predict_details(csv_path: str) -> str:
    """
    对指定的主 CSV 文件，寻找同目录下具有相同前缀但后缀带数字的其他 CSV，
    将它们中非空的 details 填入主表的 details 空值处，并保存为新文件。

    参数:
        csv_path: 主 CSV 文件的完整路径

    返回:
        新 CSV 文件的路径
    """
    # 1. 读取主表
    df_main = pd.read_csv(csv_path, dtype=str)
    if 'pdb_id' not in df_main.columns or 'details' not in df_main.columns:
        raise ValueError("输入的 CSV 必须包含 'pdb_id' 和 'details' 列")
    df_main.set_index('pdb_id', inplace=True)

    # 2. 构造同目录下的通配模式，排除自己
    dirpath, filename = os.path.split(csv_path)
    stem, ext = os.path.splitext(filename)
    # 匹配以主文件名前缀开头、后面任意字符再以 .csv 结尾
    pattern = os.path.join(dirpath, f"{stem}*.csv")
    candidates = glob.glob(pattern)
    # 不要把主文件自己也当作候选
    other_files = [f for f in candidates if os.path.abspath(f) != os.path.abspath(csv_path)]

    # 3. 逐个读取其他文件，合并 details
    for other in sorted(other_files):
        df_i = pd.read_csv(other, dtype=str)
        if 'pdb_id' not in df_i.columns or 'details' not in df_i.columns:
            continue  # 若缺列，则跳过
        df_i.set_index('pdb_id', inplace=True)
        # 用 df_i 中非空的 details 来填充主表
        # fillna 会按索引自动对齐
        df_main['details'] = df_main['details'].fillna(df_i['details'])

    # 4. 保存为新的 CSV，文件名加 "_all"
    new_stem = stem + '_all'
    new_filename = new_stem + ext
    new_path = os.path.join(dirpath, new_filename)
    # 将索引 pdb_id 重新写入列中
    df_main.reset_index().to_csv(new_path, index=False)
    print(f"已将合并后的文件保存到：{new_path}")
    return new_path

def fetch_details_for_uniprot(csv2_path, output_csv3,
                              pdb_id=None, max_workers=50,
                              batch_size=500, checkpoint_interval=1000):
    """
    高并发版本：读取 CSV，针对每行未填 details 的 UniProt ID 并行抓取，
    并在抓到 TXT 后一次性提取 InterPro 信息并拼接。
    """
    # 全局缓存，避免重复请求 InterPro
    interpro_cache = {}

    def fetch_interpro_description_cached(ipr):
        if ipr not in interpro_cache:
            interpro_cache[ipr] = fetch_interpro_description(ipr)
        return interpro_cache[ipr]

    # 加载数据，只处理 details 为空的行
    df0 = pd.read_csv(csv2_path)
    # 如果没有 details 列，就把所有行都选上；否则只选 details 为空的行
    if 'details' in df0.columns:
        mask = df0['details'].isna() | (df0['details'] == '')
    else:
        # 全部为 True，表示全部行都要处理
        mask = pd.Series(True, index=df0.index)
    df = df0[mask].copy()
    total = len(df)

    # checkpoint 文件
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

    # 准备任务列表
    tasks = []
    for idx, row in df.iterrows():
        if str(idx) not in processed:
            try:
                # target 格式 "xxx-<UniProtID>"
                up = str(row['target']).split('-')[1]
                tasks.append((idx, up))
            except:
                processed[str(idx)] = ""  # 异常留空

    # 根据任务数量调整并发
    workers = min(max_workers, max(1, len(tasks)//2))
    print(f"要处理 {len(tasks)} 行，使用 {workers} 个 worker")

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

            # 提取 InterPro ID 列表，去重
            iprs = set(re.findall(r"IPR\d{6}", txt))
            infos = []
            for ipr in iprs:
                name = fetch_interpro_description_cached(ipr)
                if name:
                    infos.append(f"{ipr}: {name}")

            # 拼接结果
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


    # 初始化进度条，总数为所有要处理的行数
    pbar = tqdm(total=len(tasks), desc="总进度")
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        # 并行提交这一批
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_task, t) for t in batch]
            # 等待这一批全部完成，但不在这里更新进度条
            for fut in as_completed(futures):
                fut.result()  # 如有需要，可以捕获异常或获取返回值

        # 整个 batch 做完后，一次性更新 pbar
        pbar.update(len(batch))
        # 每跑完一个 batch 就存一个检查点
        save_ckpt()

    pbar.close()
            
    """       
    # 并行跑批次
    pbar = tqdm(total=len(tasks), desc="总进度")
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(process_task, t) for t in batch]):
                pbar.update(1)
        save_ckpt()
    pbar.close()
    """ 
    # 写回 CSV
    df['details'] = [processed.get(str(idx), "") for idx in df.index]
    df0.loc[mask, 'details'] = df['details']
    df0.to_csv(output_csv3, index=False)

    # 清理 checkpoint
    if error_count == 0 and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    elapsed = time.time() - start
    print(f"完成：{processed_count} 条，失败 {error_count} 条，耗时 {elapsed:.1f} 秒，平均 {(elapsed/processed_count):.2f} 秒/条")


def append_ref_ted_predition(input_dir: str, csv_ref_path: str, output_dir: str = None, suffix: str = '_with_ref') -> None:
    """
    遍历指定文件夹中的所有 Excel 和 CSV 文件，将每个文件中的 DataFrame 与参考 CSV 按指定键（pdb_id vs unique_ID）合并，
    并将选定的参考列追加到原始 DataFrame 中，然后保存为新的文件。

    参数：
    - input_dir: 包含待处理文件的文件夹路径。
    - csv_ref_path: 参考 CSV 文件路径，其包含 unique_ID 列和需要追加的字段。
    - output_dir: 可选，保存结果文件的文件夹。如果为 None，则在 input_dir 下创建一个子文件夹 'output'。
    - suffix: 可选，保存文件时在文件名后追加的后缀（默认 '_with_ref'）。

    依赖：
    pandas
    """
    # 读取参考数据
    ref_df = pd.read_csv(csv_ref_path)
    print(csv_ref_path)
    print(ref_df)
    # 只保留需要追加的列（除 unique_ID 外）
    join_key = 'unique_ID'

    print('1')
    ref_cols = [c for c in ref_df.columns if c != join_key]
    print(ref_cols)
    print('2')   
    # 准备输出目录
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    print('3')
    # 遍历文件夹中的所有文件
    for fname in os.listdir(input_dir):
        file_path = os.path.join(input_dir, fname)
        base, ext = os.path.splitext(fname)
        ext = ext.lower()
        print('4')
        if ext in ['.xlsx', '.xls']:
            # 处理 Excel 文件
            excel = pd.ExcelFile(file_path)
            
            
            writer = pd.ExcelWriter(
                os.path.join(output_dir, f"{base}{suffix}.xlsx"),
                engine='openpyxl'
            )
            for sheet in excel.sheet_names:
                df = pd.read_excel(excel, sheet_name=sheet, dtype=str)
                # 重命名 df 的键列以便合并
                print('5')
                df = df.rename(columns={'pdb_id': join_key})
                print(sheet)
                print(df)
                # 合并
                merged = df.merge(
                    ref_df[[join_key] + ref_cols],
                    on=join_key,
                    how='left'
                )
                print('6')
                # 恢复列名
                merged = merged.rename(columns={join_key: 'pdb_id'})
                # 写入
                merged.to_excel(writer, sheet_name=sheet, index=False)
            writer.close()

        elif ext == '.csv':
            # 处理 CSV 文件
            df = pd.read_csv(file_path, dtype=str)
            df = df.rename(columns={'pdb_id': join_key})
            merged = df.merge(
                ref_df[[join_key] + ref_cols],
                on=join_key,
                how='left'
            )
            merged = merged.rename(columns={join_key: 'pdb_id'})
            # 保存 CSV
            merged.to_csv(
                os.path.join(output_dir, f"{base}{suffix}.csv"),
                index=False
            )
        else:
            # 跳过其他文件类型
            continue
def sequence_to_vector(seq, k=3):
    """将蛋白质序列转换为k-mer频率向量"""
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    seq_counter = Counter(kmers)
    vector = np.zeros(len(amino_acids) ** k)
    kmers_set = [''.join(x) for x in itertools.product(amino_acids, repeat=k)]
    for i, kmer in enumerate(kmers_set):
        vector[i] = seq_counter.get(kmer, 0)
    return vector

def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似性"""
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
    计算两个蛋白质序列之间的相似性，使用BLOSUM62打分矩阵并通过自比对进行归一化
    
    Parameters:
        seq1 (str): 第一个蛋白质序列
        seq2 (str): 第二个蛋白质序列
        
    Returns:
        float: 归一化的相似性分数 (0-1范围)
    """
    # 参数检查
    if not isinstance(seq1, str) or not isinstance(seq2, str):
        return 0.0
        
    if len(seq1) == 0 or len(seq2) == 0:
        return 0.0
    
    # 加载BLOSUM62矩阵
    matrix = substitution_matrices.load("BLOSUM62")
    
    try:
        # 使用BLOSUM62矩阵进行全局比对
        # 设置gap_open=-10, gap_extend=-0.5是常用参数
        alignments = pairwise2.align.globalds(seq1, seq2, 
                                            matrix, -10, -0.5,
                                            one_alignment_only=True)
        
        if not alignments:
            return 0.0
            
        # 获取比对分数
        alignment_score = alignments[0].score
        
        # 计算自比对分数，用于归一化
        self_score1 = pairwise2.align.globalds(seq1, seq1, 
                                            matrix, -10, -0.5,
                                            one_alignment_only=True)[0].score
        self_score2 = pairwise2.align.globalds(seq2, seq2, 
                                            matrix, -10, -0.5,
                                            one_alignment_only=True)[0].score
        
        # 归一化分数 (取两个自比对分数中的较小值进行归一化)
        normalized_score = alignment_score / min(self_score1, self_score2)
        
        # 确保分数在0-1范围内
        normalized_score = max(0.0, min(1.0, normalized_score))
        
        return round(normalized_score, 4)
        
    except Exception as e:
        print(f"计算序列相似性时出错: {e}")
        return 0.0

def convert_cif_to_dssp_parallel(cif_dir, dssp_dir, max_workers=None):
    """
    将目录中的所有 CIF 文件转换为 DSSP 文件，使用多进程进行并行处理。

    Args:
        cif_dir (str): CIF 文件所在目录路径。
        dssp_dir (str): DSSP 输出目录路径。
        max_workers (int, optional): 最大并行进程数，默认为 None，表示自动选择合适的数量。
    """
    cif_path = Path(cif_dir)
    dssp_path = Path(dssp_dir)

    # 检查路径是否存在
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF directory does not exist: {cif_path}")
    if not dssp_path.exists():
        dssp_path.mkdir(parents=True, exist_ok=True)  # 创建输出目录

    # 获取所有 CIF 文件
    cif_files = list(cif_path.glob("*.pdb"))
    print(max_workers)
    # 使用 ProcessPoolExecutor 进行多进程并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有转换任务
        futures = []
        for cif_file in cif_files:
            base_name = cif_file.stem  # 提取文件名（无扩展名）
            output_dssp = dssp_path / f"{base_name}.dssp"
            futures.append(executor.submit(dssp_single, cif_file, output_dssp))

        # 等待任务完成并获取结果
        for future in as_completed(futures):
            future.result()  # 获取每个任务的执行结果，如果失败则抛出异常

def dssp_single(cif_file, output_dssp):
    """
    处理单个文件的转换，运行 mkdssp 命令。

    Args:
        cif_file (Path): 输入的 CIF 文件路径。
        output_dssp (Path): 输出的 DSSP 文件路径。
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
    将目录中的所有 CIF 文件转换为 DSSP 文件。
    
    Args:
        cif_dir (str): CIF 文件所在目录路径。
        dssp_dir (str): DSSP 输出目录路径。
    """
    cif_path = Path(cif_dir)
    dssp_path = Path(dssp_dir)

    # 检查路径是否存在
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF directory does not exist: {cif_path}")
    if not dssp_path.exists():
        dssp_path.mkdir(parents=True, exist_ok=True)  # 创建输出目录

    # 遍历 CIF 文件并转换
    for cif_file in cif_path.glob("*.pdb"):
        base_name = cif_file.stem  # 提取文件名（无扩展名）
        output_dssp = dssp_path / f"{base_name}.dssp"

        # 运行 mkdssp 命令
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
    # 如果 A 或 a 为空或 None，则返回空字符串
    if not A or not a:
        print("A 或 a 为空，返回空")
        return ""
    
    # 确保 start 和 end 的范围合法
    if start < 0 or end > len(A) or start >= end:
        raise ValueError("Invalid start or end position.")
    
    # 将 A 分为三部分：替换前的部分、替换的部分和替换后的部分
    before = A[:start]
    after = A[end:]
    
    # 生成新的序列 B，start 到 end 区间由 a 替代
    B = before + a + after
    
    return B


def add_domain_info_to_target_cath_teddb_check(csv1_path, csv2_path, fasta_dict, output_path):
    # 1. 读取csv1和csv2文件
    csv1 = pd.read_csv(csv1_path)
    csv2  = pd.read_excel(csv2_path)
    csv2  = csv2.dropna(subset=['chopping_check'])
    csv2_unique = csv2.drop_duplicates(subset='target', keep='first')
    csv2_dict = csv2_unique.set_index('target')[['chopping', 'chopping_check', 'plddt', 'cath_label']].to_dict(orient='index')
    
    
    # 3. 一次性读取fasta文件并提取UniProt ID与序列长度
    #with open(fasta_path, "r") as fasta_in:
    #    fasta_records = list(SeqIO.parse(fasta_in, "fasta"))
    fasta_records = fasta_dict
    # 将FASTA记录转换为字典，key为UniProt ID，value为序列长度
    uniprot_to_len = {}
    uniprot_info = {}
    uniprot_seq = {}
    for key in fasta_records.keys():
        # 提取UniProt ID：假设FASTA的ID格式为"tr|X6BLS5|X6BLS5_9HYPH"
        record = fasta_records[key]
        uniprot_id = record.id.split('|')[1]  # 通过分割获取UniProt ID部分
        info = record.description
        uniprot_to_len[uniprot_id] = len(record.seq)
        uniprot_info[uniprot_id] = info
        uniprot_seq[uniprot_id] = str(record.seq)




    
    # 4. 合并数据到csv1
    def get_additional_info(row):
        # 从target中提取uniprot_id（假设格式为target-xxxx，取'-'后面的部分）
        target = row['target']
        source = row['source']
        target_id = target.split('-')[1]
        source_id = source.split('_')[0]
        # 1) 查找对应的ted_id信息
        ted_id = "_".join(target.split('_')[:3])
        # 查找csv2中相关ted_id的行
        if ted_id in csv2_dict:
            chopping = csv2_dict[ted_id]['chopping']
            chopping_check = csv2_dict[ted_id]['chopping_check']
            plddt = csv2_dict[ted_id]['plddt']
            cath_label = csv2_dict[ted_id]['cath_label']


            
        else:
            chopping = plddt = cath_label = chopping_check = None  # 如果没有找到ted_id，设置为None

        # 2) 获取UniProt ID的序列长度
        target_seq = uniprot_seq.get(target_id, None)
        if len(target_seq)>0:
            start_res, end_res = int(chopping_check.split('-')[0]),int(chopping_check.split('-')[1])
            target_domain_seq  = target_seq[start_res-1:end_res]
        else:
            target_domain_seq = []
            
        # 2) 获取UniProt ID的序列长度
        source_seq = uniprot_seq.get(source_id, None)
        if len(source_seq)>0:
            start_res, end_res =  int(source.split('_')[-2].split('-')[0]),int(source.split('_')[-2].split('-')[1])
            source_domain_seq  = source_seq[start_res-1:end_res]
        else:
            source_domain_seq = []
        target_len = uniprot_to_len.get(target_id, None)  # 如果没有找到该ID，返回None
        target_info = uniprot_info.get(target_id, None)
        #if target_len is None:
        #    print(uniprot_id)
        protein_seq_sim  = calculate_protein_sequence_similarity(target_seq, source_seq)
        domain_seq_sim =  calculate_protein_sequence_similarity(target_domain_seq, source_domain_seq)
        assemble_seq =  replace_subsequence(source_seq, start_res, end_res, target_domain_seq)
        assemle_protein_sim = calculate_protein_sequence_similarity(assemble_seq, source_seq)
        return pd.Series([chopping,chopping_check, plddt, cath_label, target_len,target_info,source_domain_seq,target_domain_seq,protein_seq_sim, domain_seq_sim,assemble_seq,assemle_protein_sim,target_seq,source_seq ], index=['chopping', 'chopping_check','plddt', 'cath_label', 'target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq'])
    
    # 5. 对csv1中的每一行应用get_additional_info函数

    csv1[['chopping', 'chopping_check','plddt', 'cath_label', 'target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq']] = csv1.apply(get_additional_info, axis=1)
    
    
    columns_order = ['source','target', 'FS_weight','TM_weight','Dali_weight','target_info','chopping_check', 'target_len', 'plddt', 'cath_label','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim'] + [col for col in csv1.columns if col not in['source','target', 'FS_weight','TM_weight','Dali_weight','target_info','chopping_check', 'target_len', 'plddt', 'cath_label','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim']]

    # 调整列顺序
    csv1 = csv1[columns_order]
    
    # 6. 保存合并后的csv1
    csv1.to_csv(output_path, index=False)
    print(f"输出文件已保存至: {output_path}")


def add_chopping_check_to_csv(csv_file, excel_file, sheet_name, output_csv):
    # 读取CSV文件
    df_csv = pd.read_csv(csv_file)
    
    # 读取Excel文件中的指定Sheet
    df_excel = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # 处理 Excel 中的 target 列，将其按"_"拆分去掉最后一部分
    df_excel['target'] = df_excel['target'].str.split('_').str[:-1].str.join('_')

    # 在合并前直接修改列名，给 FS_weight, TM_weight, Dali_weight 添加 _check 后缀
    df_excel = df_excel.rename(columns={
        'FS_weight': 'FS_weight_check',
        'TM_weight': 'TM_weight_check',
        'Dali_weight': 'Dali_weight_check'
    })

    # 通过 source 和 target 两列进行合并，确保只有当两者都匹配时才填充相应列
    df_merged = pd.merge(df_csv, df_excel[['source', 'target', 'chopping_check', 'FS_weight_check', 'TM_weight_check', 'Dali_weight_check']],
                         on=['source', 'target'], how='left')

    # 将需要的列放到前面
    cols_order = ['source', 'target', 'chopping', 'chopping_check', 'FS_weight_check', 'TM_weight_check', 'Dali_weight_check'] + [col for col in df_merged.columns if col not in ['source', 'target', 'chopping', 'chopping_check', 'FS_weight_check', 'TM_weight_check', 'Dali_weight_check']]
    df_merged = df_merged[cols_order]

    # 保存合并后的数据为新的CSV文件
    df_merged.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"数据已经成功更新并保存为 {output_csv}")
from tqdm.auto import tqdm
tqdm.pandas()  # 为 pandas 注册进度条

def process_fs_reslult_new(input_csv, output_csv):
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    
    # 重命名列
    df.rename(columns={'weight': 'FS_weight', 'alntmscore': 'TM_weight'}, inplace=True)
    
    # 将FS_weight和TM_weight列转换为数值型，并保留四位小数
    df['FS_weight'] = pd.to_numeric(df['FS_weight'], errors='coerce').round(4)
    df['TM_weight'] = pd.to_numeric(df['TM_weight'], errors='coerce').round(4)
    
    
    # 重新排列列顺序，将source，target，FS_weight和TM_weight移到前面
    cols = ['source', 'target', 'FS_weight', 'TM_weight'] + [col for col in df.columns if col not in ['source', 'target', 'FS_weight', 'TM_weight']]
    df = df[cols]
    
    # 筛选出FS_weight >= 0.7 且 TM_weight >= 0.5 的行
    #df_filtered = df[(df['FS_weight'] >= 0.7) & (df['TM_weight'] >= 0.5)]
    df =  df[df['source'].str.contains('Q99ZW2', na=False)]
    df_filtered = df[(df['FS_weight'] >= 0) & (df['TM_weight'] >= 0)]
    
    # 将筛选后的数据保存为新的CSV文件
    df_filtered.to_csv(output_csv, index=False)


def add_domain_info_to_target_cath_teddb(csv1_path, csv2_path, fasta_dict, output_path):
    # 1. 读取csv1和csv2文件
    csv1 = pd.read_csv(csv1_path)
    csv2 = pd.read_csv(csv2_path)
    
    # 2. 创建一个字典，索引csv2的ted_id作为key，chopping、plddt、cath_label作为值
    columns_to_extract = ['chopping', 'plddt', 'cath_label', 'Cluster_representative']
    csv2_dict = {
        row['ted_id']: {
            col: row[col] if col in row else None 
            for col in columns_to_extract
        }
        for _, row in csv2.iterrows()
    }
    
    # 3. 一次性读取fasta文件并提取UniProt ID与序列长度
    #with open(fasta_path, "r") as fasta_in:
    #    fasta_records = list(SeqIO.parse(fasta_in, "fasta"))
    fasta_records = fasta_dict
    # 将FASTA记录转换为字典，key为UniProt ID，value为序列长度
    uniprot_to_len = {}
    uniprot_info = {}
    uniprot_seq = {}
    for key in fasta_records.keys():
        # 提取UniProt ID：假设FASTA的ID格式为"tr|X6BLS5|X6BLS5_9HYPH"
        record = fasta_records[key]
        uniprot_id = record.id.split('|')[1]  # 通过分割获取UniProt ID部分
        info = record.description
        uniprot_to_len[uniprot_id] = len(record.seq)
        uniprot_info[uniprot_id] = info
        uniprot_seq[uniprot_id] = str(record.seq)
        
    # 4. 合并数据到csv1
    def get_additional_info(row):
        # 从target中提取uniprot_id（假设格式为target-xxxx，取'-'后面的部分）
        target = row['target']
        source = row['source']
        target_id = target.split('-')[1]
        source_id = source.split('_')[0]
        
        # 1) 查找对应的ted_id信息
        ted_id = "_".join(target.split('_')[:3])
        #print(f'cathdb:{ted_id}')
        # 查找csv2中相关ted_id的行
        if ted_id in csv2_dict:
            chopping = csv2_dict[ted_id]['chopping']
            plddt = csv2_dict[ted_id]['plddt']
            cath_label = csv2_dict[ted_id]['cath_label']
            Cluster_representative = csv2_dict[ted_id]['Cluster_representative']
        else:
            Cluster_representative = chopping = plddt = cath_label  = None  # 如果没有找到ted_id，设置为None

        
        # 2) 获取UniProt ID的序列长度
        target_seq = uniprot_seq.get(target_id, None)
        if len(target_seq)>0:
            target_domain_seq = []
            # 解析 chopping 信息
            ranges = []
            for segment in chopping.split('_'):
                start, end = map(int, segment.split('-'))
                ranges.append((start, end))
            for start_res, end_res in ranges:
                target_domain_seq +=  target_seq[start_res-1:end_res]
        else:
            target_domain_seq = []
        target_domain_seq = ''.join(target_domain_seq)

        
        # 2) 获取UniProt ID的序列长度
        source_seq = uniprot_seq.get(source_id, None)
        if len(source_seq)>0:
            start_res, end_res =  int(source.split('_')[-2].split('-')[0]),int(source.split('_')[-2].split('-')[1])
            source_domain_seq  = source_seq[start_res-1:end_res]
        else:
            source_domain_seq = []
        target_len = uniprot_to_len.get(target_id, None)  # 如果没有找到该ID，返回None
        target_info = uniprot_info.get(target_id, None)
        #if target_len is None:
        #    print(uniprot_id)
        protein_seq_sim  = calculate_protein_sequence_similarity(target_seq, source_seq)
        domain_seq_sim =  calculate_protein_sequence_similarity(target_domain_seq, source_domain_seq)
        assemble_seq =  replace_subsequence(source_seq, start_res, end_res, target_domain_seq)
        assemle_protein_sim = calculate_protein_sequence_similarity(assemble_seq, source_seq)
        return pd.Series([chopping,plddt, cath_label, Cluster_representative, target_len,target_info,source_domain_seq,target_domain_seq,protein_seq_sim, domain_seq_sim,assemble_seq,assemle_protein_sim,target_seq,source_seq ], index=['chopping', 'plddt', 'cath_label','Cluster_representative', 'target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq'])
        



        
    # 5. 对csv1中的每一行应用get_additional_info函数
    csv1[['chopping', 'plddt', 'cath_label', 'Cluster_representative','target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq']] = csv1.progress_apply(get_additional_info, axis=1)
    #csv1.apply(get_additional_info, axis=1)
    
    
    # 定义前几列的列名
    first_columns=['source','target', 'FS_weight','TM_weight','target_info','chopping', 'source_domain_seq','target_domain_seq','target_len', 'plddt', 'cath_label','Cluster_representative','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq']
    # 获取剩余列名
    remaining_columns = [col for col in csv1.columns if col not in first_columns]

    # 按照要求的顺序排列列
    final_columns = first_columns + remaining_columns
    

    # 调整列顺序
    csv1 = csv1[final_columns]
    
    # 6. 保存合并后的csv1
    csv1.to_csv(output_path, index=False)
    print(f"输出文件已保存至: {output_path}")
    

def add_domain_info_to_target_cluster_teddb(csv1_path, csv2_path, fasta_dict, output_path):
    # 1. 读取csv1和csv2文件
    csv1 = pd.read_csv(csv1_path)
    csv2 = pd.read_csv(csv2_path)

    # 2. 创建一个字典，索引csv2的ted_id作为key，chopping、plddt、cath_label作为值
    columns_to_extract = ['chopping', 'plddt', 'cath_label', 'Cluster_representative']
    csv2_dict = {
        row['ted_id']: {
            col: row[col] if col in row else None 
            for col in columns_to_extract
        }
        for _, row in csv2.iterrows()
    }
    # 3. 一次性读取fasta文件并提取UniProt ID与序列长度
    #with open(fasta_path, "r") as fasta_in:
    #    fasta_records = list(SeqIO.parse(fasta_in, "fasta"))
    fasta_records = fasta_dict
    # 将FASTA记录转换为字典，key为UniProt ID，value为序列长度
    uniprot_to_len = {}
    uniprot_info = {}
    uniprot_seq = {}
    for key in fasta_records.keys():
        # 提取UniProt ID：假设FASTA的ID格式为"tr|X6BLS5|X6BLS5_9HYPH"
        record = fasta_records[key]
        uniprot_id = record.id.split('|')[1]  # 通过分割获取UniProt ID部分
        info = record.description
        uniprot_to_len[uniprot_id] = len(record.seq)
        uniprot_info[uniprot_id] = info
        uniprot_seq[uniprot_id] = str(record.seq)
    
    # 4. 合并数据到csv1
    def get_additional_info(row):
        # 从target中提取uniprot_id（假设格式为target-xxxx，取'-'后面的部分）
        target = row['target']
        source = row['source']
        target_id = target.split('-')[1]
        source_id = source.split('_')[0]
        
        # 1) 查找对应的ted_id信息
        ted_id = "_".join(target.split('_')[:3])
        # 查找csv2中相关ted_id的行
        if ted_id in csv2_dict:
            chopping = csv2_dict[ted_id]['chopping']
            plddt = csv2_dict[ted_id]['plddt']
            cath_label = csv2_dict[ted_id]['cath_label']
            Cluster_representative = csv2_dict[ted_id]['Cluster_representative']
        else:
            Cluster_representative=chopping = plddt = cath_label = None  # 如果没有找到ted_id，设置为None

        # 2) 获取UniProt ID的序列长度
        target_seq = uniprot_seq.get(target_id, None)
        if target_seq is None or len(target_seq) == 0:  # 先检查是否为None
            print(ted_id)
        
        if len(target_seq)>0:
            target_domain_seq = []
            # 解析 chopping 信息
            ranges = []
            for segment in chopping.split('_'):
                start, end = map(int, segment.split('-'))
                ranges.append((start, end))
            for start_res, end_res in ranges:
                target_domain_seq +=  target_seq[start_res-1:end_res]
        else:
            target_domain_seq = []
        target_domain_seq = ''.join(target_domain_seq)

        
        # 2) 获取UniProt ID的序列长度
        source_seq = uniprot_seq.get(source_id, None)
        if len(source_seq)>0:
            start_res, end_res =  int(source.split('_')[-2].split('-')[0]),int(source.split('_')[-2].split('-')[1])
            source_domain_seq  = source_seq[start_res-1:end_res]
        else:
            source_domain_seq = []
        target_len = uniprot_to_len.get(target_id, None)  # 如果没有找到该ID，返回None
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
    
    # 5. 对csv1中的每一行应用get_additional_info函数
    csv1[['chopping', 'plddt', 'cath_label','Cluster_representative', 'target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq']] = csv1.apply(get_additional_info, axis=1)

    # 定义前几列的列名
    first_columns=['source','target', 'FS_weight','TM_weight','target_info','chopping', 'source_domain_seq','target_domain_seq','target_len', 'plddt', 'cath_label','Cluster_representative','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq']
    # 获取剩余列名
    remaining_columns = [col for col in csv1.columns if col not in first_columns]
    # 按照要求的顺序排列列
    final_columns = first_columns + remaining_columns
    # 调整列顺序
    csv1 = csv1[final_columns]
    
    # 6. 保存合并后的csv1
    csv1.to_csv(output_path, index=False)
    print(f"输出文件已保存至: {output_path}")
    
    
    
def add_domain_info_to_target_cluster_teddb_check(csv1_path, csv2_path, fasta_dict, output_path):
    # 1. 读取csv1和csv2文件
    csv1 = pd.read_csv(csv1_path)
    csv2  = pd.read_excel(csv2_path)
    csv2  = csv2.dropna(subset=['chopping_check'])
    csv2_unique = csv2.drop_duplicates(subset='target', keep='first')
    csv2_dict = csv2_unique.set_index('target')[['chopping', 'chopping_check', 'plddt', 'cath_label','Cluster_representative']].to_dict(orient='index')
    
    
    # 3. 一次性读取fasta文件并提取UniProt ID与序列长度
    #with open(fasta_path, "r") as fasta_in:
    #    fasta_records = list(SeqIO.parse(fasta_in, "fasta"))
    fasta_records = fasta_dict
    # 将FASTA记录转换为字典，key为UniProt ID，value为序列长度
    uniprot_to_len = {}
    uniprot_info = {}
    uniprot_seq = {}
    for key in fasta_records.keys():
        # 提取UniProt ID：假设FASTA的ID格式为"tr|X6BLS5|X6BLS5_9HYPH"
        record = fasta_records[key]
        uniprot_id = record.id.split('|')[1]  # 通过分割获取UniProt ID部分
        info = record.description
        uniprot_to_len[uniprot_id] = len(record.seq)
        uniprot_info[uniprot_id] = info
        uniprot_seq[uniprot_id] = str(record.seq)
    
    # 4. 合并数据到csv1
    def get_additional_info(row):
        # 从target中提取uniprot_id（假设格式为target-xxxx，取'-'后面的部分）
        target = row['target']
        source = row['source']
        target_id = target.split('-')[1]
        source_id = source.split('_')[0]
        
        # 1) 查找对应的ted_id信息
        ted_id = "_".join(target.split('_')[:3])
        print(ted_id)
        # 查找csv2中相关ted_id的行
        if ted_id in csv2_dict:
            chopping = csv2_dict[ted_id]['chopping']
            chopping_check = csv2_dict[ted_id]['chopping_check']
            plddt = csv2_dict[ted_id]['plddt']
            cath_label = csv2_dict[ted_id]['cath_label']
            Cluster_representative = csv2_dict[ted_id]['Cluster_representative']
        else:
            chopping = plddt = cath_label = chopping_check = None  # 如果没有找到ted_id，设置为None

        # 2) 获取UniProt ID的序列长度
        target_seq = uniprot_seq.get(target_id, None)
        if len(target_seq)>0:
            start_res, end_res = int(chopping_check.split('-')[0]),int(chopping_check.split('-')[1])
            target_domain_seq  = target_seq[start_res-1:end_res]
        else:
            target_domain_seq = []
            
        # 2) 获取UniProt ID的序列长度
        source_seq = uniprot_seq.get(source_id, None)
        if len(source_seq)>0:
            start_res, end_res =  int(source.split('_')[-2].split('-')[0]),int(source.split('_')[-2].split('-')[1])
            source_domain_seq  = source_seq[start_res-1:end_res]
        else:
            source_domain_seq = []
        target_len = uniprot_to_len.get(target_id, None)  # 如果没有找到该ID，返回None
        target_info = uniprot_info.get(target_id, None)
        #if target_len is None:
        #    print(uniprot_id)
        protein_seq_sim  = calculate_protein_sequence_similarity(target_seq, source_seq)
        domain_seq_sim =  calculate_protein_sequence_similarity(target_domain_seq, source_domain_seq)
        assemble_seq =  replace_subsequence(source_seq, start_res, end_res, target_domain_seq)
        assemle_protein_sim = calculate_protein_sequence_similarity(assemble_seq, source_seq)
        
        return pd.Series([chopping, chopping_check,plddt, cath_label, Cluster_representative,target_len,target_info,source_domain_seq,target_domain_seq,protein_seq_sim, domain_seq_sim,assemble_seq,assemle_protein_sim,target_seq,source_seq], index=['chopping', 'chopping_check','plddt', 'cath_label', 'Cluster_representative', 'target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq'])
    
    # 5. 对csv1中的每一行应用get_additional_info函数
    csv1[['chopping', 'chopping_check','plddt', 'cath_label', 'Cluster_representative','target_len','target_info','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim','target_seq','source_seq']] =  csv1.apply(get_additional_info, axis=1)
    
    
    columns_order = ['source','target', 'FS_weight','TM_weight','Dali_weight','target_info','chopping_check', 'target_len', 'plddt', 'cath_label','Cluster_representative','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim'] + [col for col in csv1.columns if col not in['source','target', 'FS_weight','TM_weight','Dali_weight','target_info','chopping_check', 'target_len', 'plddt', 'cath_label','Cluster_representative','source_domain_seq','target_domain_seq','protein_seq_sim','domain_seq_sim','assemble_seq','assemle_protein_sim']]

    # 调整列顺序
    csv1 = csv1[columns_order]
    
    # 6. 保存合并后的csv1
    csv1.to_csv(output_path, index=False)
    print(f"输出文件已保存至: {output_path}")

# 预加载 FASTA 文件并存入字典
def load_fasta_to_dict(fasta_file):
    fasta_dict = {}  # 存储第一个出现的序列
    with open(fasta_file, "r") as fasta_in:
        for record in SeqIO.parse(fasta_in, "fasta"):
            record_id = record.id.split('|')[1]  # 提取第二部分 ID
            if record_id not in fasta_dict:  # 只保留第一个出现的
                fasta_dict[record_id] = record
    return fasta_dict  # 返回 {id: fasta_record} 字典

def add_domain_info_to_target_cathdb(csv1_path, csv2_path, fasta_dict, output_path):
    # 1. 读取csv1和csv2文件
    csv1 = pd.read_csv(csv1_path)
    csv2 = pd.read_csv(csv2_path)
    csv2 = csv2.drop_duplicates(subset='domain_id')
    # 2. 创建一个字典，索引csv2的ted_id作为key，chopping、plddt、cath_label作为值
    csv2_dict = csv2.set_index('domain_id')[['uniprot_id',  'cath_label' ]].to_dict(orient='index')
    
    # 3. 一次性读取fasta文件并提取UniProt ID与序列长度
    #with open(fasta_path, "r") as fasta_in:
    #    fasta_records = list(SeqIO.parse(fasta_in, "fasta"))
    fasta_records = fasta_dict
    # 将FASTA记录转换为字典，key为UniProt ID，value为序列长度
    uniprot_to_len = {}
    uniprot_info = {}
    for key in fasta_records.keys():
        # 提取UniProt ID：假设FASTA的ID格式为"tr|X6BLS5|X6BLS5_9HYPH"
        record = fasta_records[key]
        uniprot_id = record.id.split('|')[1]  # 通过分割获取UniProt ID部分
        info = record.description
        uniprot_to_len[uniprot_id] = len(record.seq)
        uniprot_info[uniprot_id] = info
    
    # 4. 合并数据到csv1
    def get_additional_info(target):
        # 从target中提取uniprot_id（假设格式为target-xxxx，取'-'后面的部分）
        uniprot_id = target
        
        # 1) 查找对应的ted_id信息
        ted_id = target
        
        # 查找csv2中相关ted_id的行
        if ted_id in csv2_dict:
            uniprot = csv2_dict[ted_id]['uniprot_id']
            cath_label = csv2_dict[ted_id]['cath_label']
        else:
            uniprot = cath_label = None  # 如果没有找到ted_id，设置为None

        # 2) 获取UniProt ID的序列长度
        target_len = uniprot_to_len.get(uniprot, None)  # 如果没有找到该ID，返回None
        target_info = uniprot_info.get(uniprot, None) 
        if target_len is None:
            print(uniprot_id)
        return pd.Series([uniprot, cath_label, target_len,target_info], index=['uniprot_id', 'cath_label', 'target_len','target_info'])
    
    # 5. 对csv1中的每一行应用get_additional_info函数
    csv1[['uniprot_id', 'cath_label', 'target_len','target_info']] = csv1['target'].apply(get_additional_info)
    
    
    columns_order = ['source','target', 'FS_weight','TM_weight','Dali_weight','uniprot_id', 'target_info','target_len', 'cath_label'] + [col for col in csv1.columns if col not in ['source','target', 'FS_weight','TM_weight','Dali_weight','uniprot_id', 'target_info','target_len', 'cath_label'] ]

    # 调整列顺序
    csv1 = csv1[columns_order]
    
    # 6. 保存合并后的csv1
    csv1.to_csv(output_path, index=False)
    print(f"输出文件已保存至: {output_path}")



def add_domain_chop_to_target_cathdb(csv1_path, domain_boundaries_data,  output_path):
    # 1. 读取csv1和csv2文件
    csv1 = pd.read_csv(csv1_path)
    with open(domain_boundaries_data, "rb") as f:
        domain_boundaries = pickle.load(f)
    
    
    
    # 4. 合并数据到csv1
    def get_additional_info(target):
        # 从target中提取uniprot_id（假设格式为target-xxxx，取'-'后面的部分）

        if target[-2:] == "00":
            key = target[:-2] + "01"
        else:
            key = target
        
        chain_id = domain_boundaries[key]["chain_id"]
        ranges = domain_boundaries[key]["ranges"]
        chopping = str(ranges[0][0])+'-'+str(ranges[0][1])
        
        return pd.Series([ chain_id,chopping], index=['chain_id','chopping'])
    
    # 5. 对csv1中的每一行应用get_additional_info函数
    csv1[['chain_id','chopping']] = csv1['target'].apply(get_additional_info)
    
    
    columns_order = ['source','target', 'FS_weight','TM_weight','Dali_weight','uniprot_id', 'target_info','target_len', 'cath_label','chopping','chain_id'] + [col for col in csv1.columns if col not in ['source','target', 'FS_weight','TM_weight','Dali_weight','uniprot_id', 'target_info','target_len', 'cath_label','chopping','chain_id'] ]

    # 调整列顺序
    csv1 = csv1[columns_order]
    
    # 6. 保存合并后的csv1
    csv1.to_csv(output_path, index=False)
    print(f"输出文件已保存至: {output_path}")



def get_protein_ted_info(protein_id):
    """
    模拟从数据库或 API 获取蛋白质的详细信息。
    返回值是一个 JSON 格式的模拟数据（真实使用时可替换为实际的 API 查询）。
    """
    url = f"{TED_info_URL}/uniprot/summary/{protein_id}"  # 替换为实际的 API 地址
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # 返回 JSON 数据
    else:
        print(f"Error fetching data for {protein_id}, status code: {response.status_code}")
        return None


def process_protein_ted_info(protein_data):
    """
    处理蛋白质数据并保存为 DataFrame。

    Args:
        protein_data (dict): 包含蛋白质片段信息的字典。

    Returns:
        pd.DataFrame: 处理后的 DataFrame，包含所有片段的相关信息。
    """
    # 如果 "data" 键不存在，返回空的 DataFrame
    if "data" not in protein_data:
        print("Warning: 'data' field not found in protein_data.")
        return pd.DataFrame()

    # 获取片段列表
    fragments = protein_data["data"]

    # 创建一个列表存储表格数据
    processed_data = []

    # 遍历每个片段，提取相关信息
    for fragment in fragments:
        # 基本信息
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

        # 交互信息（如果存在）
        interactions = fragment.get("interactions", [])
        interaction_count = len(interactions)

        # 将提取的数据存储为字典
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
            "interactions": interactions  # 保留完整交互信息（如果需要）
        })

    # 将处理后的数据转换为 DataFrame
    df = pd.DataFrame(processed_data)
    return df


def process_fs_reslult_new(input_csv, output_csv):
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    
    # 重命名列
    df.rename(columns={'weight': 'FS_weight', 'alntmscore': 'TM_weight'}, inplace=True)
    
    # 将FS_weight和TM_weight列转换为数值型，并保留四位小数
    df['FS_weight'] = pd.to_numeric(df['FS_weight'], errors='coerce').round(4)
    df['TM_weight'] = pd.to_numeric(df['TM_weight'], errors='coerce').round(4)
    
    
    # 重新排列列顺序，将source，target，FS_weight和TM_weight移到前面
    cols = ['source', 'target', 'FS_weight', 'TM_weight'] + [col for col in df.columns if col not in ['source', 'target', 'FS_weight', 'TM_weight']]
    df = df[cols]
    
    # 筛选出FS_weight >= 0.7 且 TM_weight >= 0.5 的行
    #df_filtered = df[(df['FS_weight'] >= 0.7) & (df['TM_weight'] >= 0.5)]
    df =  df[df['source'].str.contains('Q99ZW2', na=False)]
    df_filtered = df[(df['FS_weight'] >= 0) & (df['TM_weight'] >= 0)]
    
    # 将筛选后的数据保存为新的CSV文件
    df_filtered.to_csv(output_csv, index=False)


def get_protein_ted_info_to_csv(input_csv, output_csv):
    """
    从输入的蛋白质 ID 文件中读取 ID 列表，获取相关蛋白质信息，
    并将处理后的数据保存到输出 CSV 文件中。

    Args:
        input_csv (str): 输入文件路径，包含蛋白质 ID 列表。
        output_csv (str): 输出文件路径，用于保存处理后的蛋白质信息。
    """
    # 确定文件是 TSV 还是 CSV 格式
    try:
        # 尝试以 TSV 格式读取
        protein_ids = pd.read_csv(input_csv, sep="\t")["Entry"].tolist()
    except Exception as e:
        # 如果失败，尝试以 CSV 格式读取
        protein_ids = pd.read_csv(input_csv)["Entry"].tolist()

    # 初始化一个空的 DataFrame 用于存储所有蛋白质数据
    all_protein_data = pd.DataFrame()

    # 遍历每个蛋白质 ID，获取数据并处理
    for idx, protein_id in enumerate(protein_ids):
        print(f"Processing protein {idx + 1}/{len(protein_ids)}: {protein_id}")

        # 调用 API 获取蛋白质数据
        protein_data = get_protein_ted_info(protein_id)  # 确保 fetch_protein_data 函数已定义

        # 如果获取数据为空，则跳过
        if not protein_data:
            print(f"Warning: No data found for protein ID {protein_id}. Skipping...")
            continue

        # 处理蛋白质数据
        protein_df = process_protein_ted_info(protein_data)  # 使用更新后的 process_protein_data 函数

        # 如果处理后的数据为空，则跳过
        if protein_df.empty:
            print(f"Warning: No fragments found for protein ID {protein_id}. Skipping...")
            continue

        # 合并当前蛋白质的数据到总数据中
        all_protein_data = pd.concat([all_protein_data, protein_df], ignore_index=True)

        # 添加延时避免请求过于频繁（可选）
        time.sleep(1)

    # 保存所有处理后的数据到输出 CSV 文件
    all_protein_data.to_csv(output_csv, index=False)
    print(f"Protein data successfully saved to {output_csv}")


def get_protein_domains(protein_id, api_url="https://api.ted-database.org/protein"):
    """
    从 TED 接口获取指定蛋白质的 domain 数据
    Args:
        protein_id (str): 目标蛋白质的 UniProt ID（例如 'Q99ZW2'）
        api_url (str): TED API 的基础 URL
    Returns:
        list: 包含 domain 信息的字典列表
    """
    url = f"{api_url}/{protein_id}/domains"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()  # 假设返回的是 JSON 格式
        return data.get("data", [])
    else:
        print(f"Failed to retrieve data for {protein_id}. Status code: {response.status_code}")
        return []

def process_protein_domains(domain_data):
    """
    处理并格式化 domain 数据
    Args:
        domain_data (list): 从 TED 获取的 domain 数据（JSON 格式）
    Returns:
        pd.DataFrame: 格式化为 Pandas DataFrame
    """
    # 提取感兴趣字段
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
    获取指定蛋白质的 domain 数据并保存到 CSV 文件
    Args:
        protein_id (str): 目标蛋白质的 UniProt ID
        output_file (str): 输出 CSV 文件路径
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
    """下载 PDB 文件，并加入重试机制"""
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
    """下载 FASTA 序列，并返回 FASTA 数据"""
    
    if pd.isna(uniprot_id) or not uniprot_id:
        print(f"Invalid UniProt ID for PDB {pdb_id}: {uniprot_id}. Skipping FASTA download.")
        return None

    fasta_url = fasta_url_template.format(uniprot_id=uniprot_id)
    
    # 尝试从标准 URL 下载 FASTA 数据，增加重试机制
    for attempt in range(max_retries):
        try:
            response = requests.get(fasta_url)
            if response.status_code == 200 and response.text.strip():  # 检查响应是否有效且非空
                # 修改 FASTA 头部，添加 PDB ID
                fasta_data = response.text
                lines = fasta_data.splitlines()

                if lines and lines[0].startswith(">"):
                    lines[0] = lines[0] + f"__{pdb_id}"
                return "\n".join(lines)
            else:
                print(f"Attempt {attempt+1}: FASTA content is empty or invalid from standard URL.")
        except requests.RequestException as e:
            print(f"Attempt {attempt+1}: Exception occurred: {e}")
        # 可以根据需要设置重试间隔，例如等待 1 秒
        time.sleep(5)

    # 如果重试多次后仍未成功，则进入 except 块，尝试从 UniSave 下载
    print("Standard FASTA URL failed after retries. Trying UniSave for multiple versions...")
    for version in range(max_versions, 0, -1):  # 从 max_versions 到 1
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
                continue  # 当前版本失败，尝试下一版本
        except requests.RequestException:
            continue  # 当前版本请求失败，尝试下一版本

    print(f"Unable to download FASTA sequence for {uniprot_id} after trying all versions.")
    return None


def download_fasta33(uniprot_id, pdb_id, fasta_url_template="https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta", max_versions=20):
    """下载 FASTA 序列，并返回 FASTA 数据"""
    
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
    """下载 FASTA 序列，并加入重试机制"""
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

    # 初始化一个列表来存储所有的 FASTA 序列
    fasta_sequences = []
    """
    # 使用线程池并行下载 PDB 文件
    with ThreadPoolExecutor() as pdb_executor:
        # 提交每个 PDB 下载任务
        pdb_futures = [pdb_executor.submit(download_pdb, pdb_id, pdb_dir) for pdb_id in pdb_ids]
        
        # 等待所有 PDB 下载任务完成
        for future in as_completed(pdb_futures):
            future.result()  # 获取每个线程的返回结果，虽然我们此时不需要结果
    """
    # 使用线程池并行下载 FASTA 序列
    
    with ThreadPoolExecutor() as fasta_executor:
        # 提交每个 FASTA 下载任务
        fasta_futures = []
        for pdb_id in pdb_ids:
            uniprot_id = df.loc[df['domain_id'].str[:4] == pdb_id, 'uniprot_id'].iloc[0]# Retrieve UniProt ID from the CSV file
            print(uniprot_id)
            if pd.notna(uniprot_id):  # 确保 uniprot_id 不为空
                fasta_futures.append(fasta_executor.submit(download_fasta, uniprot_id, pdb_id, fasta_url_template))
            else:
                print(f"Skipping FASTA download for PDB {pdb_id} as UniProt ID is NaN")


        # 收集所有返回的 FASTA 数据
        for future in as_completed(fasta_futures):
            fasta_data = future.result()
            if fasta_data:
                fasta_sequences.append(fasta_data)
                
    # 将所有 FASTA 序列写入文件
    with open(fasta_file, "w") as fasta_out:
        for fasta in fasta_sequences:
                fasta_out.writelines(fasta + "\n")
        print(f"FASTA sequences saved to {fasta_file}.")
           
                    

from Bio import SeqIO


def process_pdb_for_domain(domain_id, pdb_dir, domain_pdb_dir, domain_boundaries):
    """处理PDB文件并提取域级别的PDB"""
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
    """处理FASTA文件并提取域级别的FASTA"""
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
    """处理FASTA文件并提取域级别的FASTA"""
    try:
        domain_sequence = ""
        # 查找对应的FASTA记录
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

    # 读取完整的FASTA文件数据
    fasta_records = []
    try:
        with open(fasta_file, "r") as fasta_in:
            fasta_records = list(SeqIO.parse(fasta_in, "fasta"))
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return

    # 处理PDB文件并提取域
    # 先处理所有PDB文件，再处理FASTA文件
    pdb_futures = []
    with ThreadPoolExecutor() as pdb_executor:
        # 提交PDB处理任务
        pdb_futures = [
            pdb_executor.submit(process_pdb_for_domain, domain_id, pdb_dir, domain_pdb_dir, domain_boundaries)
            for domain_id in valid_domain_ids
        ]

    # 收集PDB处理结果，并处理FASTA文件
    domain_fasta_sequences = []
    with ThreadPoolExecutor() as fasta_executor:
        # 提交FASTA处理任务
        for future in as_completed(pdb_futures):
            result = future.result()
            if result:
                domain_id, pdb_id, ranges = result
                # 获取域的FASTA序列
                fasta_sequence = process_fasta_for_domain(domain_id, pdb_id, ranges, fasta_records)
                if fasta_sequence:
                    domain_fasta_sequences.append(fasta_sequence)

    # 写入所有的FASTA序列到文件
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
    #第一次运行时
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
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # 获取源文件夹中的所有文件
    for item in os.listdir(src_folder):
        # 构建源文件和目标文件的完整路径
        src_path = os.path.join(src_folder, item)
        dest_path = os.path.join(dest_folder, item)
        
        # 如果是文件，进行复制
        if os.path.isfile(src_path):
            shutil.copy(src_path, dest_path)
        # 如果是子文件夹，也可以递归复制
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
        

# 统一下载，然后只写入一次，会丢数据，所以下载了一次fasta，就保存一个文件，
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
        # 初始化一个列表来存储所有的 FASTA 序列
        fasta_sequences = []
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Base URL for downloading PDB data from AlphaFold
        
        # 初始化一个列表来存储所有的 FASTA 序列
        fasta_sequences = []
        
        # 使用线程池并行下载 PDB 文件

        with ThreadPoolExecutor() as pdb_executor:
            # 提交每个 PDB 下载任务
            pdb_futures = [pdb_executor.submit(download_pdb, pdb_id, output_dir, pdb_url_template = "https://alphafold.ebi.ac.uk/files/{pdb_id}.pdb" ) for pdb_id in pdb_ids]
            
            # 等待所有 PDB 下载任务完成
            for future in as_completed(pdb_futures):
                future.result()  # 获取每个线程的返回结果，虽然我们此时不需要结果

        
        # 使用线程池并行下载 FASTA 序列
        with ThreadPoolExecutor() as fasta_executor:
            # 提交每个 FASTA 下载任务
            fasta_futures = []
            # 使用 tqdm 进度条包装任务

            for pdb_id in pdb_ids:
                uniprot_id = pdb_id.split('-')[1]# Retrieve UniProt ID from the CSV file
                if pd.notna(uniprot_id):  # 确保 uniprot_id 不为空
                    print(uniprot_id)
                    fasta_futures.append(fasta_executor.submit(download_fasta, uniprot_id, pdb_id,fasta_url_template="https://www.uniprot.org/uniprot/{uniprot_id}.fasta",max_versions=20))
                else:
                    print(f"Skipping FASTA download for PDB {pdb_id} as UniProt ID is NaN")
                    
            
        # 收集所有返回的 FASTA 数据
            for future in as_completed(fasta_futures):
                fasta_data = future.result()
                if fasta_data:
                    fasta_sequences.append(fasta_data)
        # 将所有 FASTA 序列写入文件
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
    """处理PDB文件并提取域级别的PDB"""
    ted_id = row['ted_id']
    pdb_id = "_".join(ted_id.split('_')[:2])
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    domain_pdb_path = os.path.join(domain_pdb_dir, f"{ted_id}.pdb")
    
    # Skip if PDB file doesn't exist
    
    chopping = row['chopping']
    # 解析 chopping 信息
    ranges = []
    for segment in chopping.split('_'):
        start, end = map(int, segment.split('-'))
        ranges.append((start, end))
    
    # Skip if PDB file doesn't exist
    if not os.path.exists(pdb_path):
        print(f"PDB file {pdb_id} not found. Skipping.")
        return None


    if os.path.exists(domain_pdb_path):
        print(f"domain geted {domain_pdb_path}. Skipping.")
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
    """处理PDB文件并提取域级别的PDB"""
    ted_id = row['target']
    pdb_id = "_".join(ted_id.split('_')[:2])
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    domain_pdb_path = os.path.join(domain_pdb_dir, f"{ted_id}_check.pdb")
    ted_id_check = ted_id + "_check"
    # Skip if PDB file doesn't exist
    
    chopping = row['chopping_check']
    # 解析 chopping 信息
    ranges = []
    for segment in chopping.split('_'):
        start, end = map(int, segment.split('-'))
        ranges.append((start, end))
    
    # Skip if PDB file doesn't exist
    if not os.path.exists(pdb_path):
        print(f"PDB file {pdb_id} not found. Skipping.")
        return None


    if os.path.exists(domain_pdb_path):
        print(f"domain geted {domain_pdb_path}. Skipping.")
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
    将csv1转换为csv2格式
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    """
    # 读取输入CSV文件
    df1 = pd.read_csv(input_file)
    
    # 存储结果的列表
    result_data = []
    
    # 权重列名映射到weight_type
    weight_columns = {
        'FS_weight_Dali': 'FS_weight',
        'FS_weight': 'FS_weight', 
        'TM_weight_Dali': 'TM_weight',
        'TM_weight': 'TM_weight'
    }
    
    # 处理每一行
    for index, row in df1.iterrows():
        current_id = index + 1  # id从1开始
        cath_label = row['cath_label']
        
        # 确定source_file的第一部分
        if '.' in str(cath_label):
            first_part = 'TedCath_'
        else:
            first_part = 'TedCluster_'
            
        # 处理四个权重值
        for weight_col in weight_columns:
            value = row[weight_col]
            weight_type = weight_columns[weight_col]
            
            # 确定source_file的第二部分
            if weight_col in ['FS_weight_Dali', 'TM_weight_Dali']:
                second_part = 'Sim_Check'
            else:
                second_part = 'Sim'
                
            source_file = first_part + second_part
            
            # 添加到结果中
            result_data.append({
                'id': current_id,
                'value': value,
                'weight_type': weight_type,
                'source_file': source_file
            })
    
    # 创建输出DataFrame
    df2 = pd.DataFrame(result_data)
    
    # 保存到CSV文件
    df2.to_csv(output_file, index=False)
    
    print(f"转换完成！输出文件：{output_file}")
    print(f"输入行数：{len(df1)}")
    print(f"输出行数：{len(df2)}")
    
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
            raise ValueError("在 HNH_cath_in_ted 文件中未找到 'ted_id' 或 'chopping' 列。")
        
                # 读取完整的FASTA文件数据
        fasta_records = []
        try:
            with open(fasta_file, "r") as fasta_in:
                fasta_records = list(SeqIO.parse(fasta_in, "fasta"))
        except Exception as e:
            print(f"Error reading FASTA file: {e}")
            return
    
        # 处理PDB文件并提取域
        pdb_futures = []
        with ThreadPoolExecutor() as pdb_executor:
            # 提交PDB处理任务
            pdb_futures = [
                pdb_executor.submit(process_pdb_for_domain_AFDB, row, pdb_dir, domain_pdb_dir)
                for _, row in hnh_cath_df.iterrows()
            ]

        # 收集PDB处理结果，并处理FASTA文件
        domain_fasta_sequences = []
        with ThreadPoolExecutor() as fasta_executor:
            # 提交FASTA处理任务
            for future in as_completed(pdb_futures):
                result = future.result()
                if result:
                    domain_id, pdb_id, ranges = result
                    # 获取域的FASTA序列
                    fasta_sequence = process_fasta_for_domain(domain_id, pdb_id, ranges, fasta_records)
                    if fasta_sequence:
                        domain_fasta_sequences.append(fasta_sequence)

        # 写入所有的FASTA序列到文件
        with open(domain_fasta_file, "w") as fasta_out:
            fasta_out.writelines(domain_fasta_sequences)

        print("Domain-level PDB and FASTA files have been processed and saved.")
        
    except Exception as e:
        print(f"发生错误：{e}")
        

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


        # 处理PDB文件并提取域
        pdb_futures = []
        with ThreadPoolExecutor() as pdb_executor:
            # 提交PDB处理任务
            pdb_futures = [
                pdb_executor.submit(process_pdb_for_domain_check, row, pdb_dir, domain_pdb_dir)
                for _, row in hnh_cath_df.iterrows()
            ]

        # 收集PDB处理结果，并处理FASTA文件
        domain_fasta_sequences = []
        with ThreadPoolExecutor() as fasta_executor:
            # 提交FASTA处理任务
            for future in as_completed(pdb_futures):
                result = future.result()
                if result:
                    domain_id, pdb_id, ranges = result
                    # 获取域的FASTA序列
                    fasta_sequence = process_fasta_for_domain_check(domain_id, pdb_id, ranges, fasta_records)
                    if fasta_sequence:
                        domain_fasta_sequences.append(fasta_sequence)

        # 写入所有的FASTA序列到文件
        with open(domain_fasta_file, "w") as fasta_out:
            fasta_out.writelines(domain_fasta_sequences)

        print("Domain-level PDB and FASTA files have been processed and saved.")
        
    except Exception as e:
        print(f"发生错误：{e}")
        
      
   

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
            raise ValueError("在 HNH_cath_in_ted 文件中未找到 'ted_id' 或 'chopping' 列。")

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
                print(f"PDB 文件 {pdb_path} 不存在，跳过。")
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
            print(f"保存域到 {domain_pdb_path}")

            # Extract and save the corresponding FASTA sequence for the domain
            extract_fasta_sequence(fasta_file, pdb_id, ted_id, ranges,  domain_fasta_file)

    except Exception as e:
        print(f"发生错误：{e}")
   
def split_domain_by_ted_boundary(hnh_cath_in_ted_file, pdb_dir, domain_dir):
    """
    根据 HNH_cath_in_ted 文件中的 chopping 信息，从完整 PDB 数据中提取 domain。

    参数：
        hnh_cath_in_ted_file (str): 包含 ted_id 和 chopping 列的 CSV 文件路径。
        pdb_dir (str): 保存完整 PDB 文件的目录。
        domain_dir (str): 保存提取的 domain PDB 文件的目录。

    返回：
        None
    """
    try:
        # 读取 HNH_cath_in_ted 文件
        hnh_cath_df = pd.read_csv(hnh_cath_in_ted_file)
        if 'ted_id' not in hnh_cath_df.columns or 'chopping' not in hnh_cath_df.columns:
            raise ValueError("在 HNH_cath_in_ted 文件中未找到 'ted_id' 或 'chopping' 列。")

        # 确保输出目录存在
        os.makedirs(domain_dir, exist_ok=True)

        # 初始化 PDB 解析器
        parser = PDB.PDBParser(QUIET=True)
        io = PDB.PDBIO()

        # 遍历每一行提取 domain
        for _, row in tqdm(hnh_cath_df.iterrows(), desc="提取域", total=hnh_cath_df.shape[0]):
            ted_id = row['ted_id']
            chopping = row['chopping']

            # 提取完整 PDB 文件 ID
            pdb_id = "_".join(ted_id.split('_')[:2])
            pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")

            if not os.path.exists(pdb_path):
                print(f"PDB 文件 {pdb_path} 不存在，跳过。")
                continue

            # 解析 PDB 文件
            structure = parser.get_structure(pdb_id, pdb_path)

            # 解析 chopping 信息
            ranges = []
            for segment in chopping.split('_'):
                start, end = map(int, segment.split('-'))
                ranges.append((start, end))

            # 提取指定范围的残基
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

            # 保存提取的 domain 到文件
            output_path = os.path.join(domain_dir, f"{ted_id}.pdb")
            io.set_structure(domain_structure)
            io.save(output_path)
            print(f"保存域到 {output_path}")

    except Exception as e:
        print(f"发生错误：{e}")


def add_domain_range_with_ted_info(cluster_file, domain_summary_file, output_file, chunksize=100000):
    """
    扩展 cluster_file 文件中的每一行数据，在 domain_summary_file 中找到与 Cluster_member 对应的行，
    并将这些行的数据拼接到 cluster_file 中，输出为新的文件。

    参数：
        cluster_file (str): 包含 Cluster_representative 和 Cluster_member 列的 CSV 文件路径。
        domain_summary_file (str): TED domain 数据文件路径。
        output_file (str): 保存扩展后数据的文件路径。
        chunksize (int): 每次读取 domain_summary_file 的行数。

    返回：
        None
    """
    try:
        # 读取 Cluster 文件
        cluster_df = pd.read_csv(cluster_file)
        if 'Cluster_member' not in cluster_df.columns:
            raise ValueError("在 cluster_file 中未找到 'Cluster_member' 列。")

        # 获取所有 Cluster_member 的唯一值
        cluster_members = set(cluster_df['Cluster_member'].dropna().unique())

        # 初始化扩展数据存储
        expanded_data = []

        # 定义 domain_summary_file 的列名
        column_names = [
            "ted_id", "md5_domain", "consensus_level", "chopping", "nres_domain",
            "num_segments", "plddt", "num_helix_strand_turn", "num_helix", "num_strand",
            "num_helix_strand", "num_turn", "proteome_id", "cath_label", "cath_assignment_level",
            "cath_assignment_method", "packing_density", "norm_rg", "tax_common_name",
            "tax_scientific_name", "tax_lineage"
        ]

        # 分块读取 domain_summary_file 并匹配数据
        for chunk in tqdm(pd.read_csv(domain_summary_file, sep='\t', names=column_names, header=0, chunksize=chunksize),
                          desc="Processing domain summary file"):
            # 筛选出与 cluster_members 匹配的行
            filtered_chunk = chunk[chunk['ted_id'].isin(cluster_members)]

            # 拼接到结果数据
            if not filtered_chunk.empty:
                expanded_data.append(filtered_chunk)

        # 合并所有匹配的行
        if expanded_data:
            expanded_df = pd.concat(expanded_data)

            # 按 Cluster_member 将扩展数据与 cluster_file 合并
            merged_df = cluster_df.merge(expanded_df, left_on='Cluster_member', right_on='ted_id', how='left')

            # 保存合并后的数据到输出文件
            merged_df.to_csv(output_file, index=False)
            print(f"扩展后的数据已保存到 {output_file}。")
        else:
            print("未找到匹配的 TED 信息。")

    except Exception as e:
        print(f"发生错误：{e}")
        
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
    # 初始化df_merged为空DataFrame
    df_merged = pd.DataFrame()

    # 读取FS_cath_teddb，如果路径不为空且文件存在且不为空
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

    # 读取TM_cath_teddb，如果路径不为空且文件存在且不为空
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

    # 读取Dali_cath_teddb，如果路径不为空且文件存在且不为空
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

    # 如果df_merged不是空的，执行数值处理和保存
    if not df_merged.empty:
        df_merged['FS_weight'] = df_merged.get('FS_weight', pd.Series()).astype(float).round(5)
        df_merged['TM_weight'] = df_merged.get('TM_weight', pd.Series()).astype(float).round(5)
        df_merged['Dali_weight'] = df_merged.get('Dali_weight', pd.Series()).astype(float).round(5)
        
        # 保存合并后的数据为CSV文件
        df_merged.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"数据已经成功合并并保存为 {output_csv}")
    else:
        print("没有可合并的数据，输出文件为空。")


def merge_results_by_source_target(FS_cath_teddb, TM_cath_teddb, Dali_cath_teddb, output_csv):
    # 初始化df_merged为空DataFrame
    df_merged = pd.DataFrame()
    
    # 读取三个CSV文件
    df1 = pd.read_csv(FS_cath_teddb)
    df2 = pd.read_csv(TM_cath_teddb)
    df3 = pd.read_csv(Dali_cath_teddb)
    df3['source'] = df3['source'].str[:-1]
    df3['target'] = df3['target'].str[:-2]
    
    # 保留source和targe列并重命名其他列，分别加上FS_、TM_、Dali_前缀
    df1_rename = df1.rename(columns={col: f"FS_{col}" for col in df1.columns if col not in ['source', 'target']})
    df2_rename = df2.rename(columns={col: f"TM_{col}" for col in df2.columns if col not in ['source', 'target']})
    df3_rename = df3.rename(columns={col: f"Dali_{col}" for col in df3.columns if col not in ['source', 'target']})
    
    # 合并数据集，优先按source和targe列合并
    df_merged = df1_rename.merge(df2_rename, on=['source', 'target'], how='outer')
    df_merged = df_merged.merge(df3_rename, on=['source', 'target'], how='outer')
    
    df_merged['FS_weight'] = df_merged['FS_weight'].astype(float)
    df_merged['TM_weight'] = df_merged['TM_weight'].astype(float)
    df_merged['Dali_weight'] = df_merged['Dali_weight'].astype(float)
    
    df_merged['FS_weight'] = df_merged['FS_weight'].round(5)
    df_merged['TM_weight'] = df_merged['TM_weight'].round(5)
    df_merged['Dali_weight'] = df_merged['Dali_weight'].round(5)
    
    
    # 将合并后的数据保存为新的CSV文件
    df_merged.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print(f"数据已经成功合并并保存为 {output_csv}")


def filter_sequence_len_for_csv(csv1, column_name, value1, value2, csv2):
    """
    从csv1文件中读取指定列，筛选出大于value1且小于value2的行，保存到csv2文件。
    
    :param csv1: 输入CSV文件路径
    :param column_name: 要筛选的列名
    :param value1: 下限值（大于此值）
    :param value2: 上限值（小于此值）
    :param csv2: 输出CSV文件路径
    """
    # 读取csv1文件
    df = pd.read_csv(csv1)
    df['FS_weight'] = df['FS_weight'].astype(float)
    df['TM_weight'] = df['TM_weight'].astype(float)
    df['Dali_weight'] = df['Dali_weight'].astype(float)
    
    df['FS_weight'] = df['FS_weight'].round(5)
    df['TM_weight'] = df['TM_weight'].round(5)
    df['Dali_weight'] = df['Dali_weight'].round(5)
    
    # 确保目标列是数值型
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')  # 将无法转换为数字的值设置为NaNdf['TM_weight'] = df['TM_weight'].astype(float)

    # 筛选大于value1且小于value2的行
    filtered_df = df[(df[column_name] >= value1) & (df[column_name] <= value2)]
    
    # 将结果保存到csv2文件
    filtered_df.to_csv(csv2, index=False,float_format='%.5f')
    
    print(f"筛选后的数据已经保存到 {csv2}")


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

            # 打印当前查询的 ID 数量
            print(f"Requesting batch {i//batch_size + 1} with {len(chunk)} IDs.")

            r = requests.get(base_url, params=params)

            # 检查请求是否成功
            if r.status_code != 200:
                print(f"Error: Received status code {r.status_code} for batch {i//batch_size + 1}")
                continue  # 如果请求失败，跳过该批次

            r_text = r.text.strip()
            if not r_text:  # 如果返回为空，说明没有结果
                print(f"Warning: No results found for batch {i//batch_size + 1}")
                continue

            lines = r_text.split("\n")

            if not header_written:
                out.write("\n".join(lines) + "\n")
                header_written = True
            else:
                out.write("\n".join(lines[1:]) + "\n")

    # 读取保存的结果
    df = pd.read_csv(out_path, sep="\t")
    return df





    
def filter_weight_for_csv(csv1,  value1, value2,  value3, csv2):
    """
    从csv1文件中筛选出符合条件的行，保存到csv2文件。
    
    :param csv1: 输入CSV文件路径
    :param value1: FS_weight列的下限值（大于等于此值）
    :param value2: TM_weight列的下限值（大于等于此值）
    :param value3: Dali_weight列的下限值（大于等于此值）
    :param csv2: 输出CSV文件路径
    """
    # 读取csv1文件
    df = pd.read_csv(csv1)
    
    """ 查看某个值的分布
    df['FS_weight'] = pd.to_numeric(df['FS_weight'], errors='coerce')
    bins = pd.cut(df['FS_weight'], bins=10)
    value_counts = bins.value_counts(normalize=True).sort_index()
    print(value_counts)
    """
    
    # 将 None 或 NaN 值替换为 0
    df['FS_weight'] = df['FS_weight'].fillna(0)
    df['TM_weight'] = df['TM_weight'].fillna(0)
    df['Dali_weight'] = df['Dali_weight'].fillna(0)
    
    # 筛选满足条件的行
    filtered_df = df[
        (df['FS_weight'] >= value1) &
        (df['TM_weight'] >= value2) &
        (df['Dali_weight'] >= value3)
    ]
    
    # 将符合条件的行保存为csv2文件
    filtered_df.to_csv(csv2, index=False)
    
    print(f"符合条件的数据已经保存到 {csv2}")

def update_data_with_wet_id_batch2(data1_path, data2_path):
    # 读取data2文件（假设是Excel）
    data2 = pd.read_excel(data2_path)
    # 提取data2的source和target列的唯一标记以及wet_ID
    data2['unique_key'] = data2['source'].str.split('_').str[0] + '_'+ data2['target'].str.split('-').str[1]
    #wet_id_dict = dict(zip(data2['unique_key'], data2['wet_ID']))
    wet_id_dict = (
    data2
    .groupby('unique_key')['wet_ID']
    .agg(lambda ids: '+'.join(ids.astype(str)))
    .to_dict()
    )
    # 判断data1是Excel还是CSV文件
    if data1_path.endswith('.xlsx'):  # 如果是Excel文件

        # 读取Excel文件中的所有sheet
        excel_file = pd.ExcelFile(data1_path)
        os.remove( data1_path)
        with pd.ExcelWriter(data1_path, engine='openpyxl') as writer:
            for sheet_name in excel_file.sheet_names:
                df = excel_file.parse(sheet_name)
                # 为每一行添加wet_ID列
                df['unique_key'] = df['source'].str.split('_').str[0] + '_'+ df['target'].str.split('-').str[1]
                df['wet_ID'] = df['unique_key'].map(wet_id_dict)  # 根据unique_key映射wet_ID
                
                # 删除中间列 unique_key
                df.drop(columns=['unique_key'], inplace=True)
                df = df[['wet_ID'] + [col for col in df.columns if col != 'wet_ID']]
                # 保存更新后的数据到原Excel文件中
                df.to_excel(writer, sheet_name=sheet_name+'_wet_ID', index=False)
            
    elif data1_path.endswith('.csv'):  # 如果是CSV文件
        # 读取CSV文件
        df = pd.read_csv(data1_path)
        # 为每一行添加wet_ID列
        df['unique_key'] = df['source'].str.split('_').str[0] + '_'+df['target'].str.split('-').str[1]
        df['wet_ID'] = df['unique_key'].map(wet_id_dict)  # 根据unique_key映射wet_ID
        
        # 删除中间列 unique_key
        df.drop(columns=['unique_key'], inplace=True)
        df = df[['wet_ID'] + [col for col in df.columns if col != 'wet_ID']]
        # 保存更新后的CSV文件
        df.to_csv(data1_path, index=False)
    
    print(f"文件 {data1_path} 已更新。")
    
import requests
import time
import pandas as pd
from functools import lru_cache








def filter_for_nonCas9(csv_path, output_csv2, col_name,
                       filter_rules=None, 
                       dali_range=(0, float('inf'))):
    """
    读取 csv_path 文件，按规则过滤：
    1) target_info 列：根据 filter_rules 过滤掉包含关键字的行。
       filter_rules 是一个列表，元素间为或关系；每个元素是字符串列表，内部为 and 关系。
       例如: [["crispr",], ["HNH","domain-containing"]]
    2) domain_seq_sim_Dali 列：保留值在 dali_range [A1, A2] 范围内的行。
    保存过滤后结果到 output_csv2。
    """
    #df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path, encoding='latin1')

    # 文本过滤
    if filter_rules is None:
        filter_rules = [["crispr"], ["csn1"], ["cas9"], ["HNH", "domain-containing"]]
        
        
    # 如果没有 pdb_id 列，先补一个空列
    if 'pdb_id' not in df.columns:
        df['pdb_id'] = ''

    # 标记免过滤行：pdb_id 包含 '_D'
    mask_exempt = df['pdb_id'].astype(str).str.contains('_D')

    # 定义文本过滤函数
    def to_exclude(text):
        txt = str(text).lower()
        for rule in filter_rules:
            if all(keyword.lower() in txt for keyword in rule):
                return True
        return False

    # 对不免过滤的行执行文本过滤（True 表示保留）
    mask_text = (~mask_exempt) & df[col_name].apply(lambda x: not to_exclude(x))

    # Dali 范围过滤（True 表示保留）
    a1, a2 = dali_range
    mask_dali = (~mask_exempt) & df['domain_seq_sim_Dali'].between(a1, a2, inclusive='both')

    # 最终保留：免过滤行 OR (文本 & Dali)
    final_mask = mask_exempt | (mask_text & mask_dali)
    df_filtered = df[final_mask].copy()

    # 写出
    df_filtered.to_csv(output_csv2, index=False)
    print(f"Filtered data saved to {output_csv2}: {len(df_filtered)} rows.")

from Bio.PDB import MMCIFParser, PDBIO

def convert_cif_to_pdb(input_dir, output_dir):
    # 确保输出目录存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建解析器和输出对象
    parser = MMCIFParser()
    io = PDBIO()

    # 遍历输入目录中的所有 .cif 文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".cif"):
            cif_path = os.path.join(input_dir, filename)
            pdb_filename = filename.replace(".cif", ".pdb")
            pdb_path = os.path.join(output_dir, pdb_filename)

            # 解析 .cif 文件并保存为 .pdb 文件
            try:
                structure = parser.get_structure(filename, cif_path)
                io.set_structure(structure)
                io.save(pdb_path)
                print(f"Converted {filename} to {pdb_filename}")
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")


def create_fs_cytoscape_network(fs_file, output_file):
    """
    从Foldseek结果的.m8文件创建Cytoscape网络文件。

    :param fs_file: Foldseek结果的.m8文件路径
    :param output_file: 输出的Cytoscape网络文件路径
    """
    try:
        # 检查文件是否存在且不为空
        if os.path.exists(fs_file) and os.path.getsize(fs_file) > 0:
            # 读取Foldseek结果文件
            # 读取Foldseek结果文件
            fs_results = pd.read_csv(fs_file, sep="\t", header=None)
            #fs_results.columns = ['query', 'target', 'qstart', 'qend', 'qlen', 
            #                      'tstart', 'tend', 'tlen', 'qcov', 'tcov', 
            #                      'bits', 'evalue']
            print("processing fs network")
            fs_results.columns = ['query','target','fident','alnlen','mismatch','gapopen','qstart','qend','qlen','tstart',
                                'tend','tlen','qcov','tcov','bits','evalue','qca','tca','alntmscore','qtmscore',
                                'ttmscore','u','t','lddt','lddtfull','prob']

            fs_results = fs_results.drop(columns=['qca', 'tca','lddt','lddtfull'])
            # 去掉 query 和 target 列中 '_' 及其后面的字符
            #fs_results.loc[:, 'query'] = fs_results['query'].str.split('_').str[0] # HNH的过程不需要去除'_' 及其后面的字符
            #fs_results.loc[:, 'target'] = fs_results['target'].str.split('_').str[0]# HNH的过程不需要去除'_' 及其后面的字符
            
            # 去掉target=source的行
            print(f"len of ori fs{len(fs_results)}")
            fs_results = fs_results[fs_results['query'] != fs_results['target']]
            print(f"len of target!=source fs{len(fs_results)}")
            # 创建临时的排序键（不保存为新列）
            """
            fs_results['_key'] = fs_results.apply(
                lambda x: tuple(sorted([x['query'], x['target']])), 
                axis=1
            )

            # 删除重复行
            fs_results = fs_results.drop_duplicates(subset=['_key'])

            # 删除临时列
            fs_results = fs_results.drop('_key', axis=1)
            print(f"len of final fs{len(fs_results)}")
            """
            fs_results['weight'] = (fs_results['qcov'] + fs_results['tcov']) / 2
            # 调整列的顺序
            cols_order = ['query', 'target', 'weight', 'evalue', 'bits', 'qcov', 'tcov']
            other_cols = [col for col in fs_results.columns if col not in cols_order]
            fs_results = fs_results[cols_order + other_cols]

            # 将 'query' 列重命名为 'source'
            fs_results = fs_results.rename(columns={'query': 'source'})

            fs_results.to_csv(output_file, index=False)
        else:
            print(f"警告: Foldseek结果文件 {fs_file} 不存在或为空，创建一个空的结果文件")
            # 创建一个空的DataFrame并保存
            empty_df = pd.DataFrame(columns=['source', 'target', 'weight', 'alntmscore'])
            empty_df.to_csv(output_file, index=False)
            return
    
    except pd.errors.EmptyDataError:
        print(f"警告: Foldseek结果文件 {fs_file} 为空，创建一个空的结果文件")
        # 创建一个空的DataFrame并保存
        empty_df = pd.DataFrame(columns=['source', 'target', 'weight', 'alntmscore'])
        empty_df.to_csv(output_file, index=False)
        return

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


from cath_alphaflow.settings import get_default_settings

config = get_default_settings()
logging.basicConfig(level=logging.DEBUG)
FS_BINARY_PATH = config.FS_BINARY_PATH
FS_TMP_PATH = config.FS_TMP_PATH
FS_OVERLAP = config.FS_OVERLAP
DEFAULT_FS_COV_MODE = "0" # overlap over query and target
DEFAULT_FS_ALIGNER = "2" # 3di+AA (fast, accurate)
DEFAULT_FS_FORMAT_OUTPUT = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,qlen,tstart,tend,tlen,qcov,tcov,bits,evalue,qca,tca,alntmscore,qtmscore,ttmscore,u,t,lddt,lddtfull,prob"


  
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
    