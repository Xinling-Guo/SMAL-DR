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
    从 PDB 文件中提取多个指定范围的片段并保存到一个新文件

    参数：
    - input_pdb: 输入 PDB 文件路径
    - output_pdb: 输出 PDB 文件路径
    - range_str: 目标片段范围字符串，格式如 "17-114_220-312"
    - chain_id: 目标链 ID（默认 'A'）
    """
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", input_pdb)
        io = PDB.PDBIO()

        # 拆分范围字符串
        ranges = range_str.split('_')
        valid_residues = []

        for r in ranges:
            start_res, end_res = map(int, r.split('-'))

            # 提取链
            model = structure[0]  # 仅使用第一个模型
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

        # print(f"✅ 提取完成！已保存到 {output_pdb}")
        return True
    except Exception as e:
        print(f"Error processing {input_pdb}: {e}. Skipping...")
        return False


def process_dali_results(input_path, output_file, csv1_path, csv2_path):
    # 正则表达式模式
    pattern = re.compile(
        r'^(\d+):\s+(\S+)\s+(\S+)\s+(\d+)\s*-\s*(\d+)\s*<=>\s*(\d+)\s*-\s*(\d+).*$'
    )
    
    # 构建映射字典
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

    # 构建映射字典
    csv1_map = build_mapping(csv1_path)
    csv2_map = build_mapping(csv2_path)

    # 获取所有 .txt 文件
    input_files = [f for f in os.listdir(input_path) if f.endswith('.txt')]

    # 写入处理后的结果
    with open(output_file, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        # CSV头部
        writer.writerow(["source", "source_id", "sourcerange", "target", "target_id", "targetrange"])

        # 处理每个输入文件
        for file_path in input_files:
            with open(os.path.join(input_path, file_path), 'r') as f_in:
                in_section = False
                for line in f_in:
                    line = line.strip()

                    # 进入目标区域
                    if line == '# Structural equivalences':
                        print(f"Processing in {file_path}: In Structural equivalences section")
                        in_section = True
                        continue
                    # 退出目标区域
                    if line == '# Translation-rotation matrices':
                        print(f"Processing in {file_path}: In Translation-rotation matrices section")
                        in_section = False
                        continue

                    if in_section:
                        match = pattern.match(line)
                        if match:
                            # 提取各部分信息
                            source = match.group(2)
                            sourcerange = f"{match.group(4)}-{match.group(5)}"
                            target = match.group(3)
                            targetrange = f"{match.group(6)}-{match.group(7)}"

                            # 映射source列
                            source_id = csv1_map.get(source.split("-")[0], source)
                            # 映射target列
                            target_id = csv2_map.get(target.split("-")[0], target)

                            # CSV格式写入
                            writer.writerow([source, source_id, sourcerange, target, target_id, targetrange])

    print(f"Processing and mapping completed. Output written to {output_file}")

def map_and_merge(csv1_path, csv2_path, input_csv, output_csv):
    """
    将source和target列分别通过csv1和csv2映射，并生成六列输出
    :param csv1_path: source映射文件路径（key,value）
    :param csv2_path: target映射文件路径（key,value）
    :param input_csv: 输入CSV文件路径（四列：source,sourcerange,target,targetrange）
    :param output_csv: 输出CSV文件路径（六列：source,source_id,target,target_id,sourcerange,targetrange）
    """
    # 构建映射字典
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
    # 处理输入文件
    with open(input_csv, 'r') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # 写入标题行
        writer.writerow([
            'source', 'source_id','sourcerange',
            'target', 'target_id','targetrange'
        ])
        
        for row in reader:
            if len(row) != 4:
                continue  # 跳过格式错误的行
            
            source, sourcerange, target, targetrange = row
            # print(source)
            # break
            # 映射source列
            # print(csv1_map.get(source.split("-")[0]))
            source_id = csv1_map.get(source.split("-")[0], source)
            # 映射target列
            target_id = csv2_map.get(target.split("-")[0], target)
            
            writer.writerow([
                source, source_id,sourcerange,
                target, target_id,targetrange
            ])


def merge_target_ranges_parts(input_csv, output_csv,split,clas):
    """
    合并相同source和target的range区间，并根据目标范围删除不满足条件的范围
    :param input_csv: 输入CSV文件路径
    :param output_csv: 输出CSV文件路径
    """
    merged_data = {}

    # 读取并处理输入文件
    with open(input_csv, 'r') as f_in:
        reader = csv.reader(f_in)
        next(reader)  # 跳过标题行
        print(clas)
        for row in reader:
            if len(row) != 6:
                continue  # 跳过格式错误的行
            if clas not in row[1].split("_")[-1]:
                # print(row[1].split("_")[-1],clas)
                continue

            source = row[1]#.split("_")[0]
            target = row[4]
            # print(row[1])
            # 解析source range
            try:
                src_start, src_end = map(int, row[2].split('-'))
            except:
                continue  # 跳过无效的range

            # 解析target range
            try:
                tgt_start, tgt_end = map(int, row[5].split('-'))
            except:
                continue  # 跳过无效的range

            # 更新合并数据
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
    # 对每个source和target组合下的范围进行排序和合并
    final_merged_data = {}
    for (source, target), ranges in merged_data.items():
        ranges.sort(key=lambda x: x['tgt_start'])  # 按目标起始位置排序
        if not ranges:
            continue
        merged_ranges = []
        current_range = ranges[0]
        for i in range(1, len(ranges)):
            if ranges[i]['tgt_start'] - current_range['tgt_end'] <= split:
                # 视作连续区间，更新当前范围
                current_range['src_start'] = min(current_range['src_start'], ranges[i]['src_start'])
                current_range['src_end'] = max(current_range['src_end'], ranges[i]['src_end'])
                current_range['tgt_start'] = min(current_range['tgt_start'], ranges[i]['tgt_start'])
                current_range['tgt_end'] = max(current_range['tgt_end'], ranges[i]['tgt_end'])
            else:
                # 差超过20，分隔为一个新区间
                merged_ranges.append(current_range)
                current_range = ranges[i]
        # 添加最后一个范围
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
    # 写入输出文件
    # 写入输出文件，并加上 sourcerange_abs 列
    with open(output_csv, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(['source', 'sourcerange', 'target', 'targetrange', 'sourcerange_abs'])

        for (source, target, _, _), data in final_merged_data.items():
            # 提取source的绝对位置（如961-1121）
            try:
                abs_start, abs_end = map(int, source.split('_')[1].split('-'))
            except:
                abs_start = 0  # fallback
            # 计算绝对范围
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
    # 删除重复的行
    df = df.drop_duplicates()
    # df = df.sort_values(['weight', 'alntmscore'], ascending=[False, False]).groupby(['source', 'target']).first().reset_index()
    # print(f"成功删除 {input_csv} 中的重复行。")

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
    # 读取两个 CSV 文件
    df1 = pd.read_excel(csv1_path)
    df2 = pd.read_csv(csv2_path)

    # 对 df1 的 source 列进行 split 操作
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

    # 删除 df2 中 source 和 target 前 5 个字符相同的行
    # df2 = df2[df2['source'].str.split("_").str[0] != df2['target'].str.split("_").str[0]]

    # 对 csv2 的列进行重命名（除 source 和 target 外）
    csv2_columns = df2.columns.tolist()
    # print(csv2_columns)
    rename_dict = {col: col + '_dalisearch' for col in csv2_columns if col not in ['source', 'target']}
    # print(rename_dict)
    # print(df2.columns)

    df2_renamed = df2.rename(columns=rename_dict)
    # 进行合并操作，这里使用 target_check 与 target 对应，source 与 source 对应
    merged_df = pd.merge(
        df1,
        df2_renamed,
        left_on=['source', 'targetID'],
        right_on=['source', 'target'],
        how='left'
    )

    # 删除多余的 target 列（因为已经用 target_check 进行了匹配）
    # merged_df = merged_df.drop(columns=['target_y'])

    # 定义前几列的列名
    first_columns = ['source', 'targetID','target_x','target_y', 
                     'chopping_check', 'chopping_dalisearch']

    # 获取剩余列名
    remaining_columns = [col for col in merged_df.columns if col not in first_columns]

    # 按照要求的顺序排列列
    final_columns = first_columns + remaining_columns
    merged_df = merged_df[final_columns]

    # 使用字典指定列名映射关系
    # 使用字典指定列名映射关系
    new_column_names = {'target_x': 'target','weight_dalisearch':'FS_weight_dalisearch','alntmscore_dalisearch':'TM_weight_dalisearch'}
    merged_df = merged_df.rename(columns=new_column_names)

    merged_df = merged_df.drop(columns=['targetID','target_y'])

    print(merged_df.columns)

    # 保存结果
    merged_df.to_csv(output_path, index=False)
    print(f"成功合并文件，结果保存至: {output_path}")

def download_pdb(pdb_id, pdb_dir, pdb_url_template="https://files.rcsb.org/download/{pdb_id}.pdb"):
    """下载 PDB 文件，并加入重试机制"""
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
# 统一下载，然后只写入一次，会丢数据，所以下载了一次fasta，就保存一个文件，


def download_pdb_urllib(pdb_id, pdb_dir, pdb_url_template="https://alphafold.ebi.ac.uk/files/AF-{pdb_id}-F1-model_v4.pdb"):
    """使用 urllib 下载 PDB（更兼容）"""
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
        
        
        # 使用线程池并行下载 PDB 文件
        with ThreadPoolExecutor() as pdb_executor:
            # 提交每个 PDB 下载任务
            pdb_futures = [pdb_executor.submit(download_pdb_urllib, pdb_id, output_dir, pdb_url_template = "https://alphafold.ebi.ac.uk/files/AF-{pdb_id}-F1-model_v4.pdb" ) for pdb_id in pdb_ids]
            
            # 等待所有 PDB 下载任务完成
            for future in as_completed(pdb_futures):
                future.result()  # 获取每个线程的返回结果，虽然我们此时不需要结果
        


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
        # 定义要传递给脚本的参数
        param1 = Dali_work_dir
        # 构建包含脚本路径和参数的命令列表
        command = ['nice', '-n', '-20', 'sh', assemble_dali_script_path, param1]
        # 运行 CSH 脚本并传递参数
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        # 打印脚本的标准输出
        print("脚本标准输出:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"运行脚本时出错: {e}")
        print("错误输出:")
        print(e.stderr)
   
   
def process_entry_split_pdb(entry, pdb_dir,domain_pdbs_path):
    pdb_id_str, range_str, failed_downloads = entry
    # pdb_id = pdb_id_str.split('-')[1]
    pdb_id = pdb_id_str
    try:
        range_parts = range_str.split(',')# 按下划线分割范围字符串
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
    # 读取 CSV 文件
    dali_result = pd.read_csv(dali_result_path)
    dali_result_fstm = pd.read_csv(dali_result_fstm_path)
    

    # 将 dali_result 中的 sourcerange 和 targetrange 追加到 dali_result_fstm 中
    merged = dali_result_fstm.rename(columns={'weight': 'FS_weight_Dali', 'alntmscore': 'TM_weight_Dali'})
    
    # 重命名 sourcerange 和 targetrange 列
    merged = merged.rename(columns={'sourcerange': 'sourcerange_Dali', 'targetrange': 'targetrange_Dali'})
    merged['sourcerange_Dali'] = merged['source'].str.split('_').str[-1] 
    merged['targetrange_Dali'] = merged['target'].str.split('_').str[-1]
    
    # 重新排列列的顺序，将 source, target, FS_weight_Dali, TM_weight_Dali, sourcerange_Dali 和 targetrange_Dali 移到前面
    columns_order = ['source', 'target', 'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali'] + \
                    [col for col in merged.columns if col not in ['source', 'target', 'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali']]
    merged = merged[columns_order]
    
    # 保存结果为新的 CSV 文件
    merged.to_csv(output_path, index=False)
    
    print(f"处理完成，结果已保存至 {output_path}")
    

def process_dali_fstm_results(dali_result_path, dali_result_fstm_path, output_path):
    # 读取 CSV 文件
    dali_result = pd.read_csv(dali_result_path)
    dali_result_fstm = pd.read_csv(dali_result_fstm_path)
    
    # 修改 dali_result_fstm 的列名
    dali_result_fstm = dali_result_fstm.rename(columns={'weight': 'FS_weight_Dali', 'alntmscore': 'TM_weight_Dali'})
    
    # 生成唯一 ID（source 和 target 列组合成唯一 ID）
    dali_result['id'] = dali_result['source'] + '_' + dali_result['target']+'_'+dali_result['targetrange']
    dali_result_fstm['id'] = dali_result_fstm['source'] + '_' + dali_result_fstm['target']
    
    # 将 dali_result 中的 sourcerange 和 targetrange 追加到 dali_result_fstm 中
    merged = pd.merge(dali_result_fstm, dali_result[['id', 'sourcerange', 'targetrange']], on='id', how='left')
    
    # 重命名 sourcerange 和 targetrange 列
    merged = merged.rename(columns={'sourcerange': 'sourcerange_Dali', 'targetrange': 'targetrange_Dali'})
    
    # 重新排列列的顺序，将 source, target, FS_weight_Dali, TM_weight_Dali, sourcerange_Dali 和 targetrange_Dali 移到前面
    columns_order = ['source', 'target', 'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali'] + \
                    [col for col in merged.columns if col not in ['source', 'target', 'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali']]
    merged = merged[columns_order]
    
    # 保存结果为新的 CSV 文件
    merged.to_csv(output_path, index=False)
    
    print(f"处理完成，结果已保存至 {output_path}")
    

def split_target_domains_from_dali_results(dali_results_pro, Target_domains_dali, pdb_dir, max_chunksize=100, max_workers=30):
    """
    逐块处理 CSV 文件，并利用多进程进行处理

    :param dali_results_pro: 输入的 CSV 文件路径
    :param Target_domains_dali: 输出目录
    :param max_chunksize: 每个块的大小
    :param max_workers: 最大工作进程数
    """
    ensure_dir(Target_domains_dali)
    # 读取 CSV 文件并按块处理
    chunks = pd.read_csv(dali_results_pro, chunksize=max_chunksize, sep=',')
    
    # 逐块处理 CSV 文件
    for j, chunk in enumerate(tqdm(chunks, desc="Processing CSV chunks")):
        print(f"\n\n=====================Processing chunk {j + 1}=====================\n\n")
        failed_downloads = []
        
        # 准备条目列表
        entries = [(row["target"], row["targetrange"], failed_downloads) for _, row in chunk.iterrows() if pd.notna(row["targetrange"])]

        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 使用多进程处理条目
                futures = []
                for entry in entries:
                    future = executor.submit(process_entry_split_pdb2, entry, pdb_dir, Target_domains_dali)
                    futures.append(future)

                # 等待所有进程完成并检查是否有错误
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # 获取结果（如果有异常则会在此抛出）
                    except Exception as e:
                        print(f"Error processing entry: {e}")

        except Exception as e:
            print(f"Error processing chunk {j + 1}: {e}")
            continue

    print(f"Processing of {dali_results_pro} completed.")

def split_query_domains_from_dali_results(dali_results_pro, Target_domains_dali, pdb_dir, max_chunksize=5, max_workers=30):
    """
    逐块处理 CSV 文件，并利用多进程进行处理

    :param dali_results_pro: 输入的 CSV 文件路径
    :param Target_domains_dali: 输出目录
    :param max_chunksize: 每个块的大小
    :param max_workers: 最大工作进程数
    """
    ensure_dir(Target_domains_dali)
    # 读取 CSV 文件并按块处理
    all_info = pd.read_csv(dali_results_pro)
    unique_info = all_info.drop_duplicates(subset=["source", "sourcerange_abs"])
    filtered_info = unique_info[unique_info['source'].str.contains('Q99ZW2') | unique_info['source'].str.contains('J7RUA')]
    #chunks = pd.read_csv(dali_results_pro, chunksize=max_chunksize, sep=',')
    chunks = np.array_split(filtered_info, max_chunksize)
    # 逐块处理 CSV 文件
    for j, chunk in enumerate(tqdm(chunks, desc="Processing CSV chunks")):
        print(f"\n\n=====================Processing chunk {j + 1}=====================\n\n")
        failed_downloads = []
        
        # 准备条目列表
        entries = [(row["source"], row["sourcerange_abs"], failed_downloads) for _, row in chunk.iterrows() if pd.notna(row["sourcerange_abs"])] 
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                # 使用多进程处理条目
                futures = []
                for entry in entries:
                    future = executor.submit(process_entry_split_pdb2, entry, pdb_dir, Target_domains_dali)
                    futures.append(future)

                # 等待所有进程完成并检查是否有错误
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()  # 获取结果（如果有异常则会在此抛出）
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
        range_parts = range_str.split(',')  # 按下划线分割范围字符串
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
    # 读取 CSV 文件为 DataFrame df1
    df1 = pd.read_csv(csv_file)
    
    # 读取 Excel 文件的两个指定 sheet（sp-2 和 sa-2）
    df2_sp = pd.read_excel(excel_file, sheet_name='sp-2')
    df2_sa = pd.read_excel(excel_file, sheet_name='sa-2')
    
    # 修改 df2_sp 中的 source 和 target 列，按 '_' 分割并取第一个和最后一个部分，构造新的 ID
    df2_sp['source_id'] = df2_sp['source'].apply(lambda x: '_'.join(x.split('_')[:1] + x.split('_')[-1:]) if pd.notna(x) and x != '' else None)
    df2_sp['target_id'] = df2_sp['target'].apply(lambda x: x if pd.notna(x) and x != '' else None)
    
    # 修改 df2_sa 中的 source 和 target 列，按 '_' 分割并取第一个和最后一个部分，构造新的 ID
    df2_sa['source_id'] = df2_sa['source'].apply(lambda x: '_'.join(x.split('_')[:1] + x.split('_')[-1:]) if pd.notna(x) and x != '' else None)
    df2_sa['target_id'] = df2_sa['target'].apply(lambda x: x if pd.notna(x) and x != '' else None)
    
    # 修改 df1 中的 source 和 target 列，按 '_' 分割并取第一个和最后一个部分，构造新的 ID
    df1['source_id'] = df1['source'].apply(lambda x: '_'.join(x.split('_')[:1] + x.split('_')[-1:]) if pd.notna(x) and x != '' else None)
    df1['target_id'] = df1['target'].apply(lambda x: x if pd.notna(x) and x != '' else None)
    
    # 根据 df2_sp 中的 source_id 和 target_id 对应的 pair id，查找 df1 中的相应行，得到 df3_sp
    df3_sp = pd.merge(df2_sp[['source_id', 'target_id']], df1, left_on=['source_id', 'target_id'], right_on=['source_id', 'target_id'], how='left')[df1.columns]
    
    # 根据 df2_sa 中的 source_id 和 target_id 对应的 pair id，查找 df1 中的相应行，得到 df3_sa
    df3_sa = pd.merge(df2_sa[['source_id', 'target_id']], df1, left_on=['source_id', 'target_id'], right_on=['source_id', 'target_id'], how='left')[df1.columns]
    
    # 创建一个新的 Excel 文件，保存 df1, df3_sp 和 df3_sa 到不同的 sheet
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df1.to_excel(writer, sheet_name='df1', index=False)
        df3_sp.to_excel(writer, sheet_name='df3_sp', index=False)
        df3_sa.to_excel(writer, sheet_name='df3_sa', index=False)

    print(f"处理完成，结果已保存到 {output_file}")

def process_candidate_with_dali0(candidate_file, dali_check_file, output_file):
    """
    读取 candidate 和 Dali_check_fstm 文件，进行合并和处理

    :param candidate_file: 输入的 candidate CSV 文件路径
    :param dali_check_file: 输入的 Dali_check_fstm CSV 文件路径
    :param output_file: 处理后保存的输出文件路径
    """
    # 读取 CSV 文件
    candidate = pd.read_csv(candidate_file)
    dali_check = pd.read_csv(dali_check_file)

    # 生成唯一 id 列
    candidate['new_source'] = candidate['source'].apply(lambda x: '_'.join((x.split('_')[0],x.split('_')[-1]))  if isinstance(x, str) else '')
    candidate['new_target'] = candidate['target'].apply(lambda x: x.split('-')[1] if isinstance(x, str) else '')
    candidate['id'] = candidate['new_source'] + '_' + candidate['new_target']
    
    dali_check['new_source'] = dali_check['source'].apply(lambda x: '_'.join((x.split('_')[0],x.split('_')[-1]))  if isinstance(x, str) else '')
    dali_check['new_target'] =dali_check['target'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else '')
    dali_check['id'] = dali_check['new_source'] + '_' + dali_check['new_target']


    # 先初始化需要的列为空
    candidate['FS_weight_Dali'] = None
    candidate['TM_weight_Dali'] = None
    candidate['sourcerange_Dali'] = None
    candidate['targetrange_Dali'] = None
    candidate['start_dif'] = None
    candidate['end_dif'] = None
    candidate['FS_dif'] = None
    candidate['TM_dif'] = None

    # 逐行处理 candidate 数据
    for idx, row in candidate.iterrows():
        # 找到 candidate 对应的唯一 id
        candidate_id = row['id']
        
        # 在 Dali_check_fstm 中找到相同 id 的所有行
        matching_rows = dali_check[dali_check['id'] == candidate_id]

        if len(matching_rows) == 1:
            # 如果只有一行，直接把列加到 candidate
            dali_row = matching_rows.iloc[0]
        elif len(matching_rows) > 1:
            # 如果多行，首先选择 targetrange_Dali 不为空的行
            non_na_rows = matching_rows[matching_rows['targetrange_Dali'].notna()]
            
            if not non_na_rows.empty:
                # 如果存在 targetrange_Dali 不为空的行，选择第一行
                dali_row = non_na_rows.iloc[0]
            else:
                # 如果不存在 targetrange_Dali 不为空的行，选择 FS_weight_Dali 最大的那一行
                dali_row = matching_rows.loc[matching_rows['FS_weight_Dali'].idxmax()]

        # 计算 start_dif 和 end_dif
        chopping_check = row['chopping_check']
        targetrange_Dali = dali_row['target'].split('_')[1]
        
        chopping_start, chopping_end = map(int, chopping_check.split('-'))
        targetrange_start, targetrange_end = map(int, targetrange_Dali.split('-'))

        start_dif = chopping_start - targetrange_start
        end_dif = targetrange_end - chopping_end

        # 计算 FS_dif 和 TM_dif
        FS_weight = row['FS_weight']
        TM_weight = row['TM_weight']
        FS_weight_Dali = dali_row['FS_weight_Dali']
        TM_weight_Dali = dali_row['TM_weight_Dali']
        
        FS_dif = FS_weight_Dali - FS_weight
        TM_dif = TM_weight_Dali - TM_weight

        # 更新 candidate 中的行数据
        candidate.at[idx, 'FS_weight_Dali'] = dali_row['FS_weight_Dali']
        candidate.at[idx, 'TM_weight_Dali'] = dali_row['TM_weight_Dali']
        candidate.at[idx, 'sourcerange_Dali'] = dali_row['sourcerange_Dali']
        candidate.at[idx, 'targetrange_Dali'] =  dali_row['target'].split('_')[1]
        candidate.at[idx, 'start_dif'] = start_dif
        candidate.at[idx, 'end_dif'] = end_dif
        candidate.at[idx, 'FS_dif'] = FS_dif
        candidate.at[idx, 'TM_dif'] = TM_dif

    # 将特定列移到前面
    columns_order = [
        'source', 'target', 'FS_weight', 'TM_weight', 'chopping_check', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif'
    ] + [col for col in candidate.columns if col not in [
        'source', 'target', 'FS_weight', 'TM_weight', 'chopping_check', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif'
    ]]
    
    # 重排列顺序
    candidate = candidate[columns_order]

    # 将处理后的数据保存到 CSV 文件
    candidate.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")
    
    
def process_candidate_with_dali0(candidate_file, dali_check_file, output_file):
    """
    读取 candidate 和 Dali_check_fstm 文件，进行合并和处理

    :param candidate_file: 输入的 candidate CSV 文件路径
    :param dali_check_file: 输入的 Dali_check_fstm CSV 文件路径
    :param output_file: 处理后保存的输出文件路径
    """
    # 读取 CSV 文件
    candidate = pd.read_csv(candidate_file)
    dali_check = pd.read_csv(dali_check_file)

    # 生成唯一 id 列
    candidate['new_source'] = candidate['source'].apply(lambda x: '_'.join((x.split('_')[0],x.split('_')[-1]))  if isinstance(x, str) else '')
    candidate['new_target'] = candidate['target'].apply(lambda x: x.split('-')[1] if isinstance(x, str) else '')
    candidate['id'] = candidate['new_source'] + '_' + candidate['new_target']
    
    dali_check['new_source'] = dali_check['source'].apply(lambda x: '_'.join((x.split('_')[0],x.split('_')[-2]))  if isinstance(x, str) else '')
    dali_check['new_target'] =dali_check['target'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else '')
    dali_check['id'] = dali_check['new_source'] + '_' + dali_check['new_target']


    # 先初始化需要的列为空
    candidate['FS_weight_Dali'] = None
    candidate['TM_weight_Dali'] = None
    candidate['sourcerange_Dali'] = None
    candidate['targetrange_Dali'] = None
    candidate['start_dif'] = None
    candidate['end_dif'] = None
    candidate['FS_dif'] = None
    candidate['TM_dif'] = None

    # 逐行处理 candidate 数据
    for idx, row in candidate.iterrows():
        # 找到 candidate 对应的唯一 id
        candidate_id = row['id']
        
        # 在 Dali_check_fstm 中找到相同 id 的所有行
        matching_rows = dali_check[dali_check['id'] == candidate_id]

        if len(matching_rows) == 1:
            # 如果只有一行，直接把列加到 candidate
            dali_row = matching_rows.iloc[0]
        elif len(matching_rows) > 1:
            # 如果多行，首先选择 targetrange_Dali 不为空的行
            non_na_rows = matching_rows[matching_rows['targetrange_Dali'].notna()]
            
            if not non_na_rows.empty:
                # 如果存在 targetrange_Dali 不为空的行，选择第一行
                dali_row = non_na_rows.iloc[0]
            else:
                # 如果不存在 targetrange_Dali 不为空的行，选择 FS_weight_Dali 最大的那一行
                dali_row = matching_rows.loc[matching_rows['FS_weight_Dali'].idxmax()]

        # 计算 start_dif 和 end_dif
        chopping_check = row['chopping_check']
        targetrange_Dali = dali_row['target'].split('_')[1]
        
        chopping_start, chopping_end = map(int, chopping_check.split('-'))
        targetrange_start, targetrange_end = map(int, targetrange_Dali.split('-'))

        start_dif = chopping_start - targetrange_start
        end_dif = targetrange_end - chopping_end

        # 计算 FS_dif 和 TM_dif
        FS_weight = row['FS_weight']
        TM_weight = row['TM_weight']
        FS_weight_Dali = dali_row['FS_weight_Dali']
        TM_weight_Dali = dali_row['TM_weight_Dali']
        
        FS_dif = FS_weight_Dali - FS_weight
        TM_dif = TM_weight_Dali - TM_weight

        # 更新 candidate 中的行数据
        candidate.at[idx, 'FS_weight_Dali'] = dali_row['FS_weight_Dali']
        candidate.at[idx, 'TM_weight_Dali'] = dali_row['TM_weight_Dali']
        candidate.at[idx, 'sourcerange_Dali'] = dali_row['sourcerange_Dali']
        candidate.at[idx, 'targetrange_Dali'] =  dali_row['target'].split('_')[1]
        candidate.at[idx, 'start_dif'] = start_dif
        candidate.at[idx, 'end_dif'] = end_dif
        candidate.at[idx, 'FS_dif'] = FS_dif
        candidate.at[idx, 'TM_dif'] = TM_dif

    # 将特定列移到前面
    columns_order = [
        'source', 'target', 'FS_weight', 'TM_weight', 'chopping_check', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif'
    ] + [col for col in candidate.columns if col not in [
        'source', 'target', 'FS_weight', 'TM_weight', 'chopping_check', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif'
    ]]
    
    # 重排列顺序
    candidate = candidate[columns_order]

    # 将处理后的数据保存到 CSV 文件
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
def replace_subsequence(orig_seq: str, start: int, end: int, new_subseq: str) -> str:
    """将 orig_seq 的 [start-1:end] 替换为 new_subseq"""
    return orig_seq[:start-1] + new_subseq + orig_seq[end:]
def augment_dali_with_check_250703(dali_csv: str,
                            check_csv: str,
                            fasta_dict: dict[str, SeqRecord],
                            output_csv: str):
    # 1. 读取
    dali_df  = pd.read_csv(dali_csv)
    check_df = pd.read_csv(check_csv)

    # 2. 构造统一 ID
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

    # 3. 预先在 dali_df 中准备好要新增的列（全置为 NaN/None）
    new_cols = [
        # 原 check 字段
        'chopping_check','FS_weight_Check','TM_weight_Check',
        # 差值 _Check
        'start_dif_Check','end_dif_Check','FS_dif_Check','TM_dif_Check',
        # 差值 _Dali_Check
        'start_dif_Dali_Check','end_dif_Dali_Check',
        'FS_dif_Dali_Check','TM_dif_Dali_Check',
        # domain 序列 & 相似度
        'source_domain_seq_Check','target_domain_seq_Check',
        'assemble_seq_Check','domain_seq_sim_Check'
    ]
    for c in new_cols:
        dali_df[c] = None

    # 4. 构建一个 check_df 的快速访问 map
    check_df = check_df.drop_duplicates(subset='id', keep='first')
    check_map = check_df.set_index('id').to_dict(orient='index')

    # 5. 遍历 dali_df，匹配并计算
    for idx, row in dali_df.iterrows():
        rid = row['id']
        info = check_map.get(rid)
        if info is None:
            continue

        # —— 4.1 直接合并字段
        dali_df.at[idx, 'chopping_check']    = info.get('chopping_check')
        dali_df.at[idx, 'FS_weight_Check']   = info.get('FS_weight_Check')
        dali_df.at[idx, 'TM_weight_Check']   = info.get('TM_weight_Check')

        # —— 4.2 计算 _Check 差值
        try:
            # 原 chopping
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

        # —— 4.3 计算 _Dali_Check 差值
        # 需要字段 chopping_Dali, FS_weight_Dali, TM_weight_Dali
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

        # —— 4.4 提取 domain 序列 & 计算相似度（Check）
        # 固定源区间 770-905
        src_id = row['source'].split('_')[0]
        tgt_id = row['target'].split('-')[1]
        rec_src = fasta_dict.get(src_id)
        rec_tgt = fasta_dict.get(tgt_id)
        chop_ck = info.get('chopping_check')
        if rec_src and rec_tgt and isinstance(chop_ck, str):
            seq_src = str(rec_src.seq)
            seq_tgt = str(rec_tgt.seq)
            # domain 片段
            ds, de = 770, 905
            tks, tke = map(int, chop_ck.split('-'))
            dom_src = seq_src[ds-1:de]
            dom_tgt = seq_tgt[tks-1:tke]
            # 写入
            dali_df.at[idx, 'source_domain_seq_Check'] = dom_src
            dali_df.at[idx, 'target_domain_seq_Check'] = dom_tgt
            # 拼接 & 相似度
            dali_df.at[idx, 'assemble_seq_Check']   = replace_subsequence(seq_src, ds, de, dom_tgt)
            dali_df.at[idx, 'domain_seq_sim_Check'] = calculate_protein_sequence_similarity(dom_src, dom_tgt)
    # 6. 列重排：将 new_cols 放到 chopping 列之后
    cols = list(dali_df.columns)
    insert_pos = cols.index('chopping') + 1
    front = cols[:insert_pos]
    rest = [c for c in cols[insert_pos:] if c not in new_cols]
    dali_df = dali_df[front + new_cols + rest]
    
    # 6. 输出
    dali_df['unique_ID'] = 'SpMut_' + (dali_df.index + 1).astype(str).str.zfill(4)
    dali_df.insert(0, 'unique_ID', dali_df.pop('unique_ID'))
    dali_df.to_csv(output_csv, index=False)
    
    print(f"Done! 输出已保存到 {output_csv}")
def process_candidate_with_dali_250703(candidate_file, dali_check_file, csv1_file, fasta_dict, output_file):
    """
    在原 process_candidate 基础上，新增 domain 序列和相似度信息：
      - source_domain_seq_Dali
      - target_domain_seq_Dali
      - assemble_seq
      - domain_seq_sim_Dali
      - protein_seq_sim_Dali

    :param fasta_dict: 一个 dict，key 为 fasta_records 的 key，value 为 BioPython SeqRecord
    """
    # --- 1. 读取 CSV ---
    candidate = pd.read_csv(candidate_file, low_memory=False)
    dali_check = pd.read_csv(dali_check_file)
    csv1 = pd.read_csv(csv1_file)

    # --- 2. 原有 ID 生成与筛选逻辑 ---
    csv1['id'] = (csv1['source'] + '_' + 
                  csv1['sourcerange_abs'].astype(str) + '_' + 
                  csv1['target'] + '_' + 
                  csv1['targetrange'].astype(str))
    dali_check['id2'] = dali_check['source'] + '_' + dali_check['target']
    valid_ids = set(csv1['id'])
    dali_check = dali_check[dali_check['id2'].isin(valid_ids)]

    # 新增：拆分 source/target ID 片段，生成可合并的 “id”
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

    # --- 3. candidate 新 ID ---
    candidate['new_source'] = candidate['source'].apply(
        lambda x: '_'.join((x.split('_')[0], x.split('_')[-1])) if isinstance(x, str) else ''
    )
    candidate['new_target'] = candidate['target'].apply(
        lambda x: x.split('-')[1] if isinstance(x, str) else ''
    )
    candidate['id'] = candidate['new_source'] + '_' + candidate['new_target']

    # --- 4. 初始化新列 ---
    for col in [
        'FS_weight_Dali', 'TM_weight_Dali',
        'sourcerange_Dali', 'targetrange_Dali',
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif',
        # 新增 domain 信息
        'source_domain_seq_Dali', 'target_domain_seq_Dali',
        'assemble_seq_Dali', 'domain_seq_sim_Dali', 'protein_seq_sim_Dali'
    ]:
        candidate[col] = None

    # --- 5. 逐行匹配并计算 ---
    for idx, row in candidate.iterrows():
        """
        if 'Sp' in row['wet_ID'] :
            print(f"第 {idx} 行是 Sp38")
        else:
            continue
        """
        print(f"{row['id']}: starting!!!!")
        
        cid = row['id']
        match = dali_check[dali_check['id'] == cid]
        if match.empty:
            continue
        d = match.iloc[0]

        # 基本权重和差值
        candidate.at[idx, 'FS_weight_Dali'] = d['FS_weight_Dali']
        candidate.at[idx, 'TM_weight_Dali'] = d['TM_weight_Dali']
        candidate.at[idx, 'sourcerange_Dali'] = d['source'].split('_')[-1]
        candidate.at[idx, 'targetrange_Dali'] = d['target'].split('_')[-1]

        # 计算差值
        chopping = row['chopping']
        sr = d['source'].split('_')[-1]  # e.g. "100-200"
        tr = d['target'].split('_')[1]  # e.g. "50-150"
        c_start, c_end = map(int, (chopping.split('-')[0], chopping.split('-')[-1]))
        t_start, t_end = map(int, tr.split('-'))
        candidate.at[idx, 'start_dif'] = c_start - t_start
        candidate.at[idx, 'end_dif'] = t_end - c_end
        candidate.at[idx, 'FS_dif'] = d['FS_weight_Dali'] - row['FS_weight']
        candidate.at[idx, 'TM_dif'] = d['TM_weight_Dali'] - row['TM_weight']

        # --- 新增：domain 序列与相似度计算 ---
        # 从 fasta_dict 中拿到源序列和目标序列
        src_id = row['new_source'].split('_')[0]
        tgt_id = row['new_target']
        src_rec: SeqRecord = fasta_dict.get(src_id)
        tgt_rec: SeqRecord = fasta_dict.get(tgt_id)
        if src_rec and tgt_rec:
            src_seq = str(src_rec.seq)
            tgt_seq = str(tgt_rec.seq)

            # 提取 domain 片段
            ds, de = map(int, sr.split('-'))
            ts, te = map(int, tr.split('-'))
            src_dom = src_seq[ds-1:de]
            tgt_dom = tgt_seq[ts-1:te]

            # 计算相似度
            dom_sim = 0
            prot_sim = 0
            #dom_sim = calculate_protein_sequence_similarity(src_dom, tgt_dom)
            #prot_sim = calculate_protein_sequence_similarity(src_seq, tgt_seq)

            # 拼接新序列
            assembled = replace_subsequence(src_seq, ds, de, tgt_dom)

            # 写回
            candidate.at[idx, 'source_domain_seq_Dali'] = src_dom
            candidate.at[idx, 'target_domain_seq_Dali'] = tgt_dom
            candidate.at[idx, 'domain_seq_sim_Dali'] = dom_sim
            candidate.at[idx, 'protein_seq_sim_Dali'] = prot_sim
            candidate.at[idx, 'assemble_seq_Dali'] = assembled
            print(f"{d['target']}: OKOKOKOKOKOKKO!!!!")
    # --- 6. 调整列顺序并输出 ---
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
    print(f"处理完成，已保存到 {output_file}")
    
    
def process_candidate_with_dali_25070300(candidate_file, dali_check_file, csv1_file, output_file):
    """
    读取 candidate、Dali_check_fstm 和 csv1 文件，进行合并和处理

    :param candidate_file: 输入的 candidate CSV 文件路径
    :param dali_check_file: 输入的 Dali_check_fstm CSV 文件路径
    :param csv1_file: 输入的 csv1 CSV 文件路径，包含 source, sourcerange, target, targetrange, sourcerange_abs 列
    :param output_file: 处理后保存的输出文件路径
    """
    # 读取 CSV 文件
    candidate = pd.read_csv(candidate_file)
    dali_check = pd.read_csv(dali_check_file)
    csv1 = pd.read_csv(csv1_file)

    # 为csv1创建唯一ID
    csv1['id'] = csv1['source'] + '_' + csv1['sourcerange_abs'].astype(str) + '_' + csv1['target'] + '_' + csv1['targetrange'].astype(str)
    
    # 为dali_check创建id2
    dali_check['id2'] = dali_check['source'] + '_'+ dali_check['target']
    
    # 只保留id2存在于csv1的id的行
    valid_ids = set(csv1['id'])
    dali_check = dali_check[dali_check['id2'].isin(valid_ids)]
    
        
    dali_check['new_source'] = dali_check['source'].apply(lambda x: '_'.join((x.split('_')[0],x.split('_')[-2]))  if isinstance(x, str) else '')
    dali_check['new_target'] =dali_check['target'].apply(lambda x: x.split('_')[0] if isinstance(x, str) else '')
    dali_check['id'] = dali_check['new_source'] + '_' + dali_check['new_target']

    
    # 对于重复的dali_check['id']，只保留TM_weight_Dali最大的那一行
    #dali_check = dali_check.sort_values('TM_weight_Dali', ascending=False).drop_duplicates(subset=['id']).reset_index(drop=True)
    dali_check = dali_check.sort_values('tlen', ascending=False).drop_duplicates(subset=['id']).reset_index(drop=True)

    # 生成唯一 id 列用于匹配
    candidate['new_source'] = candidate['source'].apply(lambda x: '_'.join((x.split('_')[0],x.split('_')[-1])) if isinstance(x, str) else '')
    candidate['new_target'] = candidate['target'].apply(lambda x: x.split('-')[1] if isinstance(x, str) else '')
    candidate['id'] = candidate['new_source'] + '_' + candidate['new_target']
    
    # 先初始化需要的列为空
    candidate['FS_weight_Dali'] = None
    candidate['TM_weight_Dali'] = None
    candidate['sourcerange_Dali'] = None
    candidate['targetrange_Dali'] = None
    candidate['start_dif'] = None
    candidate['end_dif'] = None
    candidate['FS_dif'] = None
    candidate['TM_dif'] = None
    # 增加

    # 逐行处理 candidate 数据
    for idx, row in candidate.iterrows():
        # 找到 candidate 对应的唯一 id
        candidate_id = row['id']
        
        # 在 Dali_check_fstm 中找到相同 id 的所有行
        matching_rows = dali_check[dali_check['id'] == candidate_id]

        if len(matching_rows) >= 1:
            # 如果有匹配行，选择第一行（已经按TM_weight_Dali排序过）
            dali_row = matching_rows.iloc[0]
            
            try:
                # 计算 start_dif 和 end_dif
                #"""
                chopping_check = row['chopping']
                #chopping_check = row['chopping_check'] # 最后再计算check的
                targetrange_Dali = dali_row['target'].split('_')[1]
                
                #chopping_start, chopping_end = map(int, chopping_check.split('-'))
                chopping_start, chopping_end = map(int, (chopping_check.split('-')[0], chopping_check.split('-')[-1]))
                targetrange_start, targetrange_end = map(int, targetrange_Dali.split('-'))

                start_dif = chopping_start - targetrange_start
                end_dif = targetrange_end - chopping_end

                # 计算 FS_dif 和 TM_dif
                FS_weight = row['FS_weight']
                TM_weight = row['TM_weight']
                FS_weight_Dali = dali_row['FS_weight_Dali']
                TM_weight_Dali = dali_row['TM_weight_Dali']
                
                FS_dif = FS_weight_Dali - FS_weight
                TM_dif = TM_weight_Dali - TM_weight
                #"""
                # 更新 candidate 中的行数据
                candidate.at[idx, 'FS_weight_Dali'] = dali_row['FS_weight_Dali']
                candidate.at[idx, 'TM_weight_Dali'] = dali_row['TM_weight_Dali']
                candidate.at[idx, 'sourcerange_Dali'] = dali_row['source'].split('_')[-1]
                candidate.at[idx, 'targetrange_Dali'] = dali_row['target'].split('_')[1]
                candidate.at[idx, 'start_dif'] = start_dif
                candidate.at[idx, 'end_dif'] = end_dif
                candidate.at[idx, 'FS_dif'] = FS_dif
                candidate.at[idx, 'TM_dif'] = TM_dif
            except Exception as e:
                print(f"处理行 {idx} 时出错: {e}")
                continue

    # 将特定列移到前面
    columns_order = [
        'source', 'target', 'FS_weight', 'TM_weight', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif'
    ] + [col for col in candidate.columns if col not in [
        'source', 'target', 'FS_weight', 'TM_weight', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif'
    ]]
    
    # 重排列顺序
    candidate = candidate[columns_order]

    # 将处理后的数据保存到 CSV 文件
    candidate.to_csv(output_file, index=False)
    print(f"处理完成，结果已保存至 {output_file}")
def copy_directory(src_dir, dest_dir):
    """
    将一个目录及其内容复制到另一个目录

    :param src_dir: 源目录路径
    :param dest_dir: 目标目录路径
    """
    try:
        shutil.copytree(src_dir, dest_dir)
        print(f"目录 '{src_dir}' 成功复制到 '{dest_dir}'")
    except Exception as e:
        print(f"复制失败: {e}")

def add_sequence_2_after_check(csv1_path, fasta_dict, output_path):
    # 1. 读取csv1文件
    csv1 = pd.read_csv(csv1_path)
    
    # 2. 处理fasta字典
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
        # 初始化返回值
        target_domain_seq = ""
        source_domain_seq = ""
        combine_seq = ""
        domain_seq_sim = 0.0
        
        try:
            # 处理target部分
            if pd.notna(row['targetrange_Dali']) and row['targetrange_Dali'] != '':
                target = row['target']
                target_chopping = row['targetrange_Dali']
                target_id = target.split('-')[1]
                target_seq = uniprot_seq.get(target_id, "")
                
                if target_seq:
                    start_res, end_res = map(int, target_chopping.split('-'))
                    target_domain_seq = target_seq[start_res-1:end_res]
            
            # 处理source部分
            if pd.notna(row['sourcerange_Dali']) and row['sourcerange_Dali'] != '':
                source = row['source']
                source_chopping = row['sourcerange_Dali']
                source_id = source.split('_')[0]
                source_seq = uniprot_seq.get(source_id, "")
                
                if source_seq:
                    start_res, end_res = map(int, source_chopping.split('-'))
                    source_domain_seq = source_seq[start_res-1:end_res]
                    
                    # 计算序列相似性
                    if target_domain_seq and source_domain_seq:
                        domain_seq_sim = calculate_protein_sequence_similarity(
                            target_domain_seq, source_domain_seq
                        )
                    
                    # 生成combine_seq
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
    
    # 应用函数
    csv1[['target_domain_seq_Dali', 'source_domain_seq_Dali', 
          'combine_seq_Dali', 'domain_seq_sim_dali']] = csv1.apply(get_additional_info, axis=1)
    
    # 调整列顺序
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
    print(f"输出文件已保存至: {output_path}")
    
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
    
def add_sequence_2_after_check0(csv1_path, fasta_dict, output_path):
    # 1. 读取csv1和csv2文件
    csv1 = pd.read_csv(csv1_path)
    
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
        # 初始化默认返回值
        target_domain_seq = []
        
        # 检查targetrange_Dali是否为空
        if pd.notna(row['targetrange_Dali']) and row['targetrange_Dali'] != '':
            try:
                # 从target中提取uniprot_id（假设格式为target-xxxx，取'-'后面的部分）
                target = row['target']
                chopping_check = row['targetrange_Dali']
                target_id = target.split('-')[1]

                # 获取UniProt ID的序列长度
                target_seq = uniprot_seq.get(target_id, "")
                
                if target_seq:  # 检查target_seq是否存在且不为空
                    # 解析范围
                    start_res, end_res = map(int, chopping_check.split('-'))
                    # 提取域序列
                    target_domain_seq = target_seq[start_res-1:end_res]
                    
            except (IndexError, ValueError, AttributeError) as e:
                # 处理可能的错误：split失败、转换int失败等
                print(f"Error processing row {row.name}: {e}")
                target_domain_seq = []
        
        return pd.Series([target_domain_seq], index=['target_domain_seq'])
    
    # 5. 对csv1中的每一行应用get_additional_info函数

    csv1[['target_domain_seq_Dali']] = csv1.apply(get_additional_info, axis=1)
    
    
    # 将特定列移到前面
    columns_order = [
        'source', 'target', 'FS_weight', 'TM_weight', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif','target_domain_seq_Dali'
    ] + [col for col in csv1.columns if col not in [
        'source', 'target', 'FS_weight', 'TM_weight', 
        'FS_weight_Dali', 'TM_weight_Dali', 'sourcerange_Dali', 'targetrange_Dali', 
        'start_dif', 'end_dif', 'FS_dif', 'TM_dif','target_domain_seq_Dali'
    ]]
    # 调整列顺序
    csv1 = csv1[columns_order]
    
    # 6. 保存合并后的csv1
    csv1.to_csv(output_path, index=False)
    print(f"输出文件已保存至: {output_path}")

import shutil

import shutil

def organize_query_domains_files(query_path, out_dir):
    # 确保 query_path 和 out_dir 都存在
    if not os.path.exists(query_path):
        print(f"路径 {query_path} 不存在！")
        return []
    
    if not os.path.exists(out_dir):
        print(f"路径 {out_dir} 不存在！将创建该目录。")
        os.makedirs(out_dir)

    new_paths = []  # 用于保存每个文件的新路径
    
    # 遍历 query_path 中的所有文件
    for filename in os.listdir(query_path):
        if filename.endswith(".pdb"):
            # 获取每个文件的完整路径
            pdb_file_path = os.path.join(query_path, filename)
            
            # 获取文件名（不带扩展名）
            folder_name = os.path.splitext(filename)[0]
            
            # 创建新的文件夹路径
            new_folder_path = os.path.join(out_dir, folder_name)
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            
            # 构造新文件路径
            new_file_path = os.path.join(new_folder_path, filename)
            
            # 移动 pdb 文件到新的文件夹
            shutil.copy(pdb_file_path, new_file_path)
            
            # 将新路径添加到返回列表中
            new_paths.append(new_file_path)
            print(f"已将 {filename} 移动到文件夹 {new_folder_path}")
    
    print("所有文件已组织完成！")
    
    # 返回所有新路径
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


# 函数：并行处理单个查询PDB
def process_query_pdb(query_pdb, fs_querypdb_base_dir, main_query_dir, query_db_dir):
    """处理单个查询PDB文件"""
    # 创建唯一的数据库名称
    query_db_name = query_pdb.replace(".", "_").replace("/", "_")
    
    # 复制PDB文件到查询目录
    src_query_path = os.path.join(fs_querypdb_base_dir, query_pdb)
    
    # 为这个PDB创建单独的文件夹
    single_query_dir = os.path.join(main_query_dir, "single_pdbs", query_db_name)
    os.makedirs(single_query_dir, exist_ok=True)
    single_query_path = os.path.join(single_query_dir, query_pdb)
    
    if os.path.exists(src_query_path):
        shutil.copy(src_query_path, single_query_path)
        
        # 为这个PDB创建数据库
        db_output_dir = os.path.join(query_db_dir, query_db_name)
        os.makedirs(db_output_dir, exist_ok=True)
        gpr.convert_pdb_to_foldseek_db(single_query_dir, db_output_dir, query_db_name)
        
        return query_pdb, os.path.join(db_output_dir, f"{query_db_name}.db")
    else:
        print(f"警告: 找不到查询PDB文件: {src_query_path}")
        return query_pdb, None
    
# 函数：并行处理单个目标PDB
def process_target_pdb(target_pdb, fs_targetpdb_base_dir, main_target_dir, target_db_dir):
    """处理单个目标PDB文件"""
    # 创建唯一的数据库名称
    target_db_name = target_pdb.replace(".", "_").replace("/", "_")
    
    # 复制PDB文件到目标目录
    src_target_path = os.path.join(fs_targetpdb_base_dir, target_pdb)
    
    # 为这个PDB创建单独的文件夹
    single_target_dir = os.path.join(main_target_dir, "single_pdbs", target_db_name)
    os.makedirs(single_target_dir, exist_ok=True)
    single_target_path = os.path.join(single_target_dir, target_pdb)
    
    if os.path.exists(src_target_path):
        shutil.copy(src_target_path, single_target_path)
        
        # 为这个PDB创建数据库
        db_output_dir = os.path.join(target_db_dir, target_db_name)
        os.makedirs(db_output_dir, exist_ok=True)
        gpr.convert_pdb_to_foldseek_db(single_target_dir, db_output_dir, target_db_name)
        
        return target_pdb, os.path.join(db_output_dir, f"{target_db_name}.db")
    else:
        print(f"警告: 找不到目标PDB文件: {src_target_path}")
        return target_pdb, None

def process_pdb_pair(args):
    """处理单个PDB对"""
    pair_info, query_db_map, target_db_map, pair_result_dir, fs_tmp_path, worker_id = args
    index, row, query_pdb, target_pdb = pair_info
    
    # 获取数据库路径
    query_db_path = query_db_map[query_pdb]
    target_db_path = target_db_map[target_pdb]
    
    # 创建唯一的结果文件名
    result_name = f"{row['source']}_{row['sourcerange_abs']}__vs__{row['target']}_{row['targetrange']}"
    result_name = result_name.replace(".", "_").replace("/", "_")
    
    # 创建每个工作进程唯一的临时目录
    unique_tmp_dir = os.path.join(fs_tmp_path, f"worker_{worker_id}_{uuid.uuid4().hex}")
    os.makedirs(unique_tmp_dir, exist_ok=True)
    
    # 创建唯一的结果文件路径
    result_m8 = os.path.join(pair_result_dir, f"{result_name}.m8")
    
    try:
        # 运行foldseek，使用唯一的临时目录和原始数据文件
        gpr.run_foldseek_with_error_handling(
            query_db_path, 
            target_db_path,
            fs_rawdata=os.path.join(unique_tmp_dir, f"{result_name}.raw"),
            fs_results=result_m8,
            tmp_dir=unique_tmp_dir
        )
        
        # 清理临时目录
        shutil.rmtree(unique_tmp_dir, ignore_errors=True)
        
        return result_m8
    except Exception as e:
        print(f"处理对 {index} 时出错: {e}")
        # 确保清理临时目录
        shutil.rmtree(unique_tmp_dir, ignore_errors=True)
        return None

# 为每个批次创建一个进程
def process_batch(args):
    """处理一批PDB对"""
    batch_idx, batch, query_db_map, target_db_map, pair_result_dir, fs_tmp_path = args
    batch_results = []
    
    for i, pair_info in enumerate(batch):
        # 传递工作进程ID以创建唯一的临时目录
        result = process_pdb_pair((pair_info, query_db_map, target_db_map, pair_result_dir, fs_tmp_path, batch_idx))
        if result:
            batch_results.append(result)
            
        # 定期输出进度
        if (i + 1) % 100 == 0:
            print(f"批次 {batch_idx}: 已处理 {i+1}/{len(batch)} 对")
    
    return batch_results

      
def process_pdb_pairs_from_csv_parallel(
    csv_path, 
    fs_querypdb_base_dir, 
    fs_targetpdb_base_dir, 
    fs_results_dir,
    fs_tmp_path,
    num_processes=64  # 默认使用64个进程，可以根据实际情况调整
):
    """
    并行处理CSV中的PDB对，先去重创建数据库，再为每对并行运行foldseek查询
    
    参数:
    csv_path: 10_dali_results_pro.csv的路径
    fs_querypdb_base_dir: 查询PDB文件的基础目录
    fs_targetpdb_base_dir: 目标PDB文件的基础目录
    fs_results_dir: 结果文件的保存目录
    fs_tmp_path: foldseek临时目录的基础路径
    num_processes: 并行进程数
    """
    start_time = time.time()
    
    # 确保结果目录存在
    os.makedirs(fs_results_dir, exist_ok=True)
    
    # 为query和target创建主文件夹
    main_query_dir = os.path.join(fs_results_dir, "all_queries")
    main_target_dir = os.path.join(fs_results_dir, "all_targets")
    os.makedirs(main_query_dir, exist_ok=True)
    os.makedirs(main_target_dir, exist_ok=True)
    
    print(f"读取CSV文件: {csv_path}")
    # 读取CSV文件
    df = pd.read_csv(csv_path, encoding='latin1')
    print(f"CSV文件包含 {len(df)} 行数据")
    
    # 提取所有唯一的query和target PDB文件
    unique_query_pdbs = set()
    unique_target_pdbs = set()
    
    # 从CSV中提取所有唯一的PDB文件
    for index, row in df.iterrows():
        # 检查targetrange的长度是否满足最小长度要求
        try:
            # 假设targetrange格式为"start-end"
            target_range = row['targetrange'].split('-')
            if len(target_range) == 2:
                start_pos = int(target_range[0])
                end_pos = int(target_range[1])
                if end_pos - start_pos < 15:
                    # 如果长度小于15，跳过这一行
                    print(f"跳过处理行 {index}: targetrange长度 {end_pos - start_pos} 小于15")
                    continue
        except (ValueError, IndexError) as e:
            # 处理异常情况，例如targetrange格式不正确
            print(f"处理行 {index} 的targetrange时出错: {e}")
            continue
        
        # 只有当targetrange长度满足要求时，才处理这一行
        query_pdb = f"{row['source']}_{row['sourcerange_abs']}.pdb"
        target_pdb = f"{row['target']}_{row['targetrange']}.pdb"
        
        unique_query_pdbs.add(query_pdb)
        unique_target_pdbs.add(target_pdb)
    
    print(f"找到 {len(unique_query_pdbs)} 个唯一的查询PDB文件")
    print(f"找到 {len(unique_target_pdbs)} 个唯一的目标PDB文件")
    
    # 创建查询数据库的目录
    query_db_dir = os.path.join(main_query_dir, "db")
    os.makedirs(query_db_dir, exist_ok=True)
    
    # 创建目标数据库的目录
    target_db_dir = os.path.join(main_target_dir, "db")
    os.makedirs(target_db_dir, exist_ok=True)
    

    # 使用进程池并行处理查询PDB
    print("开始并行处理查询PDB文件...")
    # 创建部分应用函数，预先绑定一些参数
    process_query_partial = partial(
        process_query_pdb,
        fs_querypdb_base_dir=fs_querypdb_base_dir,
        main_query_dir=main_query_dir,
        query_db_dir=query_db_dir
    )
    
    with mp.Pool(processes=min(num_processes, len(unique_query_pdbs))) as pool:
        query_results = pool.map(process_query_partial, unique_query_pdbs)
    
    # 构建查询PDB到数据库路径的映射
    query_db_map = {pdb: db_path for pdb, db_path in query_results if db_path is not None}
    print(f"成功处理 {len(query_db_map)} 个查询PDB文件")
    
    # 使用进程池并行处理目标PDB
    print("开始并行处理目标PDB文件...")
    
    # 创建部分应用函数，预先绑定一些参数
    process_target_partial = partial(
        process_target_pdb,
        fs_targetpdb_base_dir=fs_targetpdb_base_dir,
        main_target_dir=main_target_dir,
        target_db_dir=target_db_dir
    )
    
    with mp.Pool(processes=min(num_processes, len(unique_target_pdbs))) as pool:
        target_results = pool.map(process_target_partial, unique_target_pdbs)
    
    # 构建目标PDB到数据库路径的映射
    target_db_map = {pdb: db_path for pdb, db_path in target_results if db_path is not None}
    print(f"成功处理 {len(target_db_map)} 个目标PDB文件")
    
    # 准备CSV行的批次以进行并行处理
    # 准备CSV行的批次以进行并行处理
    pair_data = []
    for index, row in df.iterrows():
        # 检查targetrange的长度是否满足最小长度要求
        try:
            # 假设targetrange格式为"start-end"
            target_range = row['targetrange'].split('-')
            if len(target_range) == 2:
                start_pos = int(target_range[0])
                end_pos = int(target_range[1])
                if end_pos - start_pos < 15:
                    # 如果长度小于15，跳过这一行
                    print(f"!!!!!run foldsek跳过处理行 {index}: targetrange长度 {end_pos - start_pos} 小于15")
                    continue
        except (ValueError, IndexError) as e:
            # 处理异常情况，例如targetrange格式不正确
            print(f"处理行 {index} 的targetrange时出错: {e}")
            continue
        
        # 只有当targetrange长度满足要求时，才处理这一行
        print(f"!!!!!run foldsek会要处理行 {index}: targetrange长度 {end_pos - start_pos} 大于15")
        query_pdb = f"{row['source']}_{row['sourcerange_abs']}.pdb"
        target_pdb = f"{row['target']}_{row['targetrange']}.pdb"
        
        # 只添加有效的PDB对
        if query_pdb in query_db_map and target_db_map.get(target_pdb):
            pair_data.append((index, row, query_pdb, target_pdb))
    
    # 创建结果目录
    pair_result_dir = os.path.join(fs_results_dir, "pair_results")
    os.makedirs(pair_result_dir, exist_ok=True)
    
    # 将对数据分成批次以避免创建太多进程
    batch_size = max(1, len(pair_data) // num_processes)
    batches = [pair_data[i:i + batch_size] for i in range(0, len(pair_data), batch_size)]
    
    print(f"开始并行处理 {len(pair_data)} 个PDB对，分为 {len(batches)} 个批次...")
    
    # 使用进程池并行处理PDB对
    all_m8_files = []
    
    # 准备批处理参数
    batch_args = [
        (i, batch, query_db_map, target_db_map, pair_result_dir, fs_tmp_path)
        for i, batch in enumerate(batches)
    ]
    
    with mp.Pool(processes=min(num_processes, len(batches))) as pool:
        batch_results = pool.map(process_batch, batch_args)
    
    # 展平结果列表
    for results in batch_results:
        all_m8_files.extend([r for r in results if r])
    
    print(f"成功处理 {len(all_m8_files)} 个PDB对")
    
    # 合并所有.m8文件
    print("开始合并结果文件...")
    merged_m8 = os.path.join(fs_results_dir, "merged_results.m8")
    with open(merged_m8, 'w') as outfile:
        # 直接合并所有文件内容，不特别处理标题行
        for m8_file in all_m8_files:
            try:
                with open(m8_file, 'r') as infile:
                    # 直接复制整个文件内容
                    outfile.write(infile.read())
            except Exception as e:
                print(f"合并文件 {m8_file} 时出错: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"所有结果已合并到: {merged_m8}")
        print(f"总处理时间: {total_time:.2f} 秒")
        return merged_m8

def process_pdb_pairs_from_csv_optimized(
    csv_path, 
    fs_querypdb_base_dir, 
    fs_targetpdb_base_dir, 
    fs_results_dir,
    fs_tmp_path
):
    """
    从CSV文件处理PDB对，先去重创建数据库，再为每对运行foldseek查询，并合并结果
    
    参数:
    csv_path: 10_dali_results_pro.csv的路径
    fs_querypdb_base_dir: 查询PDB文件的基础目录
    fs_targetpdb_base_dir: 目标PDB文件的基础目录
    fs_results_dir: 结果文件的保存目录
    fs_tmp_path: foldseek临时目录
    """
    # 确保结果目录存在
    os.makedirs(fs_results_dir, exist_ok=True)
    
    # 为query和target创建主文件夹
    main_query_dir = os.path.join(fs_results_dir, "all_queries")
    main_target_dir = os.path.join(fs_results_dir, "all_targets")
    os.makedirs(main_query_dir, exist_ok=True)
    os.makedirs(main_target_dir, exist_ok=True)
    
    # 读取CSV文件
    df = pd.read_csv(csv_path, encoding='latin1')
    
    # 提取所有唯一的query和target PDB文件
    unique_query_pdbs = set()
    unique_target_pdbs = set()
    
    # 创建映射字典来跟踪PDB文件和它们的数据库路径
    query_db_map = {}
    target_db_map = {}
    
    # 从CSV中提取所有唯一的PDB文件
    for index, row in df.iterrows():
        query_pdb = f"{row['source']}_{row['sourcerange_abs']}.pdb"
        target_pdb = f"{row['target']}_{row['targetrange']}.pdb"
        
        unique_query_pdbs.add(query_pdb)
        unique_target_pdbs.add(target_pdb)
    
    print(f"找到 {len(unique_query_pdbs)} 个唯一的查询PDB文件")
    print(f"找到 {len(unique_target_pdbs)} 个唯一的目标PDB文件")
    
    # 处理所有唯一的查询PDB文件
    query_db_dir = os.path.join(main_query_dir, "db")
    os.makedirs(query_db_dir, exist_ok=True)
    
    for query_pdb in unique_query_pdbs:
        # 创建唯一的数据库名称
        query_db_name = query_pdb.replace(".", "_").replace("/", "_")
        
        # 复制PDB文件到查询目录
        src_query_path = os.path.join(fs_querypdb_base_dir, query_pdb)
        dst_query_dir = os.path.join(main_query_dir, "pdbs")
        os.makedirs(dst_query_dir, exist_ok=True)
        dst_query_path = os.path.join(dst_query_dir, query_pdb)
        
        if os.path.exists(src_query_path):
            shutil.copy(src_query_path, dst_query_path)
            
            # 为这个PDB创建单独的文件夹
            single_query_dir = os.path.join(main_query_dir, "single_pdbs", query_db_name)
            os.makedirs(single_query_dir, exist_ok=True)
            single_query_path = os.path.join(single_query_dir, query_pdb)
            shutil.copy(src_query_path, single_query_path)
            
            # 为这个PDB创建数据库
            gpr.convert_pdb_to_foldseek_db(single_query_dir, query_db_dir+'/'+query_db_name, query_db_name)
            
            # 记录这个PDB对应的数据库路径
            query_db_map[query_pdb] = os.path.join(query_db_dir+'/'+query_db_name, f"{query_db_name}.db")
            
            print(f"处理查询PDB: {query_pdb}")
        else:
            print(f"警告: 找不到查询PDB文件: {src_query_path}")
    
    # 处理所有唯一的目标PDB文件
    target_db_dir = os.path.join(main_target_dir, "db")
    os.makedirs(target_db_dir, exist_ok=True)
    
    for target_pdb in unique_target_pdbs:
        # 创建唯一的数据库名称
        target_db_name = target_pdb.replace(".", "_").replace("/", "_")
        
        # 复制PDB文件到目标目录
        src_target_path = os.path.join(fs_targetpdb_base_dir, target_pdb)
        dst_target_dir = os.path.join(main_target_dir, "pdbs")
        os.makedirs(dst_target_dir, exist_ok=True)
        dst_target_path = os.path.join(dst_target_dir, target_pdb)
        
        if os.path.exists(src_target_path):
            shutil.copy(src_target_path, dst_target_path)
            
            # 为这个PDB创建单独的文件夹
            single_target_dir = os.path.join(main_target_dir, "single_pdbs", target_db_name)
            os.makedirs(single_target_dir, exist_ok=True)
            single_target_path = os.path.join(single_target_dir, target_pdb)
            shutil.copy(src_target_path, single_target_path)
            
            # 为这个PDB创建数据库
            gpr.convert_pdb_to_foldseek_db(single_target_dir, target_db_dir+'/'+target_db_name, target_db_name)
            
            # 记录这个PDB对应的数据库路径
            target_db_map[target_pdb] = os.path.join(target_db_dir+'/'+target_db_name, f"{target_db_name}.db")
            
            print(f"处理目标PDB: {target_pdb}")
        else:
            print(f"警告: 找不到目标PDB文件: {src_target_path}")
    
    # 用于存储所有m8文件路径的列表
    all_m8_files = []
    
    # 现在遍历CSV的每一行，为每对PDB运行foldseek
    for index, row in df.iterrows():
        query_pdb = f"{row['source']}_{row['sourcerange_abs']}.pdb"
        target_pdb = f"{row['target']}_{row['targetrange']}.pdb"
        
        # 检查是否有这对PDB的数据库
        if query_pdb in query_db_map and target_pdb in target_db_map:
            query_db_path = query_db_map[query_pdb]
            target_db_path = target_db_map[target_pdb]
            
            # 为这对PDB创建结果目录
            pair_result_dir = os.path.join(fs_results_dir, f"pair_results")
            os.makedirs(pair_result_dir, exist_ok=True)
            
            # 创建唯一的结果文件名
            result_name = f"{row['source']}_{row['sourcerange_abs']}__vs__{row['target']}_{row['targetrange']}"
            result_name = result_name.replace(".", "_").replace("/", "_")
            result_m8 = os.path.join(pair_result_dir, f"{result_name}.m8")
            
            # 运行foldseek
            gpr. run_foldseek_with_error_handling(
                query_db_path, 
                target_db_path,
                fs_rawdata=os.path.join(pair_result_dir, f"{result_name}.raw"),
                fs_results=result_m8,
                tmp_dir=fs_tmp_path
            )
            
            # 将结果添加到列表
            all_m8_files.append(result_m8)
            print(f"处理完成 pair {index}: {query_pdb} vs {target_pdb}")
        else:
            print(f"警告: 找不到 pair {index} 的数据库: {query_pdb} 或 {target_pdb}")
    
    # 合并所有.m8文件
    merged_m8 = os.path.join(fs_results_dir, "merged_results.m8")
    with open(merged_m8, 'w') as outfile:
        # 写入第一个文件的标题行（如果有）
        if all_m8_files:
            with open(all_m8_files[0], 'r') as first_file:
                header = first_file.readline()
                outfile.write(header)
        
        # 合并所有文件的内容
        for m8_file in all_m8_files:
            with open(m8_file, 'r') as infile:
                # 跳过标题行（如果有）
                next(infile, None)
                # 写入其余内容
                for line in infile:
                    outfile.write(line)
    
    print(f"所有结果已合并到: {merged_m8}")
    return merged_m8


def importpl_pdb_target_parallel(Dali_bin,pdb_dir, dat_dir, work_path, max_workers=8, batch_size=10):
    print("Starting import process with additional verification for dat_dir consistency.")
    print(f"max_workers =  {max_workers}")
    print(f"batch_size =  {batch_size}")
    
    os.makedirs(dat_dir, exist_ok=True)
    os.makedirs(work_path, exist_ok=True)
    
    # CSV 文件路径
    csv_file = os.path.join(work_path, "dat", "importpl_target_ids.csv")
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # 获取待处理的 PDB 文件
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    print(f"pdb_files =  {len(pdb_files)}")
    
    all_pdb_ids = list(generate_pdb_ids_target(len(pdb_files)))
    print(f"all_pdb_ids =  {len(all_pdb_ids)}")
    
    # 读取 CSV 文件中的记录
    processed_files = {}
    if os.path.exists(csv_file):
        with open(csv_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            for row in reader:
                processed_files[row[0]] = row[1]
    print(f"processed_files =  {len(processed_files)}")
    
    # 检查匹配
    mismatch_found = False
    for pdb_file, pdb_id in zip(pdb_files, all_pdb_ids):
        pdb_name = pdb_file.split('.')[0]
        if pdb_name in processed_files and processed_files[pdb_name] != pdb_id:
            print(f"Warning: Mismatch found in CSV for {pdb_name}. Expected ID: {pdb_id}, Found ID: {processed_files[pdb_name]}")
            mismatch_found = True

    if mismatch_found:
        print("Warning: Some records in the CSV file do not match the expected file-ID correspondence.")
    
    # 检查 dat_dir 中的文件
    existing_dat_files = {f[:-5] for f in os.listdir(dat_dir) if f.endswith("A.dat")}
    verified_files = set()
    
    for pdb_name, pdb_id in processed_files.items():
        if pdb_id in existing_dat_files:
            verified_files.add(pdb_name)
        else:
            print(f"Warning: {pdb_name} with ID {pdb_id} is in CSV but not found in dat_dir. It will be reprocessed.")
    
    # 筛选出待处理的文件
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
                # 分批次提交任务
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
                    
                    # 处理完成的任务
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
    digits = '123456789'  # 注意首位不为0，其他位置可以为0

    count = 0

    # 1. 全大写字母的组合
    for combo in itertools.product(uppercase, repeat=4):
        yield "".join(combo)
        count += 1
        if count >= n:
            return

    # 2. 包含 1 个小写字母的组合
    for pos in range(4):
        for combo in itertools.product(uppercase, repeat=3):
            for l in lowercase:
                temp = list(combo)
                temp.insert(pos, l)
                yield "".join(temp)
                count += 1
                if count >= n:
                    return

    # 3. 包含 1 个数字的组合
    for pos in range(4):
        for combo in itertools.product(uppercase, repeat=3):
            for d in digits:
                temp = list(combo)
                temp.insert(pos, d)
                if pos != 0 or d != '0':  # 首位不能是0
                    yield "".join(temp)
                    count += 1
                    if count >= n:
                        return

    # 4. 包含 2 个小写字母的组合
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

    # 5. 包含 2 个数字的组合
    for pos1, pos2 in itertools.combinations(range(4), 2):
        for combo in itertools.product(uppercase, repeat=2):
            for d1, d2 in itertools.product(digits, repeat=2):
                temp = list(combo)
                for p, d in zip([pos1, pos2], [d1, d2]):
                    temp.insert(p, d)
                if pos1 != 0 or d1 != '0':  # 首位不能是0
                    if pos2 != 0 or d2 != '0':  # 首位不能是0
                        yield "".join(temp)
                        count += 1
                        if count >= n:
                            return

    # 6. 包含 3 个小写字母的组合
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

    # 7. 包含 3 个数字的组合
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

    # 8. 全小写字母的组合
    for combo in itertools.product(lowercase, repeat=4):
        yield "".join(combo)
        count += 1
        if count >= n:
            return

    # 9. 全数字的组合
    for combo in itertools.product(digits + '0', repeat=4):
        if combo[0] != '0':  # 首位不为0
            yield "".join(combo)
            count += 1
            if count >= n:
                return

    # 10. 大小写字母和数字组合的复杂情况
    # 10.1 大写字母有 2 个，小写字母 1 个，数字 1 个
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
                        if temp[0] != '0':  # 首位不能是0
                            yield "".join(temp)
                            count += 1
                            if count >= n:
                                return

    # 10.2 小写字母有 2 个，大写字母 1 个，数字 1 个
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

    # 10.3 数字有 2 个，大写字母 1 个，小写字母 1 个
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


# 单个 PDB 文件的导入逻辑
def import_single_pdb(Dali_bin,pdb_file, pdb_id, pdb_dir, dat_dir, work_path):
    short_work_dir = os.path.join(work_path, f"dali_temp_{pdb_id}")
    os.makedirs(short_work_dir, exist_ok=True)

    pdb_path = os.path.join(pdb_dir, pdb_file)
    short_pdb_path = os.path.join(short_work_dir, f"{pdb_id}.pdb")
    shutil.copyfile(pdb_path, short_pdb_path)

    short_dat_dir = os.path.join(short_work_dir, "dat")
    os.makedirs(short_dat_dir, exist_ok=True)

    # 构建并运行 import.pl 的命令
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
    
    # 确保Dali的数据目录存在
    ensure_dir(dat_dir)  
    
    # 创建一个短路径的工作目录，用于运行 import.pl
    short_work_dir = os.path.join(work_path, "dali_work/query")
    if not os.path.exists(short_work_dir):
        os.makedirs(short_work_dir)
    
    # 创建ID生成器
    pdb_id_generator = generate_pdb_ids_query()

    # 获取所有PDB文件
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    csv_file = os.path.join(work_path,"dat","importpl_query_ids.csv")
    # 打开CSV文件记录原始文件名和新生成的ID
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Original_PDB_File", "Generated_PDB_ID"])  # CSV标题行
        
        for pdb_file in pdb_files:
            pdb_path = os.path.join(pdb_dir, pdb_file)
            pdb_id = next(pdb_id_generator)  # 从生成器中获取下一个唯一ID
            
            # 复制PDB文件到短路径的工作目录，并重命名
            short_pdb_path = os.path.join(short_work_dir, f"{pdb_id}.pdb")
            shutil.copyfile(pdb_path, short_pdb_path)
            
            # 确保 dat_dir 也被切换到一个相对短路径
            short_dat_dir = os.path.join(short_work_dir, "dat")
            if not os.path.exists(short_dat_dir):
                os.makedirs(short_dat_dir)

            # 构建并运行 import.pl 的命令
            cmd = [
                os.path.join(Dali_bin, "import.pl"), 
                "--pdbfile", f"./{pdb_id}.pdb",  # 使用相对路径
                "--pdbid", pdb_id, 
                "--dat", "./dat"  # 相对路径简化为 "./dat"
            ]
            
            # 切换到短路径目录进行命令执行
            original_dir = os.getcwd()
            os.chdir(short_work_dir)
            try:
                subprocess.run(cmd)
            finally:
                # 在完成后，切换回原来的工作目录
                os.chdir(original_dir)
            
            # 将生成的结果复制回到原始 dat_dir
            for file in os.listdir(short_dat_dir):
                shutil.move(os.path.join(short_dat_dir, file), dat_dir)
            
            # 将原始文件名和生成的PDB ID记录到CSV文件
            writer.writerow([pdb_file.split('.')[0], pdb_id])

    print("PDB files have been imported and logged successfully.")
def generate_pdb_ids_query():
    """ Generator that creates unique 4-character IDs in the desired sequence: 0A00, 0A01, ..., 9z99, 00AA, etc. """
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    uletters ='0'  # 扩展字符集
    digits = '0123456789'
    
    # 1. 从 0A00, 0A01, ..., 9z99 开始生成
    for digit in uletters:
        for letter in letters:
            for num in range(100):  # 生成000到999的数字
                yield f"{digit}{letter}{num:02d}"  # 格式化为 0A000, 0A001, ..., 9z999

    # 2. 生成 00AA, 00AB, ..., 99zz
    for digit1 in uletters:
        for digit2 in digits:
            for letter1 in letters:
                for num in range(10):  # 生成00到99的数字
                    yield f"{digit1}{digit2}{letter1}{num:01d}"  # 格式化为 00AA00, 00AA01, ..., 99zz99

    # 3. 生成 0AAA, 0AAB, ..., 9zzz
    for digit in uletters:
        for letter1 in letters:
            for letter2 in letters:
                for letter3 in letters:
                    yield f"{digit}{letter1}{letter2}{letter3}"  # 格式化为 0AAA0, 0AAA1, ..., 9zzz9