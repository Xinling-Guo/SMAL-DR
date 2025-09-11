import json
import os
import sys
import pandas as pd
import logging
from pathlib import Path

# Add utils path and import
utils_path = '/mnt/sdb4/protein_gen/Cas9_domain_work/code_submit/'
sys.path.append(utils_path)
import utils1 as task1_utils
import utils2 as task2_utils
import PairNet_MLP_train as task3_utils
import PairNet_MLP_inference as task4_utils
import PairNet_Transformer_train as task5_utils
import PairNet_Transformer_inference as task6_utils

class SMALDRTask1:
    def __init__(self, config_path):
        """Initialize configuration and setup working environment"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.work_dir = self.config['work_dir']
        self.protein = self.config['proteins'][0]  # Take first protein
        self.key = self.protein['key']
        self.ipr_ids = self.protein['ipr_ids']
        self.ipr_names = self.protein['ipr_names']
        
        # Ensure working directory exists
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Set up logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup basic logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.work_dir, 'smal_dr_phase1.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    

        
    def get_full_path(self, path):
        """
        Convert any path to absolute path. 
        - If already absolute: return as-is
        - If relative: resolve relative to work_dir
        - Handles ../ and ./ references correctly
        """
        if os.path.isabs(path):
            return path
        else:
            return os.path.abspath(os.path.join(self.work_dir, path))
    
    def validate_config(self):
        """Validate configuration parameters"""
        required_keys = ['work_dir', 'proteins', 'input_files']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        if len(self.config['proteins']) == 0:
            raise ValueError("No proteins defined in configuration")
    
    def validate_input_files(self):
        """Validate that all required input files exist"""
        input_files = self.config['input_files']
        
        for file_key, file_path in input_files.items():
            full_path = self.get_full_path(file_path)
            
            # Check if it's a directory (ends with '/' or has _dir in key)
            if file_path.endswith('/') or file_key.endswith('_dir'):
                if not os.path.exists(full_path):
                    self.logger.warning(f"Directory does not exist: {full_path}")
                    os.makedirs(full_path, exist_ok=True)
                    self.logger.info(f"Created directory: {full_path}")
            else:
                # It's a file
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"Input file not found: {full_path}")
                self.logger.debug(f"Found input file: {full_path}")
    
    def run_step0(self):
        """Step 0: Get true label PDB and domain-reviewed data"""
        if not self.config.get('run_step0', False):
            self.logger.info("Skipping Step 0 (run_step0 is false)")
            return
            
        self.logger.info("Running Step 0: Get true label PDB and domain-reviewed")
        true_label_dir = os.path.join(self.work_dir, "true_label")
        os.makedirs(true_label_dir, exist_ok=True)
        
        task1_utils.get_true_label_pdb_and_domain(
            self.ipr_ids, self.ipr_names, true_label_dir, self.key
        )
        self.logger.info("Step 0 completed successfully")
    
    def run_step1(self):
        """Step 1: Get true Cas9 TED CATH information"""
        if not self.config.get('run_step1', False):
            self.logger.info("Skipping Step 1 (run_step1 is false)")
            return
            
        self.logger.info("Running Step 1: Get true Cas9 TED CATH info")
        
        # Use the path resolver method
        input_file = self.get_full_path(self.config['input_files']['true_cas9_uniprot_info'])
        output_file = os.path.join(self.work_dir, "cas9_ted_info.csv")
        
        task1_utils.get_protein_ted_info_to_csv(input_file, output_file)
        self.logger.info("Step 1 completed successfully")
    
    def run_step2_cathdb(self):
        """Step 2.1: Process CATH database data"""
        if not self.config.get('run_step2_cathdb', False):
            self.logger.info("Skipping Step 2.1 CATHDB (run_step2_cathdb is false)")
            return
            
        self.logger.info("Running Step 2.1: Process CATH database data")
        
        # Use path resolver for all input files
        true_hnh_ted_info = self.get_full_path(self.config['input_files']['true_hnh_ted_info'])
        hnh_cath_in_cathdb = os.path.join(self.work_dir, "HNH_cath_in_cathdb.csv")
        protein_cathdb_dir = os.path.join(self.work_dir, "pdb_cathdb")
        domain_cathdb_dir = os.path.join(self.work_dir, "domain_cathdb")
        cath_boundary_file = self.get_full_path(self.config['input_files']['cath_boundary_file'])
        
        # Create directories
        os.makedirs(protein_cathdb_dir, exist_ok=True)
        os.makedirs(domain_cathdb_dir, exist_ok=True)
        
        # Execute CATH processing steps
        task1_utils.get_domains_from_cath(true_hnh_ted_info, hnh_cath_in_cathdb)
        task1_utils.add_uniprot_to_output(hnh_cath_in_cathdb, hnh_cath_in_cathdb)
        
        # Download and process PDB files
        pdb_cathdb_fasta = os.path.join(self.work_dir, "pdb_cathdb_fasta.fasta")
        domain_cathdb_fasta = os.path.join(self.work_dir, "domain_cathdb_fasta.fasta")
        
        task1_utils.download_pdb_and_fasta_by_cathid_parallel(
            hnh_cath_in_cathdb, protein_cathdb_dir, pdb_cathdb_fasta
        )
        
        task1_utils.split_domain_by_cath_boundary_and_save_fasta_parallel(
            protein_cathdb_dir, 
            cath_boundary_file,
            domain_cathdb_dir, 
            domain_cathdb_fasta, 
            hnh_cath_in_cathdb, 
            pdb_cathdb_fasta
        )
        
        self.logger.info("Step 2.1 completed successfully")
    
    def run_step2_cath_teddb(self):
        """Step 2.2: Process CATH-TED database data"""
        if not self.config.get('run_step2_cath_teddb', False):
            self.logger.info("Skipping Step 2.2 CATH-TEDDB (run_step2_cath_teddb is false)")
            return
            
        self.logger.info("Running Step 2.2: Process CATH-TED database data")
        
        # Get input files
        true_hnh_ted_info = self.get_full_path(self.config['input_files']['true_hnh_ted_info'])
        ted_domain_cath_info = self.get_full_path(self.config['input_files']['ted_domain_cath_info'])
        hnh_cath_in_ted = os.path.join(self.work_dir, "HNH_cath_in_ted.csv")
        
        # Get TED domains by CATH label  need more time gxl
        task1_utils.get_ted_domains_by_cath_label(true_hnh_ted_info, ted_domain_cath_info, hnh_cath_in_ted)
        
        # Setup directories
        protein_cath_teddb_dir = os.path.join(self.work_dir, "pdb_cath_teddb")
        domain_cath_teddb_dir = os.path.join(self.work_dir, "domain_cath_teddb")
        pdb_cath_teddb_fasta_dir = os.path.join(self.work_dir, "pdb_cath_teddb_fasta")
        
        os.makedirs(protein_cath_teddb_dir, exist_ok=True)
        os.makedirs(domain_cath_teddb_dir, exist_ok=True)
        os.makedirs(pdb_cath_teddb_fasta_dir, exist_ok=True)
        
        # Download and process data
        pdb_cath_teddb_fasta = os.path.join(self.work_dir, "pdb_cath_teddb_fasta.fasta")
        domain_cath_teddb_fasta = os.path.join(self.work_dir, "domain_cath_teddb_fasta.fasta")
        
        task1_utils.download_pdb_and_fasta_from_AFDB_parallel(hnh_cath_in_ted, protein_cath_teddb_dir, pdb_cath_teddb_fasta)
        
        task1_utils.split_domain_by_ted_boundary_and_save_fasta_parallel( hnh_cath_in_ted, protein_cath_teddb_dir, domain_cath_teddb_dir, domain_cath_teddb_fasta, pdb_cath_teddb_fasta )
        
        self.logger.info("Step 2.2 completed successfully")
    
    def run_step2_cluster_teddb(self):
        """Step 2.3: Process Cluster-TED database data"""
        if not self.config.get('run_step2_cluster_teddb', False):
            self.logger.info("Skipping Step 2.3 Cluster-TEDDB (run_step2_cluster_teddb is false)")
            return
            
        self.logger.info("Running Step 2.3: Process Cluster-TED database data")
        
        # Get input files
        true_hnh_ted_info = self.get_full_path(self.config['input_files']['true_hnh_ted_info'])
        ted_domain_cluster_info = self.get_full_path(self.config['input_files']['ted_domain_cluster_info'])
        up_true_hnh_ted_info = os.path.join(self.work_dir, "HNH_TED_info.xlsx") # need to get by yourself
        hnh_cluster_in_ted = os.path.join(self.work_dir, "HNH_cluster_in_ted.csv")
        ted_domain_cath_info = self.get_full_path(self.config['input_files']['ted_domain_cath_info'])

        
        # Get TED domains by cluster
        task1_utils.get_ted_domains_by_cluster(true_hnh_ted_info, ted_domain_cluster_info, hnh_cluster_in_ted)
        
        # Add domain range information
        task1_utils.add_domain_range_with_ted_info(hnh_cluster_in_ted, ted_domain_cath_info, hnh_cluster_in_ted)
        
        # Setup directories
        protein_cluster_teddb_dir = os.path.join(self.work_dir, "pdb_cluster_teddb")
        domain_cluster_teddb_dir = os.path.join(self.work_dir, "domain_cluster_teddb")
        pdb_cluster_teddb_fasta_dir = os.path.join(self.work_dir, "pdb_cluster_teddb_fasta")
        
        os.makedirs(protein_cluster_teddb_dir, exist_ok=True)
        os.makedirs(domain_cluster_teddb_dir, exist_ok=True)
        os.makedirs(pdb_cluster_teddb_fasta_dir, exist_ok=True)
        
        # Download and process data
        pdb_cluster_teddb_fasta = os.path.join(self.work_dir, "pdb_cluster_teddb_fasta.fasta")
        domain_cluster_teddb_fasta = os.path.join(self.work_dir, "domain_cluster_teddb_fasta.fasta")
        
        task1_utils.download_pdb_and_fasta_from_AFDB_parallel( hnh_cluster_in_ted, protein_cluster_teddb_dir,  pdb_cluster_teddb_fasta )
        
        task1_utils.split_domain_by_ted_boundary_and_save_fasta_parallel( hnh_cluster_in_ted, protein_cluster_teddb_dir, domain_cluster_teddb_dir, domain_cluster_teddb_fasta, pdb_cluster_teddb_fasta)
        
        self.logger.info("Step 2.3 completed successfully")
    
    def run_step3_cath_teddb(self):
        """Step 3.1: Structural similarity analysis for CATH-TEDDB candidates"""
        if not self.config.get('run_step3_cath_teddb', False):
            self.logger.info("Skipping Step 3.1 CATH-TEDDB structural similarity (run_step3_cath_teddb is false)")
            return
            
        self.logger.info("Running Step 3.1: Structural similarity analysis for CATH-TEDDB candidates")
        
        # Get paths from config
        wt_domain_dir = self.get_full_path(self.config['input_files']['wt_domain_dir'])
        domain_cath_teddb_dir = os.path.join(self.work_dir, "domain_cath_teddb")
        
        # Set up directories for structural similarity analysis
        similarity_dir = os.path.join(self.work_dir, "protein_relation_v3")
        fs_querydb_name = "HNH_truedb"
        fs_targetdb_name = "HNH_cath_teddb"
        
        fs_querydb_path = os.path.join(similarity_dir, "domain_cath_teddb", "foldseek_results", fs_querydb_name)
        fs_targetdb_path = os.path.join(similarity_dir, "domain_cath_teddb", "foldseek_results", fs_targetdb_name)
        fs_results_dirs = os.path.join(similarity_dir, "domain_cath_teddb", "foldseek_results", "results")
        
        # Ensure directories exist
        os.makedirs(fs_results_dirs, exist_ok=True)
        os.makedirs(fs_querydb_path, exist_ok=True)
        os.makedirs(fs_targetdb_path, exist_ok=True)
        
        # Convert PDB files to Foldseek databases
        task1_utils.convert_pdb_to_foldseek_db(wt_domain_dir, fs_querydb_path, fs_querydb_name)
        task1_utils.convert_pdb_to_foldseek_db(domain_cath_teddb_dir, fs_targetdb_path, fs_targetdb_name)
        
        # Run Foldseek structural alignment
        fs_result_file = os.path.join(fs_results_dirs, f"{fs_querydb_name}.m8")
        task1_utils.run_foldseek(
            os.path.join(fs_querydb_path, f"{fs_querydb_name}.db"),
            os.path.join(fs_targetdb_path, f"{fs_targetdb_name}.db"),
            fs_results=fs_result_file)
        

        # Create cytoscape network from Foldseek results
        network_dir = os.path.join(similarity_dir, "domain_cath_teddb", "cytoscape_network")
        os.makedirs(network_dir, exist_ok=True)
        
        task1_utils.create_fs_cytoscape_network(fs_result_file, os.path.join(network_dir, "cas9_fs_edge.csv"))
        
        self.logger.info("Step 3.1 completed successfully")

    def run_step3_cluster_teddb(self):
        """Step 3.2: Structural similarity analysis for Cluster-TEDDB candidates"""
        if not self.config.get('run_step3_cluster_teddb', False):
            self.logger.info("Skipping Step 3.2 Cluster-TEDDB structural similarity (run_step3_cluster_teddb is false)")
            return
            
        self.logger.info("Running Step 3.2: Structural similarity analysis for Cluster-TEDDB candidates")
        
        # Get paths from config
        wt_domain_dir = self.get_full_path(self.config['input_files']['wt_domain_dir'])
        domain_cluster_teddb_dir = os.path.join(self.work_dir, "domain_cluster_teddb")
        
        # Set up directories for structural similarity analysis
        similarity_dir = os.path.join(self.work_dir, "protein_relation_v3")
        fs_querydb_name = "HNH_truedb"
        fs_targetdb_name = "HNH_cluster_teddb"
        
        fs_querydb_path = os.path.join(similarity_dir, "domain_cluster_teddb", "foldseek_results", fs_querydb_name)
        fs_targetdb_path = os.path.join(similarity_dir, "domain_cluster_teddb", "foldseek_results", fs_targetdb_name)
        fs_results_dirs = os.path.join(similarity_dir, "domain_cluster_teddb", "foldseek_results", "results")
        
        # Ensure directories exist
        os.makedirs(fs_results_dirs, exist_ok=True)
        os.makedirs(fs_querydb_path, exist_ok=True)
        os.makedirs(fs_targetdb_path, exist_ok=True)
        
        # Convert PDB files to Foldseek databases
        task1_utils.convert_pdb_to_foldseek_db(wt_domain_dir, fs_querydb_path, fs_querydb_name)
        task1_utils.convert_pdb_to_foldseek_db(domain_cluster_teddb_dir, fs_targetdb_path, fs_targetdb_name)
        
        # Run Foldseek structural alignment
        fs_result_file = os.path.join(fs_results_dirs, f"{fs_querydb_name}.m8")
        task1_utils.run_foldseek(
            os.path.join(fs_querydb_path, f"{fs_querydb_name}.db"),
            os.path.join(fs_targetdb_path, f"{fs_targetdb_name}.db"),
             fs_results=fs_result_file
        )
        
        # Create cytoscape network from Foldseek results
        network_dir = os.path.join(similarity_dir, "domain_cluster_teddb", "cytoscape_network")
        os.makedirs(network_dir, exist_ok=True)
        
        task1_utils.create_fs_cytoscape_network(fs_result_file, os.path.join(network_dir, "cas9_fs_edge.csv"))
        
        self.logger.info("Step 3.2 completed successfully")
    
    def run_data_processing(self):
        """Final data processing and filtering"""
        if not self.config.get('run_data_processing', False):
            self.logger.info("Skipping data processing (run_data_processing is false)")
            return
            
        self.logger.info("Running final data processing and filtering")
        
        # Load FASTA data
        pdb_cath_teddb_fasta = os.path.join(self.work_dir, "pdb_cath_teddb_fasta.fasta")
        pdb_cluster_teddb_fasta = os.path.join(self.work_dir, "pdb_cluster_teddb_fasta.fasta")
        all_fasta_path = self.get_full_path(self.config['input_files']['all_fasta'])
        task1_utils.merge_fasta_files([pdb_cath_teddb_fasta,pdb_cluster_teddb_fasta ],all_fasta_path)
        fasta_dict = task1_utils.load_fasta_to_dict(all_fasta_path)
        
        # Process cath-teddb data
        data_dir = os.path.join(self.work_dir, "protein_relation_v3")
        os.makedirs(data_dir, exist_ok=True)
        network_dir = os.path.join(data_dir, "domain_cath_teddb", "cytoscape_network")
        os.makedirs(network_dir, exist_ok=True)
        all_cath_teddb = os.path.join(network_dir, "domain_cath_teddb_SpCas9.csv")
        task1_utils.process_fs_reslult_new(network_dir+"/cas9_fs_edge.csv", all_cath_teddb)
        
        # Process cluster-teddb data
        network_dir_cluster = os.path.join(data_dir, "domain_cluster_teddb", "cytoscape_network")
        os.makedirs(network_dir_cluster, exist_ok=True)
        all_cluster_teddb = os.path.join(network_dir_cluster, "domain_cluster_teddb_SpCas9.csv")
        task1_utils.process_fs_reslult_new(network_dir_cluster+"/cas9_fs_edge.csv", all_cluster_teddb)
        
        # Add domain information
        hnh_cath_in_ted = os.path.join(self.work_dir, "HNH_cath_in_ted.csv")
        hnh_cluster_in_ted = os.path.join(self.work_dir, "HNH_cluster_in_ted.csv")
        
        task1_utils.add_domain_info_to_target_cath_teddb(all_cath_teddb, hnh_cath_in_ted, fasta_dict, all_cath_teddb)
        task1_utils.add_domain_info_to_target_cluster_teddb(all_cluster_teddb, hnh_cluster_in_ted, fasta_dict, all_cluster_teddb)
        
        # Merge data
        all_teddb_variant = os.path.join(data_dir, 'domain_All_teddb_SpCas9.csv')
        
        df1 = pd.read_csv(all_cath_teddb)
        df2 = pd.read_csv(all_cluster_teddb)
        merged_df = pd.concat([df1, df2], ignore_index=True)
        
        # Define column order
        first_columns = [
            'source','target', 'FS_weight','TM_weight','target_info','chopping',
            'source_domain_seq','target_domain_seq','target_len', 'plddt',
            'cath_label','Cluster_representative','protein_seq_sim','domain_seq_sim',
            'assemble_seq','assemle_protein_sim','target_seq','source_seq'
        ]
        
        # Reorder columns
        all_columns = list(merged_df.columns)
        remaining_columns = [col for col in all_columns if col not in first_columns]
        final_columns = first_columns + remaining_columns
        final_columns_existing = [col for col in final_columns if col in merged_df.columns]
        
        merged_df = merged_df[final_columns_existing]
        merged_df.to_csv(all_teddb_variant, index=False)

        
        # Apply filtering
        all_teddb_variant_filter = os.path.join(data_dir, 'domain_All_teddb_SpCas9_filter.csv')
        df = pd.read_csv(all_teddb_variant)
        
        # Filtering conditions
        filtered_df =df[(df['FS_weight'] >= 0.7) & (df['TM_weight'] >= 0.5)]
        filtered_df.to_csv(all_teddb_variant_filter, index=False)
        
        self.logger.info(f"Data processing completed. Filtered {len(filtered_df)} candidates from {len(df)} total.")
    
    def run_task1(self):
        """Execute all pipeline steps"""
        self.logger.info(f"Starting SMAL-DR Task 1 for protein: {self.key}")
        self.logger.info(f"Task1 : Structural fold mining from the TED database identifies diverse HNH-like domains for Cas9 engineering")
        self.logger.info(f"Working directory: {self.work_dir}")
        self.logger.info("=" * 60)
        
        try:
            self.validate_config()
            #self.validate_input_files()
            
            # Execute pipeline steps
            
            self.run_step0()
            self.run_step1()
            self.run_step2_cath_teddb()
            self.run_step2_cluster_teddb()
            self.run_step3_cath_teddb() 
            self.run_step3_cluster_teddb()  
            self.run_data_processing()
            
            self.logger.info("=" * 60)
            self.logger.info("Phase 1 completed successfully!")
            self.logger.info(f"NOTE: Candidate variants can be further filtered manually or based on structural similarity metrics for subsequent experimental validation in the next phase.")

            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            self.logger.exception("Detailed error traceback:")
            raise


class SMALDRTask2:
    """
    SMAL-DR Phase 2: Refining domain boundaries using DALI structural alignment 
    to enhance recombinational compatibility
    
    This class implements Task 2 of the project: Refining domain boundaries 
    using DALI structural alignment for Cas9 engineering
    """
    
    def __init__(self, config_path):
        """Initialize configuration for Task 2"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.work_dir = self.config['work_dir']
        self.protein = self.config['proteins'][0]
        self.key = self.protein['key']
        
        # Ensure working directory exists
        os.makedirs(self.work_dir, exist_ok=True)
        
        # Set up logging
        self.setup_logging()
        
        # Task 2 specific configuration
        self.task2_config = self.config.get('task2_config', {})
    
    def setup_logging(self):
        """Setup logging for Task 2"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.work_dir, 'smal_dr_phase2.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def get_full_path(self, path):
        """Convert path to absolute path"""
        if os.path.isabs(path):
            return path
        else:
            return os.path.abspath(os.path.join(self.work_dir, path))
    
    def run_step0_refine_domains(self):
        """Step 0: Refine J7 and Q9 HNH domains"""
        if not self.task2_config.get('run_step0_refine', False):
            self.logger.info("Skipping Step 0 domain refinement")
            return
            
        self.logger.info("Running Step 0: Refining J7 and Q9 HNH domains")
        
        # Implementation for domain refinement
        # This would use the split_domain_by_ted_boundary function
        pass
    
    def run_step1_download_pdbs(self):
        """Step 1: Download all PDBs"""
        if not self.task2_config.get('run_step1_download', False):
            self.logger.info("Skipping Step 1 PDB download")
            return
            
        self.logger.info("Running Step 1: Downloading PDB files")
        
        results_before_dali = os.path.join(self.work_dir, "protein_relation_v3",self.task2_config['results_before_dali'])
        target_pdbs_dir = self.get_full_path(self.task2_config['target_pdbs_dir'])
        
        task2_utils.download_pdb_from_AFDB_parallel(results_before_dali, target_pdbs_dir)
    
    def run_step2_dali_search(self):
        """Step 2: DALI search for domain segments"""
        if not self.task2_config.get('run_step2_dali', False):
            self.logger.info("Skipping Step 2 DALI search")
            return
            
        self.logger.info("Running Step 2: DALI structural search")
        
        # Implementation for DALI search
        dali_work_dir = self.task2_config['dali_work_dir']
        dali_bin = self.task2_config['dali_bin_path']
        dali_targetpdb_dir = self.get_full_path(self.task2_config['target_pdbs_dir'])
        dali_targetdat_dir = os.path.join(dali_work_dir,self.task2_config['dali_target_dat_dir'])
        dali_querypdb_dir = self.get_full_path(self.task2_config['query_pdbs_dir'])
        dali_querydat_dir =  os.path.join(dali_work_dir,self.task2_config['dali_query_dat_dir'])
        
        
        # Import PDBs for DALI
        task2_utils.importpl_pdb_target_parallel(dali_bin, dali_targetpdb_dir, dali_targetdat_dir, dali_work_dir)
        task2_utils.importpl_pdb_query(dali_bin,dali_querypdb_dir ,dali_querydat_dir, dali_work_dir)
        
        # Run DALI search
        assemble_script = self.get_full_path(self.task2_config['assemble_dali_script'])
        task2_utils.run_dali_shell(assemble_script, dali_work_dir)
    
    def run_step3_process_dali_results(self):
        """Step 3: Process DALI search results"""
        self.logger.info("Running Step 3: Processing DALI results")
        
        dali_work_dir = self.task2_config['dali_work_dir']
        query_ids = os.path.join(dali_work_dir,self.task2_config['query_ids_file'])
        target_ids =os.path.join(dali_work_dir,self.task2_config['target_ids_file']) 
        dali_results = self.get_full_path(self.task2_config['dali_results_file'])
        task1_utils.ensure_dir(os.path.dirname(dali_results))
        
        task2_utils.process_dali_results(dali_work_dir, dali_results, query_ids, target_ids)
    
    def run_step4_merge_domains(self):
        """Step 4: Merge domain segments"""
        self.logger.info("Running Step 4: Merging domain segments")
        
        dali_results = self.get_full_path(self.task2_config['dali_results_file'])
        dali_results_pro = self.get_full_path(self.task2_config['dali_results_processed'])
        domain_threshold = self.task2_config['domain_threshold']
        domain_class = self.task2_config['domain_class']
        
        task2_utils.merge_target_ranges_parts(dali_results, dali_results_pro, domain_threshold, domain_class)
    
    def run_step5_split_domain_pdbs(self):
        """Step 5: Split domain PDBs from all PDBs"""
        self.logger.info("Running Step 5: Splitting domain PDBs")
        
        dali_results_pro = self.get_full_path(self.task2_config['dali_results_processed'])
        target_pdbs_dir = self.get_full_path(self.task2_config['target_pdbs_dir'])
        dali_querypdb_dir = self.get_full_path(self.task2_config['query_pdbs_dir'])
        target_domains_dir = self.get_full_path(self.task2_config['target_domains_dir'])
        query_domains_dir = self.get_full_path(self.task2_config['query_domains_dir'])
        
        # Split target domains
        task2_utils.split_target_domains_from_dali_results(dali_results_pro, target_domains_dir, target_pdbs_dir)
        
        # Split query domains
        task2_utils.split_query_domains_from_dali_results(dali_results_pro, query_domains_dir, dali_querypdb_dir)
    
    def run_step6_structural_analysis(self):
        """Step 6: Structural similarity analysis"""
        self.logger.info("Running Step 6: Structural similarity analysis")
        
        # Implementation for FoldSeek analysis
        query_domains_dir = self.get_full_path(self.task2_config['query_domains_dir'])
        target_domains_dir = self.get_full_path(self.task2_config['target_domains_dir'])
        fs_results_dir = self.get_full_path(self.task2_config['fs_results_dir'])
        
        # Process PDB pairs
        merged_results = task2_utils.process_pdb_pairs_from_csv_parallel(
            self.get_full_path(self.task2_config['dali_results_processed']),
            query_domains_dir,
            target_domains_dir,
            fs_results_dir,
            self.get_full_path(self.task2_config['fs_tmp_dir'])
        )
        
        # Create network
        all_fstm_result = os.path.join(fs_results_dir, "All_fstm_results.csv")
        task1_utils.create_fs_cytoscape_network(merged_results, all_fstm_result)
        
        
        # Process DALI-FSTM results
        task2_utils.process_dali_fstm_results_250703(
            self.get_full_path(self.task2_config['dali_results_processed']),
            all_fstm_result,
            all_fstm_result
        )
    
    def run_step7_merge_results(self):
        """Step 7: Merge original and new results"""
        self.logger.info("Running Step 7: Merging results")
        
        results_before_dali = os.path.join(self.work_dir, "protein_relation_v3",self.task2_config['results_before_dali'])
        results_after = self.get_full_path(self.task2_config['results_after_dali'])
        all_fstm_result = os.path.join(self.get_full_path(self.task2_config['fs_results_dir']), "All_fstm_results.csv")
        dali_results_pro = self.get_full_path(self.task2_config['dali_results_processed'])
        all_fasta = self.get_full_path(self.config['input_files']['all_fasta'])
        
        # Load FASTA data
        fasta_dict = task1_utils.load_fasta_to_dict(all_fasta)
        
        # Process candidate with DALI
        task2_utils.process_candidate_with_dali_250703(
            results_before_dali, all_fstm_result, dali_results_pro, fasta_dict, results_after
        )
    

    
    def run_task2(self):
        """Execute Task 2: Refining domain boundaries using DALI"""
        self.logger.info("=" * 80)
        self.logger.info("SMAL-DR Phase 2: Task 2 Execution")
        self.logger.info("Refining domain boundaries using DALI structural alignment")
        self.logger.info("=" * 80)
        self.logger.info("Domain boundaries refined using DALI structural alignment. Ready for experimental validation.")
        
        try:

            # Execute Task 2 pipeline steps
            self.run_step0_refine_domains()
            self.run_step1_download_pdbs()
            self.run_step2_dali_search()
            self.run_step3_process_dali_results()
            self.run_step4_merge_domains()
            self.run_step5_split_domain_pdbs()
            self.run_step6_structural_analysis()
            self.run_step7_merge_results()
            self.logger.info("=" * 80)
            self.logger.info("Task 2 completed successfully!")
            self.logger.info("Domain boundaries have been refined using DALI structural alignment")
            self.logger.info("Ready for experimental validation of refined candidates")
            
        except Exception as e:
            self.logger.error(f"Task 2 pipeline failed with error: {str(e)}")
            self.logger.exception("Detailed error traceback:")
            raise


class SMALDRTask3:
    def __init__(self, config_path):
        """初始化任务配置"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 配置目录路径等
        self.data_dir = Path(self.config['task3_config']['data_dir'])
        self.results_dir = Path(self.config['task3_config']['results_dir'])
        self.model_save_dir = self.results_dir / "model_weights"
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def run_step1_setup(self):
        """Step 1: 设置环境与配置参数"""
        self.logger.info("Setting up environment and configurations...")
        
        # 设置随机种子
        task3_utils.set_seed(42)  # 使用task3_utils中的set_seed方法
        
        # 选择设备（GPU/CPU）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据路径和模型保存路径
        self.embeddings_dir = self.data_dir / "Esm2Embedding-sp-1280"
        self.config_params = {
            "batch_size": self.config['task3_config']['batch_size'],
            "lr": self.config['task3_config']['lr'],
            "weight_decay": self.config['task3_config']['weight_decay'],
            "margin": self.config['task3_config']['margin'],
            "diff_threshold": self.config['task3_config']['diff_threshold'],
            "hidden_dim": self.config['task3_config']['hidden_dim'],
            "epochs": self.config['task3_config']['epochs'],
            "patience": self.config['task3_config']['patience']
        }
        
        # 准备数据集和数据加载器
        self.train_ds = task3_utils.PairwiseDataset(self.data_dir, ["FITNESS"], self.config_params["diff_threshold"], self.embeddings_dir)
        self.train_loader = DataLoader(self.train_ds, batch_size=self.config_params["batch_size"], shuffle=True)
        
        self.logger.info(f"Data setup completed. Training with batch size {self.config_params['batch_size']}.")

    def run_step2_build_model(self):
        """Step 2: 构建模型"""
        self.logger.info("Building model...")
        
        # 使用 task3_utils.PairNet 来初始化模型
        self.model = task3_utils.PairNet(input_dim=self.train_ds.embeddings[0].shape[0]).to(self.device)
        self.loss_fn = torch.nn.MarginRankingLoss(margin=self.config_params["margin"])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config_params["lr"], weight_decay=self.config_params["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        self.logger.info("Model built and ready for training.")

    def run_step3_train(self):
        """Step 3: 训练模型"""
        self.logger.info("Starting training...")

        best_val_loss = float('inf')
        no_improve = 0
        best_model_path = self.model_save_dir / f"best_model.pth"

        # 训练循环
        for epoch in range(1, self.config_params["epochs"] + 1):
            train_loss = task3_utils.train_epoch(self.model, self.train_loader, self.optimizer, self.loss_fn, self.device)
            self.scheduler.step(train_loss)

            self.logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")

            # 保存当前最佳模型
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                no_improve = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                no_improve += 1
                if no_improve >= self.config_params["patience"]:
                    self.logger.info("Early stopping on validation loss.")
                    break

        self.logger.info("Training completed. Best model saved.")

    def run_task3(self):
        """执行整个任务3（MLP训练）"""
        self.logger.info("Starting Task 3: MLP Model Training")
        self.run_step1_setup()   # Step 1
        self.run_step2_build_model()  # Step 2
        self.run_step3_train()  # Step 3
        self.logger.info("Task 3 completed successfully!")


class SMALDRTask4:
    def __init__(self, config_path):
        """初始化任务配置"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

        # 配置路径
        self.main_dir = Path(self.config['task4_config']['main_dir'])
        self.output_dir = Path(self.config['task4_config']['output_dir'])
        self.model_weight = Path(self.config['task4_config']['model_weight'])
        self.cache_path = Path(self.config['task4_config']['cache_path'])
        self.json_path = Path(self.config['task4_config']['json_path'])

        self.output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def run_step1_create_embedding_cache(self):
        """Step 1: 创建嵌入缓存"""
        self.logger.info("Creating embedding cache...")
        if not self.cache_path.exists():
            with open(self.json_path, "r") as f:
                metric_dict = {
                    d['Variants'] if 'Variants' in d else d['name']: d['active_numbers']
                    for d in json.load(f)
                }

            all_paths = sorted(list(self.main_dir.glob("*.npy")))
            cache = {}
            for p in all_paths:
                vid = p.stem
                arr = np.load(p)
                t = torch.tensor(arr, dtype=torch.float32)
                emb = t[1:-1].mean(dim=0) if t.ndim > 1 else t
                emb = F.normalize(emb, dim=-1)
                sup = torch.stack([t[pos] for pos in metric_dict[vid]], dim=0).mean(dim=0)
                sup = F.normalize(sup, dim=-1)
                cache[vid] = (emb, sup)
            torch.save(cache, self.cache_path)
            self.logger.info("Embedding cache saved.")
        else:
            self.logger.info("Embedding cache already exists.")

    def run_step2_inference(self):
        """Step 2: 推理"""
        self.logger.info("Starting inference...")

        # 通过 task4_utils 中的推理函数进行推理
        task4_utils.inference(self.model_weight, self.main_dir, self.cache_path, self.output_dir, self.config)

    def run_step3_merge_results(self):
        """Step 3: 合并推理结果"""
        self.logger.info("Merging inference results...")

        # 合并各 rank 的推理结果，并保存最终的结果
        files = list(self.output_dir.glob("rank*_winrates.csv"))
        dfs = [pd.read_csv(f) for f in files]
        final_df = pd.concat(dfs).groupby("variant_id", as_index=False).mean()
        final_df = final_df.sort_values("win_rate", ascending=False)
        final_df.to_csv(self.output_dir / self.config['task4_config']['output_file'], index=False)
        self.logger.info(f"Final sorted results saved to {self.output_dir / self.config['task4_config']['output_file']}")

    def run_task4(self):
        """执行整个任务4（MLP推理）"""
        self.logger.info("Starting Task 4: MLP Model Inference")
        self.run_step1_create_embedding_cache()  # Step 1: 创建嵌入缓存
        self.run_step2_inference()  # Step 2: 推理
        self.run_step3_merge_results()  # Step 3: 合并结果
        self.logger.info("Task 4 completed successfully!")
        
        
class SMALDRTask5:
    def __init__(self, config_path):
        """Initialize configuration for Task 5"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

        # Directories and configuration parameters
        self.main_dir = Path(self.config['task5_config']['main_dir'])
        self.output_dir = Path(self.config['task5_config']['output_dir'])
        self.batch_size = self.config['task5_config']['batch_size']
        self.lr = self.config['task5_config']['lr']
        self.weight_decay = self.config['task5_config']['weight_decay']
        self.margin = self.config['task5_config']['margin']
        self.epochs = self.config['task5_config']['epochs']
        self.patience = self.config['task5_config']['patience']
        
        self.model_save_dir = Path(self.config['task5_config']['model_save_dir'])
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize other components as needed (model, dataset, etc.)
        self.embedding_dir = Path(self.config['task5_config']['embedding_dir'])

    def setup_logging(self):
        """Setup basic logging configuration"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def run_step1_setup(self):
        """Step 1: Setup training environment and parameters"""
        self.logger.info("Setting up training environment...")

        # Set device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger.info(f"Using device: {self.device}")

    def run_step2_initialize_model(self):
        """Step 2: Initialize the transformer model"""
        self.logger.info("Initializing transformer model...")

        # Initialize the transformer model using the provided config
        self.model = task5_utils.PairNet(input_dim=1280, hidden_dim=self.config['task5_config']['hidden_dim']).to(self.device)
        
        # Setup loss function and optimizer
        self.loss_fn = torch.nn.MarginRankingLoss(margin=self.margin)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.logger.info("Model initialized successfully.")

    def run_step3_train(self):
        """Step 3: Training loop"""
        self.logger.info("Starting training...")

        # Initialize the dataset and DataLoader
        train_ds = task5_utils.PairwiseDataset(self.main_dir, self.embedding_dir)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        # Training loop
        best_val_loss = float('inf')
        no_improve = 0
        for epoch in range(1, self.epochs + 1):
            train_loss = task5_utils.train_epoch(self.model, train_loader, self.optimizer, self.loss_fn, self.device)
            
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                no_improve = 0
                torch.save(self.model.state_dict(), self.model_save_dir / f"best_model.pth")
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    self.logger.info("Early stopping due to no improvement.")
                    break

            self.logger.info(f"Epoch {epoch}/{self.epochs}: train_loss={train_loss:.4f}")

    def run_task5(self):
        """Execute the entire Task 5 (Transformer Training)"""
        self.logger.info("Starting Task 5: Transformer Model Training")
        self.run_step1_setup()  # Step 1: Set up the training environment
        self.run_step2_initialize_model()  # Step 2: Initialize the model
        self.run_step3_train()  # Step 3: Train the model
        self.logger.info("Task 5 completed successfully!")

class SMALDRTask6:
    def __init__(self, config_path):
        """Initialize configuration for Task 6"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.logger = logging.getLogger(__name__)
        self.setup_logging()

        # Directories and configuration parameters
        self.main_dir = Path(self.config['task6_config']['main_dir'])
        self.output_dir = Path(self.config['task6_config']['output_dir'])
        self.model_weight = Path(self.config['task6_config']['model_weight'])
        self.cache_path = Path(self.config['task6_config']['cache_path'])
        self.json_path = Path(self.config['task6_config']['json_path'])

        self.output_dir.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    def setup_logging(self):
        """Setup basic logging configuration"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def run_step1_create_embedding_cache(self):
        """Step 1: Create embedding cache"""
        self.logger.info("Creating embedding cache...")
        if not self.cache_path.exists():
            with open(self.json_path, "r") as f:
                metric_dict = {
                    d['Variants'] if 'Variants' in d else d['name']: d['active_numbers']
                    for d in json.load(f)
                }

            all_paths = sorted(list(self.main_dir.glob("*.npy")))
            cache = {}
            for p in all_paths:
                vid = p.stem
                arr = np.load(p)
                t = torch.tensor(arr, dtype=torch.float32)
                emb = t[1:-1].mean(dim=0) if t.ndim > 1 else t
                emb = torch.nn.functional.normalize(emb, dim=-1)
                sup = torch.stack([t[pos] for pos in metric_dict[vid]], dim=0).mean(dim=0)
                sup = torch.nn.functional.normalize(sup, dim=-1)
                cache[vid] = (emb, sup)
            torch.save(cache, self.cache_path)
            self.logger.info("Embedding cache saved.")
        else:
            self.logger.info("Embedding cache already exists.")

    def run_step2_inference(self):
        """Step 2: Perform inference"""
        self.logger.info("Starting inference...")

        # Perform inference using the provided model weight and configurations
        task6_utils.inference(self.model_weight, self.main_dir, self.cache_path, self.output_dir, self.config)

    def run_step3_merge_results(self):
        """Step 3: Merge inference results"""
        self.logger.info("Merging inference results...")

        # Merge results from each rank and save the final output
        files = list(self.output_dir.glob("rank*_winrates.csv"))
        dfs = [pd.read_csv(f) for f in files]
        final_df = pd.concat(dfs).groupby("variant_id", as_index=False).mean()
        final_df = final_df.sort_values("win_rate", ascending=False)
        final_df.to_csv(self.output_dir / self.config['task6_config']['output_file'], index=False)
        self.logger.info(f"Final sorted results saved to {self.output_dir / self.config['task6_config']['output_file']}")

    def run_task6(self):
        """Execute the entire Task 6 (Transformer Inference)"""
        self.logger.info("Starting Task 6: Transformer Model Inference")
        self.run_step1_create_embedding_cache()  # Step 1: Create embedding cache
        self.run_step2_inference()  # Step 2: Perform inference
        self.run_step3_merge_results()  # Step 3: Merge results
        self.logger.info("Task 6 completed successfully!")

def main(config_path):
    """Main execution function"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    #............................................... Task1 : Structural fold mining from the TED database identifies diverse HNH-like domains for Cas9 engineering..........................................................................................#
    
    pipeline_task1 = SMALDRTask1(config_path)
    #pipeline_task1.run_task1()
        
    #............................................... Task2 :  Refining domain boundaries using DALI structural alignment to enhance recombinational compatibility..........................................................................................#

    pipeline_task2 = SMALDRTask2(config_path)
    #pipeline_task2.run_task2()
   
    #............................................... Task3 :  Train PairNet(MLP) by wet-lab data..........................................................................................#
    pipeline_task3 = SMALDRTask3(config_path)
    #pipeline_task3.run_task3()
    
    #............................................... Task4 :  Inference by PairNet(MLP)..........................................................................................#
    pipeline_task4 = SMALDRTask4(config_path)
    pipeline_task4.run_task4()
    
    #............................................... Task3 :  Train PairNet(Transformer) by wet-lab data..........................................................................................#
    pipeline_task5 = SMALDRTask5(config_path)
    pipeline_task5.run_task5()
    
    #............................................... Task4 :  Inference by PairNet(Transformer)..........................................................................................#
    pipeline_task6 = SMALDRTask6(config_path)
    pipeline_task6.run_task6()
    

if __name__ == "__main__":
    config_file_path = "/mnt/sdb4/protein_gen/Cas9_domain_work/data/TED/Cas9_submit/config.json" 
    main(config_file_path)
