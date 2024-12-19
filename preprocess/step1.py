import os
import pysam
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def process_vcf_and_collect_data(directory):
    all_positions = set()
    patient_genotypes = defaultdict(lambda: defaultdict(int))
    patient_ids = []

    print("Processing VCF files and collecting genotype data...")
    for pid_dir in tqdm(os.listdir(directory), desc="Processing Patients"):
        pid = pid_dir.split('.')[0]
        patient_ids.append(pid)
        vcf_file = os.path.join(directory,pid_dir)
        
        with pysam.VariantFile(vcf_file, "r") as vcf:
            for record in vcf:
                if record.filter.keys()[0] == 'PASS':
                    id = record.chrom + '-' + str(record.pos)
                    if record.id != None:
                        id += '-' + record.id
                    all_positions.add(id)
                    genotype = record.samples[record.samples.keys()[0]]["GT"]
                    if genotype == (0, 0):
                        patient_genotypes[pid][id] = 0
                    elif genotype == (0, 1) or genotype == (1, 0):
                        patient_genotypes[pid][id] = 1
                    elif genotype == (1, 1):
                        patient_genotypes[pid][id] = 2

    
    return all_positions, patient_genotypes, patient_ids

def save_data_to_csv(all_positions, patient_genotypes, patient_ids, output_dir):
    
    
    print(f"Saving data to CSV files... total {len(all_positions)} position")
    for pos in tqdm(all_positions):
        chr_num = pos.split('-')[0]
        chr_dir = output_dir + f'/{chr_num}'

        if not os.path.exists(chr_dir):
            os.makedirs(chr_dir)
        
        csv_file = os.path.join(chr_dir, f"{pos}.csv")
        data = {'pid': [], 'id': [], 'genotype': []}



        
        for pid in patient_ids:
            data['pid'].append(pid)
            data['id'].append(pos)
            data['genotype'].append(patient_genotypes[pid].get(pos, 0))
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, mode='w', header=True, index=False)

    print(f"Data saved to CSV files in directory '{output_dir}'.")

def main():
    directory = "../vcf_files"  # VCF 파일들이 있는 디렉토리 경로
    output_dir = "./process_raw"  # CSV 파일 저장 디렉토리
    
    # VCF 파일 처리 및 데이터 수집
    all_positions, patient_genotypes, patient_ids = process_vcf_and_collect_data(directory)
    
    # 수집한 데이터를 CSV 파일로 저장
    save_data_to_csv(all_positions, patient_genotypes, patient_ids, output_dir)

    print("Process completed successfully.")

# 실행
main()
