import os
import pandas as pd
from tqdm import tqdm
from scipy.stats import f_oneway
import warnings
import argparse
from multiprocessing import Pool, cpu_count, current_process
import logging

# 특정 경고 무시하기
warnings.filterwarnings("ignore")

# 로깅 설정
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging():
    """각 프로세스마다 개별 로그 파일 생성"""
    process_name = current_process().name
    log_file = os.path.join(LOG_DIR, f"{process_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def process_chromosome(args):
    """한 염색체(chrX)에 대해 여러 p-value로 ANOVA 수행"""
    setup_logging()  # 멀티 프로세싱 로그 설정
    chr_num, input_dir, label_df, p_values = args
    significant_positions = {pv: [] for pv in p_values}

    logging.info(f"Processing chr{chr_num}...")

    target_dir = os.path.join(input_dir, f'chr{chr_num}')
    files = os.listdir(target_dir)
    logging.info(f"Total files in chr{chr_num}: {len(files)}")

    for idx, csv_file in enumerate(files):
        file_path = os.path.join(target_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            df = pd.merge(df, label_df, on='pid').dropna()

            if df.shape[0] == 550:
                if 'pid' in df.columns:
                    df = df.drop(columns=['pid', 'id'])

                groups = {label: df['genotype'][df['label'] == label].values for label in df['label'].unique()}

                if all(len(group) > 1 for group in groups.values()):
                    f_value, p_value = f_oneway(*groups.values())
                    for pv in p_values:
                        if p_value < pv:
                            significant_positions[pv].append(os.path.splitext(csv_file)[0])
            else:
                logging.warning(f"Invalid shape in {csv_file} (chr{chr_num}).")
        except Exception as e:
            logging.error(f"Error processing {csv_file} in chr{chr_num}: {e}")

        # 1000개 처리할 때마다 진행 상황 로깅
        if (idx + 1) % 1000 == 0:
            logging.info(f"Processed {idx + 1} in chr{chr_num}")

    logging.info(f"Finished processing chr{chr_num}.")
    return significant_positions


def perform_anova_parallel(input_dir, p_values):
    label_df = pd.read_csv('../meta_data/total_labels.csv')
    chroms = range(1, 23)  # chr1 ~ chr22

    print(f"Starting ANOVA on all chromosomes using {cpu_count()} CPUs...")

    args = [(chr_num, input_dir, label_df, p_values) for chr_num in chroms]
    
    # 멀티 프로세싱 풀 생성
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_chromosome, args), total=len(chroms)))

    # 각 p-value에 대해 결과 병합
    significant_positions = {pv: [] for pv in p_values}
    for result in results:
        for pv in p_values:
            significant_positions[pv].extend(result[pv])

    # 결과 저장
    for pv, positions in significant_positions.items():
        sig_pos_file = f"./sig_pos/significant_positions_{pv:.2f}.csv"
        pd.Series(positions).to_csv(sig_pos_file, index=False, header=False)
        print(f"Significant positions for p-value < {pv:.2f} saved to '{sig_pos_file}'.")
        logging.info(f"Results saved for p-value < {pv:.2f} to {sig_pos_file}")


def main():
    parser = argparse.ArgumentParser(description='Perform ANOVA with multiprocessing and multiple p-values.')
    args = parser.parse_args()

    input_dir_base = "./process_raw"  # 각 position 파일들이 저장된 디렉토리

    # p-value 범위 지정
    p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

    # 로그 디렉토리 초기화
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # 병렬 ANOVA 수행
    perform_anova_parallel(input_dir_base, p_values)

    print("ANOVA process completed successfully.")


# 실행
if __name__ == "__main__":
    main()
