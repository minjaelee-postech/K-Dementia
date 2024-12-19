import os
import pandas as pd
from tqdm import tqdm
from scipy.stats import ttest_ind
import warnings
from multiprocessing import Pool, cpu_count, current_process
import logging

# 특정 경고 무시하기
warnings.filterwarnings("ignore")

LOG_DIR = "./logs2"
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


def process_class_pair(args):
    """한 클래스 쌍(AD-MCI, MCI-CN, AD-CN)과 염색체에 대해 t-test 수행"""
    setup_logging()
    chr_num, class_pair, input_dir, label_df, p_values = args
    significant_positions = {pv: [] for pv in p_values}

    class1, class2 = class_pair
    logging.info(f"Processing chr{chr_num} for classes {class1} vs {class2}...")

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

                # Perform t-test for the given class pair
                group1 = df['genotype'][df['label'] == class1].values
                group2 = df['genotype'][df['label'] == class2].values
                if len(group1) > 1 and len(group2) > 1:
                    t_stat, p_value = ttest_ind(group1, group2)
                    for pv in p_values:
                        if p_value < pv:
                            significant_positions[pv].append(os.path.splitext(csv_file)[0])

            else:
                logging.warning(f"Invalid shape in {csv_file} (chr{chr_num}).")
        except Exception as e:
            logging.error(f"Error processing {csv_file} in chr{chr_num} for {class1} vs {class2}: {e}")

        if (idx + 1) % 1000 == 0:
            logging.info(f"Processed {idx + 1} files in chr{chr_num} for {class1} vs {class2}")

    logging.info(f"Finished processing chr{chr_num} for {class1} vs {class2}.")
    return significant_positions


def perform_ttests_parallel(input_dir, p_values):
    label_df = pd.read_csv('../meta_data/total_labels.csv')
    chroms = range(1, 23)  # chr1 ~ chr22
    class_pairs = [('AD', 'MCI'), ('MCI', 'CN'), ('AD', 'CN')]

    # 병렬 작업을 위한 인자 생성
    args = [(chr_num, class_pair, input_dir, label_df, p_values) for chr_num in chroms for class_pair in class_pairs]

    print(f"Starting t-tests on all chromosomes and class pairs using {cpu_count()} CPUs...")

    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_class_pair, args), total=len(args)))

    # 결과 병합
    significant_positions = {pv: [] for pv in p_values}
    for result in results:
        for pv in p_values:
            significant_positions[pv].extend(result[pv])

    # 저장
    for pv in p_values:
        ttest_union = list(set(significant_positions[pv]))
        sig_ttest_file = f"./sig_pos/significant_positions_2_{pv:.2f}.csv"
        pd.Series(ttest_union).to_csv(sig_ttest_file, index=False, header=False)
        print(f"T-test union significant positions for p-value < {pv:.2f} saved to '{sig_ttest_file}'.")


def main():
    input_dir_base = "./process_raw"  # 각 position 파일들이 저장된 디렉토리

    # p-value 범위 지정
    p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

    # 로그 디렉토리 초기화
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    # t-test 수행
    perform_ttests_parallel(input_dir_base, p_values)

    print("T-test process completed successfully.")


if __name__ == "__main__":
    main()
