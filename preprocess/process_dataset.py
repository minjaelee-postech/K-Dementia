# DTI / PET / T1 thickness 중 겹치는 PID 찾기 (step 1)

import pandas as pd
import os
import numpy as np
from collections import Counter
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import argparse
import torch
import warnings
import shutil
import time
import uuid
from datetime import timedelta

# PerformanceWarning 메시지 무시하기
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# 인자를 global 변수로
# 인자 파싱
parser = argparse.ArgumentParser(description="A script to process data and save output files.")
parser.add_argument("--data_savedir", type=str, default='2class',
                    help="The directory where the output files will be saved.")
parser.add_argument("--sigpos_rootdir", type=str, default='/home/minjae/genome/workspace/pre/sig_pos',
                    help="root folder containing significant positions csv files")
parser.add_argument("--modal_rootdir", type=str, default='/home/minjae/genome/newgenes/processed_image',
                    help="root folder containing DTI/PET/T1 npy files")
parser.add_argument("--raw_rootdir", type=str, default='/home/minjae/genome/newgenes/preprocess/process_raw',
                    help="root folder containing raw SNPs files")
parser.add_argument("--gwas_file_path", type=str, default='/home/minjae/genome/newgenes/meta_data/gwas_catalog.tsv',
                    help="gwas tsv file path")
parser.add_argument("--pid_label_pair_file_path", type=str, default='/home/minjae/genome/newgenes/meta_data/total_labels.csv',
                    help="pid - label pair csv file path")
parser.add_argument("--include_dti_modality", type=str, default='no')
parser.add_argument("--df_saved_dirname", type=str, default='df_saved_run')


# 쳇지피티에서 가져온 유전자 번호별 길이
# 이 길이로 위치를 나눠서 normalize 한다.

chromosome_lengths = [
    248956422,  # Chromosome 1
    242193529,  # Chromosome 2
    198295559,  # Chromosome 3
    190214555,  # Chromosome 4
    181538259,  # Chromosome 5
    170805979,  # Chromosome 6
    159345973,  # Chromosome 7
    145138636,  # Chromosome 8
    138394717,  # Chromosome 9
    133797422,  # Chromosome 10
    135086622,  # Chromosome 11
    133275309,  # Chromosome 12
    114364328,  # Chromosome 13
    107043718,  # Chromosome 14
    101991189,  # Chromosome 15
    90338345,   # Chromosome 16
    83257441,   # Chromosome 17
    80373285,   # Chromosome 18
    58617616,   # Chromosome 19
    64444167,   # Chromosome 20
    46709983,   # Chromosome 21
    50818468    # Chromosome 22
]

# 폴더 만들 때, 이름 겹치지 않게 하는 함수
# df 를 저장하는 폴더인데, 나중에는 pickle data 폴더만 남아야된다.
def create_unique_folder(base_name, directory="./temp"):
    """
    고유한 이름을 가진 폴더를 생성합니다.
    """
    # 현재 시간 기반 타임스탬프 + UUID를 추가하여 고유 이름 생성
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_id = uuid.uuid4().hex[:6]  # UUID의 앞 6자리를 사용
    unique_folder_name = f"{base_name}-{timestamp}-{unique_id}"

    folder_path = os.path.join(directory, unique_folder_name)

    return folder_path

# returns dict
def get_modalities_info():

    # 예제 파일 로드
    pids_list = []

    for i in os.walk(f'{args.modal_rootdir}/T1/thickness/'):
        pids_list.extend([k.split('_')[0] for k in i[2]]) # 144 pids
            
    for i in os.walk(f'{args.modal_rootdir}/PET/'):
        pids_list.extend([k.split('.')[0] for k in i[2]]) # 113 pids

    try:
        for i in os.walk(f'{args.modal_rootdir}/DTI_data/'):
            pids_list.extend([k.split('_')[0] for k in i[2]])
    except Exception as E:
        pass

    if args.include_dti_modality == "yes":
        pids_list = [k for k,v in Counter(pids_list).items() if v == 3]
    else:
        pids_list = [k for k,v in Counter(pids_list).items() if v == 2]

    ###########################################################################

    set_arr = []

    for i in [0.01, 0.02, 0.03, 0.04, 0.05, '2_0.01', '2_0.02', '2_0.03', '2_0.04', '2_0.05']:
        a = pd.read_csv(f'{args.sigpos_rootdir}/significant_positions_{i}.csv')
        set_arr.extend([str(u)+'.csv' for u in list(a.iloc[:, 0])])
        
    important_column_indices = list(set(set_arr)) 
    
    return {'important_column_indices' : important_column_indices, 
            'pids_list' : pids_list}


# SNPS column을 포함하는 dataframe 을 column 별로 (genotype, 유전자번호, 위치)를 저장하도록 하는 dataframe으로 변환한다. 

def convert_to_triplet_table(df):
    # 'label' 컬럼 분리
    labels = df['label']
    df = df.drop(columns=['label'])

    # 확장 작업
    expanded_df = pd.DataFrame()  # 확장된 데이터프레임

    for col in df.columns:
        original_values = df[col]
        part_1, part_2 = col.split("-")  # "chr19", "44903416"
        
        # 각 컬럼의 확장
        expanded_df[col] = original_values  # 원래 값
        expanded_df[col + "_1"] = float(part_1[3:])  # "chr19" -> 19
        expanded_df[col + "_2"] = float(part_2)      # "44903416"
        
    return expanded_df

# SNPS column을 포함하는 dataframe 을 column 별로 (genotype, 유전자번호, normalize 된 위치)를 저장하도록 하는 dataframe으로 변환한다.  
    
def convert_to_triplet_table_normalized(df):
    # 'label' 컬럼 분리
    labels = df['label']
    df = df.drop(columns=['label'])

    # 확장 작업
    expanded_df = pd.DataFrame()  # 확장된 데이터프레임

    for col in df.columns:
        original_values = df[col]
        part_1, part_2 = col.split("-")  # "chr19", "44903416"
        
        # 각 컬럼의 확장
        expanded_df[col] = original_values  # 원래 값
        expanded_df[col + "_1"] = float(part_1[3:])  # "chr19" -> 19
        idx = int(part_1[3:])
        expanded_df[col + "_2"] = float(part_2) / chromosome_lengths[idx-1]      # "44903416"
    
    return expanded_df

def return_converted_snps(df, convert_ver='triplet'):
    df = df.sort_index()
    pids = df.index
    label = df['label'].reset_index(drop=True)
    
    if convert_ver == 'triplet': # triplet table로 변환
        new_df = convert_to_triplet_table(df).reset_index(drop=True)
    elif convert_ver == 'triplet_normalized': # normalized triplet table로 변환
        new_df = convert_to_triplet_table_normalized(df).reset_index(drop=True)
    elif convert_ver == 'original': # genotype 하나만을 column별 원소로 가지는 table로 변환 (원래 테이블)
        new_df = df.reset_index(drop=True)
    else:
        raise NotImplementedError('This mode is not implemented.')
    
    label_mapping = {0 : "AD", 1 : "CN", 2 : "MCI"}
    new_df.index = pids
    return label.map(label_mapping).tolist(), new_df
    

# 병렬적으로 처리할 함수
def process_folder(g):
    
    # triplet_table =  # 길이로 나누지않은 원래 위치를 가지는 트리플렛을 포함하는 테이블
    # no_triplet_table =  # 트리플렛이 아닌, genotype 만 entry 로 포함하는 테이블
    # norm_triplet_table =  # 길이만큼으로 나눈 트리플렛을 포함하는 테이블
    
    important_column_indices = get_modalities_info()['important_column_indices']
    
    dataframes = []
    folder_path = f'{args.raw_rootdir}/{g}'
    files = os.listdir(folder_path)
    
    files = list(set(files) & set(important_column_indices))
    
    for f in tqdm(files, desc=g):
        
        # f 는 파일 이름 (예: {g}-위치.csv)
        file_path = f'{folder_path}/{f}'
        df = pd.read_csv(file_path)
        
        # ID 정규화
        df['id'] = df['id'].str.replace(r'(chr)*_.*?(-\d+)', r'\1\2', regex=True)
        
        # PID 열 기준으로 정렬
        df = df.sort_values(by='pid').reset_index(drop=True)
        
        # 피벗 테이블 생성
        df_pivot = df.pivot(index='pid', columns='id', values='genotype')
        dataframes.append(df_pivot)
    
    # 하나의 폴더 내에서 처리된 데이터프레임 병합
    folder_df = pd.concat(dataframes, axis=1)
    return folder_df

def multiprocess_and_create_df():
    # 병렬처리
    new_folders = [f'chr{i}' for i in range(1, 23)]  # chr1 ~ chr22 폴더 리스트

    # 멀티프로세싱 풀 생성
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_folder, new_folders), total=len(new_folders)))

    # 모든 결과 데이터프레임 병합
    final_df = pd.concat(results, axis=1)
    final_df = final_df.fillna(0)  # NaN 값을 0으로 채움
    
    return final_df

# 상위 2000개 feature를 고르고, 그것들에 해당하는 df를 케이스별로 pickle 파일로 저장한다.
# 케이스 : [trr_data, no_normalize_trr_data]
# K = [5, 10] 에 대해 지원

def decision_tree_top_2000(df, train_pids, unmatched_pids, combined_df):
    
    types = ['trr_data', 'no_normalize_trr_data']
    
    filtered_df = df.loc[df.index.isin(train_pids)]
    unmatched_df = df.loc[df.index.isin(unmatched_pids)]

    # 특성과 레이블 분리
    X = filtered_df.drop(columns=["label"])
    y = filtered_df["label"]

    unmatched_X = unmatched_df.drop(columns=['label'])
    unmatched_Y = unmatched_df['label']
    
    
    for k in tqdm([5, 10], desc=f'training decision tree..'):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        for depth in [2]:
            feature_importances = np.zeros(X.shape[1])
            
            for i, (train_index, test_index) in enumerate(skf.split(X, y)):
                
                # 폴드마다 별개의 model을 학습하고, 그 model로 feature selection
                
                # 단순히 붙이는 건 class imbalance가 일어날 수 있음 
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                X_train = pd.concat([X_train, unmatched_X], axis=0)
                y_train = pd.concat([y_train, unmatched_Y], axis=0)
                
                # 모델 학습
                clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
                clf.fit(X_train, y_train)
                
                # Feature 중요도
                cur_feature_importances = clf.feature_importances_
                
                # Test Accuracy 출력 (선택 사항)
                y_pred = clf.predict(X_test)
                #print(classification_report(y_test, y_pred))
                
                
                ######################################################################################################
                # 상위 2000개 feature 선택 (현재 폴드에서)
                sorted_indices = np.argsort(cur_feature_importances)[::-1][:2000]  # 중요도가 높은 순서대로 2000개 선택
                selected_features = X.columns[sorted_indices]
                
                # 선택된 feature를 포함하는 데이터프레임 생성
                # loader.py 에 들어갈 폴드 정보 (인덱스) 도 저장해 둔다.
                reduced_df = df[selected_features.tolist() + ["label"]]  # label 포함
                reduced_df_train = reduced_df.iloc[train_index] # unmatched 제외한 나머지 데이터 중 train folds를 포함함 (단 fold i 에서의 decision tree의 결정이다)
                reduced_df_test = reduced_df.iloc[test_index] # unmatched 제외한 나머지 데이터 중 test fold를 포함함.
                
                combined_df_train = combined_df.iloc[train_index]
                combined_df_test = combined_df.iloc[test_index]
                
                # converted (레이블은 다 같으므로 한번만 얻으면 된다)
                selected_train_labels, cvt_reduced_df_train = return_converted_snps(reduced_df_train, convert_ver='original')
                selected_test_labels, cvt_reduced_df_test = return_converted_snps(reduced_df_test, convert_ver='original')
                _, cvt_reduced_df_triplet_train = return_converted_snps(reduced_df_train, convert_ver='triplet')
                _, cvt_reduced_df_triplet_test = return_converted_snps(reduced_df_test, convert_ver='triplet')
                _, cvt_reduced_df_triplet_norm_train = return_converted_snps(reduced_df_train, convert_ver='triplet_normalized')
                _, cvt_reduced_df_triplet_norm_test = return_converted_snps(reduced_df_test, convert_ver='triplet_normalized')
                
                # reduced_df_train과 test를 텐서로 변환
                reduced_df_train = torch.tensor(cvt_reduced_df_train.drop(columns=['label']).values, dtype=torch.float32)
                reduced_df_test = torch.tensor(cvt_reduced_df_test.drop(columns=['label']).values, dtype=torch.float32)
                reduced_df_triplet_train = torch.tensor(cvt_reduced_df_triplet_train.values, dtype=torch.float32).view(-1, 2000, 3)
                reduced_df_triplet_test = torch.tensor(cvt_reduced_df_triplet_test.values, dtype=torch.float32).view(-1, 2000, 3)
                reduced_df_triplet_norm_train = torch.tensor(cvt_reduced_df_triplet_norm_train.values, dtype=torch.float32).view(-1, 2000, 3)
                reduced_df_triplet_norm_test = torch.tensor(cvt_reduced_df_triplet_norm_test.values, dtype=torch.float32).view(-1, 2000, 3)
                                
                # 열 이름을 기반으로 DTI, T1, PET 열 이름 자동 추출
                try:    
                    dti_columns = [col for col in combined_df.columns if col.startswith('dti_')]
                except Exception as E:
                    pass
                t1_columns = [col for col in combined_df.columns if col.startswith('t1_')]
                pet_columns = [col for col in combined_df.columns if col.startswith('pet_')]

                for typ in types:
                    # dti, t1, pet 각각 텐서로 변환
                    try:
                        dti_tensor_train = torch.tensor(combined_df_train[dti_columns].values, dtype=torch.float32).view(-1, 90, 90)
                    except Exception as E:
                        pass
                    t1_tensor_train = torch.tensor(combined_df_train[t1_columns].values, dtype=torch.float32) if typ.split('_')[0]=='no' else \
                                    minmax_normalize(torch.tensor(combined_df_train[t1_columns].values, dtype=torch.float32))
                    pet_tensor_train = torch.tensor(combined_df_train[pet_columns].values, dtype=torch.float32) if typ.split('_')[0]=='no' else \
                                    minmax_normalize(torch.tensor(combined_df_train[pet_columns].values, dtype=torch.float32))
                    try:
                        dti_tensor_test = torch.tensor(combined_df_test[dti_columns].values, dtype=torch.float32).view(-1, 90, 90)
                    except Exception as E:
                        pass
                    t1_tensor_test = torch.tensor(combined_df_test[t1_columns].values, dtype=torch.float32) if typ.split('_')[0]=='no' else \
                                    minmax_normalize(torch.tensor(combined_df_test[t1_columns].values, dtype=torch.float32))
                    pet_tensor_test = torch.tensor(combined_df_test[pet_columns].values, dtype=torch.float32) if typ.split('_')[0]=='no' else \
                                    minmax_normalize(torch.tensor(combined_df_test[pet_columns].values, dtype=torch.float32))
                    
                    
                    
                    if args.include_dti_modality == "no":                    
                        fold_tuples = [
                            ('genotype_only,train', (reduced_df_train, t1_tensor_train, pet_tensor_train, selected_train_labels)),
                            ('genotype_only,test', (reduced_df_test, t1_tensor_test, pet_tensor_test, selected_test_labels)),
                            ('triplets,train', (reduced_df_train, t1_tensor_train, pet_tensor_train, selected_train_labels)),
                            ('triplets,test', (reduced_df_test, t1_tensor_test, pet_tensor_test, selected_test_labels)),
                            ('triplets_normalized,train', (reduced_df_train, t1_tensor_train, pet_tensor_train, selected_train_labels)),
                            ('triplets_normalized,test',(reduced_df_test, t1_tensor_test, pet_tensor_test, selected_test_labels)),
                        ]
                    else:
                        try:
                            fold_tuples = [
                                ('genotype_only,train', (reduced_df_train, dti_tensor_train, t1_tensor_train, pet_tensor_train, selected_train_labels)),
                                ('genotype_only,test', (reduced_df_test, dti_tensor_test, t1_tensor_test, pet_tensor_test, selected_test_labels)),
                                ('triplets,train', (reduced_df_triplet_train, dti_tensor_train, t1_tensor_train, pet_tensor_train, selected_train_labels)),
                                ('triplets,test', (reduced_df_triplet_test, dti_tensor_test, t1_tensor_test, pet_tensor_test, selected_test_labels)),
                                ('triplets_normalized,train', (reduced_df_triplet_norm_train, dti_tensor_train, t1_tensor_train, pet_tensor_train, selected_train_labels)),
                                ('triplets_normalized,test',(reduced_df_triplet_norm_test, dti_tensor_test, t1_tensor_test, pet_tensor_test, selected_test_labels)),
                            ]
                        except Exception as E:
                            raise ValueError('DTI modality files not found in ./processed_image directory.')
                            
                    # 폴더 경로 생성
                    folder_paths = [f'./data/{args.data_savedir}/{typ}/{k}folds/genotype_only',
                                    f'./data/{args.data_savedir}/{typ}/{k}folds/triplets',
                                    f'./data/{args.data_savedir}/{typ}/{k}folds/triplets_normalized'
                                    ]
                    
                    map_dict = {'genotype_only' : 0, 'triplets' : 1, 'triplets_normalized' : 2}
                    
                    for fp in folder_paths:
                        os.makedirs(fp, exist_ok=True) # 폴더 생성 (이미 존재하면 무시)
                    
                    # 피클 파일로 바로 저장
                    for dirname, data in fold_tuples:
                        name, trtst = dirname.split(',')[0], dirname.split(',')[1]
                        base_path = folder_paths[map_dict[name]]
                        with open(f'{base_path}/{trtst}_fold={i}.pickle', 'wb') as f:
                            pickle.dump(data, f)
                

def minmax_normalize(tensor, min_val=None, max_val=None):
    """
    Min-Max Normalize a tensor.
    If min_val and max_val are not provided, calculate them from the tensor.
    """
    if min_val is None:
        min_val = tensor.min(dim=0, keepdim=True).values
    if max_val is None:
        max_val = tensor.max(dim=0, keepdim=True).values
    
    # Prevent division by zero
    range_val = max_val - min_val
    range_val[range_val == 0] = 1e-8
    
    normalized_tensor = (tensor - min_val) / range_val
    
    # Replace NaN values with 0 (or another suitable default)
    normalized_tensor = torch.nan_to_num(normalized_tensor, nan=0.0)
    
    return normalized_tensor

# 병렬적으로 처리할 함수
# gwas original (1498 개 locations) 에 대해 처리하는 함수
def process_folder_gwas(g):
    
    dataframes = []
    folder_path = f'{args.raw_rootdir}/{g}'
    files = os.listdir(folder_path)
    
    
    # processed_raw 파일들에 속하는 feature_indices의 파일들에 대해서만 process 한다. 
    files = list(set(files) & set(feature_indices))
    
    for f in tqdm(files, desc=g):
        
        # f 는 파일 이름 (예: {g}-위치.csv)
        file_path = f'{folder_path}/{f}'
        df = pd.read_csv(file_path)
        
        # ID 정규화
        df['id'] = df['id'].str.replace(r'(chr)*_.*?(-\d+)', r'\1\2', regex=True)
        
        # PID 열 기준으로 정렬
        df = df.sort_values(by='pid').reset_index(drop=True)
        
        # 피벗 테이블 생성
        df_pivot = df.pivot(index='pid', columns='id', values='genotype')
        dataframes.append(df_pivot)
    
    # 하나의 폴더 내에서 처리된 데이터프레임 병합
    folder_df = pd.concat(dataframes, axis=1)
    return folder_df


# 병렬적으로 처리할 함수
# gwas intersection (250 개 locations) 에 대해 처리하는 함수
def process_folder_gwas_intersection(g):
    
    
    dataframes = []
    folder_path = f'{args.raw_rootdir}/{g}'
    files = os.listdir(folder_path)
    
    # processed_raw 파일들에 속하는 feature_indices_intersection의 파일들에 대해서만 process 한다. 
    files = list(set(files) & set(feature_indices_intersection))
    
    for f in tqdm(files, desc=g):
        
        # f 는 파일 이름 (예: {g}-위치.csv)
        file_path = f'{folder_path}/{f}'
        df = pd.read_csv(file_path)
        
        # ID 정규화
        df['id'] = df['id'].str.replace(r'(chr)*_.*?(-\d+)', r'\1\2', regex=True)
        
        # PID 열 기준으로 정렬
        df = df.sort_values(by='pid').reset_index(drop=True)
        
        # 피벗 테이블 생성
        df_pivot = df.pivot(index='pid', columns='id', values='genotype')
        dataframes.append(df_pivot)
    
    # 하나의 폴더 내에서 처리된 데이터프레임 병합
    folder_df = pd.concat(dataframes, axis=1)
    return folder_df

# typ = ['gwas_data', 'gwas_intersection_data', 'no_normalize_gwas_data', 
# 'no_normalize_gwas_intersection_data']
def create_gwas_folds(df, train_pids, unmatched_pids, combined_df, mode=0):
    
    # 이제 위의 gwas_df 를 5-fold, 10-fold (KStratifiedFold) 로 나눈다.

    types = ['gwas_data', 'no_normalize_gwas_data'] if mode==0 else ['gwas_intersection_data', 'no_normalize_gwas_intersection_data']
    
    filtered_df = df.loc[df.index.isin(train_pids)]
    unmatched_df = df.loc[df.index.isin(unmatched_pids)]
    
    # 특성과 레이블 분리
    X = filtered_df.drop(columns=["label"])
    y = filtered_df["label"]

    unmatched_X = unmatched_df.drop(columns=['label'])
    unmatched_Y = unmatched_df['label']


    for k in tqdm([5, 10], desc=f'GWAS..'): 
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):       
            # 단순히 붙이는 건 class imbalance가 일어날 수 있음 
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            X_train = pd.concat([X_train, unmatched_X], axis=0)
            y_train = pd.concat([y_train, unmatched_Y], axis=0)
            
            # 선택된 feature를 포함하는 데이터프레임 생성
            # loader.py 에 들어갈 폴드 정보 (인덱스) 도 저장해 둔다.
            reduced_df = filtered_df
            reduced_df_train = reduced_df.iloc[train_index] # unmatched 제외한 나머지 데이터 중 train folds를 포함함 (단 fold i 에서의 decision tree의 결정이다)
            reduced_df_test = reduced_df.iloc[test_index] # unmatched 제외한 나머지 데이터 중 test fold를 포함함.
            
            combined_df_train = combined_df.iloc[train_index]
            combined_df_test = combined_df.iloc[test_index]
            
            # converted (레이블은 다 같으므로 한번만 얻으면 된다)
            selected_train_labels, cvt_reduced_df_train = return_converted_snps(reduced_df_train, convert_ver='original')
            selected_test_labels, cvt_reduced_df_test = return_converted_snps(reduced_df_test, convert_ver='original')
            _, cvt_reduced_df_triplet_train = return_converted_snps(reduced_df_train, convert_ver='triplet')
            _, cvt_reduced_df_triplet_test = return_converted_snps(reduced_df_test, convert_ver='triplet')
            _, cvt_reduced_df_triplet_norm_train = return_converted_snps(reduced_df_train, convert_ver='triplet_normalized')
            _, cvt_reduced_df_triplet_norm_test = return_converted_snps(reduced_df_test, convert_ver='triplet_normalized')
            
            # reduced_df_train과 test를 텐서로 변환
            viewsize_tup = (-1, 1498, 3) if mode==0 else (-1, 250, 3)
            reduced_df_train = torch.tensor(cvt_reduced_df_train.drop(columns=['label']).values, dtype=torch.float32)
            reduced_df_test = torch.tensor(cvt_reduced_df_test.drop(columns=['label']).values, dtype=torch.float32)
            reduced_df_triplet_train = torch.tensor(cvt_reduced_df_triplet_train.values, dtype=torch.float32).view(viewsize_tup)
            reduced_df_triplet_test = torch.tensor(cvt_reduced_df_triplet_test.values, dtype=torch.float32).view(viewsize_tup)
            reduced_df_triplet_norm_train = torch.tensor(cvt_reduced_df_triplet_norm_train.values, dtype=torch.float32).view(viewsize_tup)
            reduced_df_triplet_norm_test = torch.tensor(cvt_reduced_df_triplet_norm_test.values, dtype=torch.float32).view(viewsize_tup)
                            
            # 열 이름을 기반으로 DTI, T1, PET 열 이름 자동 추출
            try:
                dti_columns = [col for col in combined_df.columns if col.startswith('dti_')]
            except Exception as E:
                pass
            t1_columns = [col for col in combined_df.columns if col.startswith('t1_')]
            pet_columns = [col for col in combined_df.columns if col.startswith('pet_')]

            for typ in types:
                # dti, t1, pet 각각 텐서로 변환
                try:
                    dti_tensor_train = torch.tensor(combined_df_train[dti_columns].values, dtype=torch.float32).view(-1, 90, 90)
                except Exception as E:
                    pass
                t1_tensor_train = torch.tensor(combined_df_train[t1_columns].values, dtype=torch.float32) if typ.split('_')[0]=='no' else \
                                    minmax_normalize(torch.tensor(combined_df_train[t1_columns].values, dtype=torch.float32))
                pet_tensor_train = torch.tensor(combined_df_train[pet_columns].values, dtype=torch.float32) if typ.split('_')[0]=='no' else \
                                    minmax_normalize(torch.tensor(combined_df_train[pet_columns].values, dtype=torch.float32))
                try:
                    dti_tensor_test = torch.tensor(combined_df_test[dti_columns].values, dtype=torch.float32).view(-1, 90, 90)
                except Exception as E:
                    pass
                t1_tensor_test = torch.tensor(combined_df_test[t1_columns].values, dtype=torch.float32) if typ.split('_')[0]=='no' else \
                                    minmax_normalize(torch.tensor(combined_df_test[t1_columns].values, dtype=torch.float32))
                pet_tensor_test = torch.tensor(combined_df_test[pet_columns].values, dtype=torch.float32) if typ.split('_')[0]=='no' else \
                                    minmax_normalize(torch.tensor(combined_df_test[pet_columns].values, dtype=torch.float32))
                
                if args.include_dti_modality == "no":
                    fold_tuples = [
                        ('genotype_only,train', (reduced_df_train, t1_tensor_train, pet_tensor_train, selected_train_labels)),
                        ('genotype_only,test', (reduced_df_test, t1_tensor_test, pet_tensor_test, selected_test_labels)),
                        ('triplets,train', (reduced_df_train, t1_tensor_train, pet_tensor_train, selected_train_labels)),
                        ('triplets,test', (reduced_df_test, t1_tensor_test, pet_tensor_test, selected_test_labels)),
                        ('triplets_normalized,train', (reduced_df_train, t1_tensor_train, pet_tensor_train, selected_train_labels)),
                        ('triplets_normalized,test',(reduced_df_test, t1_tensor_test, pet_tensor_test, selected_test_labels)),
                    ]
                else:     
                    try:
                        fold_tuples = [
                            ('genotype_only,train', (reduced_df_train, dti_tensor_train, t1_tensor_train, pet_tensor_train, selected_train_labels)),
                            ('genotype_only,test', (reduced_df_test, dti_tensor_test, t1_tensor_test, pet_tensor_test, selected_test_labels)),
                            ('triplets,train', (reduced_df_triplet_train, dti_tensor_train, t1_tensor_train, pet_tensor_train, selected_train_labels)),
                            ('triplets,test', (reduced_df_triplet_test, dti_tensor_test, t1_tensor_test, pet_tensor_test, selected_test_labels)),
                            ('triplets_normalized,train', (reduced_df_triplet_norm_train, dti_tensor_train, t1_tensor_train, pet_tensor_train, selected_train_labels)),
                            ('triplets_normalized,test',(reduced_df_triplet_norm_test, dti_tensor_test, t1_tensor_test, pet_tensor_test, selected_test_labels)),
                        ]
                    except Exception as E:
                        raise ValueError('DTI modality files not found in ./processed_image directory.')
                    
                # 폴더 경로 생성
                folder_paths = [f'./data/{args.data_savedir}/{typ}/{k}folds/genotype_only',
                                f'./data/{args.data_savedir}/{typ}/{k}folds/triplets',
                                f'./data/{args.data_savedir}/{typ}/{k}folds/triplets_normalized'
                                ]
                
                map_dict = {'genotype_only' : 0, 'triplets' : 1, 'triplets_normalized' : 2}
                
                for fp in folder_paths:
                    os.makedirs(fp, exist_ok=True) # 폴더 생성 (이미 존재하면 무시)
                
                # 피클 파일로 바로 저장
                for dirname, data in fold_tuples:
                    name, trtst = dirname.split(',')[0], dirname.split(',')[1]
                    base_path = folder_paths[map_dict[name]]
                    with open(f'{base_path}/{trtst}_fold={i}.pickle', 'wb') as f:
                        pickle.dump(data, f)
            

#### 메인 함수 ####

if __name__ == '__main__':

    start_time = time.time()

    # ./temp/uniquename폴더에 df 가 저장됨.
    args = parser.parse_args()
    args.df_saved_dirname = create_unique_folder(args.df_saved_dirname)

    final_df = multiprocess_and_create_df()

    # snps_data/total_labels.csv 에 있는 label을 앞에서 얻은 dataframe에 들어갈 column으로 만든다.

    pid_label_pair_csv = pd.read_csv(args.pid_label_pair_file_path) # -> PID, label pair
    label_column = pid_label_pair_csv.sort_values(by='PID').reset_index(drop=True).loc[:, 'label']

    label_column = label_column.tolist()

    final_df.loc[:, 'label'] = label_column 

    # DTI, PET, T1 에 동시에 존재하는 pid에 해당하는 pid list = train_pids
    # final_df에서의 pid 중에서 train_pids 에 해당하지 않는 pid_list = unmatched_pids

    pids_list = get_modalities_info()['pids_list']

    train_pids = sorted(list(set(map(int, pids_list)) & set(final_df.index)))
    unmatched_pids = sorted(list(set(final_df.index) - set(train_pids)))

    # 데이터 로드

    t1_volume_files = [f'{args.modal_rootdir}/T1/volumes/{pid}_volumes.npy' for pid in train_pids] # 90D
    pet_dir = [f'{args.modal_rootdir}/PET/{pid}.npy' for pid in train_pids]
    
    try:
        dti_dir = [f'{args.modal_rootdir}/DTI_data/{pid}_conn.npy' for pid in train_pids]
        dti_data = np.concatenate([np.load(file).reshape(1, 8100) for file in dti_dir], axis=0)
    except Exception as E:
        pass    
        
    t1_data = [np.load(file) for file in t1_volume_files]
    z = []
    for d in t1_data:
        if d.shape == (91,):
            s = d[1:].reshape(1, 90)
        else:
            s = d.reshape(1,90)
        z.append(s)

    t1_data = np.concatenate(z, axis=0)
    pet_data = np.concatenate([np.load(file).reshape(1, 90) for file in pet_dir], axis=0)

    try:
        combined_df = pd.DataFrame(np.concatenate([dti_data, t1_data, pet_data], axis=1))
    except Exception as E:
        combined_df = pd.DataFrame(np.concatenate([t1_data, pet_data], axis=1))
    
    # 열 이름 설정
    try:
        dti_columns = [f'dti_{i}' for i in range(dti_data.shape[1])]
    except Exception as E:
        pass
    t1_columns = [f't1_{i}' for i in range(t1_data.shape[1])]
    pet_columns = [f'pet_{i}' for i in range(pet_data.shape[1])]
    try:
        combined_df.columns = dti_columns + t1_columns + pet_columns
    except Exception as E:
        combined_df.columns = t1_columns + pet_columns
    combined_df['pid'] = train_pids
    combined_df = combined_df.sort_values(by='pid').reset_index(drop=True)
    combined_df = combined_df.drop(columns=['pid'])
    combined_df.index = sorted(train_pids)

    # decision tree (depth = 2)로 각 fold 별로 상위 (importance에 대한) feature 2000개를 고르고,
    # 그 2000개의 SNPS feature에 대해 각 fold 별로 위의 concat_all_mod()를 사용해서 얻은 concat된 table을 저장한다.

    # 레이블 문자열 -> 정수 매핑
    label_mapping = {"AD": 0, "CN": 1, "MCI": 2}
    final_df["label"] = final_df["label"].map(label_mapping)                
                
    # 상위 2000개 feature 골라서 저장.
    # 피클 파일로 한번에 저장하는데, trr에서의 각 케이스에 대해 모두 저장  
    decision_tree_top_2000(final_df, train_pids, unmatched_pids, combined_df)
                  
    # TSV 파일 읽기
    data = pd.read_csv(args.gwas_file_path, sep='\t')
    data = data[data['locations']!='-']
    data['locations'] = data['locations'].apply(lambda x: 'chr'+ str(x.split(':')[0]) +'-'+str(x.split(':')[1]))
    gwas_catalog = sorted(set(data['locations'].to_list()))

    # location이 다른데 rs (방 번호)가 같은 경우
    # 5개의 경우가 있는데, 이 경우는 단순히 location 이름의 문제이다.
    # 그러니까 결론적으로 invalid 한 값은 없다.

    df = data

    # col_a에서 {문자열1} 추출
    df['prefix'] = df['riskAllele'].str.split('-').str[0]

    # 동일한 prefix를 가지는 그룹에서 col_b 값이 서로 다른 경우를 찾음
    # 1. prefix 기준으로 그룹화
    # 2. col_b의 고유값 수가 2개 이상인 그룹 선택
    condition = df.groupby('prefix')['locations'].transform('nunique') > 1

    # 조건에 맞는 데이터 슬라이싱
    result = df[condition]

    invalid_location_strs = [_ for _ in result['locations'].tolist() if len(_.split(','))!=1]

    # 위의 경우를 핸들링해준다.

    # 우선 locations의 고유한 list를 저장한다.
    gwas_catalog = sorted(set(data['locations'].to_list()))

    for r in invalid_location_strs:
        gwas_catalog.remove(r)
        
    important_features_alot = [_.split('.')[0] for _ in get_modalities_info()['important_column_indices']]   
    
    # 이제 gwas_catalog랑 원래 sig_pos에 있었던 locations 들 중 '겹치는 것만을' 뽑아낸다.
    feature_indices_intersection = list(set(gwas_catalog) & set(important_features_alot)) # 250개
    feature_indices = gwas_catalog

    # 위에서 찾은 location 들을 각각 전역변수로 저장

    feature_indices = [str(f)+'.csv' for f in feature_indices]
    feature_indices_intersection = [str(f)+'.csv' for f in feature_indices_intersection]


    new_folders = [f'chr{i}' for i in range(1, 23)]  # chr1 ~ chr22 폴더 리스트

    # 멀티프로세싱 풀 생성
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_folder_gwas, new_folders), total=len(new_folders)))

    # 모든 결과 데이터프레임 병합
    gwas_df = pd.concat(results, axis=1)
    gwas_df = gwas_df.fillna(0)  # NaN 값을 0으로 채움

    # 멀티프로세싱 풀 생성
    with Pool(processes=cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_folder_gwas_intersection, new_folders), total=len(new_folders)))

    # 모든 결과 데이터프레임 병합
    gwas_df_intersection = pd.concat(results, axis=1)
    gwas_df_intersection = gwas_df_intersection.fillna(0)  # NaN 값을 0으로 채움

    ######################################################
    
    # 문자열 label을 숫자로 매핑

    label_mapping = {"AD":0, "CN":1, "MCI":2}
    gwas_df['label'] = final_df['label']
    gwas_df_intersection['label'] = final_df['label']
    
    # typ = ['gwas_data', 'gwas_intersection_data', 'no_normalize_gwas_data', 
    # 'no_normalize_gwas_intersection_data']
    
    create_gwas_folds(gwas_df, train_pids=train_pids, unmatched_pids=unmatched_pids, combined_df=combined_df, mode=0)
    create_gwas_folds(gwas_df_intersection, train_pids=train_pids, unmatched_pids=unmatched_pids, combined_df=combined_df, mode=1)

    end_time = time.time()
    execution_time = end_time - start_time
    formatted_time = timedelta(seconds=execution_time)
    print(f"Elapsed time: {formatted_time}")