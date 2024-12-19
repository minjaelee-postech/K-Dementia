## Data 준비
```
vcf_files/
└──{patient_id}.hard_filtered.vcf.gz (환자별 유전체 데이터 정보)
meta_data/
├──total_labels.csv (Patient id - 환자 진단 상태 table)
└──gwas_catalog.tsv (gwas catalog에서 제공하는 치매 연관 SNPs 데이터 정보)
processed_image/
├──T1/
|   └──{patient_id}.npy
├──PET/
|   └──{patient_id}.npy
preprocess/ 
├──step1.py (전체 데이터에 대한 position 별 genotyping 진행 -> process_raw 디렉토리 만들어짐)
└──step2.py (전체 데이터에 대한 ANOVA test)
└──sig_pos/
    └──significant_positions_{p_value}.csv (p_value에 따른 중요 포지션 정보 (1차 필터링))
workspace/
├──svm.ipynb


```
## Data 전처리  

총 두 개의 python 파일이 있음.

```
├──process_dataset.py (3클래스 (AD, MCI, CN) 모두를 고려해서 전처리)
└──process_dataset_2class.py (2클래스 (AD, MCI, CN 중 두 개의 조합 - 3C2 = 3가지)만 고려해서 전처리)
```

만약에 각각의 프로그램에 대한 argument를 바꿔서 실행하고 싶다면, argument 로 들어가는 모든 경로의 끝에는 슬래시를 빼야 함. 각 프로그램의 자세한 argument는 아래에 기술함.

##### 예시

```
/home/minjae/genome/processed_image (O)
/home/minjae/genome/processed_image/ (X)
```

##### 실행 명령어 예시

3 클래스 데이터셋을 만들기 위한 process_dataset.py 는 다음과 같이 실행하면 됨.
(default 로 argument 가 다 주어져 있으므로, 인자 없이 python process_dataset.py 로 실행해도 됨)

```
python process_dataset.py --data_savedir {default : 3class, 데이터가 저장될 루트 폴더 이름} --sigpos_rootdir {sig_pos 폴더의 절대 경로 혹은 상대 경로} --modal_rootdir {processed_image 의 절대 경로 혹은 상대 경로} --raw_rootdir {process_raw 의 절대 경로 혹은 상대 경로} --gwas_file_path {gwas_catalog.tsv 의 절대 경로 혹은 상대 경로}--pid_label_pair_file_path {total_labels.csv 의 절대 경로 혹은 상대 경로}
```

결과 데이터셋은 프로그램 실행 (20~25분 정도 소요) 이후에 /preprocess/data/3class 폴더 안에 저장됨.

2 클래스 데이터셋을 만들기 위한 process_dataset_2class.py 는 다음과 같이 실행하면 됨.
(default 로 argument 가 다 주어져 있으므로, 인자 없이 python process_dataset_2class.py 로 실행해도 됨)

```
python process_dataset_2class.py --data_savedir {default : 2class, 데이터가 저장될 루트 폴더 이름} --sigpos_rootdir {sig_pos 폴더의 절대 경로 혹은 상대 경로} --modal_rootdir {processed_image 의 절대 경로 혹은 상대 경로} --raw_rootdir {process_raw 의 절대 경로 혹은 상대 경로} --gwas_file_path {gwas_catalog.tsv 의 절대 경로 혹은 상대 경로}--pid_label_pair_file_path {total_labels.csv 의 절대 경로 혹은 상대 경로}
--include_dti_modality {default: 'no', DTI features를 데이터셋에 포함하지 않음}
```

결과 데이터셋은 프로그램 실행 (45~50분 정도 소요) 이후에 /preprocess/data/2class 폴더 안에 저장됨.

process_dataset.py 와 process_dataset_2class.py 를 모두 실행 완료했을 때 최종 데이터셋 디렉토리 구조는 아래와 같음.

```
preprocess/data/
├── 2class/
│   ├── gwas_data/
│   │   ├── 5folds/
│   │   │   ├── genotype_only/
│   │   │   │   ├── [pickle fold files]
│   │   │   ├── triplets/
│   │   │   │   ├── [pickle fold files]
│   │   │   └── triplets_normalized/
│   │   │       ├── [pickle fold files]
│   │   └── 10folds/
│   │       ├── genotype_only/
│   │       │   ├── [pickle fold files]
│   │       ├── triplets/
│   │       │   ├── [pickle fold files]
│   │       └── triplets_normalized/
│   │           ├── [pickle fold files]
│   ├── gwas_intersection_data/ [위와 동일한 구조]
│   ├── no_normalize_gwas_data/ [위와 동일한 구조]
│   ├── no_normalize_gwas_intersection_data/ [위와 동일한 구조]
│   ├── trr_data/ [위와 동일한 구조]
│   └── no_normalize_trr_data/ [위와 동일한 구조]
│
├── 3class/
│   ├── AD_CN/
│   │   ├── gwas_data/ [위와 동일한 구조]
│   │   ├── gwas_intersection_data/ [위와 동일한 구조]
│   │   ├── no_normalize_gwas_data/ [위와 동일한 구조]
│   │   ├── no_normalize_gwas_intersection_data/ [위와 동일한 구조]
│   │   ├── trr_data/ [위와 동일한 구조]
│   │   └── no_normalize_trr_data/ [위와 동일한 구조]
│   ├── MCI_AD/
│   │   ├── gwas_data/ [위와 동일한 구조]
│   │   ├── gwas_intersection_data/ [위와 동일한 구조]
│   │   ├── no_normalize_gwas_data/ [위와 동일한 구조]
│   │   ├── no_normalize_gwas_intersection_data/ [위와 동일한 구조]
│   │   ├── trr_data/ [위와 동일한 구조]
│   │   └── no_normalize_trr_data/ [위와 동일한 구조]
│   └── MCI_CN/
│       ├── gwas_data/ [위와 동일한 구조]
│       ├── gwas_intersection_data/ [위와 동일한 구조]
│       ├── no_normalize_gwas_data/ [위와 동일한 구조]
│       ├── no_normalize_gwas_intersection_data/ [위와 동일한 구조]
│       ├── trr_data/ [위와 동일한 구조]
│       └── no_normalize_trr_data/ [위와 동일한 구조]
```


## 실행 방법

1. step 1, 2 실행
2. data 전처리 진행 
3. svm.ipynb에서 결과 확인 

## SVM.ipynb 

전처리 과정을 통해 얻어진 SNPs genotype, MRI, PET 정보를 종합하여 
SVM 을 기반으로 classification task 수행 및 shap value를 측정할 수 있는 주피터 노트북파일 
기본적으로 data load와 classification, shap value(상위 200개 값)들의 시각화 정보를 포함하고 있으며 cell 을 
자유롭게 추가하여 분석할 수 있도록 ipynb로 배포함 