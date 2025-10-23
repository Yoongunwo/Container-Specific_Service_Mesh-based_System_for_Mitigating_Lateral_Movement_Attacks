import pandas as pd
import numpy as np
from collections import defaultdict
from queue import Queue as queue

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import pathlib as Path
import sys

class NgramQueue:
    def __init__(self, maxsize):
        self.queue = queue(maxsize)
        self.syscall_dict = {}

    def update(self, ngram, label):
        if ngram in self.syscall_dict:
            old_label = self.syscall_dict[ngram]
            if label == -1 and old_label == 0: # abnormal < normal
                self.syscall_dict[ngram] = -1
            elif label == 1:
                self.syscall_dict[ngram] = 1
            
            # if label == -1: # abnormal > normal
            #     self.syscall_dict[ngram] = -1
            # elif label == 1 and old_label == 0: # normal
            #     self.syscall_dict[ngram] = 1
        else:
            self.append(ngram, label)
    
    def append(self, ngram, label):
        # 큐가 꽉 찼을 때 가장 오래된 항목 제거
        if self.queue.full():
            old_n_gram = self.queue.get()
            print('\033[95m' + f"Old ngram: {old_n_gram} removed" + '\033[0m')

            if old_n_gram in self.syscall_dict:
                del self.syscall_dict[old_n_gram]

        # 큐 추가
        self.queue.put(ngram)
        
        # 딕셔너리 추가
        self.syscall_dict[ngram] = label

    def relabeling(self, df):
        for index, row in df.iterrows():
            if row['ngram'] in self.syscall_dict:
                df.at[index, 'label'] = self.syscall_dict[row['ngram']]
        
        return df

RANDOM_STATE = 42
QUEUE_SIZE = 10000
DEPLOY_DICT = defaultdict(lambda: NgramQueue(QUEUE_SIZE))

def learningIsolationForest(X_train):
    model = IsolationForest(
        random_state=RANDOM_STATE, 
        n_estimators=100, 
        contamination=0.01
    )
    return model.fit(X_train)

def learningKmenas(X_train):
    model = KMeans(n_clusters=2, random_state=RANDOM_STATE)
    model.fit(X_train)

    return model

def learningLOF(X_train):
    model = LocalOutlierFactor(
        n_neighbors=20, 
        contamination=0.0001, 
        novelty=True
    )
    model.fit(X_train)

    return model

def learningOneClassSVM(X_train):
    model = OneClassSVM(kernel='rbf', nu=0.0001)
    model.fit(X_train)

    return model

def learningPCA(X_train):
    model = PCA(n_components=0.95)
    model.fit(X_train)

    return model

def load_data(file_path, prev_data=False):
    encodings_to_try = ['utf-8', 'cp949', sys.getfilesystemencoding()]

    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                break  # 성공적으로 읽었다면 루프 종료
        except UnicodeDecodeError:
            if encoding == encodings_to_try[-1]:  # 마지막 시도였다면
                raise  # 에러 발생
            continue  # 다음 인코딩으로 시도
    
    ngrams = []
    labels = []
    for line in lines:
        parts = line.strip().split(' : ')
        if len(parts) == 2:
            ngrams.append(parts[0])
            if prev_data:
                try:
                    labels.append(int(float((parts[1]))))
                except Exception as e:
                    print(f"Type of value: {type(parts[1])}")
                    print(f"Full line: '{line}'")
                    print(f"Error: {e}, {parts[1]}")
            else:
                labels.append(-1 if parts[1] == '-1' else 1)
    
    return pd.DataFrame({'ngram': ngrams, 'label': labels})

def evaluate_model(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    # print(y_pred)
    # y_pred = np.where(y_pred == 1, 0, 1)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal'], labels=[1, -1], zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(clf, vectorizer, directory_path, counter):
    joblib.dump(clf, directory_path + f'model_{counter}.pkl')
    joblib.dump(vectorizer, directory_path + 'vectorizer.pkl')
    print('\033[96m' + "\nModel and vectorizer have been saved." + '\033[0m')

def save_data(data, labels, directory_path, file_name, counter,):
    with open(directory_path + f'{file_name}_{counter}.txt', 'w') as f:
        for syscall, label in zip(data, labels):
            syscall_str = ' '.join(map(str, syscall))
            f.write(f"{syscall_str} : {label}\n")


def update_NgramQueue(df, deploy):
    ngramQueue = DEPLOY_DICT[deploy]
    for index, row in df.iterrows():
        ngramQueue.update(row['ngram'], row['label'])

def data_labeling(df, deploy):
    ngramQueue = DEPLOY_DICT[deploy]
    return ngramQueue.relabeling(df)

def data_filtering(df, label):
    return df[df['label'] == label]

def string_to_int_array(ngram_str):
    try:
        return [int(float(x)) for x in ngram_str.split()]
    except Exception as e:
        print(f"Error2: {e}, {ngram_str}")
        return []

def train_model(deploy, directory_path, counter):
    try:
        all_dataframes = []
        all_prev_test_dataframes = []
        all_prev_train_dataframes = []
        # 디렉토리 내의 모든 파일 나열
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # 파일인 경우에만 처리 (디렉토리 제외) 
            # dkl 제외
            if not filename.endswith('.txt'):
                continue

            if  'test' in filename:
                df = load_data(file_path, prev_data=True)
                all_prev_test_dataframes.append(df)
            elif 'train' in filename:
                df = load_data(file_path, prev_data=True)
                all_prev_train_dataframes.append(df)
            elif os.path.isfile(file_path) and not filename.endswith('.pkl'):
                try:
                    # 개별 파일 로드
                    df = load_data(file_path)
                    all_dataframes.append(df)
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        df = pd.concat(all_dataframes, ignore_index=True)
        update_NgramQueue(df, deploy)
        df = data_labeling(df, deploy)

        df = data_filtering(df, 1)
        
        # Abnormal data save
        df_abnormal = data_filtering(df, -1)
        X_abnormal = df_abnormal['ngram'].apply(string_to_int_array)
        X_abnormal = np.array(X_abnormal.tolist())
        y_abnormal = df_abnormal['label']
        save_data(X_abnormal, y_abnormal, directory_path, 'abnormal', counter)

        # Data preprocessing
        X = df['ngram'].apply(string_to_int_array)
        X = np.array(X.tolist())
        y = df['label']
        # print(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        save_data(X_train, y_train, directory_path, 'train', counter)
        save_data(X_test, y_test, directory_path, 'test', counter)

        # Vectorization
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Data splitting
        if len(all_prev_train_dataframes) != 0:
            print("concatenating previous data")
            prev_X_train = pd.concat(all_prev_train_dataframes, ignore_index=True)['ngram']
            prev_y_train = pd.concat(all_prev_train_dataframes, ignore_index=True)['label']

            prev_X_test = pd.concat(all_prev_test_dataframes, ignore_index=True)['ngram']
            prev_y_test = pd.concat(all_prev_test_dataframes, ignore_index=True)['label']

            prev_X_train = prev_X_train.apply(string_to_int_array)
            prev_X_train = np.array(prev_X_train.tolist())

            prev_X_test = prev_X_test.apply(string_to_int_array)
            prev_X_test = np.array(prev_X_test.tolist())

            prev_X_train = scaler.transform(prev_X_train)
            prev_X_test = scaler.transform(prev_X_test)

            X_train = np.concatenate((X_train, prev_X_train))
            y_train = np.concatenate((y_train, prev_y_train))

            X_test = np.concatenate((X_test, prev_X_test))
            y_test = np.concatenate((y_test, prev_y_test))

        print('\033[96m' + f"Model training start for deploy {deploy}" + '\033[0m')
        clf = learningIsolationForest(X_train)
        
        evaluate_model(clf, X_test, y_test)
        
        save_model(clf, scaler, directory_path, counter)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False