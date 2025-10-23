import collections
import joblib
import os
import re
import numpy as np
import time

def get_syscalls():
    syscalls = {}
    syscall_header = "/usr/include/x86_64-linux-gnu/asm/unistd_64.h"
    if not os.path.exists(syscall_header):
        print(f"Warning: {syscall_header} not found. Using empty syscall map.")
        return 
    
    with open(syscall_header, 'r') as f:
        for line in f:
            match = re.search(r'#define __NR_(\w+)\s+(\d+)', line)
            if match:
                name, number = match.groups()
                syscalls[name] = int(number)
    return syscalls

def string_to_int_array(ngram_str):
    return [int(x) for x in ngram_str.split()]

def detecting(data_queue, isRequestChecking, isInternalRequestChecking, 
              isDropping, netNgram_data, sysNgram_data, pause_event, requestCheck, responseCheck):
    pid_syscalls = collections.defaultdict(list)
    n = int(os.environ.get('WINDOW_SIZE', 24))

    syscalls = get_syscalls()

    def create_ngram(syscalls):
        # return ' '.join(syscalls)
        return string_to_int_array(syscalls)

    def predict_anomaly(ngram):
        try:
            # print(f"Vectorizer status: {vectorizer is not None}")  # vectorizer 상태 확인
            if vectorizer:
                # print("Transforming input with vectorizer")  # transform 시도 로그
                ngram = vectorizer.transform(ngram)
                # print("Transform successful")  # transform 성공 로그
            
            prediction = clf.predict(ngram)
            
            return prediction
        except Exception as e:
            print(f"Error in predict_anomaly: {e}")  # 에러 발생 시 상세 로그
            print(f"ngram shape: {ngram.shape}")  # 입력 데이터 shape 확인
            raise e

    networkSyscall = ["socket", "connect", "accept", "sendto", "recvfrom", "sendmsg", "recvmsg", "shutdown", "bind", "listen", "getsockname", "getpeername", "setsockopt", "getsockopt" ]
    netSyscallNums = [syscalls[syscall] for syscall in networkSyscall]

    def checkNetworkSyscall(ngram):
        if ngram in netSyscallNums:
            return True
        return False

    def data_write(data, label):
        with open(f'./model/data.txt', 'a') as f:
            # numpy array를 깔끔한 문자열로 변환
            if isinstance(data, np.ndarray):
                data_str = ' '.join(str(x) for x in data.flatten())
            elif isinstance(data, list):
                data_str = ' '.join(map(str, data))
            else:
                data_str = str(data)
            f.write(f"{data_str} : {label}\n")

    # 시작 전 비우고 시작
    with open("./model/data.txt", 'w') as f:
        f.write('')

    def load_model():
        try:
            new_clf = joblib.load('./Detect/model.pkl')
            new_vectorizer = joblib.load('./Detect/vectorizer.pkl')
            # print(f"Vectorizer type: {type(new_vectorizer)}")  # vectorizer 타입 확인
            return new_clf, new_vectorizer
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
    
    clf = joblib.load('./Detect/model.pkl')
    # vectorizer = joblib.load('./Detect/vectorizer.pkl')
    vectorizer = None

    while True:
        if not pause_event.is_set():
            print('\033[93m' + "Attack Detecting Stop" + '\033[0m')
            pause_event.wait()

            for attempt in range(3):
                print('\033[93m' + f"Attmept {attempt + 1} to reload model" + '\033[0m')
                new_clf, new_vectorizer = load_model()
                if new_clf and new_vectorizer:
                    clf = new_clf
                    vectorizer = new_vectorizer
                    break
                time.sleep(1)

            print('\033[93m' + "Attack Detecting Resume" + '\033[0m')   
            with open("./model/data.txt", 'w') as f:
                f.write('') # 파일 내용 삭제 => 수정 가능성 존재
        
        try:
            data = data_queue.get_nowait()
            pid = data['pid']
            # syscall = str(data['syscall'])
            pid_syscalls[pid].append(data['syscall'])
            
            if len(pid_syscalls[pid]) == n:
                ngram = np.array(pid_syscalls[pid]).reshape(1, -1)
                results = predict_anomaly(ngram)
                
                # result validation
                if -1 in results:
                    print('\033[31m' + f"PID: {pid}, Prediction: Attack" + '\033[0m')
                    # netNgram_data.put(ngram)
                    sysNgram_data.put(ngram)
                    isInternalRequestChecking.value = True
                    isRequestChecking.value = True
                else:
                    # print('\033[32m' + f"PID: {pid}, Prediction: Normal" + '\033[0m')
                    data_write(ngram, 1)

                # n-gram 크기를 유지하기 위해 가장 오래된 syscall 제거
                pid_syscalls[pid] = pid_syscalls[pid][1:]
            elif len(pid_syscalls[pid]) > n:
                # 만약 어떤 이유로 n보다 크게 증가했다면, n 크기로 조정
                pid_syscalls[pid] = pid_syscalls[pid][-n:]
        except:
            pass

