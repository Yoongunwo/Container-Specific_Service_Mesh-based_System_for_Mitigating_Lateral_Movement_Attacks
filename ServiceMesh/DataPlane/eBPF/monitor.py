# execve_monitor.py
from bcc import BPF
import re
import os
import ctypes as ct
import multiprocessing
import time

def monitoring(rootPID, pids, data_queue):
    PAGE_CNT = 1024
    TIME_OUT = 1

    ebpf = BPF(src_file="./eBPF/container_syscall_monitor.c")
    

    for pid in pids:
        ebpf["interested_pids"][ct.c_int(int(pid))] = ct.c_int(1)
    ebpf["root_pid"][ct.c_int(rootPID.value)] = ct.c_int(1)

    # 이벤트 콜백 함수 정의
    def print_event(cpu, data, size):
        event = ebpf["events"].event(data)
        log = {
            "pid": event.pid,
            "syscall": event.syscall_nr,
        }
        data_queue.put(log)

    # 이벤트 핸들러 등록
    ebpf["events"].open_perf_buffer(print_event, page_cnt=PAGE_CNT)

    while True:
        try:
            ebpf.perf_buffer_poll(timeout=TIME_OUT)
        except KeyboardInterrupt:
            break
