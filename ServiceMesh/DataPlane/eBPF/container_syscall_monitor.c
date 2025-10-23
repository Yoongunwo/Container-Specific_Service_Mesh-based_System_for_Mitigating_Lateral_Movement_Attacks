#include <uapi/linux/ptrace.h>
#include <linux/sched.h>
#include <linux/nsproxy.h>
#include <linux/pid_namespace.h>

#define TASK_COMM_LEN 16
#define MAX_FILENAME_LEN 256
#define RING_BUFFER_SIZE 8192

struct data_t {
    u32 pid;
    u64 syscall_nr;
    u32 padding;  
} __attribute__((aligned(8)));

BPF_PERF_OUTPUT(events);

// 관심있는 프로세스의 PID를 저장하는 해시맵
BPF_HASH(interested_pids, u32, u32);

// TRACEPOINT_PROBE(syscalls, sys_enter_clone) {
//     struct task_struct *parent = (struct task_struct *)bpf_get_current_task();
//     u32 ppid = parent->tgid;

//     // 부모 프로세스가 관심 대상인지 확인
//     u32 *interested = interested_pids.lookup(&ppid);
//     if (interested) {
//         // clone으로 생성되는 자식 프로세스도 관심 대상으로 추가
//         u32 pid = bpf_get_current_pid_tgid() >> 32;
//         interested_pids.update(&pid, &pid);
//     }

//     return 0;
// }

// 프로세스 생성 추적 (fork, clone)
TRACEPOINT_PROBE(sched, sched_process_fork) {
    struct task_struct *parent = (struct task_struct *)bpf_get_current_task();
    u32 ppid = parent->tgid;
    u32 pid = args->child_pid;

    // 부모 프로세스가 관심 대상인지 확인
    u32 *interested = interested_pids.lookup(&ppid);
    if (interested) {
        // 자식 프로세스도 관심 대상으로 추가
        interested_pids.update(&pid, &pid);
    }

    return 0;
}

TRACEPOINT_PROBE(sched, sys_exit_exit) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    // u32 pid = args->pid;
    interested_pids.delete(&pid);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_exit_group) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    interested_pids.delete(&pid);
    return 0;
}

// 시스템 콜 추적
TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;

    // 관심 있는 프로세스인지 확인
    u32 *interested = interested_pids.lookup(&pid);
    if (!interested) {
        return 0;
    }

    struct data_t data = {};

    data.pid = pid;
    
    data.syscall_nr = args->id;
    
    events.perf_submit(args, &data, sizeof(data));
    return 0;
}
