import multiprocessing
import asyncio
import aiohttp
from aiohttp import web, ClientTimeout
import logging
import json
import os
import subprocess
import urllib
from collections import defaultdict

import model_process

# 로깅 설정
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 주기적으로 파드 IP 주소를 가져올 시간 간격 (초)
INTERVAL = int(os.environ.get('INTERVAL', 10))

replicasets_pods = defaultdict(list)

manager = multiprocessing.Manager()

podIP_replicaset_dic = manager.dict()
replicaset_requestDic_dic = dict()

CPU_WEIGHT = 0.5
MEMORY_WEIGHT = 0.5
PROXY_PORT = 9011

async def fetch_pods_ip():
    try:
        cmd = [
            # "kubectl", "get", "pods", 
            # "-o", "custom-columns=REPLICASET:.metadata.ownerReferences[0].name,NAME:.metadata.name,IP:.status.podIP,NAMESPACE:.metadata.namespace", 
            # "--all-namespaces"
            "kubectl", "get", "pods", "--all-namespaces", "-o", 
            "custom-columns=NAMESPACE:.metadata.namespace,POD:.metadata.name,\
            REPLICASET:.metadata.ownerReferences[0].name,IP:.status.podIP,CONTAINERS:.spec.containers[*].name"
        ]
        pods_ip = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        return pods_ip.stdout.splitlines()[1:]  # 헤더 제외
    except subprocess.CalledProcessError as e:
        logger.error(f"Error fetching pods IP: {e}")
        return []

async def get_replicas_ip(podIP_replicaset_dic):
    pod_info = await fetch_pods_ip()

    replicasets_pods.clear()
    for line in pod_info:
        # replicaset, name, ip, namespace = line.split()
        # if replicaset != '<none>':  # ReplicaSet이 없는 Pod는 제외
        #     replicasets_pods[replicaset].append({
        #         "name": name,
        #         "ip": ip,
        #         "namespace": namespace
        #     })
        #     podIP_replicaset_dic[ip] = replicaset
        namespace, pod, replicaset, ip, containers = line.split()
        if "reverse-proxy" in containers:
            replicasets_pods[replicaset].append({
                "name": pod,
                "ip": ip,
                "namespace": namespace,
            })
            podIP_replicaset_dic[ip] = replicaset

    return 1

async def send_pods_ip(session):
    # my_app_replica = replicasets_pods.get('my-app', [])
    # if not my_app_replica:
    #     logger.error("No pods found for my-app")
    #     return
    timeout = ClientTimeout(total=3)
    for replicaset, pods in replicasets_pods.items():
        # if pods size is 1, skip
        if len(pods) <= 1:
            continue

        # if 'my-app' not in replicaset.lower():
        #     continue

        pods_ip = [{"name": pod['name'], "ip": pod['ip']} for pod in pods]
    
        for pod in pods:
            pods_ip_copy = pods_ip.copy()
            pods_ip_copy.remove({"name": pod['name'], "ip": pod['ip']})
            data = json.dumps({"name": pod['name'], "ip": pod['ip'], "pods_ip": pods_ip_copy})

            url = f"http://{pod['ip']}:{PROXY_PORT}/receive/pods_ip"
            headers = {'Content-Type': 'application/json'}
            try:
                async with session.post(url, data=data, headers=headers, timeout=timeout) as response:
                    if response.status == 200:
                        continue
                    else:
                        logger.error(f"Failed to send Pods IP to {url}, status: {response.status}, message: {await response.text()}")
            except aiohttp.ClientError as e:
                # logger.error(f"HTTP request failed: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error: {e}")

##################################### Internal Request Validation ##########################################

def create_internal_request_validation(podIP_replicaset_dic):
    async def internal_request_validation(request):
        try:
            # 요청 본문을 바이트로 읽음
            json_data = await request.json()
            
            request_src_ip = json_data.get('pod_ip')
            print(f"Received pod IP: {request_src_ip}")

            signature_data = json_data.get('signature_data', '')
            print(f"Received raw request body: {signature_data}")

            replicaset = podIP_replicaset_dic.get(request_src_ip)
            if not replicaset:
                print(f"Replicaset not found for IP: {request_src_ip}")
                return web.Response(status=400, text="Replicaset not found")

            requestDict = replicaset_requestDic_dic.get(replicaset)
            if not requestDict:
                # print(f"requestDic doesn't exist")
                requestDict = dict()
                replicaset_requestDic_dic[replicaset] = requestDict

            requestPodIPSet = requestDict.get(signature_data)
            if requestPodIPSet:
                if (len(requestPodIPSet) == 1) and (request_src_ip in requestPodIPSet):
                    print('\033[91m' + f"another pod's request doesn't exist" + '\033[0m')
                    signature_data = json.dumps({"result": "invalid"})
                else:
                    print('\033[92m' + f"another pod's request exist" + '\033[0m')
                    requestPodIPSet.add(request_src_ip)
                    signature_data = json.dumps({"result": "valid"})
            else:
                print('\033[91m' + f"First Request" + '\033[0m')
                requestPodIPSet = set()
                requestPodIPSet.add(request_src_ip)
                requestDict[signature_data] = requestPodIPSet
                signature_data = json.dumps({"result": "invalid"})

            return web.Response(body=signature_data, content_type='application/json')
        
        except Exception as e:
            logger.error(f"Internal request validation failed: {str(e)}")
            return web.Response(status=500, text=f"Internal Server Error: {str(e)}")
    return internal_request_validation
        
########################################### Initializing ###################################################

async def start_send_podIP(podIP_replicaset_dic):
    async with aiohttp.ClientSession() as session:
        while True:
            result = await get_replicas_ip(podIP_replicaset_dic)
            if result:
                await send_pods_ip(session)
            await asyncio.sleep(INTERVAL)

async def start_internal_request_validation(podIP_replicaset_dic):
    print("============== Running on http://0.0.0.0:8080 ==============")
    app = web.Application()
    app.router.add_route('POST', '/send/internal_request_body', create_internal_request_validation(podIP_replicaset_dic))
    app.router.add_route('POST', '/get/data', model_process.create_get_data())
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host='0.0.0.0', port=8080)
    await site.start()
    
    return runner 

async def start_model_process(masterNodeIP, workerNodeIPs):
    async with aiohttp.ClientSession() as session:
        while True:
            await model_process.search_deploy_for_training(session, masterNodeIP, workerNodeIPs)
            await asyncio.sleep(INTERVAL)

############################################## Main ##########################################################

async def main():
    get_nodeIPs_cmd = ["kubectl", "get", "nodes", "-o", r'jsonpath={.items[*].status.addresses[?(@.type=="InternalIP")].address}']
    nodeIPS = subprocess.run(get_nodeIPs_cmd, capture_output=True, text=True, check=True)
    
    # 결과 문자열을 리스트로 변환
    nodeIPS = nodeIPS.stdout.split()
    masterNodeIP = nodeIPS[0]
    workerNodeIPs = nodeIPS[1:]

    # 태스크 생성
    tasks = [
        asyncio.create_task(start_send_podIP(podIP_replicaset_dic)),
        asyncio.create_task(start_model_process(masterNodeIP, workerNodeIPs))
    ]

    runner = await start_internal_request_validation(podIP_replicaset_dic)

    try:
        # 모든 태스크 실행
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("Tasks cancelled")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # 정리
        for task in tasks:
            task.cancel()
        await runner.cleanup()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user")