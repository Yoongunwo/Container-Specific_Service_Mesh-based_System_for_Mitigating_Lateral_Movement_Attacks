import aiohttp
import json
import os
import subprocess
from collections import defaultdict
import aiofiles
import uuid
import train_unsupervised as train_unsupervised
import base64

TRAIN_DATA_SIZE_THREASHOLD = 20000
CPU_WEIGHT = 0.5
MEMORY_WEIGHT = 0.5
PROXY_PORT = 9011

async def search_deploy_for_training(session, masterNodeIP, workerNodeIPs):
    try:
        cmd = ["kubectl", "get", "deployments", "-A", "-o", "jsonpath={.items[*].metadata.name}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        deployments = result.stdout.split()

        for deploy in deployments:
            total_data_size = 0
            for nodeIP in workerNodeIPs:
                url = f"http://{nodeIP}:8080/get/data_size"
                post_data = json.dumps({"deploy": deploy})
                try:
                    async with session.post(url, data=post_data) as response:
                        if response.status == 200:
                            data = await response.json()
                            total_data_size += int(data.get('dataSize'))
                        # else:
                            # print(f"Failed to get data size from {url}, status: {response.status}")
                except aiohttp.ClientError as e:
                    # print(f"HTTP request failed: {e}")
                    pass
            if total_data_size > TRAIN_DATA_SIZE_THREASHOLD:
                # v1 Learning on Worker Node
                # selected_node = await find_node_for_training()
                # if await collect_train_data(session, nodeIPS, deploy, selected_node):
                #     if await train_model(session, deploy, selected_node):
                #         await deploy_model(session, deploy)
                # v2 Learning on Master Node
                print('\033[92m' + f"Online Learning for {deploy} ({total_data_size})" + '\033[0m')
                if await collect_train_data_to_master(session, masterNodeIP, workerNodeIPs, deploy):
                    if await train_model_by_master(deploy):
                        await deploy_model(session, deploy)
            elif total_data_size > 0:
                print('\033[93m' + f"Data size for {deploy} is ({total_data_size}/{TRAIN_DATA_SIZE_THREASHOLD})" + '\033[0m')

    except subprocess.CalledProcessError as e:
        print(f"Error fetching deployments: {e}")

############ v1 - Lenaring on Worker Node ################

async def find_node_for_training():
    try:
        cmd = ["kubectl", "top", "nodes"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        nodes = result.stdout.splitlines()[2:]
        
        node_resources = []
        for node in nodes:
            parts = node.split()
            name = parts[0]
            cpu_cores = float(parts[1].rstrip('m'))  # Convert millicores to cores
            cpu_percent = float(parts[2].rstrip('%'))
            memory_bytes = float(parts[3].rstrip('Mi')) * 1024 * 1024  # Convert Mi to bytes
            memory_percent = float(parts[4].rstrip('%'))
            
            # Calculate idle resources
            idle_cpu_percent = 100 - cpu_percent
            idle_memory_percent = 100 - memory_percent
            
            # You can adjust these weights based on your priorities
            idle_score = (idle_cpu_percent * CPU_WEIGHT) + (idle_memory_percent * MEMORY_WEIGHT)
            
            node_resources.append((name, idle_score, idle_cpu_percent, idle_memory_percent))
        
        # Sort nodes by idle score in descending order
        node_resources.sort(key=lambda x: x[1], reverse=True)
        
        # Return the name of the node with the highest idle score
        best_node = node_resources[0]

        cmd = ["kubectl", "get", "nodes", best_node[0], "-o", 
               r'jsonpath={.status.addresses[?(@.type=="InternalIP")].address}']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip('\n')
        return result
    
    except subprocess.CalledProcessError as e:
        print(f"Error executing kubectl: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

async def collect_train_data(session, ips, deploy, selected_node):
    await flush_data(session, deploy)
    
    for node_ip in ips:
        if node_ip == selected_node:
            continue
        url = f"http://{node_ip}:8080/post/data"
        post_data = json.dumps({"node_ip": selected_node, "deploy": deploy})
        try:
            async with session.post(url, data=post_data) as response:
                if response.status == 200:
                    print(f"Successfully sent data to {url}")
                elif response.status == 201:
                    continue
                else:
                    print(f"Failed to send data to {url}, status: {response.status}")
                    return False
        except aiohttp.ClientError as e:
            print(f"HTTP request failed: {e}")
    
    return True

async def flush_data(session, deploy):
    cmd = [
        "kubectl",
        "get",
        "pods",
        "--all-namespaces",
        "-l",
        f"app={deploy}",
        "-o",
        r'jsonpath={range .items[*]}{.metadata.namespace} {.metadata.name} {.status.podIP}{"\n"}{end}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    pods = result.stdout.splitlines()
    for pod in pods:
        ns, pod_name, pod_ip = pod.split()
        url = f"http://{pod_ip}:{PROXY_PORT}/flush"
        try:
            # get method
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"Failed to flush data from {url}, status: {response.status}")
        except aiohttp.ClientError as e:
            print(f"HTTP request failed: {e}")

async def train_model(session, deploy, selected_node):
    url = f"http://{selected_node}:8080/train/model"
    data = json.dumps({"deploy": deploy})
    whoami = subprocess.run(['whoami'], capture_output=True, text=True, check=True).stdout.strip('\n')
    save_dir = f"/home/{whoami}/model/{deploy}"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        async with session.post(url, data=data) as response:
            if response.status == 200:
                reader = aiohttp.MultipartReader.from_response(response)
                while True:
                    part = await reader.next()
                    if part is None:
                        break
                    filename = part.filename
                    if filename in ['model.pkl', 'vectorizer.pkl']:
                        file_path = os.path.join(save_dir, filename)
                        async with aiofiles.open(file_path, 'wb') as f:
                            while True:
                                chunk = await part.read_chunk()
                                if not chunk:
                                    break
                                await f.write(chunk)
                        print(f"Saved {filename}")
                return True
            else:
                print(f"Error: {response.status}")
                return False
    except aiohttp.ClientError as e:
        print(f"HTTP request failed: {e}")

async def deploy_model(session, deploy):
    namespace_cmd = [
        "kubectl", "get", "deploy", "-A",
        f"--field-selector=metadata.name={deploy}",
        "-o", "jsonpath={.items[0].metadata.namespace}"
    ]
    namespace_result = subprocess.run(namespace_cmd, capture_output=True, text=True, check=True)
    namespace = namespace_result.stdout.strip()

    # selector 찾기 위한 커맨드
    selector_cmd = [
        "kubectl", "get", "deploy", deploy, "-n", namespace,
        "-o", "jsonpath={.spec.selector.matchLabels}", 
    ]
    selector_result = subprocess.run(selector_cmd, capture_output=True, text=True, check=True)
    selector = selector_result.stdout.strip()
    selector = selector.replace('{', '').replace('}', '').replace('"', '').replace(':', '=')

    # 최종 pod 정보 가져오는 커맨드
    cmd = [
        "kubectl", "get", "pods",
        "-n", namespace,
        "--selector", selector,
        "-o", r'jsonpath={range .items[*]}{.status.podIP}{"\n"}{end}'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    pods = result.stdout.splitlines()

    whoami = os.getlogin()
    file_path = f"/home/{whoami}/model/{deploy}/"
    counter = get_filename(file_path) - 1

    for pod_ip in pods:
        url = f"http://{pod_ip}:{PROXY_PORT}/receive/model"
        try:
            async with aiofiles.open(file_path + f'model_{counter}.pkl', 'rb') as model_file:
                model_data = await model_file.read()
                model_base64 = base64.b64encode(model_data).decode('utf-8')

            async with aiofiles.open(file_path + 'vectorizer.pkl', 'rb') as vectorizer_file:
                vectorizer_data = await vectorizer_file.read()
                vectorizer_base64 = base64.b64encode(vectorizer_data).decode('utf-8')

            data = {
                'model': model_base64,
                'vectorizer': vectorizer_base64
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as response:
                    if response.status == 200:
                        print('\033[92m' + f"Successfully deployed model to {url}" + '\033[0m')
                    else:
                        print('\033[91m' + f"Failed to deploy model to {url}" + '\033[0m')

        except aiohttp.ClientError as e:
            print(f"HTTP request failed: {e}")

############ v2 Learning on Master Node ################

async def collect_train_data_to_master(session, masterNodeIP, workerNodeIPs, deploy):
    # await flush_data(session, deploy)

    whoami = os.getlogin()
    for nodeIP in workerNodeIPs:
        url = f"http://{nodeIP}:8080/post/data_to_master"
        post_data = json.dumps({"deploy": deploy})
        try:
            async with session.post(url, data=post_data) as response:
                if response.status == 200:
                    data = await response.json()
                    deploy = data.get('deploy')
                    filename = data.get('filename')
                    file_content = base64.b64decode(data.get('file_content'))
                    
                    if not os.path.exists(f'/home/{whoami}/model/{deploy}'):
                        os.makedirs(f'/home/{whoami}/model/{deploy}')

                    with open(os.path.join(f'/home/{whoami}/model/{deploy}', filename), 'wb') as f:
                        f.write(file_content)

                else:
                    print(f"Failed to get data from {url}, status: {response.status}")
        except aiohttp.ClientError as e:
            print(f"HTTP request failed: {e}")

    return True

def create_get_data():
    whoami = os.getlogin()
    async def get_data(request):
        reader = await request.multipart()

        field = await reader.next()
        assert field.name == 'deploy'
        deploy = await field.read()
        if isinstance(deploy, (bytes, bytearray)):
            deploy = deploy.decode('utf-8')

        field = await reader.next()
        assert field.name == 'file'
        filename = field.filename

        size = 0
        if not os.path.exists(f'/home/{whoami}/model/{deploy}'):
            os.makedirs(f'/home/{whoami}/model/{deploy}')

        with open(os.path.join(f'/home/{whoami}/model/{deploy}', filename), 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                f.write(chunk)

        return aiohttp.web.Response(text=f'File "{filename}" successfully saved. Size: {size} bytes')
    return get_data

def get_filename(file_path):
    counter = 0
    while True:
        filename = f'model_{counter}.pkl'
        if not os.path.exists(file_path + filename):
            return counter
        counter += 1

async def train_model_by_master(deploy):
    whoami = os.getlogin()
    file_path = f'/home/{whoami}/model/{deploy}/'
    counter = get_filename(file_path)

    return train_unsupervised.train_model(deploy, file_path, counter)

