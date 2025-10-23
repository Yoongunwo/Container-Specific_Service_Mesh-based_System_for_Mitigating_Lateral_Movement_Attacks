# -*- coding: utf-8 -*-

import asyncio
import aiohttp
from aiohttp import web
import logging
import json
import os
import subprocess
import re
import multiprocessing
import train_supervised as train_supervised
import train_unsupervised as train_unsupervised
import aiofiles
import uuid
import base64
import aiohttp.web
from pathlib import Path

# 로깅 설정
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 주기적으로 파드 IP 주소를 가져올 시간 간격 (초)
INTERVAL = int(os.environ.get('INTERVAL', 1))
PODID = set()
PROXY_PORT = 9011

def parse_pstree_output(pstree_output):
    lines = pstree_output.splitlines()
    pid_pattern = re.compile(r'\b(\d+)\b')
    exclude_pids = set()
    parent_pid = None
    
    def count_leading_spaces(line):
        return len(line) - len(line.lstrip('| '))
    
    reverse_proxy_level = None
    prev_level = 0
    prev_pid = None
    
    for line in lines:
        level = count_leading_spaces(line)
        pid = pid_pattern.search(line).group(1)
        
        if "reverseProxyServer" in line:
            if parent_pid is None:  # 첫 번째 reverseProxyServer의 parent만 저장
                if prev_level < level:  # 이전 레벨이 더 작으면 parent
                    parent_pid = prev_pid
                    exclude_pids.add(parent_pid)
            reverse_proxy_level = level
            exclude_pids.add(pid)
        elif "containerd-shim" in line or "sleep" in line or "pause" in line:
            exclude_pids.add(pid)
        elif reverse_proxy_level is not None and level > reverse_proxy_level:
            exclude_pids.add(pid)
        elif reverse_proxy_level is not None and level <= reverse_proxy_level:
            reverse_proxy_level = None

        prev_level = level
        prev_pid = pid

    return exclude_pids

def get_all_pids(pstree_output):
    pid_pattern = re.compile(r'(\S+),(\d+)')
    return set(match.group(2) for match in pid_pattern.finditer(pstree_output))

async def get_podID(session):
    global PODID
    try:
        cmd = ["crictl", "ps"]
        podInfoResult = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        #logger.info(f"Pod Info: {podInfoResult.stdout}")
        podsInfo = podInfoResult.stdout.splitlines()[1:]  # 헤더 제외
        podIDs = set()
        for podInfo in podsInfo:
            if podInfo and "reverse-proxy" in podInfo:
                index = podInfo.split().index("reverse-proxy")
                podID = podInfo.split()[index + 2]
                podIDs.add(podID)

        added_podIDs = podIDs - PODID
        if added_podIDs:
            for podID in added_podIDs:
                # logger.info(f"New PODID: {podID}")
                if not await get_PID(podID, session):
                    podIDs.remove(podID)

        PODID = podIDs
        # logger.info(f"PODID: {PODID}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error fetching containerID: {e}")
        return []

async def get_PID(podID, session):
    try:
        cmd = f"crictl inspectp {podID} | grep '\"ip\"'"
        podIP = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            check=True
        )
        # remove double quotes
        podIP = podIP.stdout.split()[1][1:-1]

        # logger.info(f"podIP: {podIP}")

        cmd = f"ps -ef | grep {podID}"
        podPID = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            check=True
        )
        for pid in podPID.stdout.split('\n'):
            if 'grep' in pid:
                continue
            else:
                podPID = pid
                break

        podPID = podPID.split()[1]

        cmd = f"pstree -ap {podPID}"
        pstree = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=True,
            check=True
        )
        # logger.info(f"pstree: {pstree.stdout}")

        exclude_pids = parse_pstree_output(pstree.stdout)
        all_pids = get_all_pids(pstree.stdout)

        final_pids = list(all_pids - exclude_pids)
        
        data = json.dumps({"rootPID": podPID, "pid": final_pids})

        url = f"http://{podIP}:{PROXY_PORT}/receive/pid"
        headers = {'Content-Type': 'application/json'}
        try:
            async with session.post(url, data=data, headers=headers) as response:
                if response.status == 200:
                    print('\033[32m' + f"Successfully sent PID to {podIP}: {data}" + '\033[0m')
                else:
                    logger.error(f"Failed to send PID to {url}, status: {response.status}")
                    return False
        except aiohttp.ClientError as e:
            # logger.error(f"HTTP request failed: {e}")
            return False

    except subprocess.CalledProcessError as e:
        print('\033[31m' + f"Failed sent PID to {url}" + '\033[0m')
        return False

    return True


async def start_control_plane_pid():
    async with aiohttp.ClientSession() as session:
        while True:
            await get_podID(session)
            await asyncio.sleep(INTERVAL)

################################### model process ###################################

def create_check_data_size():
    whoami = os.getlogin()

    async def check_data_size(request):
        data = await request.json()
        deploy = data['deploy']

        ### data 확인 ### 
        file_path = f'/home/{whoami}/model/{deploy}/data.txt'
        if not os.path.exists(file_path):
            return aiohttp.web.Response(status=404, text='Data file does not exist')
        
        result = subprocess.run(['wc', '-l', f'/home/{whoami}/model/{deploy}/data.txt'],
                                 capture_output=True, text=True, check=True)
        dataSize = result.stdout.split()[0]
        body = json.dumps({"dataSize": dataSize})

        return aiohttp.web.Response(body=body, status=200, content_type='application/json')
    return check_data_size

def create_post_data():
    whoami = os.getlogin()
    hostname = os.uname().nodename
    async def post_data(request):
        try:
            data = await request.json()
            dst_node = data['node_ip']
            deploy = data['deploy']

            if not os.path.exists(f'/home/{whoami}/model/{deploy}'):
                os.makedirs(f'/home/{whoami}/model/{deploy}')

            file_path = f'/home/{whoami}/model/{deploy}/data.txt'

            if not os.path.exists(file_path):
                return aiohttp.web.Response(status=201, text=f'File not found: {file_path}')
            
            async with aiofiles.open(file_path, mode='rb') as f:
                file_content = await f.read()

            form_data = aiohttp.FormData()
            form_data.add_field('deploy', deploy)
            form_data.add_field('file', 
                                file_content,
                                filename = hostname + "-" + os.path.basename(file_path),
                                content_type='application/octet-stream')

            async with aiohttp.ClientSession() as session:
                async with session.post(f'http://{dst_node}:8080/get/data', data=form_data) as response:
                    if response.status == 200:
                        async with aiofiles.open(file_path, mode='w') as f:
                            await f.write('') # 파일 내용 삭제 => 수정 가능성 존재
                        return aiohttp.web.Response(status=200, text='Successfully sent data')
                    else:
                        return aiohttp.web.Response(status=500, text='Failed to send data')
        except Exception as e:
            return aiohttp.web.Response(status=500, text=f'Error : {e}')    

    return post_data

def create_post_data_to_master():
    whoami = os.getlogin()
    hostname = os.uname().nodename
    async def post_data(request):
        try:
            data = await request.json()
            deploy = data['deploy']

            file_path = f'/home/{whoami}/model/{deploy}/data.txt'
            
            if not os.path.exists(file_path):
                return aiohttp.web.Response(status=201, text=f'File not found: {file_path}')
        
            async with aiofiles.open(file_path, mode='rb') as f:
                file_content = await f.read()

            response_data = {
                'deploy': deploy,
                'filename': hostname + "-" + os.path.basename(file_path),
                'file_content': base64.b64encode(file_content).decode()
            }
            
            if os.path.exists(file_path):
                new_full_path = await get_new_backup_name(file_path)
                os.rename(file_path, new_full_path)
                
            async with aiofiles.open(file_path, mode='w') as f:
                await f.write('')
            
            os.chmod(file_path, 0o777)
                
            return web.json_response(response_data)

        except Exception as e:
            print(f"Error in post_data_to_master: {e}")
            return aiohttp.web.Response(status=500, text=f'Error : {e}')    

    return post_data

async def get_new_backup_name(file_path: str) -> str:
    """
    백업 파일의 새로운 이름을 생성합니다.
    이미 존재하는 백업 파일이 있다면 숫자를 증가시켜 새 이름을 만듭니다.
    """
    path = Path(file_path)
    directory = path.parent
    base_name = path.stem  # 확장자를 제외한 파일명
    extension = path.suffix  # 확장자
    counter = 0
    
    while True:
        new_name = f"{base_name}{counter}{extension}.cp"
        new_full_path = os.path.join(directory, new_name)
            
        if not os.path.exists(new_full_path):
            return new_full_path
        counter += 1

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
        with open(os.path.join(f'/home/{whoami}/model/{deploy}', filename), 'wb') as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                size += len(chunk)
                f.write(chunk)

        return aiohttp.web.Response(text=f'File "{filename}" successfully saved. Size: {size} bytes')
    return get_data

def create_train_model():
    whoami = os.getlogin()
    async def train_model(request):
        data = await request.json()
        deploy = data['deploy']
        file_path = f'/home/{whoami}/model/{deploy}/'
    
        if train_supervised.train_model(file_path):
            boundary = uuid.uuid4().hex
            response = web.StreamResponse(
                status=200,
                reason='OK',
                headers={
                    'Content-Type': f'multipart/form-data; boundary={boundary}'
                }
            )
            await response.prepare(request)

            for filename in ['model.pkl', 'vectorizer.pkl']:
                file_path_full = os.path.join(file_path, filename)
                file_size = os.path.getsize(file_path_full)
                
                # Write multipart headers
                await response.write(
                    f'--{boundary}\r\n'
                    f'Content-Disposition: form-data; name="{filename}"; filename="{filename}"\r\n'
                    f'Content-Type: application/octet-stream\r\n'
                    f'Content-Length: {file_size}\r\n\r\n'.encode()
                )

                # Write file content
                async with aiofiles.open(file_path_full, mode='rb') as f:
                    chunk_size = 8192  # 8KB chunks
                    while True:
                        chunk = await f.read(chunk_size)
                        if not chunk:
                            break
                        await response.write(chunk)

                await response.write(b'\r\n')

            # Write final boundary
            await response.write(f'--{boundary}--\r\n'.encode())

            return response
        else:
            return web.Response(status=500, text="Model training failed")

    return train_model


async def start_control_model_data():
    print("============== Running on http://0.0.0.0:8080 ==============")
    app = aiohttp.web.Application()
    app.router.add_route('POST', '/get/data_size', create_check_data_size())
    app.router.add_route('POST', '/post/data', create_post_data())
    app.router.add_route('POST', '/post/data_to_master', create_post_data_to_master())
    app.router.add_route('POST', '/get/data', create_get_data())
    app.router.add_route('POST', '/train/model', create_train_model())

    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, '0.0.0.0', 8080)
    await site.start()

    return runner

################################### main process ###################################

async def main():
    tasks = [
        asyncio.create_task(start_control_plane_pid()),
    ]
    runner = await start_control_model_data()
        
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("Tasks cancelled")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        for task in tasks:
            task.cancel()
        await runner.cleanup()



if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Program terminated by user")