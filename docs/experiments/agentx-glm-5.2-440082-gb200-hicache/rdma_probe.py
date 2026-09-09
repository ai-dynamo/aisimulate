# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU-buffer NIXL/UCX probe inside existing experiment pods, no GPU use."""
import base64
import ctypes
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import socket
import sys
import time
import urllib.request
from nixl._api import nixl_agent, nixl_agent_config

SIZE = 64 * 1024 * 1024
PORT = 18997
buffer = ctypes.create_string_buffer(SIZE)
address = ctypes.addressof(buffer)
agent = nixl_agent('hzhou-rdma-probe-' + socket.gethostname(),
                   nixl_agent_config(backends=['UCX'], capture_telemetry=True))
registration = agent.register_memory([(address, SIZE, 0, '')], 'DRAM')


def counters():
    values = {}
    for interface in ['eth0', 'rdma0', 'rdma1', 'rdma2', 'rdma3']:
        for counter in ['rx_bytes', 'tx_bytes']:
            path = '/sys/class/net/' + interface + '/statistics/' + counter
            with open(path) as file:
                values[interface + '.' + counter] = int(file.read())
    return values


if sys.argv[1] == 'server':
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            value = {'metadata': base64.b64encode(agent.get_agent_metadata()).decode(),
                     'address': address, 'bytes': SIZE, 'counters': counters()}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def do_POST(self):
            value = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            matches = buffer.raw.count(bytes([value['pattern']]))
            result = {'matching_bytes': matches, 'bytes': SIZE,
                      'valid': matches == SIZE, 'counters': counters()}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            print(json.dumps(result), flush=True)

    server = HTTPServer(('0.0.0.0', PORT), Handler)
    server.timeout = 1
    print('RDMA_PROBE_SERVER_READY', flush=True)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        server.handle_request()
    server.server_close()
else:
    base = 'http://' + sys.argv[2] + ':' + str(PORT)
    pattern = int(sys.argv[3])
    ctypes.memset(address, pattern, SIZE)
    with urllib.request.urlopen(base, timeout=10) as response:
        remote = json.load(response)
    remote_name = agent.add_remote_agent(base64.b64decode(remote['metadata']))
    local_desc = agent.get_xfer_descs([(address, SIZE, 0)], 'DRAM')
    remote_desc = agent.get_xfer_descs([(remote['address'], SIZE, 0)], 'DRAM')
    handle = agent.initialize_xfer('WRITE', local_desc, remote_desc, remote_name,
                                   backends=['UCX'])
    before = counters()
    started = time.monotonic()
    state = agent.transfer(handle)
    while state == 'PROC' and time.monotonic() - started < 30:
        time.sleep(0.01)
        state = agent.check_xfer_state(handle)
    assert state == 'DONE', state
    elapsed = time.monotonic() - started
    request = urllib.request.Request(base, data=json.dumps({'pattern': pattern}).encode(),
                                     headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=10) as response:
        verification = json.load(response)
    assert verification['valid'], verification
    try:
        telemetry = agent.get_xfer_telemetry(handle)
        telemetry = {key: getattr(telemetry, key) for key in
                     ['startTime', 'postDuration', 'xferDuration', 'totalBytes', 'descCount']}
    except Exception as exc:
        telemetry = {'unavailable': repr(exc)}
    print(json.dumps({'state': state, 'bytes': SIZE, 'elapsed_seconds': elapsed,
                      'backend': agent.query_xfer_backend(handle),
                      'ucx_tls': os.environ.get('UCX_TLS'),
                      'gid_index': os.environ.get('UCX_IB_GID_INDEX'),
                      'telemetry': telemetry,
                      'before': before, 'after': counters(),
                      'remote_before': remote['counters'], 'remote_after': verification}), flush=True)
    agent.release_xfer_handle(handle)
agent.deregister_memory(registration)
