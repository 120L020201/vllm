# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check availability and ownership of the local benchmark server endpoint."""

import argparse
import contextlib
import socket
import time
import urllib.error
import urllib.request

import psutil


def _addresses(host: str, port: int):
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    return socket.getaddrinfo(host, port, family, socket.SOCK_STREAM)


def server_url(host: str, port: int) -> str:
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)
    authority = f"[{host}]" if ":" in host else host
    return f"http://{authority}:{port}"


def check_port_available(host: str, port: int) -> None:
    for family, socktype, proto, _, address in _addresses(host, port):
        with socket.socket(family, socktype, proto) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(address)
            except OSError as error:
                raise RuntimeError(
                    f"cannot bind {host}:{port}: {error}; the address may already "
                    "be in use. Choose another PORT or stop the old service yourself"
                ) from error


def _require_running(process: psutil.Process) -> None:
    if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
        raise RuntimeError(f"server process {process.pid} exited before readiness")


def _owns_listener(process: psutil.Process, addresses: set[str], port: int) -> bool:
    candidates = [process]
    # Linux children() scans /proc, including unrelated processes.
    with contextlib.suppress(psutil.AccessDenied, psutil.NoSuchProcess):
        candidates.extend(process.children(recursive=True))
    for candidate in candidates:
        try:
            connections = candidate.net_connections(kind="tcp")
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        for connection in connections:
            if (
                connection.status == psutil.CONN_LISTEN
                and connection.laddr.port == port
                and connection.laddr.ip in addresses
            ):
                return True
    return False


def wait_for_server(host: str, port: int, pid: int, timeout: float = 300) -> None:
    if timeout <= 0:
        raise ValueError("readiness timeout must be positive")
    process = psutil.Process(pid)
    addresses = {entry[4][0] for entry in _addresses(host, port)}
    url = f"{server_url(host, port)}/health"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while True:
        _require_running(process)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"server process {pid} did not own a healthy listener at {host}:{port} "
                f"within {timeout:g}s"
            )
        if _owns_listener(process, addresses, port):
            try:
                with opener.open(url, timeout=min(2.0, remaining)) as response:
                    healthy = response.status == 200 and response.geturl() == url
            except (urllib.error.URLError, OSError):
                healthy = False
            if healthy:
                _require_running(process)
                if not _owns_listener(process, addresses, port):
                    raise RuntimeError(
                        "server lost its listening socket during health check"
                    )
                return
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check-port", "wait"))
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    try:
        if args.action == "check-port":
            check_port_available(args.host, args.port)
            print(server_url(args.host, args.port))
        else:
            if args.pid is None or args.pid <= 0:
                raise ValueError("wait requires a positive --pid")
            wait_for_server(args.host, args.port, args.pid, args.timeout)
    except (OSError, psutil.Error, RuntimeError, ValueError) as error:
        parser.exit(2, f"server_guard: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
