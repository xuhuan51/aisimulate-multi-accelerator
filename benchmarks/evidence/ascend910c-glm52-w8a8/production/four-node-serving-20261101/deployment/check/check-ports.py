#!/usr/bin/env python3
"""用途：在加载大模型前确认 HTTP、Bootstrap 和 MemFabric 端口可以绑定。"""

from __future__ import annotations

import socket
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(f"用法: {sys.argv[0]} <监听IP> <端口> [端口...]", file=sys.stderr)
        return 2

    host = sys.argv[1]
    sockets: list[socket.socket] = []
    try:
        for value in sys.argv[2:]:
            try:
                port = int(value)
            except ValueError:
                print(f"无效端口: {value}", file=sys.stderr)
                return 2

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, port))
            except OSError as exc:
                print(f"端口预检查失败 {host}:{port}: {exc}", file=sys.stderr)
                print(
                    f"请执行 ss -ltnp 'sport = :{port}' 检查占用来源。",
                    file=sys.stderr,
                )
                return 1

        print(f"端口预检查通过: {host} {', '.join(sys.argv[2:])}")
        return 0
    finally:
        for sock in sockets:
            sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
