#!/usr/bin/env python3
import paramiko

server = "jabba.king-little.ts.net"
username = "pg2022"
password = "isthatthemars42"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(server, username=username, password=password, port=2222)
print("ok")
stdin,stdout,stderr=ssh.exec_command('python3 screen_test.py')
output = stdout.readlines()
for items in output:
    print(items)
