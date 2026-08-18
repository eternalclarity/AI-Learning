# Ubuntu 云服务器操作与部署速查笔记

> 适用场景：Ubuntu Server 云主机、深度学习训练、模型推理服务、Python 后端、前端网站、Docker 部署与日常运维。  
> 主要面向 Ubuntu LTS，命令在 22.04、24.04、26.04 等版本上通常相近。涉及驱动、CUDA、PyTorch、Docker 和 Certbot 安装时，应以对应项目的最新官方文档为准。  
> 最后整理：2026-08-05

---

## 使用方法

这份笔记采用“**快速索引 → 场景流程 → 详细解释 → 故障排查**”的结构。

忘记命令时，优先使用编辑器的全文搜索：

| 想做什么 | 推荐搜索词 |
|---|---|
| 查看端口 | `端口`、`ss`、`lsof` |
| 查看服务日志 | `journalctl` |
| 后台训练 | `tmux`、`nohup` |
| 上传文件 | `scp`、`rsync` |
| 磁盘满了 | `df`、`du`、`磁盘不足` |
| GPU 不可用 | `nvidia-smi`、`CUDA` |
| 部署 API | `systemd`、`Uvicorn` |
| 部署网站 | `Nginx`、`静态前端` |
| HTTPS | `Certbot`、`证书` |
| Docker | `docker compose` |
| 502 错误 | `502` |
| 权限不足 | `chmod`、`chown` |
| 定时任务 | `cron`、`timer` |
| 备份恢复 | `备份`、`rsync`、`pg_dump` |

### 命令标记

- ✅ **只读**：一般只查看信息，不修改系统。
- ✏️ **修改**：会安装、创建、修改或重启。
- ⚠️ **危险**：可能删除数据、断开连接或造成服务中断，执行前必须确认。

命令中的以下内容需要替换：

```text
<USER>          用户名
<SERVER_IP>     服务器公网 IP
<PORT>          端口
<DOMAIN>        域名
<PROJECT>       项目名
<PATH>          文件或目录路径
<SERVICE>       systemd 服务名
<CONTAINER>     Docker 容器名或 ID
```

---

# 第一部分：30 秒命令速查

## 1. 系统与资源

```bash
# ✅ 当前用户、主机名、工作目录
whoami
hostname
pwd

# ✅ Ubuntu 版本、内核、架构
cat /etc/os-release
uname -a
uname -m
hostnamectl

# ✅ 运行时间和负载
uptime

# ✅ CPU
lscpu
nproc

# ✅ 内存
free -h
watch -n 1 free -h

# ✅ 磁盘与挂载
df -h
df -i
lsblk
findmnt

# ✅ 某个目录占用
du -sh <PATH>
du -h --max-depth=1 <PATH> | sort -h

# ✅ GPU
nvidia-smi
watch -n 1 nvidia-smi
```

## 2. 文件和目录

```bash
pwd
ls
ls -lah
cd <PATH>
cd ..
cd ~
cd -

mkdir <DIR>
mkdir -p a/b/c

touch file.txt
cp source target
cp -r source_dir target_dir
mv old new

# ⚠️ 删除前先 ls 确认路径
rm file.txt
rm -r directory
rm -rf directory

find <PATH> -name "*.py"
find <PATH> -type f -size +1G
```

## 3. 查看和搜索文本

```bash
cat file.txt
less file.txt
head -n 20 file.txt
tail -n 50 file.txt
tail -f app.log

grep "error" app.log
grep -i "error" app.log
grep -RIn "keyword" <PATH>

wc -l file.txt
sort file.txt
sort file.txt | uniq
sort file.txt | uniq -c
```

## 4. 权限和用户

```bash
whoami
id
groups

ls -l
chmod u+x script.sh
chmod 644 file.txt
chmod 755 directory
chown <USER>:<USER> file.txt
chown -R <USER>:<USER> directory

sudo <COMMAND>
sudo -i
adduser <USER>
usermod -aG sudo <USER>
```

## 5. 软件安装

```bash
sudo apt update
sudo apt upgrade
sudo apt install <PACKAGE>
sudo apt remove <PACKAGE>
sudo apt purge <PACKAGE>
sudo apt autoremove

apt search <KEYWORD>
apt show <PACKAGE>
apt policy <PACKAGE>
dpkg -l | grep <PACKAGE>
```

## 6. 进程、端口、服务和日志

```bash
ps aux
ps aux | grep python
pgrep -af python

top
htop

ss -lntp
sudo lsof -i :8000

kill <PID>
kill -15 <PID>
kill -9 <PID>
pkill -f "train.py"

systemctl status <SERVICE>
sudo systemctl start <SERVICE>
sudo systemctl stop <SERVICE>
sudo systemctl restart <SERVICE>
sudo systemctl reload <SERVICE>
sudo systemctl enable <SERVICE>
sudo systemctl enable --now <SERVICE>

journalctl -u <SERVICE>
journalctl -u <SERVICE> -f
journalctl -u <SERVICE> --since today
journalctl -u <SERVICE> -n 200 --no-pager
```

## 7. 网络

```bash
ip addr
ip route
hostname -I

ping -c 4 8.8.8.8
curl -I https://example.com
curl http://127.0.0.1:8000/health
wget <URL>

ss -lntp
sudo lsof -i :<PORT>

dig <DOMAIN>
nslookup <DOMAIN>
nc -zv <HOST> <PORT>
```

## 8. SSH 和文件传输

```bash
ssh <USER>@<SERVER_IP>
ssh -p <PORT> <USER>@<SERVER_IP>
ssh -i ~/.ssh/id_ed25519 <USER>@<SERVER_IP>

scp file <USER>@<SERVER_IP>:/remote/path/
scp -r directory <USER>@<SERVER_IP>:/remote/path/

rsync -avhP directory/ <USER>@<SERVER_IP>:/remote/path/
rsync -avhP --delete directory/ <USER>@<SERVER_IP>:/remote/path/
```

## 9. 后台训练

```bash
tmux new -s train
tmux ls
tmux attach -t train
# tmux 内按 Ctrl+b，再按 d：退出但任务继续运行

nohup python -u train.py > logs/train.log 2>&1 &
echo $!
tail -f logs/train.log

CUDA_VISIBLE_DEVICES=0 python train.py
```

## 10. Git

```bash
git clone <REPO_URL>
git status
git pull
git add .
git commit -m "message"
git push
git log --oneline --graph -n 20
git switch <BRANCH>
```

## 11. Python 环境

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip list
python -m pip freeze > requirements.lock.txt

deactivate
```

## 12. Nginx

```bash
sudo nginx -t
sudo systemctl status nginx
sudo systemctl reload nginx
sudo systemctl restart nginx
journalctl -u nginx -n 100 --no-pager
```

## 13. Docker

```bash
docker version
docker info

docker ps
docker ps -a
docker images

docker logs -f <CONTAINER>
docker exec -it <CONTAINER> bash

docker stop <CONTAINER>
docker start <CONTAINER>
docker restart <CONTAINER>
docker rm <CONTAINER>
docker rmi <IMAGE>

docker compose up -d
docker compose down
docker compose ps
docker compose logs -f
docker compose build
docker compose pull
docker compose restart
```

---

# 第二部分：常见场景完整流程

## 场景 A：拿到一台新 Ubuntu 云服务器

### 目标流程

```text
确认系统信息
→ 更新系统
→ 创建普通 sudo 用户
→ 配置 SSH 密钥
→ 确认新用户可以登录
→ 配置防火墙和云安全组
→ 再考虑关闭 root/密码登录
→ 安装项目依赖
```

### 1. 登录并确认系统

```bash
ssh root@<SERVER_IP>

whoami
cat /etc/os-release
uname -a
lscpu
free -h
df -h
ip addr
```

### 2. 更新系统

```bash
sudo apt update
sudo apt upgrade
```

服务器更新内核、驱动或底层库后，可能需要重启：

```bash
test -f /var/run/reboot-required && cat /var/run/reboot-required
sudo reboot
```

### 3. 创建普通用户

```bash
sudo adduser <USER>
sudo usermod -aG sudo <USER>

id <USER>
```

切换用户测试：

```bash
su - <USER>
sudo whoami
```

### 4. 配置 SSH 密钥

在本地电脑生成密钥：

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

复制公钥：

```bash
ssh-copy-id <USER>@<SERVER_IP>
```

Windows 没有 `ssh-copy-id` 时，可查看公钥：

```powershell
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

服务器上写入：

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh

nano ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

先开一个**新的终端窗口**测试密钥登录，确认成功后，再修改 SSH 安全设置。

### 5. SSH 安全配置

推荐新增独立配置文件，而不是大幅修改主文件：

```bash
sudo nano /etc/ssh/sshd_config.d/99-hardening.conf
```

示例：

```text
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
```

先检查语法：

```bash
sudo sshd -t
```

再重载：

```bash
sudo systemctl reload ssh
```

> ⚠️ 在确认普通用户密钥登录成功之前，不要关闭 root 或密码登录，否则可能把自己锁在服务器外。

### 6. 防火墙

云服务器通常有两层防火墙：

1. 云厂商安全组；
2. Ubuntu 本机 UFW。

先允许 SSH，再启用 UFW：

```bash
sudo ufw allow OpenSSH
sudo ufw enable
sudo ufw status verbose
```

网站常用：

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

若 SSH 使用自定义端口：

```bash
sudo ufw allow <PORT>/tcp
```

---

## 场景 B：把深度学习项目传到服务器并开始训练

### 目标流程

```text
检查 GPU
→ 获取代码
→ 创建独立环境
→ 安装匹配的 PyTorch
→ 准备数据
→ 先小规模测试
→ 使用 tmux 正式训练
→ 监控日志、GPU、磁盘
→ 定期保存 checkpoint
```

### 1. 检查 GPU

```bash
nvidia-smi
```

重点查看：

- GPU 型号；
- 驱动版本；
- 显存总量与已使用显存；
- 当前占用 GPU 的进程。

持续刷新：

```bash
watch -n 1 nvidia-smi
```

只显示部分信息：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv
```

查看 GPU 上的计算进程：

```bash
nvidia-smi pmon
```

### 2. 获取代码

Git：

```bash
git clone <REPO_URL>
cd <PROJECT>
```

从本地同步：

```bash
rsync -avhP \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude ".venv" \
  ./ <USER>@<SERVER_IP>:/srv/<PROJECT>/
```

### 3. 创建 Python 环境

#### 方案一：venv

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip

cd /srv/<PROJECT>
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

#### 方案二：Conda

```bash
conda create -n ai python=3.11 -y
conda activate ai

python -m pip install -r requirements.txt
```

导出环境：

```bash
conda env export --from-history > environment.yml
python -m pip freeze > requirements.lock.txt
```

> 不要对系统 Python 使用 `sudo pip install`，否则容易污染系统环境并破坏 APT 管理的软件。

### 4. 安装 PyTorch

PyTorch、CUDA 轮子和支持版本会变化。应进入 PyTorch 官方安装选择器，根据以下条件复制当前命令：

```text
OS: Linux
Package: Pip
Language: Python
Compute Platform: 与服务器和官方支持相匹配的 CUDA
```

安装后测试：

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("torch CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
PY
```

理解两件事：

- `nvidia-smi` 主要反映 NVIDIA 驱动和 GPU 状态；
- `torch.cuda.is_available()` 反映当前 Python 环境中的 PyTorch 是否能使用 CUDA。

### 5. 训练前快速检查

```bash
pwd
which python
python --version
python -m pip show torch

ls -lah
df -h
free -h
nvidia-smi
```

先跑一个小实验：

```bash
python train.py \
  --epochs 1 \
  --batch-size 8
```

确认：

- 数据路径正确；
- 模型可以前向和反向传播；
- GPU 被正确使用；
- 日志和 checkpoint 路径可写；
- 验证流程可以完成。

### 6. 使用 tmux 正式训练

安装：

```bash
sudo apt install tmux
```

创建会话：

```bash
tmux new -s train
```

进入项目：

```bash
cd /srv/<PROJECT>
source .venv/bin/activate
```

开始训练并同时记录日志：

```bash
mkdir -p logs checkpoints

python -u train.py 2>&1 | tee logs/train_$(date +%F_%H-%M-%S).log
```

退出但保持任务运行：

```text
Ctrl+b
然后按 d
```

重新连接：

```bash
tmux ls
tmux attach -t train
```

关闭会话：

```bash
tmux kill-session -t train
```

### 7. nohup 方案

```bash
mkdir -p logs

nohup python -u train.py \
  > logs/train.log \
  2>&1 &

echo $!
```

查看：

```bash
tail -f logs/train.log
pgrep -af train.py
```

停止：

```bash
kill -15 <PID>
```

> tmux 通常比 nohup 更方便，因为可以重新进入原来的终端环境。

### 8. 指定 GPU

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```

使用物理 GPU 1 和 3，并在程序中重新编号为 `cuda:0`、`cuda:1`：

```bash
CUDA_VISIBLE_DEVICES=1,3 python train.py
```

多进程分布式训练示例：

```bash
torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py
```

前提是训练代码已正确实现 DistributedDataParallel。

### 9. 训练监控

```bash
# GPU
watch -n 1 nvidia-smi

# CPU、内存
htop

# 磁盘
watch -n 5 df -h

# 日志
tail -f logs/train.log

# 进程
pgrep -af train.py
ps -o pid,ppid,etime,%cpu,%mem,cmd -p <PID>
```

### 10. Checkpoint 与断点恢复

项目至少应保存：

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "best_metric": best_metric,
}
```

建议：

- 保存 `latest.pt`：用于异常恢复；
- 保存 `best.pt`：用于最终评估和部署；
- 不要只在训练结束时保存；
- 检查磁盘空间，避免 checkpoint 写一半失败；
- 记录代码提交号、配置文件和随机种子。

### 11. 显存不足 OOM 排查

按优先级处理：

```text
减小 batch size
→ 使用梯度累积
→ 使用自动混合精度
→ 降低输入尺寸或序列长度
→ 开启梯度检查点
→ 减少模型规模
→ 检查是否保存了不必要的计算图
```

查看是否有其他进程占用：

```bash
nvidia-smi
```

不要随意结束其他用户的进程。

---

## 场景 C：部署一个 Python 模型 API

推荐架构：

```text
浏览器/客户端
    ↓ HTTPS 443
Nginx
    ↓ 127.0.0.1:8000
Uvicorn / FastAPI
    ↓
模型推理代码
```

优点：

- 应用端口不直接暴露公网；
- Nginx 处理域名、HTTPS、静态文件和反向代理；
- systemd 管理应用开机启动、自动重启和日志。

### 1. 项目目录

```text
/srv/model-api/
├── app/
│   ├── __init__.py
│   └── main.py
├── models/
│   └── best.pt
├── requirements.txt
└── .venv/
```

### 2. 创建环境

```bash
sudo mkdir -p /srv/model-api
sudo chown -R <USER>:<USER> /srv/model-api

cd /srv/model-api
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. 本地启动测试

```bash
/srv/model-api/.venv/bin/uvicorn \
  app.main:app \
  --host 127.0.0.1 \
  --port 8000
```

另开终端测试：

```bash
curl http://127.0.0.1:8000/health
```

查看端口：

```bash
ss -lntp | grep 8000
```

### 4. 创建 systemd 服务

```bash
sudo nano /etc/systemd/system/model-api.service
```

示例：

```ini
[Unit]
Description=Model API
After=network.target

[Service]
Type=simple
User=<USER>
Group=<USER>
WorkingDirectory=/srv/model-api
EnvironmentFile=-/etc/model-api.env
ExecStart=/srv/model-api/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

加载并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now model-api
sudo systemctl status model-api
```

实时日志：

```bash
journalctl -u model-api -f
```

更新服务文件后：

```bash
sudo systemctl daemon-reload
sudo systemctl restart model-api
```

> GPU 模型通常不要盲目增加多个 worker。每个 worker 可能分别加载一份模型并占用一份显存。应根据模型大小、显存和并发方式决定 worker 数量。

### 5. 环境变量与密钥

```bash
sudo nano /etc/model-api.env
sudo chmod 600 /etc/model-api.env
```

示例：

```text
MODEL_PATH=/srv/model-api/models/best.pt
LOG_LEVEL=INFO
API_SECRET=replace-me
```

不要把密钥提交到 Git。

---

## 场景 D：Nginx 反向代理与 HTTPS

### 1. 安装 Nginx

```bash
sudo apt update
sudo apt install nginx

sudo systemctl enable --now nginx
sudo systemctl status nginx
```

### 2. 创建站点配置

```bash
sudo nano /etc/nginx/sites-available/model-api
```

示例：

```nginx
server {
    listen 80;
    listen [::]:80;

    server_name <DOMAIN>;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;

        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 30s;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

启用：

```bash
sudo ln -s \
  /etc/nginx/sites-available/model-api \
  /etc/nginx/sites-enabled/model-api
```

可选：删除默认站点：

```bash
sudo rm /etc/nginx/sites-enabled/default
```

先测试再重载：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

### 3. 防火墙

```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw status
```

同时确保云厂商安全组允许 80 和 443。

### 4. 配置域名

在 DNS 服务商处添加：

```text
A 记录：
<DOMAIN> → <SERVER_IP>
```

检查：

```bash
dig <DOMAIN>
```

### 5. Certbot 配置 HTTPS

Certbot 官方对常见 Linux + Nginx 环境推荐 snap 安装方式：

```bash
sudo snap install --classic certbot
sudo ln -s /snap/bin/certbot /usr/local/bin/certbot
```

申请并自动修改 Nginx：

```bash
sudo certbot --nginx
```

测试自动续期：

```bash
sudo certbot renew --dry-run
```

查看 timer：

```bash
systemctl list-timers | grep certbot
```

申请证书前应确认：

- 域名已指向服务器；
- 80 端口公网可访问；
- Nginx 配置正确；
- 云安全组和 UFW 均放行 80/443。

---

## 场景 E：部署静态前端和后端 API

推荐结构：

```text
/var/www/myapp/dist      前端构建产物
/srv/myapp-api           Python 后端
```

### 1. 构建前端

通常在本地或 CI 中构建：

```bash
npm ci
npm run build
```

上传：

```bash
rsync -avhP --delete \
  dist/ \
  <USER>@<SERVER_IP>:/tmp/myapp-dist/
```

服务器部署：

```bash
sudo mkdir -p /var/www/myapp
sudo rsync -avh --delete \
  /tmp/myapp-dist/ \
  /var/www/myapp/dist/

sudo chown -R www-data:www-data /var/www/myapp
```

### 2. Nginx 配置 SPA + API

```nginx
server {
    listen 80;
    listen [::]:80;

    server_name <DOMAIN>;

    root /var/www/myapp/dist;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;

        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_read_timeout 300s;
    }
}
```

检查并重载：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

---

## 场景 F：使用 Docker Compose 部署

### 1. 安装 Docker Engine

生产服务器优先使用 Docker 官方 APT 仓库，而不是一键安装脚本。

当前官方流程的核心形式：

```bash
sudo apt update
sudo apt install ca-certificates curl

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL \
  https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources >/dev/null <<EOF_DOCKER
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF_DOCKER

sudo apt update
sudo apt install \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin
```

验证：

```bash
sudo systemctl status docker
sudo docker run --rm hello-world
docker compose version
```

### 2. 非 root 使用 Docker

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

验证：

```bash
docker run --rm hello-world
```

> ⚠️ `docker` 用户组通常等价于拥有很高的主机权限，只应加入可信用户。

### 3. 常用 Docker 命令

```bash
docker ps
docker ps -a
docker images
docker stats

docker logs -f <CONTAINER>
docker inspect <CONTAINER>
docker exec -it <CONTAINER> bash

docker stop <CONTAINER>
docker restart <CONTAINER>
docker rm <CONTAINER>
docker rmi <IMAGE>
```

### 4. Compose 示例

```yaml
services:
  api:
    build: .
    restart: unless-stopped
    env_file:
      - .env
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data
```

启动：

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f
```

停止：

```bash
docker compose down
```

更新单个服务：

```bash
docker compose build api
docker compose up -d --no-deps api
```

### 5. Docker 与 UFW 的重要注意事项

Docker 发布端口时，流量处理可能绕过常规 UFW 规则。安全做法：

- 后端服务尽量绑定 `127.0.0.1`；
- 对公网只开放 Nginx 的 80/443；
- 不要直接写 `"8000:8000"`，除非确实需要公网访问；
- 同时检查云安全组；
- 对生产环境进一步了解 Docker 的 `DOCKER-USER` 防火墙链。

### 6. GPU 容器

先确认宿主机驱动：

```bash
nvidia-smi
```

安装 NVIDIA Container Toolkit 后配置 Docker：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

用适合当前环境的 NVIDIA CUDA 镜像测试：

```bash
docker run --rm --gpus all <NVIDIA_CUDA_IMAGE> nvidia-smi
```

不要照抄过期 CUDA 镜像标签，应在 NVIDIA 官方镜像仓库中选择存在且适合的标签。

---

# 第三部分：Ubuntu 常见命令详解

## 1. 命令行基础

### 命令基本结构

```text
命令 [选项] [参数]
```

例如：

```bash
ls -lah /var/log
```

- `ls`：命令；
- `-lah`：选项；
- `/var/log`：参数。

### 查看帮助

```bash
<COMMAND> --help
man <COMMAND>
info <COMMAND>
```

在 `man` 中：

```text
q       退出
/word   搜索
n       下一个匹配
N       上一个匹配
```

### 常见快捷键

| 快捷键 | 作用 |
|---|---|
| `Tab` | 自动补全 |
| `↑ / ↓` | 浏览历史命令 |
| `Ctrl+C` | 终止当前前台程序 |
| `Ctrl+D` | 发送 EOF / 退出 shell |
| `Ctrl+L` | 清屏 |
| `Ctrl+A` | 移到行首 |
| `Ctrl+E` | 移到行尾 |
| `Ctrl+R` | 反向搜索历史命令 |
| `Ctrl+U` | 删除光标前内容 |
| `Ctrl+K` | 删除光标后内容 |
| `Ctrl+Z` | 暂停前台任务 |

### 命令历史

```bash
history
history | tail -n 30
history | grep docker
```

再次执行某条历史命令：

```bash
!123
```

执行前一条命令：

```bash
!!
```

使用 `sudo` 重跑上一条：

```bash
sudo !!
```

执行前应确认上一条命令是什么。

---

## 2. 路径和目录结构

### 路径符号

| 符号 | 含义 |
|---|---|
| `/` | 根目录 |
| `~` | 当前用户家目录 |
| `.` | 当前目录 |
| `..` | 上一级目录 |
| `-` | 上一次所在目录 |

### 服务器常见目录

| 路径 | 常见用途 |
|---|---|
| `/home/<USER>` | 普通用户文件 |
| `/root` | root 家目录 |
| `/etc` | 系统和服务配置 |
| `/var/log` | 日志 |
| `/var/lib` | 服务持久化状态 |
| `/var/www` | 网站静态文件 |
| `/srv` | 自己部署的服务和项目 |
| `/opt` | 第三方软件 |
| `/tmp` | 临时文件，可能被自动清理 |
| `/mnt` | 临时或额外挂载点 |
| `/usr/local/bin` | 本机安装的可执行命令 |
| `/etc/systemd/system` | 自定义 systemd 服务 |

项目建议：

```text
/srv/project-name        项目代码
/srv/project-name/.venv  Python 环境
/srv/project-name/data   数据
/srv/project-name/logs   应用日志
/srv/project-name/models 模型文件
```

---

## 3. 文件与目录

### 查看

```bash
ls
ls -l
ls -la
ls -lah
ls -lt
ls -lhS
```

- `-a`：包括隐藏文件；
- `-h`：人类可读大小；
- `-t`：按时间排序；
- `-S`：按文件大小排序。

### 创建

```bash
mkdir directory
mkdir -p parent/child
touch file.txt
```

### 复制与移动

```bash
cp source target
cp -r source_dir target_dir
cp -a source_dir target_dir

mv old_name new_name
mv file target_directory/
```

`cp -a` 尽量保留权限、时间戳和符号链接。

### 删除

```bash
rm file
rm -r directory
rm -rf directory
```

> ⚠️ `rm` 不经过回收站。尤其是 `sudo rm -rf`，必须先用 `pwd` 和 `ls` 核对目标路径。

更安全的两步：

```bash
TARGET=/srv/old-project
printf '%s\n' "$TARGET"
ls -lah "$TARGET"
rm -rf -- "$TARGET"
```

### 链接

软链接：

```bash
ln -s <TARGET> <LINK_NAME>
```

查看：

```bash
ls -l <LINK_NAME>
readlink -f <LINK_NAME>
```

Nginx 启用站点就是常见软链接场景。

---

## 4. 查看、搜索和处理文本

### 查看文件

```bash
cat file
less file
head file
head -n 20 file
tail file
tail -n 100 file
tail -f app.log
```

日志实时查看优先：

```bash
tail -f app.log
```

### grep

```bash
grep "error" app.log
grep -i "error" app.log
grep -n "error" app.log
grep -v "debug" app.log
grep -RIn "TODO" .
grep -E "error|warning" app.log
```

常用参数：

| 参数 | 作用 |
|---|---|
| `-i` | 忽略大小写 |
| `-n` | 显示行号 |
| `-R` | 递归目录 |
| `-v` | 反向选择 |
| `-E` | 扩展正则表达式 |
| `-C 3` | 显示前后各 3 行 |

### find

```bash
find . -name "*.py"
find . -type d -name "__pycache__"
find /var/log -type f -mtime -1
find /srv -type f -size +1G
```

### 管道与重定向

```bash
command1 | command2
```

示例：

```bash
ps aux | grep python
du -h --max-depth=1 /srv | sort -h
```

重定向：

```bash
command > output.txt
command >> output.txt
command 2> error.txt
command > all.log 2>&1
command 2>&1 | tee all.log
```

- `>`：覆盖；
- `>>`：追加；
- `2>`：只写标准错误；
- `2>&1`：把错误合并到正常输出；
- `tee`：终端显示的同时写文件。

---

## 5. 编辑文件

### nano

```bash
nano file.txt
sudo nano /etc/nginx/sites-available/myapp
```

常用：

```text
Ctrl+O   保存
Enter    确认文件名
Ctrl+X   退出
Ctrl+W   搜索
```

### vim 基础

```bash
vim file.txt
```

```text
i        进入插入模式
Esc      回到普通模式
:w       保存
:q       退出
:wq      保存并退出
:q!      不保存退出
/word    搜索
n        下一个匹配
dd       删除当前行
yy       复制当前行
p        粘贴
```

---

## 6. 用户、组与权限

### 用户信息

```bash
whoami
id
groups
who
w
last
```

### 创建和管理用户

```bash
sudo adduser <USER>
sudo usermod -aG sudo <USER>
sudo passwd <USER>
sudo deluser <USER>
```

添加到其他组：

```bash
sudo usermod -aG docker <USER>
```

`-aG` 中的 `a` 很重要，表示追加组；漏掉可能覆盖原有附加组。

### 权限表示

```text
r = 4  读
w = 2  写
x = 1  执行
```

常见：

```bash
chmod 644 file
chmod 600 secret.env
chmod 755 directory
chmod u+x script.sh
chmod -R 755 directory
```

含义：

```text
644 = 所有者读写，其他人只读
600 = 只有所有者读写
755 = 所有者读写执行，其他人读执行
```

修改所有者：

```bash
sudo chown <USER>:<GROUP> file
sudo chown -R <USER>:<GROUP> directory
```

### umask

```bash
umask
```

它决定新文件的默认权限掩码。通常不必随意修改全局 umask。

---

## 7. APT 软件管理

### 更新与升级

```bash
sudo apt update
sudo apt upgrade
```

区别：

- `apt update`：更新“有哪些软件版本”的索引；
- `apt upgrade`：升级已安装软件。

脚本中更适合使用：

```bash
sudo apt-get update
sudo apt-get install -y <PACKAGE>
```

### 安装、卸载、搜索

```bash
sudo apt install <PACKAGE>
sudo apt remove <PACKAGE>
sudo apt purge <PACKAGE>
sudo apt autoremove

apt search <KEYWORD>
apt show <PACKAGE>
apt policy <PACKAGE>
```

安装本地 `.deb`，优先让 APT 处理依赖：

```bash
sudo apt install ./package.deb
```

### 查看已安装软件

```bash
dpkg -l
dpkg -l | grep nginx
dpkg -L nginx
dpkg -S /usr/sbin/nginx
```

### 缓存清理

```bash
sudo apt clean
sudo apt autoclean
sudo apt autoremove
```

不要把清理命令当作固定日常任务，先确认磁盘问题来源。

---

## 8. 进程与任务控制

### 查看进程

```bash
ps aux
ps -ef
pgrep -af python
pstree -p
```

按资源排序：

```bash
ps aux --sort=-%cpu | head
ps aux --sort=-%mem | head
```

### 前台与后台

后台启动：

```bash
python app.py &
```

查看当前 shell 的任务：

```bash
jobs -l
```

把暂停任务转后台：

```bash
bg
```

转回前台：

```bash
fg
```

普通 `&` 在 SSH 断开后不一定可靠，长期任务优先 tmux 或 systemd。

### 停止进程

优先优雅结束：

```bash
kill -15 <PID>
```

仍不退出时才考虑：

```bash
kill -9 <PID>
```

按命令匹配：

```bash
pkill -f "python train.py"
```

> `kill -9` 不给程序保存 checkpoint、关闭文件或释放资源的机会，不应作为第一选择。

### 进程优先级

```bash
nice -n 10 command
renice 10 -p <PID>
```

数值越大，CPU 调度优先级通常越低。

---

## 9. systemd 服务

### 常用操作

```bash
systemctl status <SERVICE>
sudo systemctl start <SERVICE>
sudo systemctl stop <SERVICE>
sudo systemctl restart <SERVICE>
sudo systemctl reload <SERVICE>

sudo systemctl enable <SERVICE>
sudo systemctl disable <SERVICE>
sudo systemctl enable --now <SERVICE>
```

区别：

- `start`：本次启动；
- `enable`：设置开机启动；
- `enable --now`：设置开机启动并立即启动；
- `reload`：让程序重新加载配置，不一定中断连接；
- `restart`：完整重启，可能短暂中断。

修改 unit 文件后：

```bash
sudo systemctl daemon-reload
```

### 查看服务是否启用

```bash
systemctl is-active <SERVICE>
systemctl is-enabled <SERVICE>
systemctl list-units --type=service --state=running
systemctl list-unit-files --type=service
```

### 服务失败

```bash
systemctl status <SERVICE> --no-pager -l
journalctl -u <SERVICE> -n 200 --no-pager
```

---

## 10. 日志

### journalctl

```bash
journalctl
journalctl -b
journalctl -b -1

journalctl -u <SERVICE>
journalctl -u <SERVICE> -f
journalctl -u <SERVICE> -n 100
journalctl -u <SERVICE> --since today
journalctl -u <SERVICE> --since "1 hour ago"
journalctl -u <SERVICE> --since "2026-08-05 10:00"
```

只看错误级别：

```bash
journalctl -p err -b
```

内核日志：

```bash
journalctl -k
dmesg -T | tail -n 100
```

### 常见日志位置

| 服务 | 常见日志 |
|---|---|
| Nginx | `/var/log/nginx/access.log`、`error.log` |
| SSH | journal 或 `/var/log/auth.log` |
| APT/dpkg | `/var/log/apt/`、`/var/log/dpkg.log` |
| 自定义 systemd 服务 | `journalctl -u 服务名` |
| Docker | `docker logs` |

---

## 11. 网络、端口与 DNS

### 查看地址和路由

```bash
ip addr
ip route
hostname -I
```

### 连通性

```bash
ping -c 4 <HOST>
curl -I https://example.com
curl -v http://127.0.0.1:8000
wget <URL>
```

### DNS

```bash
dig <DOMAIN>
dig +short <DOMAIN>
nslookup <DOMAIN>
```

### 端口

```bash
ss -lntp
ss -lunp
sudo lsof -i :8000
sudo fuser -v 8000/tcp
```

参数：

```text
-l  监听
-n  不解析服务名
-t  TCP
-u  UDP
-p  显示进程
```

测试远程端口：

```bash
nc -zv <HOST> <PORT>
```

### 查看公网 IP

```bash
curl -4 ifconfig.me
```

依赖第三方服务，结果仅供参考。

---

## 12. UFW 防火墙

```bash
sudo ufw status
sudo ufw status verbose
sudo ufw status numbered

sudo ufw allow OpenSSH
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

sudo ufw deny 8000/tcp
sudo ufw delete allow 8000/tcp
```

只允许特定 IP 访问端口：

```bash
sudo ufw allow from <TRUSTED_IP> to any port <PORT> proto tcp
```

启用和禁用：

```bash
sudo ufw enable
sudo ufw disable
```

> 启用前必须先放行当前 SSH 端口。

---

## 13. SSH

### 连接

```bash
ssh <USER>@<SERVER_IP>
ssh -p <PORT> <USER>@<SERVER_IP>
ssh -i ~/.ssh/id_ed25519 <USER>@<SERVER_IP>
```

### 本地 SSH 配置

编辑本地：

```bash
nano ~/.ssh/config
```

示例：

```text
Host ai-server
    HostName <SERVER_IP>
    User <USER>
    Port 22
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

之后：

```bash
ssh ai-server
scp file ai-server:/srv/project/
```

### 端口转发

访问服务器仅监听本机的 Jupyter：

```bash
ssh -L 8888:127.0.0.1:8888 <USER>@<SERVER_IP>
```

本地浏览器访问：

```text
http://127.0.0.1:8888
```

访问服务器 API：

```bash
ssh -L 8000:127.0.0.1:8000 <USER>@<SERVER_IP>
```

### 保持连接

客户端命令：

```bash
ssh -o ServerAliveInterval=60 <USER>@<SERVER_IP>
```

更稳妥的长期任务仍应使用 tmux 或 systemd。

---

## 14. scp、rsync 与压缩

### scp

```bash
scp file <USER>@<SERVER_IP>:/remote/path/
scp <USER>@<SERVER_IP>:/remote/file ./
scp -r directory <USER>@<SERVER_IP>:/remote/path/
```

### rsync

推荐大文件、数据集和重复同步：

```bash
rsync -avhP source/ <USER>@<SERVER_IP>:/target/
```

参数：

```text
-a  归档模式
-v  显示过程
-h  人类可读
-P  进度 + 支持部分续传
-z  传输时压缩，局域网或已压缩文件未必更快
```

镜像同步：

```bash
rsync -avhP --delete source/ target/
```

> ⚠️ `--delete` 会删除目标端中源端不存在的文件。先用 `--dry-run`：

```bash
rsync -avhP --delete --dry-run source/ target/
```

### tar

打包压缩：

```bash
tar -czf archive.tar.gz directory/
```

解压：

```bash
tar -xzf archive.tar.gz
```

查看内容：

```bash
tar -tzf archive.tar.gz
```

### zip

```bash
zip -r archive.zip directory/
unzip archive.zip
unzip -l archive.zip
```

---

## 15. 磁盘与存储

### 查看空间

```bash
df -h
df -i
lsblk
findmnt
```

- `df -h`：磁盘容量；
- `df -i`：inode 是否耗尽；
- `lsblk`：磁盘和分区；
- `findmnt`：挂载关系。

### 查找大目录

```bash
sudo du -xhd1 / | sort -h
sudo du -xhd1 /var | sort -h
du -h --max-depth=1 /srv | sort -h
```

查找大文件：

```bash
sudo find / -xdev -type f -size +1G -printf '%s %p\n' 2>/dev/null \
  | sort -n \
  | tail
```

### 常见空间来源

- `/var/lib/docker`：镜像、容器、volume；
- `/var/log`：日志；
- 数据集和 checkpoint；
- pip/conda 缓存；
- 旧内核和 APT 缓存；
- 删除后仍被进程占用的文件。

查看被删除但仍占用空间的文件：

```bash
sudo lsof +L1
```

### Docker 空间

```bash
docker system df
docker image prune
docker container prune
docker builder prune
docker volume ls
```

一键清理未使用资源：

```bash
docker system prune
```

> ⚠️ `docker system prune -a --volumes` 可能删除未使用镜像和 volume，执行前必须确认。

---

## 16. CPU、内存与性能

### 常用工具

```bash
top
htop
free -h
vmstat 1
iostat -xz 1
```

安装扩展工具：

```bash
sudo apt install htop sysstat
```

### 判断瓶颈

```text
GPU 利用率低、CPU 很高
→ 数据加载或预处理可能成为瓶颈

GPU 利用率忽高忽低
→ DataLoader、磁盘读取、网络存储或 batch 太小

内存持续上涨
→ 内存泄漏、缓存过多或日志对象未释放

load average 高但 CPU 不满
→ 可能有 I/O 等待
```

---

## 17. 环境变量与 Shell 配置

临时设置：

```bash
export APP_ENV=production
export CUDA_VISIBLE_DEVICES=0
```

查看：

```bash
echo "$APP_ENV"
env
printenv APP_ENV
```

只对一个命令生效：

```bash
APP_ENV=production python app.py
```

用户 shell 配置：

```bash
nano ~/.bashrc
source ~/.bashrc
```

系统服务的环境变量更推荐放在：

```text
/etc/<PROJECT>.env
```

并使用 systemd 的 `EnvironmentFile=` 读取。

---

## 18. Git 项目操作

### 基础

```bash
git clone <URL>
git status
git add .
git commit -m "message"
git pull
git push
```

### 分支

```bash
git branch
git branch -a
git switch <BRANCH>
git switch -c <NEW_BRANCH>
```

### 查看历史和差异

```bash
git log --oneline --graph --decorate -n 30
git diff
git diff --staged
git show <COMMIT>
```

### 服务器更新代码

生产服务器不要直接修改大量源代码。常见流程：

```bash
cd /srv/<PROJECT>
git status
git pull --ff-only
source .venv/bin/activate
python -m pip install -r requirements.txt
sudo systemctl restart <SERVICE>
journalctl -u <SERVICE> -n 100 --no-pager
```

更新前记录当前提交：

```bash
git rev-parse HEAD
```

需要回滚时可切回旧提交或重新部署旧镜像。

---

## 19. tmux

### 会话

```bash
tmux new -s train
tmux ls
tmux attach -t train
tmux kill-session -t train
```

### 常用快捷键

先按 `Ctrl+b`，松开后再按：

| 按键 | 作用 |
|---|---|
| `d` | 离开会话 |
| `c` | 新建窗口 |
| `n` | 下一个窗口 |
| `p` | 上一个窗口 |
| `0-9` | 切换窗口 |
| `"` | 水平分屏 |
| `%` | 垂直分屏 |
| `[` | 进入滚动模式，按 `q` 退出 |

列出窗口：

```bash
tmux list-windows -t train
```

---

## 20. 定时任务

### cron

编辑当前用户任务：

```bash
crontab -e
```

查看：

```bash
crontab -l
```

格式：

```text
分 时 日 月 周 命令
```

示例：每天凌晨 2 点备份：

```cron
0 2 * * * /usr/local/bin/backup.sh >> /var/log/my-backup.log 2>&1
```

cron 环境变量很少，因此脚本中使用绝对路径：

```bash
#!/usr/bin/env bash
set -euo pipefail

/usr/bin/rsync ...
```

### systemd timer

查看定时器：

```bash
systemctl list-timers
```

对于需要日志、失败状态和依赖管理的任务，systemd timer 通常比 cron 更容易运维。

---

## 21. 数据库备份常用命令

### PostgreSQL

备份单个数据库：

```bash
pg_dump -Fc \
  -h 127.0.0.1 \
  -U <DB_USER> \
  -d <DB_NAME> \
  -f backup.dump
```

恢复：

```bash
pg_restore \
  -h 127.0.0.1 \
  -U <DB_USER> \
  -d <DB_NAME> \
  --clean \
  backup.dump
```

### MySQL / MariaDB

备份：

```bash
mysqldump \
  -u <DB_USER> \
  -p \
  --single-transaction \
  <DB_NAME> \
  > backup.sql
```

恢复：

```bash
mysql \
  -u <DB_USER> \
  -p \
  <DB_NAME> \
  < backup.sql
```

备份完成后应检查文件非空，并定期测试恢复流程。

---

## 22. 备份项目与模型

推荐至少备份：

```text
源代码或 Git 提交
配置文件
环境定义
模型 checkpoint
数据库
用户上传文件
Nginx 配置
systemd unit
密钥的安全副本
```

使用 rsync：

```bash
rsync -avhP \
  /srv/<PROJECT>/ \
  <BACKUP_USER>@<BACKUP_HOST>:/backups/<PROJECT>/
```

本机压缩：

```bash
tar -czf \
  <PROJECT>_$(date +%F).tar.gz \
  /srv/<PROJECT>
```

不要把备份只放在同一块云盘上。

---

# 第四部分：故障排查手册

## 1. SSH 连不上

按顺序检查：

### 本地

```bash
ssh -vvv <USER>@<SERVER_IP>
ping -c 4 <SERVER_IP>
nc -zv <SERVER_IP> 22
```

### 云平台

检查：

- 实例是否运行；
- 公网 IP 是否变化；
- 安全组是否允许 SSH 端口；
- 是否使用了正确用户名；
- 密钥是否匹配。

### 服务器控制台

```bash
sudo systemctl status ssh
sudo sshd -t
sudo ss -lntp | grep ssh
sudo ufw status
journalctl -u ssh -n 100 --no-pager
```

---

## 2. 服务启动失败

```bash
sudo systemctl status <SERVICE> --no-pager -l
journalctl -u <SERVICE> -n 200 --no-pager
```

检查：

- `ExecStart` 路径是否存在；
- `User` 是否有权限；
- `WorkingDirectory` 是否正确；
- 环境变量文件是否可读；
- 端口是否被占用；
- Python 环境依赖是否完整；
- 模型或配置文件路径是否正确。

手动用服务用户运行命令：

```bash
sudo -u <USER> \
  /srv/<PROJECT>/.venv/bin/python \
  -m app
```

---

## 3. 端口被占用

```bash
sudo ss -lntp | grep :8000
sudo lsof -i :8000
sudo fuser -v 8000/tcp
```

确认进程后，优先通过其服务管理器停止：

```bash
sudo systemctl stop <SERVICE>
docker stop <CONTAINER>
```

不要直接杀死不明进程。

---

## 4. Nginx 502 Bad Gateway

502 一般表示 Nginx 无法正常访问上游应用。

检查：

```bash
sudo nginx -t
sudo systemctl status nginx

curl -v http://127.0.0.1:8000/health
ss -lntp | grep 8000

systemctl status model-api
journalctl -u model-api -n 200 --no-pager

tail -n 100 /var/log/nginx/error.log
```

常见原因：

- 应用没有启动；
- `proxy_pass` 端口写错；
- 应用只监听了其他地址；
- 应用启动后立即崩溃；
- socket/文件权限错误；
- 模型加载时间过长；
- Nginx timeout 太短；
- 容器端口映射错误。

---

## 5. 域名不能访问

```bash
dig +short <DOMAIN>
curl -I http://<DOMAIN>
curl -I https://<DOMAIN>
```

检查：

- DNS A 记录是否正确；
- DNS 是否仍在传播；
- 云安全组是否放行 80/443；
- UFW 是否放行；
- Nginx 是否监听；
- `server_name` 是否正确；
- 域名是否经过代理/CDN。

---

## 6. Certbot 失败

```bash
sudo nginx -t
sudo certbot certificates
sudo certbot renew --dry-run
```

检查：

- 域名是否已解析到当前服务器；
- 80 端口是否公网可访问；
- Nginx 是否正常；
- 是否存在重复或冲突的 server block；
- 云安全组和 UFW；
- 是否短时间内反复申请触发限制。

---

## 7. 磁盘不足

```bash
df -h
df -i
sudo du -xhd1 / | sort -h
sudo du -xhd1 /var | sort -h
sudo lsof +L1
docker system df
```

处理前先识别来源，不要直接删除系统目录。

可能动作：

```bash
sudo journalctl --vacuum-time=14d
sudo apt clean
docker image prune
```

训练项目可检查：

```bash
find checkpoints -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort
```

保留 best/latest，清理策略应写入训练代码或运维脚本。

---

## 8. GPU 不可用

### 第一步：驱动

```bash
nvidia-smi
```

若命令不存在或报错，先处理驱动问题。

### 第二步：Python 环境

```bash
which python
python -m pip show torch
```

### 第三步：PyTorch

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

### 常见原因

- 安装了 CPU 版 PyTorch；
- NVIDIA 驱动不可用；
- 容器没有启用 GPU；
- 当前 shell 使用了错误的 Python 环境；
- `CUDA_VISIBLE_DEVICES` 隐藏了 GPU；
- 重启后驱动模块未正常加载；
- 服务器本身没有挂载 GPU。

---

## 9. CUDA Out of Memory

```bash
nvidia-smi
```

检查：

- batch size；
- 输入尺寸和序列长度；
- 是否保存了每个 batch 的 Tensor 而未 `detach()`；
- 验证时是否使用 `inference_mode()`；
- 是否重复加载模型；
- DataParallel/多 worker 是否复制模型；
- 其他进程是否占用显存。

程序中只在必要场景使用：

```python
torch.cuda.empty_cache()
```

它不会释放仍被 Tensor 引用的显存，也不能代替修复内存泄漏。

---

## 10. Permission denied

先查看：

```bash
ls -ld <PATH>
ls -l <FILE>
namei -l <PATH>
id
```

可能处理：

```bash
sudo chown -R <USER>:<GROUP> <PATH>
chmod u+x script.sh
chmod 600 secret.env
```

不要为了省事直接：

```bash
chmod -R 777 ...
```

这会造成不必要的安全风险。

---

## 11. APT 被锁定

查看占用：

```bash
ps aux | grep -E "apt|dpkg"
```

可能是自动更新正在运行。优先等待，不要直接删除 lock 文件。

查看状态：

```bash
systemctl status unattended-upgrades
journalctl -u unattended-upgrades -n 100 --no-pager
```

若安装被异常中断：

```bash
sudo dpkg --configure -a
sudo apt --fix-broken install
```

---

## 12. Docker 容器反复重启

```bash
docker ps -a
docker logs --tail 200 <CONTAINER>
docker inspect <CONTAINER>
docker compose config
docker compose ps
docker compose logs --tail 200
```

检查：

- 启动命令；
- 环境变量；
- volume 路径；
- 文件权限；
- 数据库是否就绪；
- healthcheck；
- 端口冲突；
- 内存不足；
- GPU runtime。

---

# 第五部分：安全与生产环境清单

## 上线前

- [ ] 使用普通 sudo 用户，而不是长期使用 root；
- [ ] SSH 密钥登录已验证；
- [ ] 修改 SSH 配置前保留一个现有连接；
- [ ] 云安全组只开放必要端口；
- [ ] UFW 只开放 SSH、80、443 等必要端口；
- [ ] 数据库、Redis、模型 API 不直接暴露公网；
- [ ] 应用只监听 `127.0.0.1`，由 Nginx 反向代理；
- [ ] HTTPS 已启用，续期测试成功；
- [ ] 密钥放在权限为 600 的环境变量文件中；
- [ ] 不把 `.env`、密钥和数据库密码提交到 Git；
- [ ] systemd 或 Compose 设置了合理重启策略；
- [ ] 服务日志可以查询；
- [ ] 健康检查接口可用；
- [ ] 数据库、模型和用户文件有异机备份；
- [ ] 已经实际测试过恢复；
- [ ] 更新和回滚流程明确；
- [ ] 磁盘和 GPU 监控可用。

## 深度学习训练前

- [ ] `nvidia-smi` 正常；
- [ ] `torch.cuda.is_available()` 为 True；
- [ ] Python 环境和依赖已固定；
- [ ] 数据路径正确；
- [ ] 小规模实验能完成；
- [ ] checkpoint 定期保存；
- [ ] best/latest 分开保存；
- [ ] 日志带时间戳；
- [ ] 使用 tmux 或任务调度器；
- [ ] 磁盘空间足够；
- [ ] 训练命令和代码版本已记录。

## 更新服务前

```bash
git rev-parse HEAD
systemctl status <SERVICE>
curl http://127.0.0.1:<PORT>/health
```

更新后：

```bash
sudo systemctl restart <SERVICE>
journalctl -u <SERVICE> -n 100 --no-pager
curl http://127.0.0.1:<PORT>/health
sudo nginx -t
```

---

# 第六部分：建议安装的基础工具

```bash
sudo apt update

sudo apt install \
  curl \
  wget \
  git \
  vim \
  nano \
  htop \
  tmux \
  tree \
  unzip \
  zip \
  rsync \
  jq \
  ca-certificates \
  build-essential \
  python3 \
  python3-pip \
  python3-venv
```

网络排查工具：

```bash
sudo apt install \
  dnsutils \
  netcat-openbsd \
  lsof \
  traceroute
```

性能工具：

```bash
sudo apt install sysstat
```

工具作用：

| 工具 | 用途 |
|---|---|
| `curl` | HTTP/API 测试 |
| `wget` | 下载文件 |
| `git` | 代码版本管理 |
| `tmux` | 持久终端、后台训练 |
| `htop` | CPU 和内存监控 |
| `tree` | 查看目录树 |
| `rsync` | 增量同步和备份 |
| `jq` | 处理 JSON |
| `lsof` | 查询文件和端口占用 |
| `dig` | DNS 查询 |
| `iostat` | 磁盘 I/O 监控 |

---

# 第七部分：一页式操作流程

## 新服务器

```bash
ssh root@<SERVER_IP>

apt update
apt upgrade

adduser <USER>
usermod -aG sudo <USER>

ufw allow OpenSSH
ufw enable

# 配好并测试 SSH 密钥后，再关闭 root/密码登录
```

## 深度学习训练

```bash
nvidia-smi

git clone <REPO_URL>
cd <PROJECT>

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

tmux new -s train
python -u train.py 2>&1 | tee logs/train.log
```

## 模型 API

```bash
cd /srv/model-api
source .venv/bin/activate

uvicorn app.main:app --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health

sudo systemctl enable --now model-api
journalctl -u model-api -f
```

## Nginx

```bash
sudo nginx -t
sudo systemctl reload nginx

curl -I http://<DOMAIN>
```

## HTTPS

```bash
sudo certbot --nginx
sudo certbot renew --dry-run
```

## Docker Compose

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f
```

---

# 官方资料与更新入口

安装步骤和版本支持会变化，以下官方资料应作为最终依据：

- Ubuntu Server 文档  
  https://ubuntu.com/server/docs/

- Ubuntu OpenSSH Server  
  https://ubuntu.com/server/docs/how-to/security/openssh-server/

- Ubuntu APT 软件管理  
  https://ubuntu.com/server/docs/how-to/software/package-management/

- Ubuntu UFW 防火墙  
  https://documentation.ubuntu.com/server/how-to/security/firewalls/

- Docker Engine on Ubuntu  
  https://docs.docker.com/engine/install/ubuntu/

- Docker Compose 生产部署  
  https://docs.docker.com/compose/how-tos/production/

- NVIDIA Container Toolkit  
  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

- PyTorch 安装选择器  
  https://pytorch.org/get-started/locally/

- Nginx 官方文档  
  https://docs.nginx.com/nginx/admin-guide/web-server/

- Certbot 官方安装指引  
  https://certbot.eff.org/instructions

- systemd / journalctl 手册  
  https://www.freedesktop.org/software/systemd/man/latest/journalctl.html

---

# 最后总结

日常管理 Ubuntu 云服务器，最应该形成的操作习惯：

1. 执行修改命令前，先用只读命令确认当前状态；
2. 删除文件前检查 `pwd`、目标变量和 `ls`；
3. 长时间训练使用 tmux，并记录日志和 checkpoint；
4. 遇到服务问题先看 `systemctl status` 和 `journalctl`；
5. 遇到网络问题按“应用 → 本机端口 → Nginx → UFW → 云安全组 → DNS”逐层检查；
6. 深度学习问题优先检查 GPU、Python 环境、PyTorch 版本和 Tensor device；
7. 模型 API 不直接暴露公网，使用 Nginx 反向代理；
8. 密钥不进 Git，数据库和内部服务不开放公网；
9. 部署前准备健康检查、备份、恢复和回滚流程；
10. 版本敏感的安装命令始终重新核对官方文档。
