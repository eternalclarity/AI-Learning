# Git 日常操作完整流程速查笔记





# 一、 Git 的核心工作流程

<img src="D:\code.py\workspace\AI-Learning\week01_python_review\notes\屏幕截图 2026-08-18 100525.jpg" alt="屏幕截图 2026-08-18 100525" style="zoom:80%;" />

Git 日常操作最核心的流程只有这一条：

```
远程仓库 GitHub
      ↓ clone / pull
本地仓库
      ↓
工作区 Working Directory
      ↓ git add
暂存区 Staging Area
      ↓ git commit
本地仓库 Local Repository
      ↓ git push
远程仓库 Remote Repository
```

最常用的几个命令：

```
git status
git add .
git commit -m "xxx"
git pull
git push
```

可以直接记成：

```
修改代码
  ↓
git status
  ↓
git add .
  ↓
git commit -m "说明"
  ↓
git pull --rebase
  ↓
git push
```

# 二、第一次使用 Git

## 1. 查看 Git 是否安装

```
git --version
```

例如：

```
git version 2.46.0
```

## 2. 配置用户名

```
git config --global user.name "你的名字"
```

例如：

```
git config --global user.name "Hongming Zhao"
```

## 3. 配置邮箱

```
git config --global user.email "你的邮箱"
```

例如：

```
git config --global user.email "example@gmail.com"
```

GitHub 提交记录会根据邮箱关联你的 GitHub 账号。

## 4. 查看当前配置

```
git config --global --list
```

或者：

```
git config --global user.name
git config --global user.email
```

# 三、创建 Git 项目的两种情况

# 情况 A：本地已经有一个项目

比如：

```
C:\workspace\my_project
```

进入项目：

```
cd my_project
```

初始化 Git：

```
git init
```

这时候项目里会生成：

```
.git/
```

`.git` 保存 Git 的版本历史。

然后：

```
git add .
git commit -m "Initial commit"
```

# 情况 B：GitHub 上已经有项目

这种情况一般直接：

```
git clone 仓库地址
```

例如：

```
git clone https://github.com/username/project.git
```

然后：

```
cd project
```

项目默认已经是 Git 仓库，不需要：

```
git init
```

# 四、每天最常用的 Git 工作流程

这是最重要的一部分。

假设今天开始继续开发项目。

## Step 1：进入项目

```
cd 项目路径
```

例如：

```
cd ~/projects/my_project
```

Windows：

```
cd D:\code\my_project
```

## Step 2：查看当前分支

```
git branch
```

例如：

```
* main
```

`*` 表示当前所在分支。

更推荐：

```
git status
```

因为它同时告诉你：

- 当前分支；
- 修改了哪些文件；
- 哪些文件已暂存；
- 哪些文件未跟踪。

## Step 3：开始工作前同步远程代码

```
git pull
```

更推荐：

```
git pull --rebase
```

作用：

```
GitHub 最新代码
       ↓
更新到本地
```

`--rebase` 可以减少不必要的 merge commit，让提交历史更加整洁。

# 五、修改代码之后

例如今天修改了：

```
train.py
model.py
README.md
```

先查看：

```
git status
```

可能看到：

```
modified: train.py
modified: model.py
modified: README.md
```

# 六、查看自己到底修改了什么

非常推荐养成这个习惯。

```
git diff
```

表示：

> 查看工作区中还没有 `git add` 的修改。

如果已经 `git add`：

```
git diff --cached
```

查看：

> 已经进入暂存区、准备 commit 的修改。

# 七、git add：加入暂存区

## 添加一个文件

```
git add train.py
```

## 添加多个文件

```
git add train.py model.py
```

## 添加当前目录全部修改

最常用：

```
git add .
```

## 添加所有修改

```
git add -A
```

日常开发中：

```
git add .
```

基本已经够用了。

# 八、再次检查

```
git status
```

如果看到：

```
Changes to be committed:
```

说明已经成功进入暂存区。

# 九、git commit：保存一个版本

```
git commit -m "提交说明"
```

例如：

```
git commit -m "Add ResNet training pipeline"
```

或者：

```
git commit -m "Fix data loading bug"
```

# 十、Commit 信息怎么写

建议：

```
动作 + 修改内容
```

例如：

```
git commit -m "Add CIFAR-10 dataset loader"
git commit -m "Fix training accuracy calculation"
git commit -m "Update README"
git commit -m "Refactor ResNet model"
git commit -m "Remove unused code"
```

常见前缀：

```
feat:     新功能
fix:      修复 Bug
docs:     文档修改
refactor: 代码重构
test:     测试代码
chore:    杂项
style:    格式修改
```

例如：

```
git commit -m "feat: add ResNet model"
git commit -m "fix: correct validation accuracy"
git commit -m "docs: update training instructions"
```

对于自己的学习项目，不必过度纠结规范，最重要的是：

> **让未来的自己看到 commit 就知道当时改了什么。**

# 十一、Push 前同步远程仓库

如果只有你一个人开发，可能直接：

```
git push
```

就行。

但是养成习惯更推荐：

```
git pull --rebase
```

然后：

```
git push
```

完整流程：

```
git add .
git commit -m "feat: add training pipeline"

git pull --rebase

git push
```

# 十二、git push：上传 GitHub

第一次推送：

```
git push -u origin main
```

以后：

```
git push
```

即可。

其中：

```
origin
```

通常代表远程 GitHub 仓库。

```
main
```

代表分支。

# 十三、最标准的每日 Git 流程

以后可以直接照这个执行：

```
# 1. 进入项目
cd my_project

# 2. 查看状态
git status

# 3. 获取远程最新代码
git pull --rebase

# ---------------------
# 开始写代码
# ---------------------

# 4. 查看修改
git status
git diff

# 5. 添加修改
git add .

# 6. 再次检查
git status

# 7. 提交
git commit -m "feat: add xxx"

# 8. 再同步一次远程代码
git pull --rebase

# 9. 推送
git push
```

整个过程可以概括成：

```
pull
 ↓
修改代码
 ↓
status
 ↓
diff
 ↓
add
 ↓
commit
 ↓
pull --rebase
 ↓
push
```

# 十四、查看 Git 提交历史

最简单：

```
git log
```

但内容比较多。

推荐：

```
git log --oneline
```

例如：

```
9c14f83 feat: add ResNet model
62d18ac fix: correct training loop
fa76c10 docs: update README
```

非常实用。

## 更漂亮的提交树

```
git log --oneline --graph --decorate --all
```

例如：

```
* 39bc2d1 (HEAD -> main) Add evaluation
* 29ab313 Add training
* 13bd21a Initial commit
```

# 十五、查看某个 commit 改了什么

```
git show commit编号
```

例如：

```
git show 9c14f83
```

也可以：

```
git show HEAD
```

查看最近一次提交。

# 十六、HEAD 是什么

Git 中：

```
HEAD
```

表示：

> 当前所在的 commit。

例如：

```
A → B → C
        ↑
       HEAD
```

那么：

```
HEAD
```

就是 C。

```
HEAD~1
```

就是 B。

```
HEAD~2
```

就是 A。

# 十七、撤销还没有 git add 的修改

假设你把：

```
train.py
```

改乱了，希望恢复到上一次 commit。

执行：

```
git restore train.py
```

全部恢复：

```
git restore .
```

注意：

> 修改会被直接删除。

所以使用前确认这些代码真的不要了。

# 十八、撤销 git add

假如：

```
git add train.py
```

之后突然发现：

> 我现在还不想 commit 它。

执行：

```
git restore --staged train.py
```

全部取消暂存：

```
git restore --staged .
```

代码不会丢。

只是：

```
暂存区
 ↓
重新放回工作区
```

# 十九、修改最近一次 commit

假设刚刚执行：

```
git commit -m "Add model"
```

结果忘记添加一个文件。

可以：

```
git add README.md
git commit --amend
```

或者直接修改 commit 信息：

```
git commit --amend -m "feat: add ResNet model"
```

注意：

如果 commit 已经 `push` 到公共仓库，要谨慎 amend，因为它会改变 commit ID。

# 二十、撤销 commit，但保留代码

假设：

```
A → B → C
```

刚提交 C，但现在觉得不应该提交。

执行：

```
git reset --soft HEAD~1
```

变成：

```
A → B
```

但是 C 的代码修改还保留在暂存区。

非常适合：

> commit 提交错了，想重新整理后再提交。

# 二十一、撤销 commit，并取消 git add

```
git reset HEAD~1
```

或者：

```
git reset --mixed HEAD~1
```

效果：

```
commit 删除
+
代码保留
+
回到工作区
```

# 二十二、彻底回退 commit

```
git reset --hard HEAD~1
```

意味着：

```
commit 删除
+
代码修改也删除
```

这是危险操作。

所以：

```
git reset --hard
```

一定谨慎。

# 二十三、已经 push 的 commit 怎么撤销

如果已经 push 到 GitHub，团队项目中更推荐：

```
git revert commit编号
```

例如：

```
git revert 9c14f83
```

它不会删除历史，而是创建一个新的 commit：

```
A → B → C → Revert C
```

因此：

```
reset
```

更多用于修改**本地历史**。

```
revert
```

更适合撤销**已经公开的历史**。

# 二十四、reset 和 revert 的区别

可以记：

```
git reset
=
修改历史
```

而：

```
git revert
=
不修改旧历史
而是新增一个反向 commit
```

所以：

```
没 push：
git reset
已经 push：
git revert
```

是非常实用的经验规则。

# 二十五、Git 分支

大型项目不要所有东西都直接在：

```
main
```

上开发。

可以创建：

```
main
│
├── feature/resnet
├── feature/detection
└── fix/dataloader
```

# 二十六、查看所有本地分支

```
git branch
```

例如：

```
* main
  feature/resnet
  feature/detection
```

# 二十七、创建新分支

传统写法：

```
git checkout -b feature/resnet
```

现在更推荐：

```
git switch -c feature/resnet
```

相当于：

```
创建分支
+
切换过去
```

# 二十八、切换分支

```
git switch main
```

旧写法：

```
git checkout main
```

# 二十九、典型分支开发流程

例如开发一个目标检测模块：

```
git switch main

git pull --rebase

git switch -c feature/detection
```

然后开发代码。

完成后：

```
git add .

git commit -m "feat: add object detection module"

git push -u origin feature/detection
```

GitHub 上创建 Pull Request。

最后合并进：

```
main
```

# 三十、合并分支

假设：

```
feature/resnet
```

已经完成。

先切换：

```
git switch main
```

更新 main：

```
git pull
```

合并：

```
git merge feature/resnet
```

然后：

```
git push
```

# 三十一、删除已经完成的分支

本地：

```
git branch -d feature/resnet
```

如果 Git 不允许删除：

```
git branch -D feature/resnet
```

`-D` 是强制删除，要谨慎。

远程：

```
git push origin --delete feature/resnet
```

# 三十二、查看远程分支

```
git branch -r
```

查看所有：

```
git branch -a
```

# 三十三、查看远程仓库

```
git remote -v
```

例如：

```
origin  https://github.com/user/project.git (fetch)
origin  https://github.com/user/project.git (push)
```

# 三十四、添加远程仓库

如果项目是：

```
git init
```

创建的，需要关联 GitHub。

```
git remote add origin 仓库地址
```

例如：

```
git remote add origin https://github.com/user/project.git
```

然后：

```
git branch -M main
git push -u origin main
```

# 三十五、修改远程仓库地址

```
git remote set-url origin 新地址
```

例如：

```
git remote set-url origin git@github.com:user/project.git
```

# 三十六、删除远程仓库

```
git remote remove origin
```

只会解除关联，不会把 GitHub 仓库删除。

# 三十七、git fetch 和 git pull

这是很容易混淆的一组。

## git fetch

```
git fetch
```

作用：

```
下载远程最新信息
```

但是：

> 不修改你当前代码。

## git pull

```
git pull
```

基本可以理解为：

```
git fetch
+
git merge
```

也就是：

```
获取远程代码
+
合并到当前分支
```

# 三十八、为什么推荐 git pull --rebase

普通：

```
git pull
```

可能产生：

```
Merge branch 'main'...
```

这种额外提交。

而：

```
git pull --rebase
```

会让历史更加接近：

```
A → B → C → D
```

而不是：

```
      C
     / \
A → B   M
     \ /
      D
```

个人项目以及简单协作项目中：

```
git pull --rebase
```

通常很好用。

# 三十九、冲突 Conflict

假设你修改了：

```
train.py
```

别人也修改了：

```
train.py
```

Git 无法自动判断应该保留谁，就会产生：

```
CONFLICT
```

文件里可能出现：

```
<<<<<<< HEAD

你的代码

=======

远程代码

>>>>>>> origin/main
```

你需要人工决定最终代码。

例如修改成：

```
最终正确代码
```

然后删除：

```
<<<<<<<
=======
>>>>>>>
```

# 四十、解决 merge 冲突

修改完冲突文件后：

```
git add .
```

然后：

```
git commit
```

如果是普通 merge。

# 四十一、解决 rebase 冲突

如果执行：

```
git pull --rebase
```

产生冲突。

先修改冲突。

然后：

```
git add .
```

继续：

```
git rebase --continue
```

如果想放弃：

```
git rebase --abort
```

恢复到 rebase 之前。

# 四十二、git stash：临时保存修改

这是非常实用的命令。

例如你正在写代码：

```
train.py 修改了一半
```

突然需要：

```
git pull
```

或者切换分支。

但是现在又不想 commit。

执行：

```
git stash
```

Git 会临时把修改收起来。

现在工作区变干净。

然后：

```
git pull
```

完成后恢复：

```
git stash pop
```

# 四十三、查看 stash

```
git stash list
```

例如：

```
stash@{0}: WIP on main
stash@{1}: WIP on feature/resnet
```

# 四十四、恢复 stash

恢复并删除：

```
git stash pop
```

只恢复不删除：

```
git stash apply
```

# 四十五、带说明保存 stash

推荐：

```
git stash push -m "ResNet training unfinished"
```

之后：

```
git stash list
```

就比较容易看懂。

# 四十六、删除 stash

删除一个：

```
git stash drop stash@{0}
```

全部删除：

```
git stash clear
```

# 四十七、.gitignore

做 Python / PyTorch 项目时非常重要。

有很多东西一般不应该提交，例如：

```
__pycache__/
.idea/
.vscode/
.env
*.pyc
checkpoints/
datasets/
wandb/
```

在项目根目录创建：

```
.gitignore
```

例如：

```
# Python
__pycache__/
*.py[cod]

# Virtual environment
venv/
.venv/

# Conda
*.conda

# IDE
.idea/
.vscode/

# Environment variables
.env

# Jupyter
.ipynb_checkpoints/

# PyTorch checkpoints
*.pth
*.pt
*.ckpt

# Logs
logs/
*.log

# Datasets
data/
datasets/

# Weights
weights/

# Weights & Biases
wandb/

# OS
.DS_Store
Thumbs.db
```

# 四十八、已经 git add 的文件加入 .gitignore 为什么没用

因为：

> `.gitignore` 只忽略尚未被 Git 跟踪的文件。

假设：

```
data/
```

之前已经提交。

后来才写：

```
data/
```

Git 还是会继续跟踪。

需要：

```
git rm -r --cached data/
```

然后：

```
git add .
git commit -m "chore: stop tracking dataset"
```

注意：

```
--cached
```

表示只从 Git 中删除，不删除电脑里的文件。

# 四十九、查看 Git 跟踪了哪些文件

```
git ls-files
```

# 五十、删除文件

直接：

```
rm file.py
```

然后：

```
git add .
git commit -m "Remove unused file"
```

也可以：

```
git rm file.py
```

然后：

```
git commit -m "Remove unused file"
```

# 五十一、重命名文件

```
git mv old.py new.py
```

然后：

```
git commit -m "Rename old.py to new.py"
```

实际上你直接在 IDE 里重命名也可以，Git 通常能够识别。

# 五十二、查看某个文件历史

```
git log -- train.py
```

查看改动：

```
git log -p -- train.py
```

# 五十三、查是谁修改了某一行

```
git blame train.py
```

团队开发时比较有用。

# 五十四、恢复被误删的代码——reflog

这是 Git 的“后悔药”。

如果不小心：

```
git reset --hard
```

然后发现：

> 完了，代码没了。

可以：

```
git reflog
```

例如：

```
e912a10 HEAD@{0}: reset: moving to HEAD~1
fe123ab HEAD@{1}: commit: add ResNet model
```

找到：

```
fe123ab
```

然后：

```
git reset --hard fe123ab
```

很多情况下能把代码找回来。

所以：

> **误操作以后先不要慌，先** `**git reflog**`**。**

# 五十五、标签 Tag

例如项目发布：

```
v1.0
```

创建标签：

```
git tag v1.0
```

查看：

```
git tag
```

推送：

```
git push origin v1.0
```

推送所有标签：

```
git push origin --tags
```

删除本地：

```
git tag -d v1.0
```

删除远程：

```
git push origin --delete v1.0
```

# 五十六、比较两个 commit

```
git diff commit1 commit2
```

例如：

```
git diff a381cc2 b2719ef
```

# 五十七、比较两个分支

```
git diff main feature/resnet
```

# 五十八、只提交部分修改

假设：

```
train.py
```

里面同时修改了两个功能，但是你只想 commit 一个。

可以：

```
git add -p
```

Git 会一块一块询问：

```
Stage this hunk?
```

常见：

```
y = 添加
n = 不添加
q = 退出
```

这是比较高级但非常好用的功能。

# 五十九、Cherry-pick

假设：

```
feature-A
```

里面有一个 commit：

```
abc123
```

你想把这个 commit 单独复制到：

```
main
```

执行：

```
git switch main
git cherry-pick abc123
```

它相当于：

```
只把某一个 commit 拿过来
```

# 六十、GitHub 项目的推荐分支流程

如果自己做学习项目：

```
main
```

直接开发其实完全可以。

推荐：

```
修改代码
 ↓
git add .
 ↓
git commit
 ↓
git pull --rebase
 ↓
git push
```

如果项目逐渐复杂：

```
main
│
├── feature/classification
├── feature/detection
├── feature/segmentation
└── fix/training-loop
```

流程：

```
main
 ↓
创建 feature 分支
 ↓
开发
 ↓
commit
 ↓
push
 ↓
Pull Request
 ↓
merge main
 ↓
删除 feature 分支
```

# 六十一、一个实际 PyTorch 项目的完整 Git 示例

假设项目：

```
deep_learning_project/
├── configs/
├── data/
├── datasets/
├── models/
├── checkpoints/
├── train.py
├── evaluate.py
├── requirements.txt
├── README.md
└── .gitignore
```

每天开始：

```
cd deep_learning_project

git status

git pull --rebase
```

开发代码。

例如修改：

```
models/resnet.py
train.py
```

查看：

```
git status

git diff
```

加入：

```
git add models/resnet.py train.py
```

检查：

```
git status
```

提交：

```
git commit -m "feat: add ResNet training support"
```

同步：

```
git pull --rebase
```

推送：

```
git push
```

完成。

# 六十二、一个功能最好一个 commit

不要写一天代码最后：

```
git add .
git commit -m "update"
```

里面同时包含：

```
增加模型
修 Bug
改 README
重构 Dataset
改训练流程
```

更推荐：

```
git add models/
git commit -m "feat: add ResNet model"

git add datasets/
git commit -m "refactor: simplify dataset loader"

git add train.py
git commit -m "fix: correct validation loop"

git add README.md
git commit -m "docs: update training guide"
```

这样以后特别容易：

```
查 Bug
回退
Cherry-pick
查看历史
```

# 六十三、不建议提交 GitHub 的东西

机器学习项目尤其注意。

一般不要提交：

```
数据集
模型大权重
API Key
.env
密码
数据库密码
服务器密码
SSH 私钥
缓存
临时文件
IDE 配置
训练生成的大量日志
```

尤其：

```
OPENAI_API_KEY
GITHUB_TOKEN
HF_TOKEN
```

绝对不要提交 Git。

# 六十四、如果 API Key 不小心 commit 了

仅仅：

```
git rm
```

是不够的。

因为历史 commit 里仍然存在。

第一件事情应该是：

```
立即撤销 / Rotate / 删除这个 API Key
```

重新生成。

不要继续使用已经泄露的 Key。

# 六十五、大模型权重怎么办

例如：

```
model.pth    2GB
```

不应该直接 Git 提交。

可以：

```
*.pth
*.pt
*.ckpt
```

大型文件可以考虑：

```
Git LFS
Hugging Face Hub
云存储
对象存储
```

# 六十六、常见错误：push 被拒绝

例如：

```
! [rejected] main -> main
```

通常因为：

> GitHub 有新的 commit，而你的本地没有。

执行：

```
git pull --rebase
```

如果没有冲突：

```
git push
```

即可。

# 六十七、常见错误：本地修改导致无法 pull

例如：

```
Your local changes would be overwritten by merge
```

说明你有未提交修改。

方案一：提交。

```
git add .
git commit -m "WIP: save current changes"

git pull --rebase
```

方案二：暂存。

```
git stash

git pull --rebase

git stash pop
```

# 六十八、常见错误：detached HEAD

如果看到：

```
HEAD detached at abc123
```

说明你现在直接停在某个 commit 上，而不是正常分支。

如果只是看看：

```
git switch main
```

即可。

如果在这个状态下写了重要代码：

```
git switch -c recovery
```

先创建一个分支保存它。

# 六十九、强制推送

存在：

```
git push --force
```

但危险。

如果确实需要强制推送，更推荐：

```
git push --force-with-lease
```

因为它比：

```
--force
```

安全。

日常开发：

> **能不用 force 就不要用。**

# 七十、Git 常见区域关系

一定理解：

```
工作区
Working Directory
      │
      │ git add
      ↓
暂存区
Staging Area
      │
      │ git commit
      ↓
本地仓库
Local Repository
      │
      │ git push
      ↓
远程仓库
GitHub
```

反方向：

```
GitHub
   │
   │ git fetch / git pull
   ↓
本地
```

因此：

```
git add
```

不是上传 GitHub。

```
git commit
```

也不是上传 GitHub。

真正上传 GitHub：

```
git push
```

# 七十一、几个核心命令的本质

```
git status
```

→ 我现在项目是什么状态？

```
git diff
```

→ 我到底修改了什么？

```
git add
```

→ 哪些修改准备提交？

```
git commit
```

→ 在本地保存一个版本。

```
git pull
```

→ 把 GitHub 最新版本拉下来。

```
git push
```

→ 把我的 commit 上传 GitHub。

```
git branch
```

→ 管理不同开发路线。

```
git merge
```

→ 把两条开发路线合起来。

```
git stash
```

→ 临时保存未完成代码。

```
git restore
```

→ 撤销工作区修改。

```
git reset
```

→ 回退本地历史。

```
git revert
```

→ 创建一个新 commit 抵消旧 commit。

# 七十二、最值得记住的 20 个 Git 命令

如果以后全部忘了，优先查这里：

```
# 查看状态
git status

# 查看修改
git diff

# 下载项目
git clone URL

# 获取最新代码
git pull --rebase

# 添加全部修改
git add .

# 提交
git commit -m "说明"

# 推送
git push

# 查看提交记录
git log --oneline

# 查看分支
git branch

# 创建并切换分支
git switch -c feature/test

# 切换分支
git switch main

# 合并分支
git merge feature/test

# 临时保存
git stash

# 恢复临时修改
git stash pop

# 撤销文件修改
git restore file.py

# 撤销 add
git restore --staged file.py

# 回退最近 commit，保留代码
git reset --soft HEAD~1

# 安全撤销已发布 commit
git revert commit_id

# 查看远程仓库
git remote -v

# Git 后悔药
git reflog
```

# 七十三、我推荐你真正养成的 Git 使用习惯

对于日常 Python / PyTorch 项目，可以固定使用这一套：

```
# ==============================
# 开始今天的开发
# ==============================

git status
git pull --rebase


# ==============================
# 写代码……
# ==============================


# ==============================
# 准备提交
# ==============================

git status
git diff

git add .

git status

git commit -m "feat: xxx"

git pull --rebase

git push
```

如果一个功能比较大：

```
git switch main
git pull --rebase

git switch -c feature/xxx

# 开发……

git add .
git commit -m "feat: add xxx"

git push -u origin feature/xxx
```

# 七十四、最终记忆口诀

Git 日常使用真正需要形成肌肉记忆的是：

```
开始工作：

pull
写完代码：

status
↓
diff
↓
add
↓
commit
↓
pull --rebase
↓
push
```

出现问题：

```
代码改错
→ restore

add 错了
→ restore --staged

commit 错了但没 push
→ reset

commit 已经 push
→ revert

代码写一半要切分支
→ stash

代码好像被我弄没了
→ reflog

远程代码比我新
→ pull --rebase

出现冲突
→ 手动修改
→ add
→ rebase --continue
```

对于绝大多数日常开发工作，只要真正掌握：

```
git status
git diff
git add
git commit
git pull --rebase
git push
git branch
git switch
git merge
git stash
git restore
git reset
git revert
git reflog
```

基本就已经可以非常顺畅地使用 Git 进行个人项目和团队开发了。