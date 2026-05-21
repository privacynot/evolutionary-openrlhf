# 快速开始

## 安装依赖
```bash
# 克隆仓库
git clone https://github.com/你的用户名/evolutionary-openrlhf.git
cd evolutionary-openrlhf

# 创建虚拟环境（推荐）
python -m venv venv
# Windows 激活:
venv\Scripts\activate
# Linux/MacOS 激活:
source venv/bin/activate

# 安装项目及主要依赖
pip install -r requirements.txt
pip install -e .
# 安装 OpenRLHF（从 GitHub 源码）
pip install git+https://github.com/OpenRLHF/OpenRLHF.git
