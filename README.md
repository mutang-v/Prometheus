# Project Prometheus — v0.1

一个最小可运行的 AI 研究代理脚手架。它先验证研究闭环，不追求“自主进化”的宏大能力。

## 设计原则
- 默认只使用 Python 标准库，无需 `pip install`。
- 通过环境变量配置 OpenAI-compatible API；不假设你的供应商、URL 或模型名。
- API 调用需要你明确填写配置。
- 第一版只运行内置的安全实验，不执行模型生成的任意代码。
- 每轮结果写入 `runs/`，便于复现和比较。
- 设有轮数限制；不自动安装软件、不修改系统配置、不自行购买服务。

## 运行环境
Windows 10/11 + Python 3.10+（建议 3.11 或更新版本）。

## 第一步：检查 Python
在项目目录打开 PowerShell：
```powershell
python --version
python main.py --check
```

## 第二步：配置 API（仅在确认接口信息后）
复制 `config.example.json` 为 `config.json`，填入：
- `base_url`: 服务商给出的 OpenAI-compatible API 根地址，通常包含 `/v1`
- `model`: 服务商要求的模型 ID
- `api_key_env`: 保存密钥的环境变量名称（不要把密钥直接写进配置文件）
- `timeout_seconds`: 超时秒数

在 PowerShell 中为当前终端设置密钥：
```powershell
$env:PROMETHEUS_API_KEY="你的API密钥"
```
不要把密钥发给聊天窗口，也不要提交到 Git。

如果供应商并非 OpenAI-compatible 接口，先不要猜 URL 或改代码；确认服务商文档后再适配。

## 第三步：验证本地闭环（不调用 API、不花钱）
```powershell
python main.py --demo
```
这会运行一个内置的算法参数搜索实验，输出基准结果与改进结果，并把记录保存到 `runs/`。

## 第四步：测试 API 连通性
```powershell
python main.py --check-api
```
这会发送一次短请求，可能产生少量 API 费用。运行前先确认模型价格和账户额度。

## 第五步：运行一轮研究任务
```powershell
python main.py --goal "在给定测试预算下，寻找更低误差的参数组合" --rounds 1
```
v0.1 只允许模型输出 JSON 格式的候选参数；实际评分由本地固定的评估器完成。模型不能向执行器提交代码。

## 文件说明
- `main.py`：命令行入口、实验与记录。
- `prometheus/api_client.py`：OpenAI-compatible API 客户端（标准库实现）。
- `prometheus/experiment.py`：内置、可重复的安全实验与评分。
- `prometheus/storage.py`：运行记录。
- `config.example.json`：配置模板。
- `runs/`：运行后自动创建，保存 JSON 结果。

## v0.1 的限制
- 这是研究代理原型，不是通用自主智能，也没有意识。
- 只能做内置参数优化实验，不能自动修改自身程序或训练模型。
- 结果的“更好”仅针对代码里定义的指标。
- API 调用可能收费；请自行确认服务商计费方式。
