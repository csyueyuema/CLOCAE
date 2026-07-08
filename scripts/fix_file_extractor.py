#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import yaml
import requests
import logging
from typing import Tuple, List
from tqdm import tqdm

# =========================
# CAEFaultLoc 路径配置
# =========================
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
CONFIG_FILE = os.path.join(base_dir, "config", "config.yaml")
CORPUS_DIR = os.path.join(base_dir, "issue_corpus")
LOG_DIR = os.path.join(base_dir, "logs")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# =========================
# 日志分流配置
# =========================
# 创建日志记录器
logger = logging.getLogger("CAEFaultLoc")
logger.setLevel(logging.DEBUG)  # 设置总级别为最低，由 handler 自行过滤

# 1. 文件 Handler: 记录所有详细信息 (INFO 级别)
file_handler = logging.FileHandler(os.path.join(
    LOG_DIR, "fix_extractor.log"), encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_fmt)

# 2. 控制台 Handler: 仅显示严重错误，避免干扰 tqdm 进度条
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # 只有错误才会打印到屏幕
console_fmt = logging.Formatter("%(levelname)s: %(message)s")
console_handler.setFormatter(console_fmt)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# =========================
# 凭据与逻辑
# =========================


def load_credentials() -> Tuple[str, str]:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    github = config.get("github", {})
    return github.get("username"), github.get("token")


def get_fixing_files(owner: str, repo: str, issue_num: str, user: str, token: str) -> List[str]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    auth = (user, token)

    timeline_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}/timeline"
    # 在日志中记录 API 调用详情
    logger.info(f"Requesting timeline for {owner}/{repo}#{issue_num}")

    res = requests.get(timeline_url, auth=auth, headers=headers, timeout=20)

    pr_numbers = []
    if res.status_code == 200:
        for event in res.json():
            if event.get("event") in ["cross-referenced", "connected"]:
                source = event.get("source", {})
                if source.get("type") == "issue" and "pull_request" in source.get("issue", {}):
                    pr_num = source["issue"]["number"]
                    pr_numbers.append(pr_num)
                    logger.info(f"Found linked PR: #{pr_num}")

    fixing_files = []
    for pr_num in set(pr_numbers):
        files_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}/files"
        f_res = requests.get(files_url, auth=auth, headers=headers, timeout=20)
        if f_res.status_code == 200:
            for file_item in f_res.json():
                filename = file_item.get("filename")
                if any(filename.endswith(ext) for ext in ['.cpp', '.hpp', '.c', '.h', '.py', '.f90', '.f', '.cu']):
                    fixing_files.append(filename)

    if fixing_files:
        logger.info(
            f"Successfully located {len(fixing_files)} fixing files for Issue #{issue_num}")
    else:
        logger.info(f"No fixing files found for Issue #{issue_num}")

    return list(set(fixing_files))


def update_markdown_with_fix(file_path: str, files: List[str]):
    if not files:
        return
    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n## 【Fixing Files (Ground Truth)】\n\n")
        for file in files:
            f.write(f"- `{file}`\n")
        f.write("\n")


def main():
    try:
        user, token = load_credentials()
    except Exception as e:
        logger.error(f"Failed to load credentials: {e}")
        return

    for project in os.listdir(CORPUS_DIR):
        proj_path = os.path.join(CORPUS_DIR, project)
        if not os.path.isdir(proj_path):
            continue

        md_files = [f for f in os.listdir(proj_path) if f.endswith(".md")]

        # 使用 tqdm 接管控制台进度显示
        for md_name in tqdm(md_files, desc=f"Processing {project}", unit="file"):
            match = re.search(r"(.+)_issue_(\d+)\.md", md_name)
            if not match:
                continue

            repo_name, issue_num = match.groups()
            md_full_path = os.path.join(proj_path, md_name)

            with open(md_full_path, "r", encoding="utf-8") as f:
                content = f.read()
                owner_match = re.search(r"Repository: (.+)/", content)
                owner = owner_match.group(1) if owner_match else project

            try:
                fix_files = get_fixing_files(
                    owner, repo_name, issue_num, user, token)
                update_markdown_with_fix(md_full_path, fix_files)
            except Exception as e:
                # 错误信息会同时出现在日志和屏幕（因为级别为 ERROR）
                logger.error(f"Error processing {md_name}: {e}")


if __name__ == "__main__":
    main()
