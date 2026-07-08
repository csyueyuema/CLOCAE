#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
import subprocess
from utils import env  # 导入你定义的工具类实例

# 获取标准化的 Logger
logger = env.get_logger("RepoCloner", "repo_clone.log")


def extract_repo_url(issue_url):
    """
    从 Issue URL 中解析出基础仓库的 .git 地址
    输入: https://github.com/KratosMultiphysics/Kratos/issues/11544
    输出: https://github.com/KratosMultiphysics/Kratos.git
    """
    if pd.isna(issue_url):
        return None
    # 匹配 GitHub 仓库主页路径
    match = re.match(r"(https://github\.com/[^/]+/[^/]+)", str(issue_url))
    if match:
        return f"{match.group(1)}.git"
    return None


def clone_repos():
    logger.info("="*40)
    logger.info("Starting Automated CAE Repository Cloning")
    logger.info("="*40)

    # 获取代理配置
    proxy = env.get_proxy()
    if proxy:
        logger.info(f"Proxy detected in config: {proxy}")
    else:
        logger.warning(
            "No proxy detected in config.yaml, proceeding with direct connection.")

    # 扫描 raw_data 目录下的所有 Excel
    if not os.path.exists(env.raw_data_dir):
        logger.error(f"Raw data directory not found: {env.raw_data_dir}")
        return

    excel_files = [f for f in os.listdir(env.raw_data_dir)
                   if f.endswith('.xlsx') and not f.startswith('~$')]

    for file in excel_files:
        # 提取项目名作为文件夹名 (例如 Kratos_crash.xlsx -> Kratos)
        project_name = file.split('_')[0]
        target_repo_path = os.path.join(env.repos_dir, project_name)

        # 如果仓库已存在则跳过
        if os.path.exists(target_repo_path):
            logger.info(
                f"[{project_name}] Repository already exists at {target_repo_path}. Skipping.")
            continue

        # 读取 Excel 提取第一条 URL 用于推断仓库地址
        excel_path = os.path.join(env.raw_data_dir, file)
        try:
            # 假设你的 Sheet 名字是 crash，如果不是请修改
            df = pd.read_excel(excel_path)
            if 'url' not in df.columns:
                logger.error(
                    f"[{project_name}] Column 'url' not found in {file}")
                continue

            sample_url = df['url'].dropna(
            ).iloc[0] if not df['url'].dropna().empty else None
            repo_git_url = extract_repo_url(sample_url)

            if not repo_git_url:
                logger.warning(
                    f"[{project_name}] Could not infer Git URL from sample: {sample_url}")
                continue

            # 构建 Git 克隆命令
            # 使用 --bare 模式克隆：只保留 Git 元数据，不检出源码，节省大量空间
            cmd = ["git"]

            # 如果有代理，动态注入 git 配置参数
            if proxy:
                cmd += ["-c", f"http.proxy={proxy}",
                        "-c", f"https.proxy={proxy}"]

            cmd += ["clone", "--bare", repo_git_url, target_repo_path]

            logger.info(
                f"[{project_name}] Attempting to clone: {repo_git_url}")

            # 执行克隆
            subprocess.run(cmd, check=True)
            logger.info(
                f"[{project_name}] SUCCESS: Cloned to {target_repo_path}")

        except Exception as e:
            logger.error(f"[{project_name}] FAILED to clone: {str(e)}")

    logger.info("="*40)
    logger.info("Cloning Task Completed.")
    logger.info("="*40)


if __name__ == "__main__":
    clone_repos()
