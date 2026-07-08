#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
from loguru import logger
from tqdm import tqdm


class _TqdmSink:
    # English comment: make loguru output compatible with tqdm progress bar
    def write(self, message: str):
        msg = message.rstrip("\n")
        if msg:
            tqdm.write(msg)

    def flush(self):
        pass


class ProjectEnv:
    def __init__(self):
        # Path init
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.dirname(self.script_dir)
        self.raw_data_dir = os.path.join(self.base_dir, "raw_data")
        self.repos_dir = os.path.join(self.base_dir, "repos")
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.corpus_dir = os.path.join(self.base_dir, "issue_corpus")
        self.config_file = os.path.join(self.base_dir, "config", "config.yaml")

        # Create dirs
        for d in [self.repos_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)

        # Load config
        self.config = self._load_config()

        # Configure loguru only once
        self._configured = False

    def _load_config(self):
        """Load config.yaml safely."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Error loading config: {e}")
        return {}

    def get_proxy(self):
        """Return proxy string (http) if exists."""
        proxy_cfg = self.config.get("proxy", {})
        return proxy_cfg.get("http")

    def _setup_loguru(self, filename: str):
        if self._configured:
            return

        log_path = os.path.join(self.log_dir, filename)
        os.makedirs(self.log_dir, exist_ok=True)

        logger.remove()

        # ✅ Console: only WARNING/ERROR (no spam, tqdm-safe)
        logger.add(
            _TqdmSink(),
            level="WARNING",
            enqueue=True,
            backtrace=False,
            diagnose=False,
            colorize=False,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level}</level> | {message}"
        )

        # ✅ File: overwrite each run (NOT append)
        logger.add(
            log_path,
            level="INFO",
            encoding="utf-8",
            enqueue=True,
            backtrace=False,
            diagnose=False,
            mode="w",  # ✅ overwrite each run
            rotation="50 MB",
            retention="14 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}"
        )

        self._configured = True

    def get_logger(self, name, filename):
        """
        Compatibility wrapper:
        You can keep using: logger = env.get_logger("AtomicExtractor", "fix_extraction.log")
        'name' is kept for compatibility; loguru uses global logger.
        """
        self._setup_loguru(filename)
        return logger


env = ProjectEnv()
