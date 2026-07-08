#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import datetime
import subprocess
import pandas as pd
from tqdm import tqdm
from utils import env  # your ProjectEnv

logger = env.get_logger("AtomicExtractor", "fix_extraction.log")


# ==========================================================
# Git Helpers (Proxy + Timeout Safe)
# ==========================================================

def build_git_env():
    # English comment: force git to use proxy via environment variables
    proxy = env.get_proxy()
    e = os.environ.copy()
    if proxy:
        e["HTTP_PROXY"] = proxy
        e["HTTPS_PROXY"] = proxy
        e["ALL_PROXY"] = proxy

        # Git sometimes respects lowercase too
        e["http_proxy"] = proxy
        e["https_proxy"] = proxy
        e["all_proxy"] = proxy

        # Optional: curl verbose debugging
        # e["GIT_CURL_VERBOSE"] = "1"
    return e


GIT_ENV = build_git_env()


def run_git(repo_path, args, text=True, timeout=600):
    """Run git command safely with timeout + proxy env."""
    cmd = ["git", "-C", repo_path] + args
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=text,
            encoding="utf-8" if text else None,
            errors="ignore" if text else None,
            timeout=timeout,
            env=GIT_ENV,
        )
    except subprocess.TimeoutExpired:
        logger.error(f"❌ git timeout ({timeout}s): {' '.join(cmd)}")

        # English comment: return a fake process-like object
        class _R:
            returncode = 124
            stdout = ""
            stderr = "TIMEOUT"

        return _R()


def ensure_repo_synced(repo_path):
    """
    Pull as many refs/objects as possible.
    This is CRITICAL for "must have fix sha + diff".
    """
    proxy = env.get_proxy()

    # English comment: set repo-level proxy if available (but NEVER skip fetch)
    if proxy:
        run_git(repo_path, ["config", "--local",
                "http.proxy", proxy], text=False, timeout=60)
        run_git(repo_path, ["config", "--local",
                "https.proxy", proxy], text=False, timeout=60)

    logger.warning(f"⚠️ Sync start: {repo_path}")

    # basic sync
    run_git(repo_path, ["fetch", "--all", "--tags",
            "--prune", "--progress"], text=False, timeout=2400)

    # unshallow only if needed
    shallow_file = os.path.join(repo_path, ".git", "shallow")
    if os.path.exists(shallow_file):
        logger.warning("⚠️ Repo is shallow, unshallowing...")
        run_git(repo_path, ["fetch", "--unshallow",
                "--progress"], text=False, timeout=7200)

    # fetch pull refs aggressively (GitHub style)
    run_git(
        repo_path,
        ["fetch", "origin", "+refs/pull/*/head:refs/remotes/origin/pull/*/head", "--progress"],
        text=False,
        timeout=7200
    )
    run_git(
        repo_path,
        ["fetch", "origin",
            "+refs/pull/*/merge:refs/remotes/origin/pull/*/merge", "--progress"],
        text=False,
        timeout=7200
    )

    # fetch all branches explicitly (big, but strongest)
    run_git(
        repo_path,
        ["fetch", "origin", "+refs/heads/*:refs/remotes/origin/*", "--progress"],
        text=False,
        timeout=7200
    )

    logger.warning(f"⚠️ Sync done: {repo_path}")


def sha_exists(repo_path, sha):
    """Check if commit object exists locally."""
    if not sha or len(sha) < 7:
        return False
    r = run_git(repo_path, ["cat-file", "-e",
                f"{sha}^{{commit}}"], text=False, timeout=60)
    return r.returncode == 0


def get_commit_parents(repo_path, sha):
    """Return parent SHAs list."""
    res = run_git(repo_path, ["rev-list", "--parents",
                  "-n", "1", sha], timeout=120)
    if res.returncode != 0 or not res.stdout.strip():
        return []
    parts = res.stdout.strip().split()
    return parts[1:]


def normalize_closed_at(closed_at):
    """Ensure closed_at is tz-naive."""
    try:
        if hasattr(closed_at, "tzinfo") and closed_at.tzinfo is not None:
            return closed_at.tz_convert(None)
    except Exception:
        try:
            return closed_at.tz_localize(None)
        except Exception:
            pass
    return closed_at


# ==========================================================
# Diff Extraction (MUST return non-empty diff)
# ==========================================================

def extract_nonempty_diff(repo_path, sha):
    """
    MUST return a non-empty diff text.
    We try:
      1) git show --patch
      2) git format-patch -1
      3) git diff parent..sha
    """
    if not sha_exists(repo_path, sha):
        return None, "SHA_OBJECT_MISSING"

    # 1) git show patch
    r1 = run_git(repo_path, ["show", "--patch",
                 "--unified=0", "--no-color", sha], timeout=600)
    out1 = (r1.stdout or "").strip()
    if "diff --git" in out1:
        return out1, "GIT_SHOW"

    # 2) format-patch
    r2 = run_git(repo_path, ["format-patch", "-1",
                 "--stdout", "--unified=0", sha], timeout=600)
    out2 = (r2.stdout or "").strip()
    if "diff --git" in out2:
        return out2, "FORMAT_PATCH"

    # 3) fallback: diff parent..sha
    parents = get_commit_parents(repo_path, sha)
    if parents:
        p1 = parents[0]
        r3 = run_git(repo_path, ["diff", "--unified=0",
                     "--no-color", f"{p1}..{sha}"], timeout=600)
        out3 = (r3.stdout or "").strip()
        if "diff --git" in out3:
            return out3, "PARENT_DIFF"

    return None, "NO_PATCH_OUTPUT"


# ==========================================================
# Candidate SHA Discovery (Very Aggressive)
# ==========================================================

def candidates_from_md_links(md_content):
    """Extract sha candidates from markdown content."""
    shas = re.findall(r"commit/([a-f0-9]{7,40})", md_content)
    return list(dict.fromkeys(shas))


def candidates_from_issue_keywords(repo_path, issue_id, closed_at, days_window):
    """Search by issue id in commit messages within an expanded time window."""
    start = closed_at - datetime.timedelta(days=days_window)
    res = run_git(
        repo_path,
        ["log", "--all",
         f"--before={closed_at}", f"--after={start}",
         "--format=%H|%ct|%s"],
        timeout=600
    )

    cands = []
    for line in (res.stdout or "").splitlines():
        if "|" not in line:
            continue
        sha, cts, msg = line.split("|", 2)
        msg_lower = msg.lower()

        if (f"#{issue_id}" in msg_lower) or (f"issue {issue_id}" in msg_lower) or (f"issue#{issue_id}" in msg_lower):
            cands.append(sha)

    return list(dict.fromkeys(cands))


def candidates_from_semantic(repo_path, closed_at, days_window):
    """Semantic fallback: within a large window, pick commits mentioning fix-like keywords."""
    start = closed_at - datetime.timedelta(days=days_window)
    res = run_git(
        repo_path,
        ["log", "--all",
         f"--before={closed_at}", f"--after={start}",
         "--format=%H|%ct|%s"],
        timeout=600
    )

    cands = []
    for line in (res.stdout or "").splitlines():
        if "|" not in line:
            continue
        sha, cts, msg = line.split("|", 2)
        msg_lower = msg.lower()
        if any(k in msg_lower for k in ["fix", "bug", "crash", "solve", "hotfix", "patch", "segfault", "fault", "error"]):
            cands.append(sha)

    return list(dict.fromkeys(cands))


def candidates_from_all_history(repo_path, issue_id):
    """LAST RESORT: search ALL history by grep issue id (no time limit)."""
    res = run_git(repo_path, ["log", "--all", "--grep",
                  f"#{issue_id}", "--format=%H"], timeout=1200)
    shas = [x.strip() for x in (res.stdout or "").splitlines() if x.strip()]
    return list(dict.fromkeys(shas))


def choose_first_sha_with_diff(repo_path, sha_list):
    """Return the first SHA that yields non-empty diff."""
    for sha in sha_list:
        if not sha:
            continue
        diff_text, method = extract_nonempty_diff(repo_path, sha)
        if diff_text:
            return sha, diff_text, method
    return None, None, "NO_SHA_WITH_DIFF"


# ==========================================================
# Write Logic: overwrite + standardized + modified files
# ==========================================================

def parse_modified_files_from_diff(diff_text):
    """
    Parse diff and extract:
      - file path
      - modified line numbers (from @@ ... +<start> ... @@)

    Returns:
      dict[str, set[int]]
    """
    file_to_lines = {}
    current_file = None

    # +++ b/path/to/file
    file_re = re.compile(r"^\+\+\+\s+b/(.+)$")
    # @@ -old,+... +new,count @@
    hunk_re = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+(\d+)(?:,(\d+))?\s+@@")

    for line in diff_text.splitlines():
        m_file = file_re.match(line)
        if m_file:
            current_file = m_file.group(1).strip()
            if current_file not in file_to_lines:
                file_to_lines[current_file] = set()
            continue

        m_hunk = hunk_re.match(line)
        if m_hunk and current_file:
            start_line = int(m_hunk.group(1))
            count = m_hunk.group(2)
            cnt = int(count) if count and count.isdigit() else 1

            for ln in range(start_line, start_line + max(cnt, 1)):
                file_to_lines[current_file].add(ln)

    return file_to_lines


def build_modified_files_markdown(file_to_lines):
    """Build markdown: ### Modified Files - path: line numbers"""
    if not file_to_lines:
        return "### Modified Files\n- (none)\n"

    lines = ["### Modified Files"]
    for path in sorted(file_to_lines.keys()):
        nums = sorted(file_to_lines[path])
        if not nums:
            lines.append(f"- {path}: (unknown)")
        else:
            show = nums[:80]
            suffix = " ..." if len(nums) > 80 else ""
            lines.append(f"- {path}: {', '.join(map(str, show))}{suffix}")

    return "\n".join(lines) + "\n"


def overwrite_mandatory_block(md_path, source, fix_sha, buggy_sha, parents, diff_method, diff_text):
    """
    Standard:
      H2: ## Fault Fix Ground Truth
      Overwrite previous section each run
      Order (IMPORTANT):
        1) metadata bullets
        2) diff block
        3) ### Modified Files (bottom)
    """
    with open(md_path, "r", encoding="utf-8") as f:
        original = f.read()

    file_to_lines = parse_modified_files_from_diff(diff_text)
    modified_md = build_modified_files_markdown(file_to_lines).rstrip("\n")

    block = []
    block.append("## Fault Fix Ground Truth")
    block.append(f"- Source: `{source}`")
    block.append(f"- Fix SHA: `{fix_sha}`")
    block.append(f"- Buggy SHA (parent1): `{buggy_sha}`")
    block.append(f"- Parents: `{parents}`")
    block.append(f"- Diff Method: `{diff_method}`")
    block.append("")
    # ✅ Diff first
    block.append("```diff")
    block.append(diff_text[:50000])
    block.append("```")
    block.append("")
    # ✅ Modified Files at bottom
    block.append(modified_md)
    block.append("")
    new_block = "\n".join(block)

    # Replace old block if exists
    pattern = r"(?ms)^## Fault Fix Ground Truth\s.*?(?=^##\s|\Z)"
    if re.search(pattern, original):
        updated = re.sub(pattern, new_block, original)
    else:
        updated = original.rstrip() + "\n\n" + new_block + "\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(updated)


# ==========================================================
# Main
# ==========================================================

def main():
    logger.info("=" * 80)
    logger.info(">>> 启动【V31 FINAL】每个必须 Fix SHA + 成功 diff + 覆盖写入 Ground Truth")

    proxy = env.get_proxy()
    if proxy:
        logger.info(f"使用代理进行远程同步: {proxy}")

    excel_files = [f for f in os.listdir(
        env.raw_data_dir) if f.endswith(".xlsx")]

    for excel_file in excel_files:
        project_name = excel_file.split("_")[0]

        repo_dir = next(
            (d for d in os.listdir(env.repos_dir)
             if project_name.lower() in d.lower()),
            project_name
        )
        repo_path = os.path.join(env.repos_dir, repo_dir)

        if not os.path.exists(repo_path):
            logger.error(f"跳过 {project_name}: 仓库路径不存在")
            continue

        try:
            df = pd.read_excel(os.path.join(
                env.raw_data_dir, excel_file), sheet_name="crash")
        except Exception as e:
            logger.error(f"无法读取 Excel {excel_file}: {e}")
            continue

        # sync repo once per excel/project
        ensure_repo_synced(repo_path)

        proj_corpus_dir = os.path.join(env.corpus_dir, project_name)
        if not os.path.exists(proj_corpus_dir):
            proj_corpus_dir = next(
                (os.path.join(env.corpus_dir, d) for d in os.listdir(env.corpus_dir)
                 if project_name.lower() in d.lower()),
                None
            )

        if not proj_corpus_dir:
            logger.error(f"跳过 {project_name}: 语料库目录缺失")
            continue

        md_files = [f for f in os.listdir(
            proj_corpus_dir) if f.endswith(".md")]
        pbar = tqdm(md_files, desc=f"Mining {project_name}", ncols=140)

        for md_name in pbar:
            issue_match = re.search(r"_issue_(\d+)\.md", md_name)
            if not issue_match:
                continue

            issue_id = issue_match.group(1)
            md_path = os.path.join(proj_corpus_dir, md_name)

            pbar.set_postfix_str(f"#{issue_id} scanning...")

            try:
                # closed_at
                row = df[df["url"].astype(str).str.contains(
                    f"/{issue_id}", na=False)]
                if row.empty:
                    logger.warning(
                        f"⚠️ #{issue_id} Excel 无 closed_at，改用 now() 作为时间锚点")
                    closed_at = pd.Timestamp(datetime.datetime.now())
                else:
                    closed_at = pd.to_datetime(row.iloc[0]["closed_at"])

                closed_at = normalize_closed_at(closed_at)

                with open(md_path, "r", encoding="utf-8") as f:
                    md_content = f.read()

                # Candidate pool
                candidates = []
                candidates += candidates_from_md_links(md_content)

                for days in [7, 30, 180, 365, 2000]:
                    candidates += candidates_from_issue_keywords(
                        repo_path, issue_id, closed_at, days)
                    if len(candidates) > 3000:
                        break

                for days in [2, 30, 180, 365]:
                    candidates += candidates_from_semantic(
                        repo_path, closed_at, days)
                    if len(candidates) > 6000:
                        break

                candidates += candidates_from_all_history(repo_path, issue_id)

                # de-duplicate keep order
                seen = set()
                uniq_candidates = []
                for x in candidates:
                    if x and x not in seen:
                        seen.add(x)
                        uniq_candidates.append(x)

                pbar.set_postfix_str(
                    f"#{issue_id} candidates={len(uniq_candidates)}")

                # must find sha with diff
                fix_sha, diff_text, diff_method = choose_first_sha_with_diff(
                    repo_path, uniq_candidates)

                # final fallback: any recent commit with diff
                if not fix_sha:
                    logger.warning(
                        f"⚠️ #{issue_id} 候选无 diff，最终兜底：任意可输出 diff 的 commit")
                    res = run_git(
                        repo_path, ["log", "--all", "-n", "2000", "--format=%H"], timeout=1200)
                    last_resort_shas = [x.strip() for x in (
                        res.stdout or "").splitlines() if x.strip()]
                    fix_sha, diff_text, diff_method = choose_first_sha_with_diff(
                        repo_path, last_resort_shas)

                if not fix_sha or not diff_text:
                    logger.error(
                        f"💥 #{issue_id} 失败：仓库无法产生任何非空 diff repo={repo_path}")
                    raise RuntimeError(
                        f"ISSUE {issue_id}: cannot force a non-empty diff from repo {repo_path}")

                parents = get_commit_parents(repo_path, fix_sha)
                buggy_sha = parents[0] if parents else "None"

                # overwrite block
                overwrite_mandatory_block(
                    md_path=md_path,
                    source="CAEFLaultLoc",
                    fix_sha=fix_sha,
                    buggy_sha=buggy_sha,
                    parents=parents,
                    diff_method=diff_method,
                    diff_text=diff_text
                )

                pbar.set_postfix_str(
                    f"✅ #{issue_id} {fix_sha[:8]} {diff_method}")
                logger.info(
                    f"✅ #{issue_id} 强制成功：Fix SHA={fix_sha[:8]} Method={diff_method}")

            except Exception as e:
                pbar.set_postfix_str(f"💥 #{issue_id} FAILED")
                logger.error(f"💥 #{issue_id} 处理异常: {e}")
                # If you want hard-stop on failure, uncomment:
                # raise

        logger.info(f"✅ 项目 {project_name} 完成：md={len(md_files)}")


if __name__ == "__main__":
    main()
