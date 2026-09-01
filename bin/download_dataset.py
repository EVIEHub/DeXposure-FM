#!/usr/bin/env python3
"""
DeXposure 数据集下载脚本

自动从指定 URL 下载 DeXposure 数据集文件
"""

import os
import sys
from pathlib import Path
import urllib.request
import hashlib
import json
import shutil
from typing import Dict, Optional


# 数据集配置
DATASET_FILES = {
    # 主要数据文件
    "historical-network_week_2025-07-01.json": {
        "url": "hf://data/historical-network_week_2025-07-01.json",
        "sha256": "d77920a7212847dd3cbfbbaabc8622c25e6523d2e7b2006d79d71b85851e1d0b",
        "size_mb": 76
    },
    "historical-network_week_2020-03-30.json": {
        "url": "hf://data/historical-network_week_2020-03-30.json",
        "sha256": "aa330bbb8fbf99719fc85d49625d7df7bd68f2b042f5806ded080cec99bad3f8",
        "size_mb": 1100
    },
    "meta_df.csv": {
        "url": "hf://data/meta_df.csv",
        "sha256": "a8306889fc4473972e843d8e847c9db68776cb86014746f1029fee574e254305",
        "size_mb": 0.1
    },

    # mapping 目录
    "mapping/id_to_info.json": {
        "url": "https://github.com/losdwind/graph-dexposure/releases/download/v1.0.0/id_to_info.json",
        "md5": None,
        "size_mb": 0.5
    },
    "mapping/rev_map.json": {
        "url": "https://github.com/losdwind/graph-dexposure/releases/download/v1.0.0/rev_map.json",
        "md5": None,
        "size_mb": 0.05
    },
    "mapping/token_to_protocol.json": {
        "url": "https://github.com/losdwind/graph-dexposure/releases/download/v1.0.0/token_to_protocol.json",
        "md5": None,
        "size_mb": 2.5
    },

    # network_data 目录
    "network_data/filtered_edges_ftx.csv": {
        "url": "https://github.com/losdwind/graph-dexposure/releases/download/v1.0.0/filtered_edges_ftx.csv",
        "md5": None,
        "size_mb": 3.2
    },
    "network_data/filtered_edges_terra.csv": {
        "url": "https://github.com/losdwind/graph-dexposure/releases/download/v1.0.0/filtered_edges_terra.csv",
        "md5": None,
        "size_mb": 2.9
    },
    "network_data/filtered_graph_data.csv": {
        "url": "https://github.com/losdwind/graph-dexposure/releases/download/v1.0.0/filtered_graph_data.csv",
        "md5": None,
        "size_mb": 59
    },
    "network_data/filtered_nodes_ftx.csv": {
        "url": "https://github.com/losdwind/graph-dexposure/releases/download/v1.0.0/filtered_nodes_ftx.csv",
        "md5": None,
        "size_mb": 1.0
    },
    "network_data/filtered_nodes_terra.csv": {
        "url": "https://github.com/losdwind/graph-dexposure/releases/download/v1.0.0/filtered_nodes_terra.csv",
        "md5": None,
        "size_mb": 0.7
    }
}

V2_REQUIRED_FILES = [
    "historical-network_week_2020-03-30.json",
    "historical-network_week_2025-07-01.json",
    "meta_df.csv",
]


def download_file(url: str, dest_path: Path, desc: str = "文件") -> bool:
    """
    下载文件并显示进度

    Args:
        url: 下载链接
        dest_path: 目标路径
        desc: 文件描述

    Returns:
        是否下载成功
    """
    try:
        print(f"\n正在下载 {desc}...")
        print(f"  URL: {url}")
        print(f"  保存至: {dest_path}")

        def progress_hook(block_num, block_size, total_size):
            """下载进度回调"""
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                print(f"\r  进度: {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)", end='')

        # 创建临时文件
        temp_path = dest_path.with_suffix(dest_path.suffix + '.tmp')

        # 下载文件
        if url.startswith("hf://"):
            from huggingface_hub import hf_hub_download

            cached_path = hf_hub_download(
                repo_id="EVIEHub/DeXposure-FM",
                filename=url.removeprefix("hf://"),
            )
            shutil.copy2(cached_path, temp_path)
        else:
            urllib.request.urlretrieve(url, temp_path, reporthook=progress_hook)
        print()  # 换行

        # 重命名为最终文件名
        temp_path.rename(dest_path)

        print(f"  ✓ 下载完成: {dest_path.name}")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n  ✗ HTTP 错误: {e.code} - {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"\n  ✗ URL 错误: {e.reason}")
        return False
    except Exception as e:
        print(f"\n  ✗ 下载失败: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def verify_file(
    file_path: Path,
    expected_md5: Optional[str] = None,
    expected_sha256: Optional[str] = None,
) -> bool:
    """
    验证下载的文件

    Args:
        file_path: 文件路径
        expected_md5: 期望的 MD5 值(可选)

    Returns:
        是否验证通过
    """
    if not file_path.exists():
        return False

    # 检查文件大小
    size = file_path.stat().st_size
    if size == 0:
        print(f"  ✗ 文件为空")
        return False

    print(f"  文件大小: {size / (1024*1024):.2f} MB")

    # MD5 校验(如果提供)
    if expected_md5:
        print(f"  正在计算 MD5...")
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        actual_md5 = md5.hexdigest()

        if actual_md5 != expected_md5:
            print(f"  ✗ MD5 校验失败: 期望 {expected_md5}, 实际 {actual_md5}")
            return False
        print(f"  ✓ MD5 校验通过")

    if expected_sha256:
        print("  正在计算 SHA-256...")
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            print(
                f"  ✗ SHA-256 校验失败: 期望 {expected_sha256}, "
                f"实际 {actual_sha256}"
            )
            return False
        print("  ✓ SHA-256 校验通过")

    return True


def download_dataset(
    data_dir: str = ".",
    force: bool = False,
    files: Optional[list] = None
) -> bool:
    """
    下载 DeXposure 数据集

    Args:
        data_dir: 数据目录
        force: 是否强制重新下载(覆盖已存在的文件)
        files: 要下载的文件列表,None 表示下载所有文件

    Returns:
        是否全部下载成功
    """
    data_path = Path(data_dir).resolve()
    print(f"\n{'='*60}")
    print(f"DeXposure 数据集下载工具")
    print(f"{'='*60}")
    print(f"\n目标目录: {data_path}")

    # 创建数据目录
    data_path.mkdir(parents=True, exist_ok=True)

    # 确定要下载的文件
    files_to_download = files if files else V2_REQUIRED_FILES

    success_count = 0
    failed_files = []

    for filename in files_to_download:
        if filename not in DATASET_FILES:
            print(f"\n⚠️  未知的文件: {filename}")
            failed_files.append(filename)
            continue

        config = DATASET_FILES[filename]
        dest_path = data_path / filename

        # 如果文件在子目录中,创建子目录
        if '/' in filename:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

        # 检查文件是否已存在
        if dest_path.exists() and not force:
            print(f"\n{filename} 已存在,跳过下载")
            print(f"  如需重新下载,请使用 --force 参数")
            success_count += 1
            continue

        # 下载文件
        if download_file(config['url'], dest_path, filename):
            # 验证文件
            if verify_file(
                dest_path,
                config.get("md5"),
                config.get("sha256"),
            ):
                success_count += 1
            else:
                failed_files.append(filename)
        else:
            failed_files.append(filename)

    # 总结
    print(f"\n{'='*60}")
    print(f"下载完成!")
    print(f"{'='*60}")
    print(f"成功: {success_count}/{len(files_to_download)}")

    if failed_files:
        print(f"\n失败的文件:")
        for f in failed_files:
            print(f"  - {f}")
        print(f"\n提示:")
        print(f"  1. 检查网络连接")
        print(f"  2. 确认已接受 Hugging Face 仓库的访问条件并完成登录")
        print(f"  3. 可以手动下载后放到 {data_path}")
        return False

    return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="下载 DeXposure 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载所有数据集到当前目录
  python bin/download_dataset.py

  # 下載到指定目录
  python bin/download_dataset.py --data-dir /path/to/data

  # 强制重新下载
  python bin/download_dataset.py --force

  # 仅下载特定文件
  python bin/download_dataset.py --files meta_df.csv

注意:
  - 数据集文件约 1.2GB,请确保网络连接稳定
  - 下载前请确保目标目录有足够空间
  - 默认下载 v2 复现所需的三个固定输入，并检查 SHA-256
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='数据目录路径 (默认: 当前目录)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重新下载,覆盖已存在的文件'
    )

    parser.add_argument(
        '--files',
        nargs='+',
        help='仅下载指定的文件名'
    )

    args = parser.parse_args()

    # 下载数据集
    success = download_dataset(
        data_dir=args.data_dir,
        force=args.force,
        files=args.files
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
