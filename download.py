#!/usr/bin/env python3
import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

import yaml


def extract_urls_from_yaml(yaml_path: str) -> List[str]:
    """Extract all URLs from the YAML configuration file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    urls = []

    # Get training data URLs
    if 'data' in config and 'paths' in config['data']:
        urls.extend(config['data']['paths'])

    # Get eval data URLs
    if 'evaluators' in config:
        for evaluator in config['evaluators']:
            if 'type' in evaluator and evaluator['type'] == 'downstream':
                continue

            if 'data' in evaluator and 'datasets' in evaluator['data']:
                for dataset_urls in evaluator['data']['datasets'].values():
                    if isinstance(dataset_urls, list):
                        urls.extend(dataset_urls)

    return urls


async def download_file(url: str, target_path: Path) -> bool:
    """Download a single file using wget."""
    try:
        print(f"Downloading {url}")
        process = await asyncio.create_subprocess_exec(
            'wget', '-c', '-O', str(target_path), url,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        await process.communicate()

        if process.returncode != 0:
            print(f"Error downloading {url}", file=sys.stderr)
            return False

        print(f"Downloaded {url}")
        return True

    except Exception as e:
        print(f"Error downloading {url}: {str(e)}", file=sys.stderr)
        return False


async def download_all_data(urls: List[str], data_root: Path, max_concurrent: int) -> Tuple[int, int]:
    """Download all data files using wget."""
    if not urls:
        return 0, 0

    print(f"\nFound {len(urls)} files to download")

    downloaded = 0
    failed = 0

    # Process URLs in batches to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_download(url: str, target_path: Path) -> Tuple[bool, str]:
        async with semaphore:
            # Ensure directory exists
            os.makedirs(target_path.parent, exist_ok=True)

            # Let wget handle resume logic
            success = await download_file(url, target_path)
            return success, url

    tasks = []
    for url in urls:
        parsed_url = urlparse(url)
        relative_path = parsed_url.path.lstrip('/')
        target_path = data_root / relative_path

        task = asyncio.create_task(bounded_download(url, target_path))
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            failed += 1
            print(f"Download failed with error: {str(result)}", file=sys.stderr)
        else:
            success, url = result
            if success:
                downloaded += 1
            else:
                failed += 1

    return downloaded, failed


async def main_async():
    parser = argparse.ArgumentParser(description='Download OLMo data.')
    parser.add_argument('yaml_path', help='Path to the OLMo YAML configuration file')
    parser.add_argument('data_root', help='Root directory for downloading the dataset')
    parser.add_argument('--parallel', type=int, default=8,
                        help='Number of parallel downloads (default: 8)')
    args = parser.parse_args()

    try:
        urls = extract_urls_from_yaml(args.yaml_path)
        data_root = Path(args.data_root)

        downloaded, failed = await download_all_data(urls, data_root, args.parallel)

        print("\n=== Download Summary ===")
        print(f"Successfully downloaded: {downloaded}")
        print(f"Failed downloads: {failed}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


def main():
    """Entry point that runs the async main function."""
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
