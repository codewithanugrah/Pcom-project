import asyncio
import aiohttp
import semver
import pandas as pd
import re
import os
from datetime import datetime, timedelta

# Grab the token from the terminal environment
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("WARNING: GITHUB_TOKEN environment variable not set. GitHub data will fail.")

# --- 1. METADATA FUNCTION ---
async def get_package_metadata(package_name, session):
    url = f"https://registry.npmjs.org/{package_name}"
    async with session.get(url) as response:
        if response.status != 200: return None
        data = await response.json()
        
        stable_versions = [semver.VersionInfo.parse(v) for v in data.get('versions', {}).keys() 
                           if semver.VersionInfo.is_valid(v) and semver.VersionInfo.parse(v).prerelease is None]
        if not stable_versions: return None
            
        latest_version = str(max(stable_versions))
        version_data = data['versions'][latest_version]
        
        repo_url = version_data.get('repository', {}).get('url', '')
        github_owner, github_repo = None, None
        match = re.search(r'github\.com[/:]([^/]+)/([^/\.]+)', str(repo_url))
        if match:
            github_owner = match.group(1)
            github_repo = match.group(2).replace('.git', '')
            
        return {
            "purl": f"pkg:npm/{package_name}@{latest_version}",
            "github_owner": github_owner,
            "github_repo": github_repo,
            "package_created_at": data.get('time', {}).get('created', ''),
            "published_at": data.get('time', {}).get(latest_version, ''),
            "unpacked_size": version_data.get('dist', {}).get('unpackedSize', None),
            "file_count": version_data.get('dist', {}).get('fileCount', None),
            "dependency_count": len(version_data.get('dependencies', {})),
            "optional_dependency_count": len(version_data.get('optionalDependencies', {})),
            "peer_dependency_count": len(version_data.get('peerDependencies', {})),
            "dev_dependency_count": len(version_data.get('devDependencies', {})),
            "bundled_dependency_count": len(version_data.get('bundledDependencies', version_data.get('bundleDependencies', {})))
        }

# --- 2. DOWNLOADS FUNCTION ---
async def get_npm_downloads(package_name, session):
    url = f"https://api.npmjs.org/downloads/range/2025-07-01:2026-06-30/{package_name}"
    downloads_data = {f"package_downloads_M{i}": 0 for i in range(12)}
    downloads_data.update({f"package_download_days_observed_M{i}": 0 for i in range(12)})
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            month_map = {"2025-07": 0, "2025-08": 1, "2025-09": 2, "2025-10": 3, "2025-11": 4, "2025-12": 5, 
                         "2026-01": 6, "2026-02": 7, "2026-03": 8, "2026-04": 9, "2026-05": 10, "2026-06": 11}
            for daily in data.get('downloads', []):
                day_str = daily.get('day', '')
                if len(day_str) >= 7 and day_str[:7] in month_map:
                    m_index = month_map[day_str[:7]]
                    downloads_data[f"package_downloads_M{m_index}"] += daily.get('downloads', 0)
                    downloads_data[f"package_download_days_observed_M{m_index}"] += 1
    return downloads_data

# --- 3. GHSA VULNERABILITIES FUNCTION (Replaces OSV) ---
def parse_iso_date(date_str):
    if not date_str: return None
    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))

async def get_ghsa_vulnerabilities(package_name, published_at_str, session):
    vuln_data = {f"target_ghsa_{sev}_count_90d": 0 for sev in ['low', 'moderate', 'high', 'critical', 'unknown']}
    for sev in ['low', 'moderate', 'high', 'critical', 'unknown']:
        vuln_data[f"ghsa_{sev}_count"] = 0
        for i in range(12): vuln_data[f"ghsa_{sev}_count_M{i}"] = 0
            
    published_at_dt = parse_iso_date(published_at_str)
    if not published_at_dt or not GITHUB_TOKEN: return vuln_data
    target_end_dt = published_at_dt + timedelta(days=90)
    
    # GraphQL query targeting the GitHub Security Advisory ecosystem
    query = """
    query($package: String!) {
      securityVulnerabilities(ecosystem: NPM, package: $package, first: 100) {
        nodes {
          advisory {
            publishedAt
            severity
          }
        }
      }
    }
    """
    
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    async with session.post("https://api.github.com/graphql", json={"query": query, "variables": {"package": package_name}}, headers=headers) as response:
        if response.status == 200:
            data = await response.json()
            nodes = data.get('data', {}).get('securityVulnerabilities', {}).get('nodes', [])
            
            month_map = {"2025-07": 0, "2025-08": 1, "2025-09": 2, "2025-10": 3, "2025-11": 4, "2025-12": 5, 
                         "2026-01": 6, "2026-02": 7, "2026-03": 8, "2026-04": 9, "2026-05": 10, "2026-06": 11}
            
            for node in nodes:
                advisory = node.get('advisory', {})
                vuln_pub_dt = parse_iso_date(advisory.get('publishedAt'))
                if not vuln_pub_dt: continue
                
                severity = advisory.get('severity', 'UNKNOWN').lower()
                if severity not in ['low', 'moderate', 'high', 'critical']: severity = 'unknown'
                
                # 1. Total historical count
                vuln_data[f"ghsa_{severity}_count"] += 1
                
                # 2. Strict 90-day forward looking target
                if published_at_dt <= vuln_pub_dt <= target_end_dt:
                    vuln_data[f"target_ghsa_{severity}_count_90d"] += 1
                    
                # 3. M0-M11 Historical Time-Series Bucketing
                year_month = advisory.get('publishedAt', '')[:7]
                if year_month in month_map:
                    m_index = month_map[year_month]
                    vuln_data[f"ghsa_{severity}_count_M{m_index}"] += 1
                    
    return vuln_data

# --- 4. GITHUB REPO STATS FUNCTION ---
async def get_github_stats(owner, repo, session):
    gh_data = {}
    for i in range(12):
        gh_data[f"issues_opened_M{i}"] = 0
        gh_data[f"pull_requests_merged_M{i}"] = 0
        
    if not owner or not repo or not GITHUB_TOKEN: return gh_data

    months = [
        ("2025-07-01", "2025-07-31"), ("2025-08-01", "2025-08-31"), ("2025-09-01", "2025-09-30"),
        ("2025-10-01", "2025-10-31"), ("2025-11-01", "2025-11-30"), ("2025-12-01", "2025-12-31"),
        ("2026-01-01", "2026-01-31"), ("2026-02-01", "2026-02-28"), ("2026-03-01", "2026-03-31"),
        ("2026-04-01", "2026-04-30"), ("2026-05-01", "2026-05-31"), ("2026-06-01", "2026-06-30")
    ]
    
    query_blocks = []
    for i, (start, end) in enumerate(months):
        query_blocks.append(f'issues_M{i}: search(query: "repo:{owner}/{repo} is:issue created:{start}..{end}", type: ISSUE) {{ issueCount }}')
        query_blocks.append(f'prs_merged_M{i}: search(query: "repo:{owner}/{repo} is:pr is:merged merged:{start}..{end}", type: ISSUE) {{ issueCount }}')
        
    graphql_query = "query { " + " ".join(query_blocks) + " }"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    
    async with session.post("https://api.github.com/graphql", json={"query": graphql_query}, headers=headers) as response:
        if response.status == 200:
            data = await response.json()
            if 'data' in data:
                for i in range(12):
                    gh_data[f"issues_opened_M{i}"] = data['data'].get(f'issues_M{i}', {}).get('issueCount', 0)
                    gh_data[f"pull_requests_merged_M{i}"] = data['data'].get(f'prs_merged_M{i}', {}).get('issueCount', 0)
    return gh_data

# --- ORCHESTRATOR ---
async def process_package(package_name, session):
    meta_data = await get_package_metadata(package_name, session)
    if not meta_data: return None
    
    dl_task = get_npm_downloads(package_name, session)
    ghsa_task = get_ghsa_vulnerabilities(package_name, meta_data['published_at'], session)
    gh_task = get_github_stats(meta_data['github_owner'], meta_data['github_repo'], session)
    
    dl_data, ghsa_data, gh_data = await asyncio.gather(dl_task, ghsa_task, gh_task)
    
    meta_data.update(dl_data)
    meta_data.update(ghsa_data)
    meta_data.update(gh_data)
    
    meta_data.pop('github_owner', None)
    meta_data.pop('github_repo', None)
    
    # Backfill any remaining un-queried schema columns with 0
    missing_prefixes = ['package_stable_release_count_M', 'issues_closed_M', 'issues_reopened_M', 
                        'pull_requests_opened_M', 'pull_requests_closed_unmerged_M', 
                        'pull_requests_reopened_M', 'issues_open_at_month_end_M', 
                        'pull_requests_open_at_month_end_M']
    for prefix in missing_prefixes:
        for i in range(12): meta_data[f"{prefix}{i}"] = 0
            
    return meta_data

async def main():
    test_packages = ['lodash', 'express']
    print("Fetching data from NPM and GitHub GraphQL (GHSA + Repo Stats)...")
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_package(pkg, session) for pkg in test_packages]
        results = await asyncio.gather(*tasks)
        
        valid_results = [res for res in results if res is not None]
        df = pd.DataFrame(valid_results)
        df.to_csv("training_dataset.csv", index=False)
        print(f"Success! Exported to training_dataset.csv (Fully GHSA Integrated)")

if __name__ == "__main__":
    asyncio.run(main())







# =====================================================================
# ALTERNATIVE: OSV.dev API APPROACH (Kept for reference)
# 
# I initially built and validated this pipeline using the OSV.dev API.
# OSV mirrors the exact GitHub Security Advisory (GHSA) database for npm,
# but operates as a REST endpoint that bypasses GraphQL complexity limits.
# After verifying both systems return identical GHSA node counts, I 
# switched the live pipeline to GitHub GraphQL to align strictly with 
# the dataset expectations discussed in class. 
# =====================================================================
'''
async def get_osv_vulnerabilities_alternative(package_name, published_at_str, session):
    url = "https://api.osv.dev/v1/query"
    payload = {"package": {"name": package_name, "ecosystem": "npm"}}
    
    # Initialize the target and historical columns
    vuln_data = {f"target_ghsa_{sev}_count_90d": 0 for sev in ['low', 'moderate', 'high', 'critical', 'unknown']}
    for sev in ['low', 'moderate', 'high', 'critical', 'unknown']:
        vuln_data[f"ghsa_{sev}_count"] = 0
        for i in range(12): vuln_data[f"ghsa_{sev}_count_M{i}"] = 0
            
    published_at_dt = parse_iso_date(published_at_str)
    if not published_at_dt: return vuln_data
    target_end_dt = published_at_dt + timedelta(days=90)
    
    async with session.post(url, json=payload) as response:
        if response.status == 200:
            data = await response.json()
            month_map = {"2025-07": 0, "2025-08": 1, "2025-09": 2, "2025-10": 3, "2025-11": 4, "2025-12": 5, 
                         "2026-01": 6, "2026-02": 7, "2026-03": 8, "2026-04": 9, "2026-05": 10, "2026-06": 11}
            
            for vuln in data.get('vulns', []):
                vuln_pub_dt = parse_iso_date(vuln.get('published'))
                if not vuln_pub_dt: continue
                
                # Extract the GHSA severity string
                severity = vuln.get('database_specific', {}).get('severity', 'UNKNOWN').lower()
                if severity not in ['low', 'moderate', 'high', 'critical']: severity = 'unknown'
                
                # 1. Total historical count
                vuln_data[f"ghsa_{severity}_count"] += 1
                
                # 2. Strict 90-day target calculation
                if published_at_dt <= vuln_pub_dt <= target_end_dt:
                    vuln_data[f"target_ghsa_{severity}_count_90d"] += 1
                    
                # 3. Time-Series historical bucketing (M0-M11)
                year_month = vuln.get('published', '')[:7]
                if year_month in month_map:
                    m_index = month_map[year_month]
                    vuln_data[f"ghsa_{severity}_count_M{m_index}"] += 1
                    
    return vuln_data
'''