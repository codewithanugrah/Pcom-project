import asyncio
import aiohttp
import semver
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import re
import os
from datetime import datetime, timedelta, timezone

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("WARNING: GITHUB_TOKEN environment variable not set. GitHub data will fail.")

def parse_dt(dt_str):
    if not dt_str: return None
    return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))

# --- RATE LIMIT HANDLERS ---
async def fetch_graphql_with_retry(session, query, variables=None):
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    payload = {"query": query}
    if variables: payload["variables"] = variables
        
    for attempt in range(3):
        async with session.post("https://api.github.com/graphql", json=payload, headers=headers) as response:
            remaining = int(response.headers.get('x-ratelimit-remaining', -1))
            reset_time = int(response.headers.get('x-ratelimit-reset', 0))
            
            if response.status in [403, 429] and remaining == 0:
                wait_time = max(0, reset_time - int(datetime.now(timezone.utc).timestamp()))
                print(f"\n⚠️ GitHub Rate Limit Hit! Sleeping for {wait_time}s...\n")
                await asyncio.sleep(wait_time + 5)
                continue
                
            if response.status == 200:
                data = await response.json()
                if 'errors' in data and any(err.get('type') == 'RATE_LIMITED' for err in data['errors']):
                    print(f"\n⚠️ GraphQL Node Limit Hit! Sleeping for 60s...\n")
                    await asyncio.sleep(60)
                    continue
                return data
            await asyncio.sleep(5)
    return None

async def fetch_npm_with_retry(session, url):
    for attempt in range(3):
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Ensure it's a dictionary, not a string or list
                    if isinstance(data, dict):
                        return data
                    return None
                elif response.status == 429:
                    await asyncio.sleep(10)
                else:
                    await asyncio.sleep(2)
        except Exception:
            await asyncio.sleep(2)
    return None

# --- 1. NPM METADATA & VERSIONS ---
async def get_package_registry_data(package_name, session):
    url = f"https://registry.npmjs.org/{package_name}"
    data = await fetch_npm_with_retry(session, url)
    if not data or not isinstance(data, dict): 
        return None, None, None, None
    
    created_at = parse_dt(data.get('time', {}).get('created'))
    stable_versions = []
    
    for v_str, v_data in data.get('versions', {}).items():
        try:
            parsed = semver.VersionInfo.parse(v_str)
            if parsed.prerelease is None:
                pub_date = parse_dt(data.get('time', {}).get(v_str))
                if pub_date:
                    stable_versions.append({
                        "version": v_str,
                        "published_at": pub_date,
                        "purl": f"pkg:npm/{package_name}@{v_str}",
                        "unpacked_size": v_data.get('dist', {}).get('unpackedSize'),
                        "file_count": v_data.get('dist', {}).get('fileCount'),
                        "dependency_count": len(v_data.get('dependencies')) if isinstance(v_data.get('dependencies'), dict) else 0,
                        "dev_dependency_count": len(v_data.get('devDependencies')) if isinstance(v_data.get('devDependencies'), dict) else 0,
                        "optional_dependency_count": len(v_data.get('optionalDependencies')) if isinstance(v_data.get('optionalDependencies'), dict) else 0,
                        "peer_dependency_count": len(v_data.get('peerDependencies')) if isinstance(v_data.get('peerDependencies'), dict) else 0,
                        "bundled_dependency_count": len(v_data.get('bundledDependencies', v_data.get('bundleDependencies'))) if isinstance(v_data.get('bundledDependencies', v_data.get('bundleDependencies')), dict) else 0
                    })
        except ValueError: continue
        
    if not stable_versions: return None, None, None, None
    
    latest_version = str(max([semver.VersionInfo.parse(v["version"]) for v in stable_versions]))
    repo_field = data.get('versions', {}).get(latest_version, {}).get('repository', '')
    
    repo_url = ''
    if isinstance(repo_field, dict):
        repo_url = repo_field.get('url', '')
    elif isinstance(repo_field, str):
        repo_url = repo_field
    
    owner, repo = None, None
    match = re.search(r'github\.com[/:]([^/]+)/([^/.]+)', str(repo_url))
    if match:
        owner, repo = match.group(1), match.group(2).replace('.git', '')
        
    return stable_versions, owner, repo, created_at

# --- 2. NPM BULK DOWNLOADS PAGINATION ---
async def get_all_downloads(package, created_at, session):
    """Chunks history into 365-day blocks to obey NPM's 18-month max limit."""
    daily_downloads = {}
    if not created_at: created_at = datetime.now(timezone.utc) - timedelta(days=365*5)
    
    current = created_at
    now = datetime.now(timezone.utc)
    
    while current < now:
        end = min(current + timedelta(days=365), now)
        url = f"https://api.npmjs.org/downloads/range/{current.strftime('%Y-%m-%d')}:{end.strftime('%Y-%m-%d')}/{package}"
        data = await fetch_npm_with_retry(session, url)
        if data and 'downloads' in data:
            for d in data['downloads']:
                daily_downloads[d['day']] = d.get('downloads', 0)
        current = end + timedelta(days=1)
        await asyncio.sleep(0.1) # Be gentle to NPM API
    return daily_downloads

# --- 3. GITHUB TIMELINE EXTRACTION ---
async def fetch_entire_github_timeline(owner, repo, session):
    if not owner or not repo or not GITHUB_TOKEN: return pd.DataFrame(), pd.DataFrame()
    issues, prs = [], []
    query_template = """
    query($owner: String!, $repo: String!, $cursor: String) {
      repository(owner: $owner, name: $repo) {
        items: %s(first: 100, after: $cursor) {
          pageInfo { hasNextPage endCursor }
          nodes { createdAt closedAt %s }
        }
      }
    }
    """
    for item_type, extra_field in [("issues", ""), ("pullRequests", "mergedAt")]:
        query = query_template % (item_type, extra_field)
        has_next, cursor = True, None
        while has_next:
            data = await fetch_graphql_with_retry(session, query, {"owner": owner, "repo": repo, "cursor": cursor})
            if not data or not data.get('data', {}).get('repository'): break
            connection = data['data']['repository'].get('items', {})
            for node in connection.get('nodes', []):
                node['createdAt'] = parse_dt(node.get('createdAt'))
                node['closedAt'] = parse_dt(node.get('closedAt'))
                if item_type == "pullRequests":
                    node['mergedAt'] = parse_dt(node.get('mergedAt'))
                    prs.append(node)
                else: issues.append(node)
            has_next = connection.get('pageInfo', {}).get('hasNextPage', False)
            cursor = connection.get('pageInfo', {}).get('endCursor')
    return pd.DataFrame(issues), pd.DataFrame(prs)

# --- 4. GHSA VULNERABILITIES EXTRACTION ---
async def fetch_all_ghsa_vulnerabilities(package_name, session):
    query = """
    query($package: String!) {
      securityVulnerabilities(ecosystem: NPM, package: $package, first: 100) {
        nodes { advisory { publishedAt, severity } }
      }
    }
    """
    data = await fetch_graphql_with_retry(session, query, variables={"package": package_name})
    vulns = []
    if data:
        for node in data.get('data', {}).get('securityVulnerabilities', {}).get('nodes', []):
            adv = node.get('advisory', {})
            dt = parse_dt(adv.get('publishedAt'))
            if dt: vulns.append({"published_at": dt, "severity": adv.get('severity', 'UNKNOWN').lower()})
    return pd.DataFrame(vulns)

# --- 5. COMPREHENSIVE DYNAMIC SLICING ---
def slice_version_metrics(df_issues, df_prs, df_vulns, daily_downloads, v_data, all_versions, package_name, created_at):
    pub_dt = v_data['published_at']
    row = {
        "package_name": package_name,
        "version": v_data['version'],
        "purl": v_data['purl'],
        "package_created_at": created_at.isoformat() if created_at else None,
        "published_at": pub_dt.isoformat(),
        "unpacked_size": v_data['unpacked_size'],
        "file_count": v_data['file_count'],
        "dependency_count": v_data['dependency_count'],
        "optional_dependency_count": v_data['optional_dependency_count'],
        "peer_dependency_count": v_data['peer_dependency_count'],
        "dev_dependency_count": v_data['dev_dependency_count'],
        "bundled_dependency_count": v_data['bundled_dependency_count']
    }
    
    target_end_dt = pub_dt + timedelta(days=90)
    sevs = ['low', 'moderate', 'high', 'critical', 'unknown']
    
    for s in sevs: 
        row[f"target_ghsa_{s}_count_90d"] = 0
        row[f"ghsa_{s}_count"] = 0
        
    if not df_vulns.empty:
        forward = df_vulns[(df_vulns['published_at'] >= pub_dt) & (df_vulns['published_at'] <= target_end_dt)]
        past = df_vulns[df_vulns['published_at'] < pub_dt]
        for s in sevs:
            row[f"target_ghsa_{s}_count_90d"] = len(forward[forward['severity'] == s])
            row[f"ghsa_{s}_count"] = len(past[past['severity'] == s])

    for i in range(12):
        w_end = pub_dt - timedelta(days=30 * i)
        w_start = pub_dt - timedelta(days=30 * (i + 1))
        
        # GitHub Metrics
        if not df_issues.empty:
            row[f"issues_opened_M{i}"] = len(df_issues[(df_issues['createdAt'] >= w_start) & (df_issues['createdAt'] < w_end)])
            row[f"issues_closed_M{i}"] = len(df_issues[(df_issues['closedAt'] >= w_start) & (df_issues['closedAt'] < w_end)])
            row[f"issues_open_at_month_end_M{i}"] = len(df_issues[(df_issues['createdAt'] < w_end) & (df_issues['closedAt'].isna() | (df_issues['closedAt'] >= w_end))])
        else:
            row[f"issues_opened_M{i}"] = row[f"issues_closed_M{i}"] = row[f"issues_open_at_month_end_M{i}"] = 0
            
        if not df_prs.empty:
            row[f"pull_requests_opened_M{i}"] = len(df_prs[(df_prs['createdAt'] >= w_start) & (df_prs['createdAt'] < w_end)])
            row[f"pull_requests_merged_M{i}"] = len(df_prs[df_prs['mergedAt'].notna() & (df_prs['mergedAt'] >= w_start) & (df_prs['mergedAt'] < w_end)])
            row[f"pull_requests_closed_unmerged_M{i}"] = len(df_prs[df_prs['mergedAt'].isna() & df_prs['closedAt'].notna() & (df_prs['closedAt'] >= w_start) & (df_prs['closedAt'] < w_end)])
            row[f"pull_requests_open_at_month_end_M{i}"] = len(df_prs[(df_prs['createdAt'] < w_end) & (df_prs['closedAt'].isna() | (df_prs['closedAt'] >= w_end))])
        else:
            row[f"pull_requests_opened_M{i}"] = row[f"pull_requests_merged_M{i}"] = row[f"pull_requests_closed_unmerged_M{i}"] = row[f"pull_requests_open_at_month_end_M{i}"] = 0

        # Unfetchable GitHub schema fillers
        row[f"issues_reopened_M{i}"] = 0
        row[f"pull_requests_reopened_M{i}"] = 0
        
        # NPM Release count
        row[f"package_stable_release_count_M{i}"] = len([v for v in all_versions if w_start <= v['published_at'] < w_end])
        
        # Past Vulnerabilities
        for s in sevs: row[f"ghsa_{s}_count_M{i}"] = 0
        if not df_vulns.empty:
            w_vulns = df_vulns[(df_vulns['published_at'] >= w_start) & (df_vulns['published_at'] < w_end)]
            for s in sevs: row[f"ghsa_{s}_count_M{i}"] = len(w_vulns[w_vulns['severity'] == s])
                
        # Download Volume
        d_count, d_days = 0, 0
        curr_day = w_start
        while curr_day < w_end:
            day_str = curr_day.strftime('%Y-%m-%d')
            if day_str in daily_downloads:
                d_count += daily_downloads[day_str]
                d_days += 1
            curr_day += timedelta(days=1)
        row[f"package_downloads_M{i}"] = d_count
        row[f"package_download_days_observed_M{i}"] = d_days

    return row

# --- ORCHESTRATOR ---
async def process_package(package_name, session, semaphore):
    async with semaphore:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {package_name}...")
        try:
            versions, owner, repo, created_at = await get_package_registry_data(package_name, session)
            if not versions: return []
            
            task_gh = asyncio.create_task(fetch_entire_github_timeline(owner, repo, session))
            task_ghsa = asyncio.create_task(fetch_all_ghsa_vulnerabilities(package_name, session))
            task_dl = asyncio.create_task(get_all_downloads(package_name, created_at, session))
            
            df_issues, df_prs = await task_gh
            df_vulns = await task_ghsa
            daily_downloads = await task_dl
            
            package_rows = [slice_version_metrics(df_issues, df_prs, df_vulns, daily_downloads, v, versions, package_name, created_at) for v in versions]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished {package_name} ({len(package_rows)} versions).")
            return package_rows
        except Exception as e:
            print(f"Error on {package_name}: {e}")
            return []

async def main():
    csv_file = "training_dataset_comprehensive.csv"
    
    # 1. Load target packages
    with open("packages.txt", "r") as f:
        target_packages = list(set([line.strip() for line in f if line.strip() and not line.startswith("#")]))
        
    # 2. Check for already processed packages to enable safe resuming!
    processed_packages = set()
    if os.path.exists(csv_file):
        try:
            df_existing = pd.read_csv(csv_file, usecols=['package_name'], keep_default_na=False)
            processed_packages = set(df_existing['package_name'].unique())
            print(f"resuming ETL: Found {len(processed_packages)} packages already processed in {csv_file}.")
        except Exception:
            pass
            
    remaining_packages = [pkg for pkg in target_packages if pkg not in processed_packages]
    print(f"Total packages: {len(target_packages)} | Already done: {len(processed_packages)} | Remaining to process: {len(remaining_packages)}")
    
    if not remaining_packages:
        print("✅ All packages are already fully processed!")
        return

    semaphore = asyncio.Semaphore(10)
    timeout = aiohttp.ClientTimeout(total=600)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [process_package(pkg, session, semaphore) for pkg in remaining_packages]
        
        total_rows = 0
        for coro in asyncio.as_completed(tasks):
            package_rows = await coro
            if package_rows:
                df_pkg = pd.DataFrame(package_rows)
                df_pkg = df_pkg.fillna(0)
                # Append safely without deleting existing data
                write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
                df_pkg.to_csv(csv_file, mode='a', header=write_header, index=False)
                total_rows += len(package_rows)
                
        print(f"\n✅ ETL Batch Complete! Appended {total_rows} new rows to {csv_file}.")

if __name__ == "__main__":
    asyncio.run(main())