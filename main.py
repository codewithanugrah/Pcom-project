import asyncio
import aiohttp
import semver
import pandas as pd
import re
import os
from datetime import datetime, timedelta, timezone

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("WARNING: GITHUB_TOKEN environment variable not set. GitHub data will fail.")

# --- RATE LIMIT HANDLER FOR GITHUB GRAPHQL ---
async def fetch_graphql_with_retry(session, query, variables=None):
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
        
    for attempt in range(3):
        async with session.post("https://api.github.com/graphql", json=payload, headers=headers) as response:
            remaining = int(response.headers.get('x-ratelimit-remaining', -1))
            reset_time = int(response.headers.get('x-ratelimit-reset', 0))
            
            if response.status in [403, 429] and remaining == 0:
                current_time = int(datetime.now(timezone.utc).timestamp())
                wait_time = max(0, reset_time - current_time)
                print(f"\n⚠️ GITHUB RATE LIMIT HIT! Sleeping for {wait_time}s...\n")
                await asyncio.sleep(wait_time + 5)
                continue
                
            if response.status == 200:
                data = await response.json()
                if 'errors' in data:
                    if any(err.get('type') == 'RATE_LIMITED' for err in data['errors']):
                        print(f"\n⚠️ GRAPHQL NODE LIMIT HIT! Sleeping for 60s...\n")
                        await asyncio.sleep(60)
                        continue
                    else:
                        print(f"GraphQL Query Error: {data['errors'][0].get('message')}")
                return data
                
            await asyncio.sleep(5)
    return None

# --- 1. METADATA & VERSION EXTRACTION ---
async def get_all_stable_versions(package_name, session):
    """Fetches all stable versions and their release timestamps."""
    url = f"https://registry.npmjs.org/{package_name}"
    async with session.get(url) as response:
        if response.status != 200: return None
        data = await response.json()
        
        stable_versions = []
        for v in data.get('versions', {}).keys():
            try:
                parsed = semver.VersionInfo.parse(v)
                if parsed.prerelease is None:
                    pub_date = data.get('time', {}).get(v)
                    if pub_date:
                        dt = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                        stable_versions.append({
                            "version": v,
                            "published_at": dt,
                            "purl": f"pkg:npm/{package_name}@{v}"
                        })
            except ValueError: continue
            
        latest_version = str(max([semver.VersionInfo.parse(v["version"]) for v in stable_versions]))
        repo_url = data.get('versions', {}).get(latest_version, {}).get('repository', {}).get('url', '')
        
        github_owner, github_repo = None, None
        match = re.search(r'github\.com[/:]([^/]+)/([^/\.]+)', str(repo_url))
        if match:
            github_owner = match.group(1)
            github_repo = match.group(2).replace('.git', '')
            
        return stable_versions, github_owner, github_repo

# --- 2. GITHUB TIMELINE EXTRACTION (FETCH ONCE) ---
async def fetch_entire_github_timeline(owner, repo, session):
    """Paginates through ALL issues and PRs ever created in the repo."""
    if not owner or not repo or not GITHUB_TOKEN: return pd.DataFrame(), pd.DataFrame()
    
    issues, prs = [], []
    
    # Query template for pagination
    query_template = """
    query($owner: String!, $repo: String!, $cursor: String) {
      repository(owner: $owner, name: $repo) {
        issues(first: 100, after: $cursor) {
          pageInfo { hasNextPage endCursor }
          nodes { createdAt closedAt state }
        }
        pullRequests(first: 100, after: $cursor) {
          pageInfo { hasNextPage endCursor }
          nodes { createdAt closedAt mergedAt state }
        }
      }
    }
    """
    
    # We paginate Issues and PRs separately to avoid node exhaustion
    for item_type in ["issues", "pullRequests"]:
        has_next = True
        cursor = None
        
        query = query_template.replace("issues(first: 100", f"items: {item_type}(first: 100")
        
        while has_next:
            data = await fetch_graphql_with_retry(session, query, {"owner": owner, "repo": repo, "cursor": cursor})
            if not data or not data.get('data', {}).get('repository'): break
                
            connection = data['data']['repository'].get('items', {})
            nodes = connection.get('nodes', [])
            
            for node in nodes:
                node['createdAt'] = datetime.fromisoformat(node['createdAt'].replace('Z', '+00:00')) if node.get('createdAt') else None
                node['closedAt'] = datetime.fromisoformat(node['closedAt'].replace('Z', '+00:00')) if node.get('closedAt') else None
                if item_type == "pullRequests":
                    node['mergedAt'] = datetime.fromisoformat(node['mergedAt'].replace('Z', '+00:00')) if node.get('mergedAt') else None
                    prs.append(node)
                else:
                    issues.append(node)
                    
            has_next = connection.get('pageInfo', {}).get('hasNextPage', False)
            cursor = connection.get('pageInfo', {}).get('endCursor')
            
    df_issues = pd.DataFrame(issues)
    df_prs = pd.DataFrame(prs)
    print(f"[{owner}/{repo}] Fetched {len(df_issues)} issues and {len(df_prs)} PRs.")
    return df_issues, df_prs

# --- 3. GHSA VULNERABILITIES EXTRACTION (FETCH ONCE) ---
async def fetch_all_ghsa_vulnerabilities(package_name, session):
    """Fetches all vulnerabilities for a package once."""
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
        nodes = data.get('data', {}).get('securityVulnerabilities', {}).get('nodes', [])
        for node in nodes:
            adv = node.get('advisory', {})
            dt = datetime.fromisoformat(adv['publishedAt'].replace('Z', '+00:00')) if adv.get('publishedAt') else None
            if dt:
                vulns.append({"published_at": dt, "severity": adv.get('severity', 'UNKNOWN').lower()})
    return pd.DataFrame(vulns)

# --- 4. DYNAMIC SLICING (IN-MEMORY) ---
def slice_version_metrics(df_issues, df_prs, df_vulns, version_data, package_name):
    """Slices the dataframes into relative M0-M11 windows based on the version publish date."""
    pub_dt = version_data['published_at']
    row = {
        "package_name": package_name,
        "version": version_data['version'],
        "purl": version_data['purl'],
        "published_at": pub_dt.isoformat()
    }
    
    # Target Variable: 90-day forward looking vulnerabilities (The Poisson Target)
    target_end_dt = pub_dt + timedelta(days=90)
    for sev in ['low', 'moderate', 'high', 'critical']:
        row[f"target_ghsa_{sev}_count_90d"] = 0
        
    if not df_vulns.empty:
        forward_vulns = df_vulns[(df_vulns['published_at'] >= pub_dt) & (df_vulns['published_at'] <= target_end_dt)]
        for sev in ['low', 'moderate', 'high', 'critical']:
            row[f"target_ghsa_{sev}_count_90d"] = len(forward_vulns[forward_vulns['severity'] == sev])

    # Relative Feature Engineering: M0 (T-30d to T), M1 (T-60d to T-30d), etc.
    for i in range(12):
        window_end = pub_dt - timedelta(days=30 * i)
        window_start = pub_dt - timedelta(days=30 * (i + 1))
        
        # Slicing Issues
        if not df_issues.empty:
            opened_issues = df_issues[(df_issues['createdAt'] >= window_start) & (df_issues['createdAt'] < window_end)]
            closed_issues = df_issues[(df_issues['closedAt'] >= window_start) & (df_issues['closedAt'] < window_end)]
            open_at_end = df_issues[(df_issues['createdAt'] < window_end) & (df_issues['closedAt'].isna() | (df_issues['closedAt'] >= window_end))]
            
            row[f"issues_opened_M{i}"] = len(opened_issues)
            row[f"issues_closed_M{i}"] = len(closed_issues)
            row[f"issues_open_at_month_end_M{i}"] = len(open_at_end)
        else:
            row[f"issues_opened_M{i}"] = row[f"issues_closed_M{i}"] = row[f"issues_open_at_month_end_M{i}"] = 0
            
        # Slicing PRs
        if not df_prs.empty:
            opened_prs = df_prs[(df_prs['createdAt'] >= window_start) & (df_prs['createdAt'] < window_end)]
            merged_prs = df_prs[df_prs['mergedAt'].notna() & (df_prs['mergedAt'] >= window_start) & (df_prs['mergedAt'] < window_end)]
            closed_unmerged_prs = df_prs[df_prs['mergedAt'].isna() & df_prs['closedAt'].notna() & (df_prs['closedAt'] >= window_start) & (df_prs['closedAt'] < window_end)]
            open_pr_at_end = df_prs[(df_prs['createdAt'] < window_end) & (df_prs['closedAt'].isna() | (df_prs['closedAt'] >= window_end))]
            
            row[f"pull_requests_opened_M{i}"] = len(opened_prs)
            row[f"pull_requests_merged_M{i}"] = len(merged_prs)
            row[f"pull_requests_closed_unmerged_M{i}"] = len(closed_unmerged_prs)
            row[f"pull_requests_open_at_month_end_M{i}"] = len(open_pr_at_end)
        else:
            row[f"pull_requests_opened_M{i}"] = row[f"pull_requests_merged_M{i}"] = row[f"pull_requests_closed_unmerged_M{i}"] = row[f"pull_requests_open_at_month_end_M{i}"] = 0

    return row

# --- ORCHESTRATOR ---
async def process_package(package_name, session):
    print(f"Processing {package_name}...")
    
    # 1. Fetch metadata & stable versions
    meta_result = await get_all_stable_versions(package_name, session)
    if not meta_result: return []
    versions, owner, repo = meta_result
    
    # 2. Fetch entire timelines concurrently
    task_gh = asyncio.create_task(fetch_entire_github_timeline(owner, repo, session))
    task_ghsa = asyncio.create_task(fetch_all_ghsa_vulnerabilities(package_name, session))
    
    # Note: NPM Downloads is temporarily omitted to avoid the 18-month API cap per request. 
    # To include relative download windows, we'd loop NPM's API backwards year-by-year 
    # to package creation and slice the days exactly like the GH data.
    
    df_issues, df_prs = await task_gh
    df_vulns = await task_ghsa
    
    # 3. Depth Pivot: Create a row for EVERY stable version
    package_rows = []
    for v_data in versions:
        row = slice_version_metrics(df_issues, df_prs, df_vulns, v_data, package_name)
        package_rows.append(row)
        
    return package_rows

async def main():
    test_packages = ['lodash', 'express']
    print("Executing Depth Pivot ETL pipeline (Local Timeline Slicing)...")
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_package(pkg, session) for pkg in test_packages]
        results_lists = await asyncio.gather(*tasks)
        
        # Flatten the list of lists
        all_rows = [row for package_rows in results_lists for row in package_rows]
        
        df = pd.DataFrame(all_rows)
        # Handle zero-inflation for missing historical blocks
        df = df.fillna(0)
        df.to_csv("training_dataset_depth_pivot.csv", index=False)
        print(f"\nSuccess! Exported {len(all_rows)} version-specific rows to training_dataset_depth_pivot.csv")

if __name__ == "__main__":
    asyncio.run(main())