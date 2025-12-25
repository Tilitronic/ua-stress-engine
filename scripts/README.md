# Git LFS Management Scripts

Automated tools to manage Git LFS storage and keep within GitHub's 1GB free tier limit.

## Scripts

### 1. `lfs_storage_monitor.py`

Monitors current LFS storage usage and warns when approaching limits.

**Usage:**

```bash
python scripts/lfs_storage_monitor.py
```

**Output:**

```
📊 Git LFS Storage Check:
   Used: 391.0 MB / 1024 MB (38.2%)
   ✅ Storage healthy
```

**Alerts:**

- ✅ < 75%: Healthy
- ⚡ 75-90%: Warning
- ⚠️ 90-100%: Urgent - cleanup needed
- ❌ > 100%: Error - push will fail

### 2. `lfs_cleanup.py`

Analyzes LFS files and shows which old versions can be removed.

**Usage:**

```bash
# Dry run - see what would be removed (safe)
python scripts/lfs_cleanup.py --dry-run

# Keep only 3 versions per file (default)
python scripts/lfs_cleanup.py --dry-run --keep-versions 3

# Keep only latest version
python scripts/lfs_cleanup.py --dry-run --keep-versions 1
```

**Example Output:**

```
📁 Found 2 unique LFS file(s):

  src/nlp/stress_service/stress.lmdb/data.mdb
    Total versions: 2
    ✓ Keeping all 2 version(s) (within policy)

  src/nlp/stress_service/stress.lmdb/lock.mdb
    Total versions: 4
    ✓ Keeping 2 version(s):
      - 2bc8de3142... (8.2 KB) from 2025-12-21
      - a74ed7d9e6... (8.2 KB) from 2025-12-21
    ✗ Removing 2 old version(s):
      - 1ace413300... (8.2 KB) from 2025-12-21
      - e535377d15... (8.2 KB) from 2025-12-21
```

## Storage Policy

**Default:** Keep 3 most recent versions of each LFS file

**Why 3 versions?**

- Current version (HEAD)
- Previous version (rollback capability)
- One more for safety (in case of issues)

**When to cleanup:**

- Storage > 75%: Consider reducing to 2 versions
- Storage > 90%: Reduce to 1-2 versions
- Storage > 100%: Immediate cleanup required

## Automatic Monitoring

### Option 1: Pre-Push Hook (Recommended)

Add to `.git/hooks/pre-push`:

```bash
#!/bin/sh
python scripts/lfs_storage_monitor.py || exit 1
```

This checks storage before every push and warns if approaching limits.

### Option 2: GitHub Actions

Create `.github/workflows/lfs-monitor.yml`:

```yaml
name: LFS Storage Monitor
on: [push]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          lfs: true
      - run: python scripts/lfs_storage_monitor.py
```

## Current Status

**LFS Files Tracked:**

- `*.mdb` (LMDB database files)

**Current Usage:**

- 391 MB / 1024 MB (38.2%)
- 2 data.mdb versions (175 MB + 216 MB)
- 4 lock.mdb versions (~32 KB total)

**Available Space:** 633 MB (62%)

## Manual Cleanup (Advanced)

If automated scripts aren't sufficient, manually remove old versions:

### 1. Identify commits with old versions

```bash
git log --all -- src/nlp/stress_service/stress.lmdb/data.mdb
```

### 2. Rewrite history (DANGEROUS)

```bash
# Use git-filter-repo (recommended)
pip install git-filter-repo
git filter-repo --path-glob '*.mdb' --invert-paths

# Or use BFG Repo-Cleaner
java -jar bfg.jar --delete-files '*.mdb'
```

### 3. Force push

```bash
git push origin main --force-with-lease
```

### 4. Cleanup local LFS cache

```bash
git lfs prune --verify-remote
```

⚠️ **Warning:** This rewrites history. All collaborators must re-clone!

## Troubleshooting

### "Push rejected - file too large"

1. Run: `python scripts/lfs_cleanup.py --dry-run`
2. Identify large files
3. Ensure `.gitattributes` tracks `*.mdb`
4. Run: `git lfs migrate import --include="*.mdb" --everything`

### "Out of LFS storage"

1. Check usage: `python scripts/lfs_storage_monitor.py`
2. Analyze versions: `python scripts/lfs_cleanup.py --dry-run`
3. Keep fewer versions or upgrade GitHub plan

### "LFS files not downloading"

```bash
# Download all LFS files
git lfs pull

# Check what's tracked
git lfs ls-files
```

## GitHub LFS Pricing

**Free Tier:**

- 1 GB storage
- 1 GB/month bandwidth

**Paid ($5/month per pack):**

- +50 GB storage
- +50 GB/month bandwidth

**Current project:** Well within free tier ✅

## See Also

- [Git LFS Documentation](https://git-lfs.com/)
- [GitHub LFS Guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [git-filter-repo](https://github.com/newren/git-filter-repo)
