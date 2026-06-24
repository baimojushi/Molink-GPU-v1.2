---
name: github-push-with-pat
description: Push a new git repo to GitHub using PAT when embedding token in URL fails due to credential prompts
source: auto-skill
extracted_at: '2026-06-22T11:55:20.011Z'
---

## Problem

Embedding PAT directly in the remote URL (e.g. `https://<TOKEN>@github.com/user/repo.git`) can still trigger an interactive password prompt that causes tools to time out, because git's credential system does not extract the password from the URL.

## Solution

Use an inline credential helper to supply username and PAT without any interactive prompt:

```bash
cd /path/to/repo
git remote set-url origin https://github.com/<user>/<repo>.git
git -c credential.helper='!f() { echo "username=<user>"; echo "password=<PAT>"; }; f' push -u origin main
```

## Why this works

The `-c credential.helper=...` flag injects a one-shot credential provider directly into the push command. Git calls the helper function, receives username+password, and authenticates non-interactively. This bypasses any system credential store or URL-embedded token parsing.

## Validation steps

1. Verify token is valid before pushing:
```bash
curl -s -H "Authorization: token <PAT>" https://api.github.com/user | python3 -c "import sys,json; print(json.load(sys.stdin)['login'])"
```

2. Verify repo exists and default branch:
```bash
curl -s -H "Authorization: token <PAT>" https://api.github.com/repos/<user>/<repo> | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('default_branch','main'))"
```

3. After push, confirm remote SHA matches local:
```bash
curl -s -H "Authorization: token <PAT>" https://api.github.com/repos/<user>/<repo>/git/refs/heads/main | python3 -c "import sys,json; print(json.load(sys.stdin)['object']['sha'][:7])"
```

## Network connectivity diagnosis

On AutoDL machines in China, direct GitHub access often fails with:
- `GnuTLS recv error (-110): The TLS connection was non-properly terminated`
- `Failed to connect to github.com port 443 after 129934 ms: Connection timed out`

**Pre-push connectivity check:**
```bash
curl -vI --connect-timeout 10 https://github.com 2>&1 | grep -E "Trying|connect to|Closing"
```

If the above shows connection timeout to `20.205.243.166:443`, GitHub is blocked.

**Workarounds:**

1. **If a local proxy is available** (e.g. clash, v2ray on the machine):
```bash
export https_proxy=http://127.0.0.1:<proxy_port>
git -c credential.helper='!f() { echo "username=<user>"; echo "password=<PAT>"; }; f' push origin main
```

2. **SSH via proxy command** (if SSH key is configured and proxy supports it):
```bash
git config core.sshCommand "ssh -o ProxyCommand='nc -X connect -x <proxy_host>:<proxy_port> %h %p'"
git remote set-url origin git@github.com:<user>/<repo>.git
git push origin main
```

3. **Defer push** — commit locally and push when network allows. The local commit is safe; push can be retried later.
