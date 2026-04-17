#!/bin/bash
set -e
cd "$(dirname "$0")"

# Syntax check all JS
echo "🔍 Checking JS syntax..."
for f in js/*.js; do
  node -c "$f" || { echo "❌ Syntax error in $f"; exit 1; }
done

# Check inline scripts in HTML
python3 -c "
import re, subprocess, sys
for html in ['result.html','analyze.html','upload.html','index.html']:
    try:
        content = open(html).read()
    except: continue
    scripts = re.findall(r'<script>(.*?)</script>', content, re.DOTALL)
    for i, s in enumerate(scripts):
        with open(f'/tmp/_check_{html}_{i}.js','w') as f: f.write(s)
        r = subprocess.run(['node','-c',f'/tmp/_check_{html}_{i}.js'], capture_output=True)
        if r.returncode != 0:
            print(f'❌ Syntax error in {html} script block {i}')
            print(r.stderr.decode()[:200])
            sys.exit(1)
print('✅ All inline scripts OK')
"

# Git commit & push
git add -A
git diff --cached --quiet && echo "No changes to commit" && exit 0
echo -n "Commit message: "
read msg
git commit -m "$msg"
git push origin main

# Deploy
echo "🚀 Deploying to Cloudflare Pages..."
source ~/.bashrc
wrangler pages deploy . --project-name danceflow --branch main

echo "✅ Deployed!"
