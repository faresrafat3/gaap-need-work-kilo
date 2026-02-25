import requests
import re
import urllib.parse

def deep_js_scan(url):
    print(f"🕵️ Deep Scanning JS (No-BS4) at: {url}")
    try:
        response = requests.get(url, timeout=10)
        # Find script tags with src attribute using Regex
        scripts = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', response.text)
        
        findings = []
        for script_url in scripts:
            if not script_url.endswith('.js'): continue
            
            full_url = urllib.parse.urljoin(url, script_url)
            try:
                js_content = requests.get(full_url, timeout=5).text
                
                # High-Value Patterns
                aws_key = re.findall(r'AKIA[0-9A-Z]{16}', js_content)
                firebase = re.findall(r'[a-z0-9-]+\.firebaseio\.com', js_content)
                # Look for potential sensitive keywords
                secrets = re.findall(r'(?i)(api_key|client_secret|access_token|db_password|db_user)["\']\s*:\s*["\']([^"\']+)["\']', js_content)
                
                if aws_key: findings.append(f"[!] AWS KEY FOUND in {full_url}: {aws_key}")
                if firebase: findings.append(f"[!] FIREBASE DB FOUND in {full_url}: {firebase}")
                if secrets: findings.append(f"[!] POTENTIAL SECRETS FOUND in {full_url}: {secrets[:2]}...")
            except:
                continue
            
        return findings
    except Exception as e:
        return [f"Error scanning {url}: {e}"]

def main():
    targets = ["https://indeed.com", "https://employer.indeed.com", "https://apply.indeed.com"]
    for t in targets:
        results = deep_js_scan(t)
        if results:
            for r in results: print(r)
        else:
            print(f"[-] No secrets found in {t}")
        print("-" * 30)

if __name__ == "__main__":
    main()
