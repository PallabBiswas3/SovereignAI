"""TLS-verifying Python probe usable in WSL without installing dependencies."""
import hashlib
import json
import ssl
import urllib.request

failed = False
print(json.dumps({"openssl": ssl.OPENSSL_VERSION}))
for url in ["https://registry.npmjs.org/typescript", "https://deb.debian.org/debian/dists/bookworm/main/binary-amd64/Packages.xz"]:
    size = 0
    sha = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            while chunk := response.read(65536):
                size += len(chunk)
                sha.update(chunk)
        print(json.dumps({"url": url, "bytes": size, "sha256": sha.hexdigest(), "verified_tls": True}))
    except Exception as exc:
        failed = True
        print(json.dumps({"url": url, "bytes": size, "error": str(exc)}))
raise SystemExit(1 if failed else 0)
