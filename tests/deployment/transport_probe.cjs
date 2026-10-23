// Same TLS-verifying probe runs on the host and in untouched base images.
const https = require('node:https');
const crypto = require('node:crypto');
const urls = [
  'https://registry.npmjs.org/-/ping',
  'https://registry.npmjs.org/typescript',
  'https://deb.debian.org/debian/dists/bookworm/main/binary-amd64/Packages.xz'
];
async function probe(url, mode) {
  return new Promise(resolve => {
    const hash = crypto.createHash('sha256');
    let bytes = 0;
    const options = { timeout: 30000, rejectUnauthorized: true };
    if (mode === 'chacha') options.ciphers = 'TLS_CHACHA20_POLY1305_SHA256:ECDHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-CHACHA20-POLY1305';
    if (mode === 'tls12') options.maxVersion = 'TLSv1.2';
    const request = https.get(url, options, response => {
      const cert = response.socket.getPeerCertificate();
      const tls = { authorized: response.socket.authorized, protocol: response.socket.getProtocol(), cipher: response.socket.getCipher()?.name,
        issuer: cert?.issuer?.CN, subject: cert?.subject?.CN };
      response.on('data', chunk => { bytes += chunk.length; hash.update(chunk); });
      response.on('end', () => resolve({url, mode, status: response.statusCode, bytes, sha256: hash.digest('hex'), ...tls}));
      response.on('error', error => resolve({url, mode, bytes, error: error.code, ...tls}));
    });
    request.on('timeout', () => request.destroy(Object.assign(new Error('timeout'), {code: 'TIMEOUT'})));
    request.on('error', error => resolve({url, mode, bytes, error: error.code}));
  });
}
(async () => {
  console.log(JSON.stringify({node: process.version, openssl: process.versions.openssl, platform: process.platform, arch: process.arch}));
  let failed = false;
  for (const mode of (process.argv.includes('--all-ciphers') ? ['default', 'chacha', 'tls12'] : ['default'])) {
    for (const url of urls) {
      const result = await probe(url, mode);
      console.log(JSON.stringify(result));
      failed ||= Boolean(result.error) || result.authorized !== true || result.status !== 200;
    }
  }
  process.exitCode = failed ? 1 : 0;
})();
