/**
 * Warn (once) if a privileged OAIY_SERVER_TOKEN would be sent as a cleartext Bearer
 * header to a NON-loopback host over plain HTTP. oaiy-server binds loopback, so a
 * remote OAIY_SERVER_URL only works behind a tunnel/proxy — but if an operator points
 * it at a plain `http://host` the token (which is equivalent to RCE on the server)
 * is trivially sniffable. Loopback http and any https stay silent; warn-only so
 * existing 127.0.0.1 setups are unaffected. Goes to stderr (never the result channel).
 */
let warned = false;
export function warnIfPlaintextRemoteToken(url: string, hasToken: boolean): void {
  if (warned || !hasToken) return;
  let u: URL;
  try {
    u = new URL(url);
  } catch {
    return;
  }
  if (u.protocol !== 'http:') return; // https is fine
  const host = u.hostname.replace(/^\[|\]$/g, '');
  if (host === 'localhost' || host === '127.0.0.1' || host === '::1') return; // loopback http is fine
  warned = true;
  process.stderr.write(
    `[oaiy] WARNING: OAIY_SERVER_TOKEN is being sent as a cleartext Bearer header to ${u.origin} ` +
      `over plain HTTP. Use https:// or an SSH tunnel for a remote oaiy-server, or unset the token.\n`,
  );
}
