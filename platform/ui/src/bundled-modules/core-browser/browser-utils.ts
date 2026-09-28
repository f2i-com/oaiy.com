/**
 * Browser utilities for OAIY workflow browser nodes (Browser-compatible version)
 * Uses DOMParser instead of cheerio for HTML parsing
 */

// ============================================
// Browser Profile Definitions
// ============================================

export interface BrowserProfileConfig {
  name: string;
  userAgent: string;
  headers: Record<string, string>;
}

export const BROWSER_PROFILES: Record<string, BrowserProfileConfig> = {
  chrome_windows: {
    name: 'Chrome (Windows)',
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    headers: {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9',
    }
  },
  chrome_mac: {
    name: 'Chrome (macOS)',
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    headers: {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9',
    }
  },
  firefox_windows: {
    name: 'Firefox (Windows)',
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
    headers: {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.5',
    }
  },
  firefox_mac: {
    name: 'Firefox (macOS)',
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0',
    headers: {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.5',
    }
  },
  safari_mac: {
    name: 'Safari (macOS)',
    userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15',
    headers: {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9',
    }
  },
  edge_windows: {
    name: 'Edge (Windows)',
    userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0',
    headers: {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9',
    }
  },
  mobile_ios: {
    name: 'Safari (iOS)',
    userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',
    headers: {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9',
    }
  },
  mobile_android: {
    name: 'Chrome (Android)',
    userAgent: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Mobile Safari/537.36',
    headers: {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9',
    }
  },
  custom: {
    name: 'Custom',
    userAgent: '',
    headers: {}
  }
};

// ============================================
// Cookie Management
// ============================================

export interface CookieEntry {
  name: string;
  value: string;
  domain: string;
  path: string;
  expires?: Date;
  httpOnly: boolean;
  secure: boolean;
  sameSite?: 'Strict' | 'Lax' | 'None';
}

export class CookieJar {
  private cookies: Map<string, Map<string, CookieEntry>> = new Map();

  setCookie(setCookieHeader: string, requestDomain: string): void {
    const parts = setCookieHeader.split(';').map(p => p.trim());
    if (parts.length === 0) return;

    const [nameValue, ...attributes] = parts;
    const eqIndex = nameValue.indexOf('=');
    if (eqIndex === -1) return;

    const name = nameValue.substring(0, eqIndex).trim();
    const value = nameValue.substring(eqIndex + 1).trim();

    const cookie: CookieEntry = {
      name,
      value,
      domain: requestDomain,
      path: '/',
      httpOnly: false,
      secure: false
    };

    for (const attr of attributes) {
      const [attrName, attrValue] = attr.split('=').map(s => s.trim());
      const lowerAttrName = attrName.toLowerCase();

      switch (lowerAttrName) {
        case 'domain':
          if (attrValue) {
            cookie.domain = attrValue.startsWith('.') ? attrValue.substring(1) : attrValue;
          }
          break;
        case 'path':
          if (attrValue) cookie.path = attrValue;
          break;
        case 'expires':
          if (attrValue) {
            const date = new Date(attrValue);
            if (!isNaN(date.getTime())) {
              cookie.expires = date;
            }
          }
          break;
        case 'max-age':
          if (attrValue) {
            const seconds = parseInt(attrValue, 10);
            if (!isNaN(seconds)) {
              cookie.expires = new Date(Date.now() + seconds * 1000);
            }
          }
          break;
        case 'httponly':
          cookie.httpOnly = true;
          break;
        case 'secure':
          cookie.secure = true;
          break;
        case 'samesite':
          if (attrValue) {
            const sameSite = attrValue.toLowerCase();
            if (sameSite === 'strict') cookie.sameSite = 'Strict';
            else if (sameSite === 'lax') cookie.sameSite = 'Lax';
            else if (sameSite === 'none') cookie.sameSite = 'None';
          }
          break;
      }
    }

    if (!this.cookies.has(cookie.domain)) {
      this.cookies.set(cookie.domain, new Map());
    }
    this.cookies.get(cookie.domain)!.set(cookie.name, cookie);
  }

  getCookieHeader(url: string): string {
    try {
      const parsedUrl = new URL(url);
      const domain = parsedUrl.hostname;
      const path = parsedUrl.pathname;
      const isSecure = parsedUrl.protocol === 'https:';
      const now = new Date();

      const matchingCookies: CookieEntry[] = [];

      for (const [cookieDomain, domainCookies] of this.cookies) {
        if (domain === cookieDomain || domain.endsWith('.' + cookieDomain)) {
          for (const cookie of domainCookies.values()) {
            if (cookie.expires && cookie.expires < now) continue;
            if (!path.startsWith(cookie.path)) continue;
            if (cookie.secure && !isSecure) continue;
            matchingCookies.push(cookie);
          }
        }
      }

      return matchingCookies.map(c => `${c.name}=${c.value}`).join('; ');
    } catch {
      return '';
    }
  }

  toJSON(): string {
    const data: Record<string, CookieEntry[]> = {};
    for (const [domain, domainCookies] of this.cookies) {
      data[domain] = Array.from(domainCookies.values());
    }
    return JSON.stringify(data);
  }

  static fromJSON(json: string): CookieJar {
    const jar = new CookieJar();
    try {
      const data = JSON.parse(json) as Record<string, CookieEntry[]>;
      for (const [domain, cookies] of Object.entries(data)) {
        const domainMap = new Map<string, CookieEntry>();
        for (const cookie of cookies) {
          if (cookie.expires) {
            cookie.expires = new Date(cookie.expires);
          }
          domainMap.set(cookie.name, cookie);
        }
        jar.cookies.set(domain, domainMap);
      }
    } catch {
      // Return empty jar on parse error
    }
    return jar;
  }

  clear(): void {
    this.cookies.clear();
  }

  get size(): number {
    let count = 0;
    for (const domainCookies of this.cookies.values()) {
      count += domainCookies.size;
    }
    return count;
  }
}

// ============================================
// HTML Extraction Utilities (using DOMParser)
// ============================================

export interface FormField {
  name: string;
  type: string;
  value: string;
  id?: string;
  required?: boolean;
}

export interface ExtractedForm {
  selector: string;
  action: string;
  method: string;
  fields: FormField[];
}

export interface ExtractedLink {
  text: string;
  href: string;
}

function parseHTML(html: string): Document {
  const parser = new DOMParser();
  return parser.parseFromString(html, 'text/html');
}

export function extractBySelector(
  html: string,
  selector: string,
  target: 'text' | 'html' | 'attribute' = 'text',
  attributeName?: string
): string[] {
  const doc = parseHTML(html);
  const results: string[] = [];

  try {
    const elements = doc.querySelectorAll(selector);
    elements.forEach(element => {
      let value: string;

      switch (target) {
        case 'text':
          value = (element.textContent || '').trim();
          break;
        case 'html':
          value = element.innerHTML || '';
          break;
        case 'attribute':
          value = attributeName ? (element.getAttribute(attributeName) || '') : '';
          break;
        default:
          value = (element.textContent || '').trim();
      }

      if (value) {
        results.push(value);
      }
    });
  } catch (e) {
    console.error('Selector error:', e);
  }

  return results;
}

export function extractAllLinks(html: string, baseUrl?: string): ExtractedLink[] {
  const doc = parseHTML(html);
  const links: ExtractedLink[] = [];

  doc.querySelectorAll('a[href]').forEach(element => {
    const href = element.getAttribute('href') || '';
    const text = (element.textContent || '').trim();

    if (href) {
      let resolvedHref = href;
      if (baseUrl && !href.startsWith('http://') && !href.startsWith('https://') && !href.startsWith('//')) {
        resolvedHref = resolveUrl(baseUrl, href);
      }
      links.push({ text, href: resolvedHref });
    }
  });

  return links;
}

export function extractAllForms(html: string): ExtractedForm[] {
  const doc = parseHTML(html);
  const forms: ExtractedForm[] = [];

  doc.querySelectorAll('form').forEach((form, index) => {
    let selector = 'form';
    const id = form.getAttribute('id');
    const name = form.getAttribute('name');
    const action = form.getAttribute('action');

    if (id) {
      selector = `form#${id}`;
    } else if (name) {
      selector = `form[name="${name}"]`;
    } else if (action) {
      selector = `form[action="${action}"]`;
    } else {
      selector = `form:nth-of-type(${index + 1})`;
    }

    const fields = extractFormFieldsFromElement(form);

    forms.push({
      selector,
      action: form.getAttribute('action') || '',
      method: (form.getAttribute('method') || 'GET').toUpperCase(),
      fields
    });
  });

  return forms;
}

export function extractFormFields(html: string, formSelector: string): FormField[] {
  const doc = parseHTML(html);
  const form = doc.querySelector(formSelector);

  if (!form) {
    return [];
  }

  return extractFormFieldsFromElement(form);
}

function extractFormFieldsFromElement(form: Element): FormField[] {
  const fields: FormField[] = [];

  // Input elements
  form.querySelectorAll('input').forEach(el => {
    const type = el.getAttribute('type') || 'text';
    const name = el.getAttribute('name');

    if (['button', 'submit', 'reset', 'image'].includes(type) && !name) {
      return;
    }

    if (name) {
      fields.push({
        name,
        type,
        value: el.getAttribute('value') || '',
        id: el.getAttribute('id') || undefined,
        required: el.hasAttribute('required')
      });
    }
  });

  // Textarea elements
  form.querySelectorAll('textarea').forEach(el => {
    const name = el.getAttribute('name');
    if (name) {
      fields.push({
        name,
        type: 'textarea',
        value: el.textContent || '',
        id: el.getAttribute('id') || undefined,
        required: el.hasAttribute('required')
      });
    }
  });

  // Select elements
  form.querySelectorAll('select').forEach(el => {
    const name = el.getAttribute('name');
    if (name) {
      const selected = el.querySelector('option[selected]');
      const value = selected ? (selected.getAttribute('value') || selected.textContent || '') : '';

      fields.push({
        name,
        type: 'select',
        value,
        id: el.getAttribute('id') || undefined,
        required: el.hasAttribute('required')
      });
    }
  });

  return fields;
}

export function buildFormBody(
  fields: FormField[],
  values: Record<string, string>,
  format: 'urlencoded' | 'json' | 'multipart' = 'urlencoded',
  includeHidden: boolean = true
): string {
  const mergedValues: Record<string, string> = {};

  for (const field of fields) {
    if (!includeHidden && field.type === 'hidden' && !(field.name in values)) {
      continue;
    }

    if (field.name in values) {
      mergedValues[field.name] = values[field.name];
    } else if (field.value) {
      mergedValues[field.name] = field.value;
    }
  }

  for (const [key, value] of Object.entries(values)) {
    if (!(key in mergedValues)) {
      mergedValues[key] = value;
    }
  }

  switch (format) {
    case 'json':
      return JSON.stringify(mergedValues);

    case 'multipart':
      return JSON.stringify(mergedValues);

    case 'urlencoded':
    default:
      return Object.entries(mergedValues)
        .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
        .join('&');
  }
}

// ============================================
// URL Utilities
// ============================================

export function resolveUrl(base: string, relative: string): string {
  try {
    return new URL(relative, base).href;
  } catch {
    return relative;
  }
}

export function extractDomain(url: string): string {
  try {
    return new URL(url).hostname;
  } catch {
    return '';
  }
}

// ============================================
// Regex Extraction
// ============================================

// Maximum content size for regex operations to prevent ReDoS attacks
// 1MB is generous for text extraction but prevents massive CPU hangs
const MAX_REGEX_CONTENT_SIZE = 1_000_000;

export function extractByRegex(
  content: string,
  pattern: string,
  flags: string = 'g'
): string[] {
  try {
    // ReDoS Protection: Limit input size to prevent catastrophic backtracking
    // A malicious regex like (a+)+$ on large input can freeze the UI indefinitely
    const safeContent = content.length > MAX_REGEX_CONTENT_SIZE
      ? content.substring(0, MAX_REGEX_CONTENT_SIZE)
      : content;

    const regex = new RegExp(pattern, flags);
    const results: string[] = [];

    if (flags.includes('g')) {
      let match;
      // Additional safety: limit number of matches to prevent infinite loops
      let matchCount = 0;
      const maxMatches = 10000;
      while ((match = regex.exec(safeContent)) !== null && matchCount < maxMatches) {
        results.push(match[1] !== undefined ? match[1] : match[0]);
        matchCount++;
        // Prevent infinite loop on zero-width matches
        if (match.index === regex.lastIndex) {
          regex.lastIndex++;
        }
      }
    } else {
      const match = regex.exec(safeContent);
      if (match) {
        results.push(match[1] !== undefined ? match[1] : match[0]);
      }
    }

    return results;
  } catch {
    return [];
  }
}
