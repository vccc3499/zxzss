const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

const ROOT = process.cwd();
const PROFILE_DIR = path.join(ROOT, 'avito_profile');
const COOKIES_PATH = path.join(ROOT, 'avito_cookies.json');

function toSameSite(value) {
  const s = String(value || '').toLowerCase();
  if (s === 'lax') return 'Lax';
  if (s === 'strict') return 'Strict';
  if (s === 'none' || s === 'no_restriction') return 'None';
  return undefined;
}

async function applyCookiesFromFile(context) {
  if (!fs.existsSync(COOKIES_PATH)) return 0;
  const raw = JSON.parse(fs.readFileSync(COOKIES_PATH, 'utf8').replace(/^\uFEFF/, ''));
  if (!Array.isArray(raw) || !raw.length) return 0;

  const converted = raw
    .filter((c) => c && c.name && c.value && c.domain)
    .map((c) => {
      const cookie = {
        name: String(c.name),
        value: String(c.value),
        domain: String(c.domain),
        path: c.path || '/',
        httpOnly: Boolean(c.httpOnly),
        secure: Boolean(c.secure),
      };
      const sameSite = toSameSite(c.sameSite);
      if (sameSite) cookie.sameSite = sameSite;
      if (typeof c.expirationDate === 'number' && Number.isFinite(c.expirationDate)) {
        cookie.expires = Math.floor(c.expirationDate);
      }
      return cookie;
    });

  await context.addCookies(converted);
  return converted.length;
}

async function gotoMessenger(page) {
  const urls = [
    'https://www.avito.ru/profile/messenger',
    'https://www.avito.ru/profile',
    'https://www.avito.ru/',
  ];

  for (const url of urls) {
    try {
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 30000 });
      await page.waitForTimeout(1500);
      const body = (await page.textContent('body')) || '';
      if (!body.includes('Такой страницы не существует')) return;
    } catch {}
  }
}

async function openFirstChat(page) {
  const chatSelectors = [
    'a[href*="/messenger/"]',
    'a[href*="/messages/"]',
    '[data-marker*="chat"] a',
    '[class*="chat"] a',
  ];

  for (const sel of chatSelectors) {
    const loc = page.locator(sel).first();
    if (await loc.count()) {
      await loc.click({ timeout: 5000 });
      await page.waitForTimeout(1000);
      return true;
    }
  }
  return false;
}

async function clickAny(page, patterns) {
  for (const p of patterns) {
    const byRole = page.getByRole('button', { name: p }).first();
    if (await byRole.count()) {
      await byRole.click({ timeout: 3000 });
      return true;
    }

    const txt = page.getByText(p).first();
    if (await txt.count()) {
      await txt.click({ timeout: 3000 });
      return true;
    }

    const menu = page.locator(`[aria-label*="${p}"]`).first();
    if (await menu.count()) {
      await menu.click({ timeout: 3000 });
      return true;
    }
  }
  return false;
}

async function deleteCurrentChat(page) {
  const menuOpened = await clickAny(page, [/ещ[её]/i, /действия/i, /меню/i, /more/i])
    || await clickAny(page, [/\.\.\./, /⋯/]);

  if (!menuOpened) {
    // Try context menu on header/chat area
    const header = page.locator('header, [class*="header"], [data-marker*="chat-title"]').first();
    if (await header.count()) {
      await header.click({ button: 'right', timeout: 2000 });
      await page.waitForTimeout(500);
    }
  }

  const deleteClicked = await clickAny(page, [
    /удалить чат/i,
    /удалить диалог/i,
    /удалить переписк/i,
    /удалить/i,
    /delete/i,
  ]);
  if (!deleteClicked) return false;

  await page.waitForTimeout(500);

  const confirmed = await clickAny(page, [
    /^удалить$/i,
    /подтвердить/i,
    /да/i,
    /delete/i,
  ]);

  await page.waitForTimeout(1200);
  return confirmed || true;
}

(async () => {
  const context = await chromium.launchPersistentContext(PROFILE_DIR, {
    headless: false,
    channel: 'msedge',
    viewport: { width: 1440, height: 900 },
  });

  const imported = await applyCookiesFromFile(context);
  const page = context.pages()[0] || await context.newPage();
  await gotoMessenger(page);

  console.log(`cookies imported: ${imported}`);

  let deleted = 0;
  for (let i = 0; i < 300; i += 1) {
    await gotoMessenger(page);
    const hasChat = await openFirstChat(page);
    if (!hasChat) break;

    const ok = await deleteCurrentChat(page);
    if (!ok) {
      console.log('delete action not found, stopping');
      break;
    }

    deleted += 1;
    console.log(`deleted: ${deleted}`);
    await page.waitForTimeout(1200);
  }

  console.log(`done, total deleted: ${deleted}`);
  await context.close();
})();
