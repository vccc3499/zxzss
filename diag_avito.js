const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const context = await chromium.launchPersistentContext(path.join(process.cwd(),'avito_profile'), { headless: true, channel: 'msedge' });
  const page = context.pages()[0] || await context.newPage();

  const cookiesPath = path.join(process.cwd(), 'avito_cookies.json');
  if (fs.existsSync(cookiesPath)) {
    const raw = JSON.parse(fs.readFileSync(cookiesPath, 'utf8').replace(/^\uFEFF/, ''));
    const cookies = raw.filter(c=>c.name&&c.value&&c.domain).map(c=>({
      name:String(c.name), value:String(c.value), domain:String(c.domain), path:c.path||'/', httpOnly:!!c.httpOnly, secure:!!c.secure,
      sameSite: (String(c.sameSite||'').toLowerCase()==='lax'?'Lax':String(c.sameSite||'').toLowerCase()==='strict'?'Strict':(String(c.sameSite||'').toLowerCase()==='none'||String(c.sameSite||'').toLowerCase()==='no_restriction')?'None':undefined),
      expires: typeof c.expirationDate==='number'?Math.floor(c.expirationDate):undefined,
    }));
    await context.addCookies(cookies);
  }

  const targets = ['https://www.avito.ru/profile/messenger','https://www.avito.ru/profile','https://www.avito.ru/'];
  for (const t of targets) {
    try {
      await page.goto(t, { waitUntil: 'domcontentloaded', timeout: 30000 });
      await page.waitForTimeout(1200);
      const data = await page.evaluate(() => {
        const title = document.title;
        const url = location.href;
        const body = (document.body?.innerText || '').slice(0, 800);
        const buttons = Array.from(document.querySelectorAll('button,[role="button"],a')).map(el => (el.innerText||el.getAttribute('aria-label')||'').trim()).filter(Boolean).slice(0, 80);
        const links = Array.from(document.querySelectorAll('a[href]')).map(a=>a.href).filter(h=>/messenger|messages|profile/.test(h)).slice(0,80);
        return {title,url,body,buttons,links};
      });
      console.log('===');
      console.log(JSON.stringify({target:t, ...data}, null, 2));
    } catch (e) {
      console.log('ERR', t, e.message);
    }
  }

  await context.close();
})();
