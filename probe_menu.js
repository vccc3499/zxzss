const fs=require('fs'); const path=require('path'); const {chromium}=require('playwright');
(async()=>{
 const ctx=await chromium.launchPersistentContext(path.join(process.cwd(),'avito_profile'),{headless:false,channel:'msedge',viewport:{width:1400,height:900}});
 const p=ctx.pages()[0]||await ctx.newPage();
 const cp=path.join(process.cwd(),'avito_cookies.json');
 if(fs.existsSync(cp)){const raw=JSON.parse(fs.readFileSync(cp,'utf8').replace(/^\uFEFF/,''));const c=raw.filter(x=>x.name&&x.value&&x.domain).map(x=>({name:String(x.name),value:String(x.value),domain:String(x.domain),path:x.path||'/',httpOnly:!!x.httpOnly,secure:!!x.secure,sameSite:(String(x.sameSite||'').toLowerCase()==='lax'?'Lax':String(x.sameSite||'').toLowerCase()==='strict'?'Strict':(String(x.sameSite||'').toLowerCase()==='none'||String(x.sameSite||'').toLowerCase()==='no_restriction')?'None':undefined),expires:typeof x.expirationDate==='number'?Math.floor(x.expirationDate):undefined}));await ctx.addCookies(c);} 
 await p.goto('https://www.avito.ru/profile/messenger',{waitUntil:'domcontentloaded', timeout:90000}); await p.waitForTimeout(2500);
 const href=await p.locator('a[href*="/profile/messenger/channel/"]').nth(1).getAttribute('href');
 console.log('href',href);
 if(href){await p.goto(href.startsWith('http')?href:`https://www.avito.ru${href}`,{waitUntil:'domcontentloaded',timeout:90000});await p.waitForTimeout(2000);} 
 console.log('url',p.url());
 console.log('label count',await p.getByLabel('Открыть меню').count());
 console.log('role button ellipsis',await p.getByRole('button',{name:/меню|ещ|more/i}).count());
 await ctx.close();
})();
