const fs=require('fs'); const path=require('path'); const {chromium}=require('playwright');
(async()=>{
 const ctx=await chromium.launchPersistentContext(path.join(process.cwd(),'avito_profile'),{headless:false,channel:'msedge',viewport:{width:1400,height:900}});
 const p=ctx.pages()[0]||await ctx.newPage();
 const cp=path.join(process.cwd(),'avito_cookies.json');
 if(fs.existsSync(cp)){const raw=JSON.parse(fs.readFileSync(cp,'utf8').replace(/^\uFEFF/,''));const c=raw.filter(x=>x.name&&x.value&&x.domain).map(x=>({name:String(x.name),value:String(x.value),domain:String(x.domain),path:x.path||'/',httpOnly:!!x.httpOnly,secure:!!x.secure,sameSite:(String(x.sameSite||'').toLowerCase()==='lax'?'Lax':String(x.sameSite||'').toLowerCase()==='strict'?'Strict':(String(x.sameSite||'').toLowerCase()==='none'||String(x.sameSite||'').toLowerCase()==='no_restriction')?'None':undefined),expires:typeof x.expirationDate==='number'?Math.floor(x.expirationDate):undefined}));await ctx.addCookies(c);} 
 await p.goto('https://www.avito.ru/profile/messenger',{waitUntil:'domcontentloaded', timeout:90000}); await p.waitForTimeout(2500);
 const first = p.locator('a[href*="/profile/messenger/channel/"]').first();
 console.log('firstCount', await first.count());
 if(await first.count()){ await first.click({timeout:8000}); await p.waitForTimeout(2000); }
 const om = p.getByText('Открыть меню').first();
 console.log('openMenuCount', await om.count());
 if(await om.count()){ await om.click({timeout:6000}); await p.waitForTimeout(1000); }
 const texts = await p.evaluate(()=>Array.from(document.querySelectorAll('button,[role="button"],a,[role="menuitem"],li,div')).map(el=>(el.innerText||el.getAttribute('aria-label')||'').trim()).filter(Boolean).filter(t=>/удал|очист|архив|блок|жалоб|меню|диалог|чат/i.test(t)).slice(0,200));
 console.log(JSON.stringify(texts,null,2));
 await ctx.close();
})();
