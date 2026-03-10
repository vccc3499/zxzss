const fs=require('fs'); const path=require('path'); const {chromium}=require('playwright');
(async()=>{
 const ctx=await chromium.launchPersistentContext(path.join(process.cwd(),'avito_profile'),{headless:false,channel:'msedge',viewport:{width:1400,height:900}});
 const p=ctx.pages()[0]||await ctx.newPage();
 const cp=path.join(process.cwd(),'avito_cookies.json');
 if(fs.existsSync(cp)){const raw=JSON.parse(fs.readFileSync(cp,'utf8').replace(/^\uFEFF/,''));const c=raw.filter(x=>x.name&&x.value&&x.domain).map(x=>({name:String(x.name),value:String(x.value),domain:String(x.domain),path:x.path||'/',httpOnly:!!x.httpOnly,secure:!!x.secure,sameSite:(String(x.sameSite||'').toLowerCase()==='lax'?'Lax':String(x.sameSite||'').toLowerCase()==='strict'?'Strict':(String(x.sameSite||'').toLowerCase()==='none'||String(x.sameSite||'').toLowerCase()==='no_restriction')?'None':undefined),expires:typeof x.expirationDate==='number'?Math.floor(x.expirationDate):undefined}));await ctx.addCookies(c);} 
 await p.goto('https://www.avito.ru/profile/messenger',{waitUntil:'domcontentloaded', timeout:90000}); await p.waitForTimeout(3000);
 const data=await p.evaluate(()=>({url:location.href,title:document.title,hasOpenMenu:!!Array.from(document.querySelectorAll('button,[role="button"],a')).find(el=>((el.innerText||'')+(el.getAttribute('aria-label')||'')).includes('Открыть меню')), menuText:Array.from(document.querySelectorAll('button,[role="button"],a')).map(el=>(el.innerText||el.getAttribute('aria-label')||'').trim()).filter(Boolean).filter(t=>/меню|удалить|чат|диалог/i.test(t)).slice(0,80)}));
 console.log(JSON.stringify(data,null,2));
 await ctx.close();
})();
