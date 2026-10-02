"""Controlled Chromium replacement of the actual sandboxed plugin iframe.

Actual PluginScreenPage, private channel and generated SDK; MOCK services only.
No real host/API/model/voice/microphone jobs or external network.
"""
import asyncio
import base64
import json
import os
from pathlib import Path
import subprocess
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parents[1]


async def main():
    compiled = subprocess.run(['node', 'scripts/plugin-document-browser-bundle.mjs'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    if compiled.returncode:
        raise RuntimeError(compiled.stderr)
    checks, errors, requests = [], [], []
    def passed(name, condition):
        assert condition, name
        checks.append(name)
        print('PASS', name, flush=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path=os.environ.get('CHROMIUM_PATH') or pw.chromium.executable_path, headless=True, args=['--no-sandbox'])
        for suppress_unload in [False, True]:
            page = await browser.new_page()
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.on('request', lambda request: requests.append(request.url))
            await page.route('**/*', lambda route: route.abort())
            await page.set_content('<!doctype html><html><body><div id="root"></div></body></html>')
            await page.add_script_tag(content=compiled.stdout)
            await page.wait_for_selector('iframe')
            frame = await (await page.locator('iframe').element_handle()).content_frame()
            await frame.wait_for_selector('#voice')
            passed(f'MOCK mode {suppress_unload}: closing-style CSS injection cannot execute before the bootstrap', await frame.evaluate('typeof __cssBeforeBootstrap==="undefined"&&document.head.querySelectorAll("script").length===1'))
            passed(f'MOCK mode {suppress_unload}: private bootstrap executes before inline body code', await frame.evaluate('__bootstrapBeforeBody===true'))
            await frame.wait_for_function('__initialSnapshotReady===true')
            await page.wait_for_timeout(50)
            passed(f'MOCK mode {suppress_unload}: inline body RPC binds the first bootstrap port; copied nonce cannot replace that port', await frame.evaluate('__duplicatePortReplies.length===0') and await page.evaluate('__mock.opens===0&&__mock.completions===0&&__mock.commands===0'))
            await page.evaluate('window.__originalSource=document.querySelector("iframe").contentWindow')
            passed(f'MOCK mode {suppress_unload}: original opaque document has zero voice/model/command calls on mount', await page.evaluate('__mock.opens===0&&__mock.completions===0&&__mock.commands===0'))
            await frame.locator('#voice').click()
            await page.wait_for_function('__mock.opens===1&&__mock.records===1')
            await page.evaluate('__mock.emit({sessionId:"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",requestId:"first",type:"transcript",text:"MOCK accepted original transcript"})')
            await frame.wait_for_function('document.getElementById("transcript").textContent === "MOCK accepted original transcript"')
            passed(f'MOCK mode {suppress_unload}: legitimate first document receives only its correlated private callback', await frame.evaluate('__voiceEvents.length===1'))
            await frame.locator('#next').click()
            await frame.locator('#pending').click()
            await page.wait_for_function('__mock.records===2&&__mock.completions===1&&__mock.commands===1')
            nonce = await page.locator('iframe').evaluate('el=>JSON.parse(el.srcdoc.match(/window\.__oaiyDocumentNonce=("[^"]+")/)[1])')
            if suppress_unload:
                await frame.evaluate('''()=>{for(const type of ['pagehide','beforeunload'])window.addEventListener(type,event=>event.stopImmediatePropagation(),true);}''')
            replacement = f'''<!doctype html><html><body><h1>MOCK replacement</h1><script>
window.__leaks=[];addEventListener('message',event=>__leaks.push(event.data));
const nonce={json.dumps(nonce)};
for(const method of ['voice.open','aiComplete','command'])parent.postMessage({{__pluginHost:1,documentNonce:nonce,id:'forged-'+method,method,args:[]}},'*');
const channel=new MessageChannel();channel.port1.onmessage=event=>__leaks.push(event.data);
parent.postMessage({{__pluginHost:1,documentNonce:nonce,id:'new-connect',method:'document.connect',args:[]}},'*',[channel.port2]);
for(const method of ['voice.open','aiComplete','command'])channel.port1.postMessage({{__pluginHost:1,documentNonce:nonce,id:'port-forged-'+method,method,args:[]}});
</script></body></html>'''
            target = 'data:text/html;base64,' + base64.b64encode(replacement.encode()).decode()
            await frame.evaluate('(target)=>{location.href=target}', target)
            await page.wait_for_function('__mock.voiceDisposed===1&&__mock.aiDisposed===1')
            replacement_frame = await (await page.locator('iframe').element_handle()).content_frame()
            await replacement_frame.wait_for_selector('h1')
            passed(f'MOCK mode {suppress_unload}: self-navigation retains WindowProxy but revokes AI and voice without reopening', await page.evaluate('__originalSource===document.querySelector("iframe").contentWindow&&__mock.opens===1&&__mock.completions===1&&__mock.commands===1'))
            await page.evaluate('''()=>{__mock.emit({sessionId:'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',requestId:'second',type:'transcript',text:'PRIVATE old transcript'});__mock.finishAi({text:'PRIVATE old model'});__mock.finishCommand({ok:true,result:{ok:true,data:'PRIVATE old command'}});}''')
            await page.wait_for_timeout(100)
            passed(f'MOCK mode {suppress_unload}: replacement cannot receive late private voice, AI or command replies, even with copied nonce', await replacement_frame.evaluate('__leaks.length===0') and await page.evaluate('__mock.opens===1&&__mock.completions===1&&__mock.commands===1'))
            passed(f'MOCK mode {suppress_unload}: parent shows permanent document revocation', 'pending deliveries were revoked' in await page.locator('#root').inner_text())
            await page.close()
        passed('Controlled browser used no external requests or actual host/model/voice jobs', not requests and not errors)
        output = ROOT / 'test-results' / 'plugin-document-browser.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'mockOnly': True, 'passed': len(checks), 'tests': checks, 'errors': errors, 'externalRequests': requests}, indent=2)+'\n', encoding='utf-8')
        await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
