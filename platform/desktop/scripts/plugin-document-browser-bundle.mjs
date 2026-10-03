/** Actual PluginScreenPage + SDK, browser MOCK services; writes no app artifact. */
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const snapshot = { plugins: [{ id: 'mock', state: 'running', manifest: { name: 'MOCK plugin', capabilities: ['oaiy.voice.session', 'oaiy.ai.complete'], ui: { nav: [{ id: 'screen', screen: 'screen' }], screens: [{ id: 'screen', entry: 'index.html', files: ['index.html', 'theme.css'] }] } } }] };
const css = '.safe{} </StYlE><script>window.__cssBeforeBootstrap=true;</script><style>';
const body = `<button id="voice">MOCK voice request</button><button id="next">MOCK next recording</button><button id="pending">MOCK pending calls</button><p id="transcript"></p><script>
window.__bootstrapBeforeBody=!!window.PluginHost;
window.__initialSnapshotReady=false;window.__duplicatePortReplies=[];
PluginHost.snapshot().then(snapshot=>window.__initialSnapshotReady=snapshot.id==='mock').catch(()=>{});
const copiedNonce=JSON.parse(document.head.querySelector('script:last-child').textContent.split('=')[1].slice(0,-1));
const duplicateChannel=new MessageChannel();duplicateChannel.port1.onmessage=event=>__duplicatePortReplies.push(event.data);
parent.postMessage({__pluginHost:1,documentNonce:copiedNonce,id:'inline-duplicate-connect',method:'document.connect',args:[]},'*',[duplicateChannel.port2]);
for(const method of ['voice.open','aiComplete','command'])duplicateChannel.port1.postMessage({__pluginHost:1,documentNonce:copiedNonce,id:'inline-forged-'+method,method,args:[]});
window.__voiceEvents=[]; window.__session=null;
document.getElementById('voice').onclick=async()=>{await PluginHost.voice.subscribe(e=>{__voiceEvents.push(e);document.getElementById('transcript').textContent=e.text||e.type;});__session=await PluginHost.voice.open();await PluginHost.voice.record({sessionId:__session.sessionId,requestId:'first'});};
document.getElementById('next').onclick=()=>PluginHost.voice.record({sessionId:__session.sessionId,requestId:'second'});
document.getElementById('pending').onclick=()=>{PluginHost.aiComplete({requestId:'pending',sourceId:'provider:mock',prompt:'MOCK only',maxOutputChars:100}).catch(()=>{});PluginHost.command('read').catch(()=>{});};
</script>`;
const mocks = {
  api: `export const API_BASE='http://mock.invalid';export const plugins={list:async()=>(${JSON.stringify(snapshot)})};export const bridge={connectorRequest:()=>{__mock.commands++;return new Promise(r=>__mock.finishCommand=r);},events:async()=>({events:[],next:0})};export const companion={};export const engines={};export const voices={};`,
  Toasts: `const toast={push(){}};export const useToast=()=>toast;`,
  useModules: `export const useModules=()=>null;export const moduleOn=()=>false;`,
  PluginVoiceControls: `export const PluginVoiceControls=()=>null;`,
  pluginAi: `export class PluginAiSession{complete(){__mock.completions++;return new Promise(r=>__mock.finishAi=r);}dispose(){__mock.aiDisposed++;}}`,
  pluginVoice: `export class PluginVoiceSession{constructor(id,base,emit){__mock.emit=emit;}subscribe(){__mock.subscriptions++;return true;}unsubscribe(){return true;}open(){__mock.opens++;return Promise.resolve({sessionId:'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',state:'awaiting-opt-in',sttReady:true,ttsReady:true});}record(request){__mock.records++;return{...request,state:'awaiting-start'};}dispose(){__mock.voiceDisposed++;}}`,
};
const entry = `import React from 'react';import {createRoot} from 'react-dom/client';import PluginScreenPage from './src/PluginScreenPage';
window.__mock={opens:0,records:0,subscriptions:0,completions:0,commands:0,voiceDisposed:0,aiDisposed:0,emit:null};
let nonceSequence=0;Object.defineProperty(crypto,'randomUUID',{configurable:true,value:()=>\`00000000-0000-4000-8000-\${String(++nonceSequence).padStart(12,'0')}\`});
window.fetch=async(url)=>({ok:true,text:async()=>String(url).endsWith('/theme.css')?${JSON.stringify(css)}:${JSON.stringify(body)},arrayBuffer:async()=>new ArrayBuffer(0)});
createRoot(document.getElementById('root')).render(React.createElement(PluginScreenPage,{pluginId:'mock',navId:'screen'}));`;
const result = await require('esbuild').build({ stdin: { contents: entry, resolveDir: process.cwd(), sourcefile: 'plugin-document-mock.tsx', loader: 'tsx' }, write: false, bundle: true, format: 'iife', platform: 'browser', target: 'es2022', jsx: 'automatic', define: { 'process.env.NODE_ENV': '"production"' }, logLevel: 'silent', plugins: [{ name: 'explicit-mock-host-services', setup(build) { build.onResolve({ filter: /^\.\/(api|Toasts|useModules|PluginVoiceControls|pluginAi|pluginVoice)$/ }, args => args.importer.endsWith('PluginScreenPage.tsx') ? { path: args.path.slice(2), namespace: 'mock' } : undefined); build.onLoad({ filter: /.*/, namespace: 'mock' }, args => ({ contents: mocks[args.path], loader: 'js' })); } }] });
process.stdout.write(result.outputFiles[0].text);
