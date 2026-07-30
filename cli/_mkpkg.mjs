import { zipSync, strToU8 } from 'fflate';
import fs from 'node:fs';
const S = process.argv[2];
const main = {
  id: 'main', name: 'Main',
  graph: {
    nodes: [
      {id:'t',type:'template',position:{x:0,y:0},data:{template:'seed'}},
      {id:'sf',type:'subflow',position:{x:200,y:0},data:{flowId:'helper',flowName:'Helper'}},
      {id:'out',type:'output',position:{x:400,y:0},data:{label:'text',outputType:'text'}}
    ],
    edges: [
      {id:'e1',source:'t',target:'sf',sourceHandle:'result',targetHandle:'input'},
      {id:'e2',source:'sf',target:'out',sourceHandle:'result',targetHandle:'result'}
    ]
  }
};
const helper = {
  id:'helper', name:'Helper',
  graph:{ nodes:[
      {id:'hin',type:'input_text',position:{x:0,y:0},data:{label:'in',value:'H'}},
      {id:'htpl',type:'template',position:{x:200,y:0},data:{template:'helper saw {{input}}'}},
      {id:'hout',type:'output',position:{x:400,y:0},data:{label:'text',outputType:'text'}}
    ], edges:[
      {id:'he1',source:'hin',target:'htpl',sourceHandle:'result',targetHandle:'input'},
      {id:'he2',source:'htpl',target:'hout',sourceHandle:'result',targetHandle:'result'}
    ]}
};
const manifest = {
  formatVersion:'1.0', id:'com.example.pkg', name:'Example Package', version:'1.0.0',
  entryFlow:'flows/main.flow.json',
  flows:['flows/main.flow.json','flows/helper.flow.json'],
  permissions:['network'], isolation:{sandboxed:true,networkAccess:true}
};
fs.writeFileSync(S+'/pkg.oaiy', Buffer.from(zipSync({
  'manifest.json': strToU8(JSON.stringify(manifest,null,2)),
  'flows/main.flow.json': strToU8(JSON.stringify(main,null,2)),
  'flows/helper.flow.json': strToU8(JSON.stringify(helper,null,2)),
})));
// package whose entryFlow uses a windows separator
const m2 = {...manifest, entryFlow:'flows\main.flow.json'};
fs.writeFileSync(S+'/pkg-bs.oaiy', Buffer.from(zipSync({
  'manifest.json': strToU8(JSON.stringify(m2,null,2)),
  'flows/main.flow.json': strToU8(JSON.stringify(main,null,2)),
})));
// a code node package (logic_block) to check the warning
const code = {nodes:[{id:'c',type:'logic_block',position:{x:0,y:0},data:{code:'return 1'}}],edges:[]};
fs.writeFileSync(S+'/pkg-code.oaiy', Buffer.from(zipSync({
  'manifest.json': strToU8(JSON.stringify({...manifest, entryFlow:'flows/c.json', flows:['flows/c.json']})),
  'flows/c.json': strToU8(JSON.stringify(code)),
})));
console.log('written');
