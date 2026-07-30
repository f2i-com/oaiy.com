import { zipSync, strToU8 } from 'fflate';
import fs from 'node:fs';
const S = process.argv[2];
const main = {
  id: 'main', name: 'Main',
  graph: {
    nodes: [
      { id: 't', type: 'template', position: { x: 0, y: 0 }, data: { template: 'seed' } },
      { id: 'out', type: 'output', position: { x: 400, y: 0 }, data: { label: 'text', outputType: 'text' } },
    ],
    edges: [{ id: 'e2', source: 't', target: 'out', sourceHandle: 'result', targetHandle: 'result' }],
  },
};
const BS = String.fromCharCode(92); // backslash
const manifest = {
  formatVersion: '1.0', id: 'com.example.pkg', name: 'BS Package', version: '1.0.0',
  entryFlow: 'flows' + BS + 'main.flow.json',
  flows: ['flows' + BS + 'main.flow.json'],
};
console.log('entryFlow is:', JSON.stringify(manifest.entryFlow));
fs.writeFileSync(S + '/pkg-bs2.oaiy', Buffer.from(zipSync({
  'manifest.json': strToU8(JSON.stringify(manifest, null, 2)),
  'flows/main.flow.json': strToU8(JSON.stringify(main, null, 2)),
})));

// package with a custom node module declared -> CLI must refuse
const m3 = { ...manifest, entryFlow: 'flows/main.flow.json', flows: ['flows/main.flow.json'], nodes: [{ id: 'n1', path: 'nodes/n1' }] };
fs.writeFileSync(S + '/pkg-custom.oaiy', Buffer.from(zipSync({
  'manifest.json': strToU8(JSON.stringify(m3, null, 2)),
  'flows/main.flow.json': strToU8(JSON.stringify(main, null, 2)),
})));

// package with NO manifest.json
fs.writeFileSync(S + '/pkg-nomanifest.oaiy', Buffer.from(zipSync({
  'flows/main.flow.json': strToU8(JSON.stringify(main, null, 2)),
})));
console.log('written');
