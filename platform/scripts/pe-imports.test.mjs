// pe-imports.mjs: the DLLs a Windows executable imports, read from a PE image built here (the tests run on Linux too).
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { importedDlls, nvidiaBeyondTheDriver, PeError } from './pe-imports.mjs';

/** A minimal PE32+ image: one section at RVA 0x1000 (file 0x200) holding an import directory for `dlls`. */
function image(dlls) {
  const buf = Buffer.alloc(0x600);
  buf.write('MZ', 0, 'latin1');
  buf.writeUInt32LE(0x40, 0x3c);
  buf.writeUInt32LE(0x00004550, 0x40);
  buf.writeUInt16LE(0x8664, 0x44);
  buf.writeUInt16LE(1, 0x46);
  buf.writeUInt16LE(240, 0x54);
  const opt = 0x58;
  buf.writeUInt16LE(0x20b, opt);
  buf.writeUInt32LE(16, opt + 108);
  buf.writeUInt32LE(0x1000, opt + 112 + 8);
  buf.writeUInt32LE((dlls.length + 1) * 20, opt + 112 + 12);
  const section = opt + 240;
  buf.write('.idata', section, 'latin1');
  buf.writeUInt32LE(0x400, section + 8);
  buf.writeUInt32LE(0x1000, section + 12);
  buf.writeUInt32LE(0x400, section + 16);
  buf.writeUInt32LE(0x200, section + 20);
  let name = 0x200 + (dlls.length + 1) * 20;
  dlls.forEach((d, i) => {
    const desc = 0x200 + i * 20;
    buf.writeUInt32LE(0x1000 + (name - 0x200), desc + 12);
    buf.writeUInt32LE(0x1000, desc + 16);
    buf.write(`${d}\0`, name, 'latin1');
    name += d.length + 1;
  });
  return buf;
}

describe('the DLLs a Windows executable imports', () => {
  it('are read from its import directory, in order', () => {
    assert.deepEqual(importedDlls(image(['KERNEL32.dll', 'nvcuda.dll', 'ntdll.dll'])), ['KERNEL32.dll', 'nvcuda.dll', 'ntdll.dll']);
    assert.deepEqual(importedDlls(image([])), []);
  });

  it('are refused for a file that is not a PE image', () => {
    assert.throws(() => importedDlls(Buffer.from('not a program')), PeError);
    const notPe = image(['a.dll']);
    notPe.writeUInt32LE(0, 0x40);
    assert.throws(() => importedDlls(notPe), /no PE signature/);
  });

  it('name an NVIDIA library beyond the driver when the engine would need the CUDA toolkit', () => {
    assert.deepEqual(nvidiaBeyondTheDriver(['KERNEL32.dll', 'nvcuda.dll']), []);
    assert.deepEqual(nvidiaBeyondTheDriver(['nvcuda.dll', 'cublas64_12.dll', 'nvrtc64_120_0.dll', 'cudart64_12.dll']), ['cublas64_12.dll', 'nvrtc64_120_0.dll', 'cudart64_12.dll']);
  });
});
