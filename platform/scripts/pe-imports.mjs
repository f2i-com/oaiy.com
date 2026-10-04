#!/usr/bin/env node
// The DLLs a Windows executable imports (its PE import directory), one a line.
//
//   node platform/scripts/pe-imports.mjs <file.exe> [--only-nvidia-driver]
//
// The release's NVIDIA engine must need nothing from NVIDIA where it runs but the driver's nvcuda.dll: its kernels are
// compiled when it is built and its GEMM is its own, so a build that imports cuBLAS, NVRTC or the CUDA runtime would
// start only on a computer with the CUDA toolkit. With --only-nvidia-driver this exits 1 when it imports any NVIDIA
// library but nvcuda.dll (and lists it), else 0. The runner has no NVIDIA driver, so starting the program cannot show it.
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

export class PeError extends Error {}

/** The names of the DLLs `bytes` (a PE32+ or PE32 image) imports, in the order its import directory lists them. */
export function importedDlls(bytes) {
  const buf = Buffer.from(bytes);
  const need = (ok, what) => { if (!ok) throw new PeError(what); };
  need(buf.length >= 0x40 && buf.readUInt16LE(0) === 0x5a4d, 'not a Windows executable (no MZ header)');
  const pe = buf.readUInt32LE(0x3c);
  need(pe + 24 <= buf.length && buf.readUInt32LE(pe) === 0x00004550, 'not a PE image (no PE signature)');
  const sections = buf.readUInt16LE(pe + 6);
  const optSize = buf.readUInt16LE(pe + 20);
  const opt = pe + 24;
  const magic = buf.readUInt16LE(opt);
  need(magic === 0x20b || magic === 0x10b, `an optional header of unknown kind (${magic.toString(16)})`);
  const dirs = opt + (magic === 0x20b ? 112 : 96);
  const count = buf.readUInt32LE(dirs - 4);
  if (count < 2) return [];
  const importRva = buf.readUInt32LE(dirs + 8);
  if (importRva === 0) return [];
  const table = opt + optSize;
  const at = (rva) => {
    for (let i = 0; i < sections; i++) {
      const s = table + i * 40;
      const va = buf.readUInt32LE(s + 12), raw = buf.readUInt32LE(s + 16), ptr = buf.readUInt32LE(s + 20);
      const size = Math.max(raw, buf.readUInt32LE(s + 8));
      if (rva >= va && rva < va + size) return ptr + (rva - va);
    }
    throw new PeError(`address ${rva.toString(16)} is in no section`);
  };
  const names = [];
  for (let d = at(importRva); ; d += 20) {
    need(d + 20 <= buf.length, 'the import directory runs past the end of the file');
    const nameRva = buf.readUInt32LE(d + 12);
    if (nameRva === 0 && buf.readUInt32LE(d) === 0 && buf.readUInt32LE(d + 16) === 0) break;
    const start = at(nameRva);
    const end = buf.indexOf(0, start);
    need(end > start, 'an import names no DLL');
    names.push(buf.toString('latin1', start, end));
  }
  return names;
}

/** The NVIDIA libraries in `dlls` other than the driver's nvcuda.dll (cuBLAS, NVRTC, the CUDA runtime, ...). */
export function nvidiaBeyondTheDriver(dlls) {
  return dlls.filter((d) => /^(cu|nv)/i.test(d) && d.toLowerCase() !== 'nvcuda.dll');
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const [file, flag] = process.argv.slice(2);
  try {
    if (!file) throw new PeError('usage: pe-imports.mjs <file.exe> [--only-nvidia-driver]');
    const dlls = importedDlls(fs.readFileSync(file));
    for (const d of dlls) console.log(d);
    if (flag === '--only-nvidia-driver') {
      const beyond = nvidiaBeyondTheDriver(dlls);
      if (beyond.length) {
        console.error(`::error::${file} imports ${beyond.join(', ')}: the NVIDIA engine must need nothing from NVIDIA but the driver's nvcuda.dll`);
        process.exit(1);
      }
    }
  } catch (e) {
    console.error(`::error::${e.message}`);
    process.exit(2);
  }
}
