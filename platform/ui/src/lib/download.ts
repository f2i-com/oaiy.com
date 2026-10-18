/** Saving a file from the page, as a download (the web editor's shim and the nodes that save what a run made). */

/** Hand `blob` to the browser as a download named `filename` (in the page, so Firefox starts it; revoked after a beat, so WebKit has it). */
export function triggerDownloadFromBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Browser needs a beat to start the download before revoking.
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}
