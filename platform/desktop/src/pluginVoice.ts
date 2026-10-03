import { PluginCommandError } from './pluginRpc';
import { BrowserVoiceMedia, voiceBase64, type VoiceMedia, type VoiceRecording } from './pluginVoiceMedia';

export type VoiceEventType = 'enabled' | 'recording' | 'transcribing' | 'transcript' | 'speaking' | 'finished' | 'cancelled' | 'closed' | 'failed';
export interface PluginVoiceEvent { sessionId: string; requestId?: string; type: VoiceEventType; text?: string; code?: string; message?: string; elapsedMs?: number }
export interface VoiceAvailability { sttReady: boolean; ttsReady: boolean; reason: string | null }
export interface VoiceView extends VoiceAvailability {
  sessionId: string; enabled: boolean; state: 'awaiting-opt-in' | 'enabled' | 'awaiting-start' | 'recording' | 'transcribing' | 'speaking' | 'failed';
  elapsedMs: number; message?: string;
}
interface VoiceRequest { sessionId: string; requestId: string }
interface Pending extends VoiceRequest { controller: AbortController; recording?: VoiceRecording; timer?: number; interval?: number; started?: number; starting?: boolean }
const active = new Set<PluginVoiceSession>();
const fail = (code: string, message: string) => new PluginCommandError({code,message});
const sessionId = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/i.test(value);
const requestId = (value: unknown): value is string => typeof value === 'string' && /^[a-zA-Z0-9_.:-]{1,96}$/.test(value);
function input(value: unknown, speaking = false): VoiceRequest & {text?: string} {
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).some(k => !['sessionId','requestId',...(speaking ? ['text'] : [])].includes(k))) throw fail('voice_invalid_request','Supply a current voice session and fresh request ID.');
  const data = value as VoiceRequest & {text?: string};
  if (!sessionId(data.sessionId) || !requestId(data.requestId) || (speaking && (typeof data.text !== 'string' || data.text.length > 1600 || !data.text.trim() || [...data.text].length > 800 || /[\u0000-\u0008\u000b-\u001f\u007f]/.test(data.text)))) throw fail('voice_invalid_request','Supply a current voice session, fresh request ID and speech text up to 800 characters.');
  return data;
}

/** One mounted document. Owner consent and media never belong to frame RPCs. */
export class PluginVoiceSession {
  private readonly base: string;
  private disposed = false;
  private subscribed = false;
  private view: VoiceView | null = null;
  private pending: Pending | undefined;
  private opening = false;
  private enabling = false;
  private readonly recent = new Set<string>();
  private readonly requests = new Set<AbortController>();
  private monitor: number | undefined;
  private expiry: number | undefined;
  private readonly hidden = () => {if (document.hidden && this.view) void this.close(this.view.sessionId);};
  private readonly pageHide = () => this.dispose();
  private media: VoiceMedia;
  constructor(private readonly plugin: string, apiBase: string, private readonly emit: (event: PluginVoiceEvent) => void,
    private readonly update: (view: VoiceView | null) => void, private readonly makeMedia: () => VoiceMedia = () => new BrowserVoiceMedia()) {
    this.base = `${apiBase}/api/plugins/${encodeURIComponent(plugin)}/voice`;
    this.media = makeMedia();
    document.addEventListener('visibilitychange',this.hidden);
    window.addEventListener('pagehide',this.pageHide);
  }
  private publish(event: PluginVoiceEvent): void { if (!this.disposed && this.subscribed && this.view?.sessionId === event.sessionId) this.emit(event); }
  private render(changes: Partial<VoiceView> = {}): void {if (!this.disposed && this.view) {this.view = {...this.view,...changes}; this.update({...this.view});}}
  private current(id: string): void {if (this.disposed || !this.view || this.view.sessionId !== id) throw fail('voice_session_unavailable','Open and enable the current voice session first.');}
  subscribe(): boolean {if (this.disposed) throw fail('voice_session_unavailable','The plugin screen closed.'); this.subscribed = true; return true;}
  unsubscribe(): boolean {this.subscribed = false; return true;}
  async status(): Promise<VoiceAvailability> {
    return (await this.availability()).ready;
  }
  private async availability(id?: string): Promise<{ready:VoiceAvailability;live:boolean}> {
    const value = await this.json(id ? `status?sessionId=${encodeURIComponent(id)}` : 'status', undefined, undefined, 4000) as VoiceAvailability & {sessionLeaseValid?:boolean};
    if (!value || Object.keys(value).some(k => !['sttReady','ttsReady','reason',...(id ? ['sessionLeaseValid'] : [])].includes(k)) || typeof value.sttReady !== 'boolean' || typeof value.ttsReady !== 'boolean' || (value.reason !== null && (typeof value.reason !== 'string' || value.reason.length > 300)) || (id && typeof value.sessionLeaseValid !== 'boolean')) throw fail('voice_invalid_response','The desktop returned invalid voice availability.');
    // Internal lease state stays in the trusted parent, outside the SDK/view.
    return {ready:{sttReady:value.sttReady,ttsReady:value.ttsReady,reason:value.reason},live:id ? value.sessionLeaseValid! : true};
  }
  async open(): Promise<{sessionId:string;state:'awaiting-opt-in';sttReady:boolean;ttsReady:boolean}> {
    if (this.disposed || this.view || this.opening) throw fail('voice_busy','Close the current voice session before opening another.');
    this.opening = true; const id = crypto.randomUUID();
    try {
      const [ready, result] = await Promise.all([this.status(), this.json('open',{sessionId:id},undefined,4000)]) as [VoiceAvailability,{sessionId?:unknown}];
      if (this.disposed || result?.sessionId !== id) throw fail('voice_session_unavailable','The voice screen closed or returned an invalid session.');
      const view: VoiceView & {state:'awaiting-opt-in'} = {sessionId:id,state:'awaiting-opt-in',enabled:false,elapsedMs:0,...ready};
      this.view = view; this.update({...view});
      this.expiry = window.setTimeout(()=>{if(this.view?.sessionId===id)void this.close(id);},900_000);
      // Permission loss closes parent media even if the plugin does not call
      // snapshot. No service start/model load is performed by this probe.
      let probing = false;
      this.monitor = window.setInterval(() => {
        if (probing || !this.view) return; probing = true;
        void this.availability(id).then(({ready:availability,live}) => {
          if (this.view?.sessionId !== id) return;
          // Check the session's original owned-service/process lease through
          // buffered playback, after the native request has already completed.
          if (!live) {void this.close(id);return;}
          this.render(availability);
          const pending=this.pending;
          // Aokie's busy health reflects the mutex held by our own admitted
          // inference. Stop capture on lost readiness; native process/service
          // leases fence an already admitted STT/TTS operation continuously,
          // and the independent session lease also fences parent playback.
          if (pending && ['awaiting-start','recording'].includes(this.view.state) && !availability.sttReady) {
            pending.controller.abort();pending.recording?.cancel();
            this.publish({sessionId:id,requestId:pending.requestId,type:'failed',code:'voice_unavailable',message:'The local voice lane is no longer ready. This request stopped.'});
            void this.cleanup('cancel',{sessionId:id,requestId:pending.requestId});this.finish(pending);
          }
        })
          .catch(() => {if (this.view?.sessionId === id) void this.close(id);}).finally(() => {probing = false;});
      },2000);
      return {sessionId:id,state:'awaiting-opt-in',sttReady:ready.sttReady,ttsReady:ready.ttsReady};
    } catch (error) {void this.cleanup('close',{sessionId:id}); throw error;}
    finally {this.opening = false;}
  }
  /** Called only by the visible trusted-parent Enable button. */
  async enable(): Promise<void> {
    if (!this.view || this.enabling || this.view.enabled || this.disposed) return;
    const id = this.view.sessionId; this.enabling = true;
    try {
      await this.media.enable(); this.current(id);
      this.render({enabled:true,state:'enabled',message:undefined}); this.publish({sessionId:id,type:'enabled'});
    } catch {
      if (this.view?.sessionId === id && !this.disposed) {this.render({enabled:false,state:'failed',message:'Microphone permission was declined or is unavailable. Enable this session explicitly to try again.'}); this.publish({sessionId:id,type:'failed',code:'microphone_denied',message:'Microphone permission was declined or is unavailable.'});}
    } finally {this.enabling = false;}
  }
  decline(): void {if (this.view) {this.publish({sessionId:this.view.sessionId,type:'failed',code:'microphone_denied',message:'The owner declined this microphone session.'}); void this.close(this.view.sessionId);}}
  private begin(value: unknown, speaking: boolean): Pending {
    const data = input(value,speaking); this.current(data.sessionId);
    if (!this.view!.enabled) throw fail('voice_session_unavailable','The owner must explicitly enable this microphone session.');
    if (this.pending || [...active].some(session=>session.plugin===this.plugin) || active.size >= 2 || this.recent.size >= 512) throw fail('voice_busy','A bounded voice request is already active. Cancel it or wait.');
    if (this.recent.has(data.requestId)) throw fail('voice_request_repeated','Use a fresh request ID for every voice attempt.');
    if (speaking ? !this.view!.ttsReady : !this.view!.sttReady) throw fail('voice_unavailable','OAIY Voice is not already loaded and ready. Start it explicitly from Services when other work is idle.');
    const pending = {...data,controller:new AbortController()}; this.pending = pending; active.add(this); this.recent.add(data.requestId); return pending;
  }
  record(value: unknown): {sessionId:string;requestId:string;state:'awaiting-start'} {
    const pending = this.begin(value,false); this.render({state:'awaiting-start',elapsedMs:0,message:undefined});
    return {sessionId:pending.sessionId,requestId:pending.requestId,state:'awaiting-start'};
  }
  /** Called only by visible trusted-parent Start, never the frame RPC. */
  async start(): Promise<void> {
    const pending = this.pending;
    if (!pending || pending.starting || this.view?.state !== 'awaiting-start' || pending.controller.signal.aborted) return;
    pending.starting = true;
    this.render({message:'Waiting for microphone permission.'});
    try {
      const recording = await this.media.record();
      if (this.pending !== pending || pending.controller.signal.aborted || this.disposed) {recording.cancel();return;}
      pending.recording = recording; pending.started = Date.now(); this.render({state:'recording',message:undefined});
      this.publish({sessionId:pending.sessionId,requestId:pending.requestId,type:'recording',elapsedMs:0});
      pending.interval = window.setInterval(() => this.render({elapsedMs:Math.min(15000,Date.now()-pending.started!)}),100);
      pending.timer = window.setTimeout(() => {void this.stop();},15000);
    } catch {this.failed(pending,fail('microphone_denied','Microphone permission was declined or capture is unavailable.'));}
  }
  async stop(): Promise<void> {
    const pending = this.pending;
    if (!pending?.recording || this.view?.state !== 'recording') return;
    this.clearTimers(pending); this.render({state:'transcribing'}); this.publish({sessionId:pending.sessionId,requestId:pending.requestId,type:'transcribing'});
    try {
      const wav = await pending.recording.stop(); pending.recording = undefined;
      if (this.pending !== pending || pending.controller.signal.aborted) return;
      const result = await this.json('transcribe',{sessionId:pending.sessionId,requestId:pending.requestId,audio:voiceBase64(wav)},pending.controller.signal,20_000) as {sessionId?:unknown;requestId?:unknown;text?:unknown};
      if (this.pending !== pending || pending.controller.signal.aborted || this.disposed) return;
      if (!result || Object.keys(result).some(k=>!['sessionId','requestId','text'].includes(k)) || result.sessionId !== pending.sessionId || result.requestId !== pending.requestId || typeof result.text !== 'string' || !result.text.trim() || result.text.length > 4000 || [...result.text].length > 2000) throw fail('voice_invalid_response','The desktop returned an invalid bounded transcript.');
      this.publish({sessionId:pending.sessionId,requestId:pending.requestId,type:'transcript',text:result.text}); this.finish(pending);
    } catch (error) {this.failed(pending,error);}
  }
  speak(value: unknown): {sessionId:string;requestId:string;state:'speaking'} {
    const data = input(value,true); const pending = this.begin(data,true); this.render({state:'speaking',message:undefined});
    this.publish({sessionId:pending.sessionId,requestId:pending.requestId,type:'speaking'});
    pending.timer=window.setTimeout(()=>{
      if(this.pending!==pending||this.disposed)return;
      pending.controller.abort();
      this.publish({sessionId:pending.sessionId,requestId:pending.requestId,type:'failed',code:'voice_timeout',message:'The bounded speech session timed out.'});
      void this.cleanup('cancel',{sessionId:pending.sessionId,requestId:pending.requestId});this.finish(pending);
    },30_000);
    void (async () => {
      try {
        const pcm = await this.bytes('speak',data,pending.controller.signal,30_000,true);
        if (this.pending !== pending || pending.controller.signal.aborted || this.disposed) return;
        // Native generation/download have their own30s bound. The parent's
        // overall30s timer also includes at most20s of cancellable playback.
        await this.media.play(pcm,pending.controller.signal);
        if (this.pending !== pending || pending.controller.signal.aborted || this.disposed) return;
        this.publish({sessionId:pending.sessionId,requestId:pending.requestId,type:'finished'}); this.finish(pending);
      } catch (error) {this.failed(pending,error);}
    })();
    return {sessionId:pending.sessionId,requestId:pending.requestId,state:'speaking'};
  }
  async cancel(value: unknown): Promise<{sessionId:string;requestId:string;cancelled:boolean}> {
    const data = input(value); this.current(data.sessionId); const pending = this.pending;
    const cancelled = !!pending && pending.requestId === data.requestId;
    if (cancelled) {pending.controller.abort(); pending.recording?.cancel(); this.publish({...data,type:'cancelled'}); this.finish(pending);}
    void this.cleanup('cancel',data); return {...data,cancelled};
  }
  async close(id: unknown): Promise<{sessionId:string;closed:boolean}> {
    if (!sessionId(id)) throw fail('voice_invalid_request','Supply the current voice session ID.');
    this.current(id); this.publish({sessionId:id,type:'closed'});
    this.release(); void this.cleanup('close',{sessionId:id}); return {sessionId:id,closed:true};
  }
  dispose(): void {
    if (this.disposed) return; const id = this.view?.sessionId; this.release(); this.disposed = true;
    document.removeEventListener('visibilitychange',this.hidden);window.removeEventListener('pagehide',this.pageHide);
    for (const controller of this.requests) controller.abort(); this.requests.clear();
    if (id) void this.cleanup('close',{sessionId:id});
  }
  private clearTimers(pending: Pending): void {window.clearTimeout(pending.timer);window.clearInterval(pending.interval);}
  private finish(pending: Pending): void {
    if (this.pending !== pending) return; this.clearTimers(pending); this.pending = undefined; active.delete(this);
    this.render({state:'enabled',elapsedMs:0});
  }
  private failed(pending: Pending, error: unknown): void {
    if (this.pending !== pending || this.disposed || pending.controller.signal.aborted) return;
    const details = error instanceof PluginCommandError ? error : fail('voice_failed','The bounded local voice request failed.');
    this.publish({sessionId:pending.sessionId,requestId:pending.requestId,type:'failed',code:details.code,message:details.message.slice(0,300)});
    void this.cleanup('cancel',{sessionId:pending.sessionId,requestId:pending.requestId}); this.finish(pending); this.render({state:'failed',message:details.message.slice(0,300)});
  }
  private release(): void {
    if (this.monitor) window.clearInterval(this.monitor); this.monitor = undefined;
    window.clearTimeout(this.expiry);this.expiry=undefined;
    const pending = this.pending;
    if (pending) {pending.controller.abort();pending.recording?.cancel();this.clearTimers(pending);void this.cleanup('cancel',{sessionId:pending.sessionId,requestId:pending.requestId});}
    this.pending = undefined; active.delete(this);this.recent.clear(); this.media.dispose(); this.media = this.makeMedia(); this.view = null; if (!this.disposed) this.update(null);
  }
  private async cleanup(method: 'cancel'|'close', data: unknown): Promise<void> {try {await this.json(method,data,undefined,4000);} catch { /* native deadline remains authoritative */ }}
  private async json(method: string, body: unknown, signal: AbortSignal|undefined, timeout: number): Promise<unknown> {
    const bytes = await this.bytes(method,body,signal,timeout,false);
    try {return JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(bytes));} catch {throw fail('voice_invalid_response','The desktop returned invalid voice response data.');}
  }
  private async bytes(method: string, body: unknown, signal: AbortSignal|undefined, timeout: number, pcm: boolean): Promise<Uint8Array> {
    const controller = new AbortController(); this.requests.add(controller); let reader: ReadableStreamDefaultReader<Uint8Array>|undefined;
    const abort = () => controller.abort(); signal?.addEventListener('abort',abort,{once:true}); if (signal?.aborted) abort();
    const timer = window.setTimeout(abort,timeout);
    const cancelled = () => fail(signal?.aborted ? 'voice_cancelled':'voice_timeout',signal?.aborted ? 'The voice request was cancelled.':'The bounded voice request timed out.');
    const fence = () => {if (controller.signal.aborted) throw cancelled();};
    let rejectAbort!:()=>void;
    const aborted = new Promise<never>((_resolve,reject) => {rejectAbort=()=>{void reader?.cancel().catch(()=>{});reject(cancelled());};controller.signal.addEventListener('abort',rejectAbort,{once:true});});
    const operation = (async () => {
      fence(); const response = await fetch(`${this.base}/${method}`,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),signal:controller.signal});
      if (controller.signal.aborted) {void response.body?.cancel().catch(()=>{});fence();}
      if (!response.body) throw fail('voice_invalid_response','The desktop returned no voice response.');
      if (pcm && response.ok && (response.headers.get('Content-Type')?.split(';')[0] !== 'audio/pcm' || response.headers.get('X-Sample-Rate') !== '24000')) {void response.body.cancel().catch(()=>{});throw fail('voice_invalid_response','The desktop returned invalid speech audio.');}
      reader = response.body.getReader(); const chunks:Uint8Array[]=[]; let total=0; const limit=pcm&&response.ok?960_000:16*1024;
      try {
        while (true) {fence();const chunk=await reader.read();fence();if(chunk.done)break;total+=chunk.value.byteLength;if(total>limit)throw fail('voice_invalid_response','The desktop voice response exceeded its size limit.');chunks.push(chunk.value);}
      } catch (error) {void reader.cancel().catch(()=>{});throw error;} finally {try{reader.releaseLock();}catch{}}
      const bytes=new Uint8Array(total);let offset=0;for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}
      fence(); if (!response.ok) {
        let error:{code?:unknown;message?:unknown}|undefined;
        try {error=JSON.parse(new TextDecoder().decode(bytes))?.error;} catch {}
        throw fail(typeof error?.code==='string'&&/^[a-zA-Z0-9_.:-]{1,64}$/.test(error.code)?error.code:'voice_failed',typeof error?.message==='string'?error.message.slice(0,300).replace(/[\u0000-\u001f\u007f]/g,' '):'The local voice request failed.');
      }
      if(pcm&&(!bytes.length||bytes.length%2))throw fail('voice_invalid_response','The desktop returned invalid speech audio.');return bytes;
    })();
    try {return await Promise.race([operation,aborted]);} finally {window.clearTimeout(timer);signal?.removeEventListener('abort',abort);controller.signal.removeEventListener('abort',rejectAbort);this.requests.delete(controller);}
  }
}
