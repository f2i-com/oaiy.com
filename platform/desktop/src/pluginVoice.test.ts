import {afterEach,beforeEach,describe,expect,it,vi} from 'vitest';
import {PluginVoiceSession,type PluginVoiceEvent,type VoiceView,type VoiceAvailability} from './pluginVoice';
import {voiceWav,type VoiceMedia} from './pluginVoiceMedia';

const sessions:PluginVoiceSession[]=[];
const ready:VoiceAvailability={sttReady:true,ttsReady:true,reason:null};
const isStatus=(url:string)=>new URL(url).pathname.endsWith('/status');
const metadata=(url:string,value=ready,live=true)=>({...value,...(new URL(url).searchParams.has('sessionId')?{sessionLeaseValid:live}:{})});
const flush=async()=>{await vi.advanceTimersByTimeAsync(0);};
let fetcher:ReturnType<typeof vi.fn>;
beforeEach(()=>{
  vi.useFakeTimers();
  fetcher=vi.fn().mockImplementation((url:string,options?:RequestInit)=>{
    const data=options?.body?JSON.parse(String(options.body)):{};
    if(isStatus(url))return Promise.resolve(new Response(JSON.stringify(metadata(url))));
    if(url.endsWith('/speak'))return Promise.resolve(new Response(new Uint8Array([0,0,1,0]),{headers:{'Content-Type':'audio/pcm','X-Sample-Rate':'24000'}}));
    return Promise.resolve(new Response(JSON.stringify(url.endsWith('/transcribe')?{sessionId:data.sessionId,requestId:data.requestId,text:'Synthetic source-grounded question'}:data)));
  });vi.stubGlobal('fetch',fetcher);
});
afterEach(()=>{sessions.splice(0).forEach(s=>s.dispose());vi.clearAllTimers();vi.useRealTimers();vi.unstubAllGlobals();});
async function fixture(plugin='probe'){
  const recording={stop:vi.fn().mockResolvedValue(voiceWav(new Float32Array([0,.1]),16000)),cancel:vi.fn()};
  const media:VoiceMedia={enable:vi.fn().mockResolvedValue(undefined),record:vi.fn().mockResolvedValue(recording),play:vi.fn().mockResolvedValue(undefined),dispose:vi.fn()};
  const emit=vi.fn<(event:PluginVoiceEvent)=>void>();let view:VoiceView|null=null;
  const session=new PluginVoiceSession(plugin,'http://127.0.0.1:45678',emit,v=>{view=v;},()=>media);sessions.push(session);
  session.subscribe();const opened=await session.open();
  return {session,opened,emit,media,recording,view:()=>view,request:(requestId='one')=>({sessionId:opened.sessionId,requestId})};
}
describe('trusted-parent scoped voice sessions',()=>{
  it('opens metadata without microphone use and rejects frame capture before owner consent',async()=>{
    const f=await fixture();expect(f.opened).toEqual({sessionId:expect.any(String),state:'awaiting-opt-in',sttReady:true,ttsReady:true});
    expect(f.media.enable).not.toHaveBeenCalled();expect(f.media.record).not.toHaveBeenCalled();
    expect(()=>f.session.record(f.request())).toThrow(expect.objectContaining({code:'voice_session_unavailable'}));
    await f.session.enable();expect(f.emit).toHaveBeenCalledExactlyOnceWith({sessionId:f.opened.sessionId,type:'enabled'});
    expect(f.session.record(f.request())).toMatchObject({state:'awaiting-start'});expect(f.media.record).not.toHaveBeenCalled();
  });
  it('declining native permission never reports enabled or records',async()=>{
    const f=await fixture();vi.mocked(f.media.enable).mockRejectedValue(new Error('Sensitive platform detail'));
    await f.session.enable();expect(f.view()?.enabled).toBe(false);expect(f.emit).toHaveBeenCalledWith({sessionId:f.opened.sessionId,type:'failed',code:'microphone_denied',message:'Microphone permission was declined or is unavailable.'});
    expect(f.emit.mock.calls.some(([e])=>e.type==='enabled')).toBe(false);expect(f.media.record).not.toHaveBeenCalled();
  });
  it('explicit owner Start/Stop sends a canonical bounded WAV only to the scoped native route and returns transcript only',async()=>{
    const f=await fixture();await f.session.enable();f.session.record(f.request());await f.session.start();await f.session.stop();
    const call=fetcher.mock.calls.find(([url])=>url.endsWith('/transcribe'))!;expect(call[0]).toBe('http://127.0.0.1:45678/api/plugins/probe/voice/transcribe');
    const data=JSON.parse(call[1].body);expect(Object.keys(data).sort()).toEqual(['audio','requestId','sessionId']);expect(atob(data.audio).slice(0,4)).toBe('RIFF');
    expect(f.emit).toHaveBeenCalledWith({...f.request(),type:'transcript',text:'Synthetic source-grounded question'});
    expect(f.emit.mock.calls.every(([event])=>!('audio'in event)&&!('url'in event))).toBe(true);
  });
  it('max15 seconds stops the microphone and submits once',async()=>{
    const f=await fixture();await f.session.enable();f.session.record(f.request());await f.session.start();await vi.advanceTimersByTimeAsync(15000);await flush();
    expect(f.recording.stop).toHaveBeenCalledTimes(1);expect(fetcher.mock.calls.filter(([url])=>url.endsWith('/transcribe'))).toHaveLength(1);
  });
  it('cancel stops capture, sends bounded native cancellation and forbids request replay',async()=>{
    const f=await fixture();await f.session.enable();f.session.record(f.request());await f.session.start();await f.session.cancel(f.request());
    expect(f.recording.cancel).toHaveBeenCalledOnce();expect(f.emit).toHaveBeenCalledWith({...f.request(),type:'cancelled'});
    expect(()=>f.session.record(f.request())).toThrow(expect.objectContaining({code:'voice_request_repeated'}));
    expect(fetcher.mock.calls.some(([url])=>url.endsWith('/transcribe'))).toBe(false);
  });
  it('closing while microphone permission is pending stops the late capture without events',async()=>{
    const f=await fixture();await f.session.enable();let resolve!:(recording:typeof f.recording)=>void;vi.mocked(f.media.record).mockImplementation(()=>new Promise(r=>{resolve=r;}));
    f.session.record(f.request());const starting=f.session.start();await f.session.close(f.opened.sessionId);const count=f.emit.mock.calls.length;
    resolve(f.recording);await starting;expect(f.recording.cancel).toHaveBeenCalledOnce();expect(f.emit).toHaveBeenCalledTimes(count);
  });
  it('dispose aborts a live transcript and suppresses an abort-ignoring late response',async()=>{
    const f=await fixture();await f.session.enable();let finish!:(r:Response)=>void;let signal!:AbortSignal;
    fetcher.mockImplementation((url:string,options:RequestInit)=>{if(url.endsWith('/transcribe')){signal=options.signal!;return new Promise(r=>{finish=r;});}return Promise.resolve(new Response('{}'));});
    f.session.record(f.request());await f.session.start();const stopping=f.session.stop();await flush();f.session.dispose();const count=f.emit.mock.calls.length;
    expect(signal.aborted).toBe(true);finish(new Response(JSON.stringify({...f.request(),text:'late'})));await stopping;expect(f.emit).toHaveBeenCalledTimes(count);
  });
  it('unsubscribed private callbacks do not leak into another screen',async()=>{
    const f=await fixture();f.session.unsubscribe();await f.session.enable();f.session.record(f.request());await f.session.start();await f.session.stop();expect(f.emit).not.toHaveBeenCalled();
  });
  it.each(['wrong-session','wrong-request','oversized','extra-audio'])('rejects %s native transcript',async(kind)=>{
    const f=await fixture();await f.session.enable();const result={...f.request(),text:'text'} as Record<string,unknown>;
    if(kind==='wrong-session')result.sessionId=crypto.randomUUID();if(kind==='wrong-request')result.requestId='other';if(kind==='oversized')result.text='x'.repeat(2001);if(kind==='extra-audio')result.audio='raw';
    fetcher.mockImplementation((url:string)=>Promise.resolve(new Response(JSON.stringify(url.endsWith('/transcribe')?result:ready))));
    f.session.record(f.request());await f.session.start();await f.session.stop();expect(f.emit.mock.calls.some(([e])=>e.type==='transcript')).toBe(false);expect(f.emit).toHaveBeenCalledWith(expect.objectContaining({type:'failed',code:'voice_invalid_response'}));
  });
  it('voice text is bounded, closed, uses host playback and never passes audio to the frame',async()=>{
    const f=await fixture();await f.session.enable();expect(()=>f.session.speak({...f.request(),text:'x'.repeat(801)})).toThrow();expect(()=>f.session.speak({...f.request(),text:'hello',url:'http://other'})).toThrow();
    expect(f.session.speak({...f.request(),text:'Grounded source narration'})).toMatchObject({state:'speaking'});await flush();expect(f.media.play).toHaveBeenCalledWith(new Uint8Array([0,0,1,0]),expect.any(AbortSignal));expect(f.emit).toHaveBeenCalledWith({...f.request(),type:'finished'});
  });
  it('the overall30-second speech deadline includes playback and suppresses a late finished callback',async()=>{
    const f=await fixture();await f.session.enable();let finish!:()=>void;let signal!:AbortSignal;
    vi.mocked(f.media.play).mockImplementation((_audio,s)=>{signal=s;return new Promise(r=>{finish=r;});});
    f.session.speak({...f.request(),text:'Bounded narration'});await flush();await vi.advanceTimersByTimeAsync(30_000);
    expect(signal.aborted).toBe(true);expect(f.emit).toHaveBeenCalledWith(expect.objectContaining({type:'failed',code:'voice_timeout'}));
    finish();await flush();expect(f.emit.mock.calls.some(([e])=>e.type==='finished')).toBe(false);
  });
  it('native permission revocation closes active parent capture',async()=>{
    const f=await fixture();await f.session.enable();f.session.record(f.request());await f.session.start();
    fetcher.mockImplementation((url:string)=>Promise.resolve(new Response(JSON.stringify(isStatus(url)?{error:{code:'capability_unavailable',message:'Permission revoked.'}}:{}),{status:isStatus(url)?403:200})));
    await vi.advanceTimersByTimeAsync(2000);expect(f.recording.cancel).toHaveBeenCalled();expect(f.media.dispose).toHaveBeenCalled();expect(f.view()).toBeNull();
  });
  it('loaded lane loss stops capture before sending audio',async()=>{
    const f=await fixture();await f.session.enable();f.session.record(f.request());await f.session.start();
    fetcher.mockImplementation((url:string)=>Promise.resolve(new Response(JSON.stringify(metadata(url,{sttReady:false,ttsReady:false,reason:'Local lane unavailable.'})))));
    await vi.advanceTimersByTimeAsync(2000);expect(f.recording.cancel).toHaveBeenCalledOnce();expect(f.emit).toHaveBeenCalledWith(expect.objectContaining({type:'failed',code:'voice_unavailable'}));expect(fetcher.mock.calls.some(([url])=>url.endsWith('/transcribe'))).toBe(false);
  });
  it.each(['transcribing','speaking'])('busy health during our own %s does not cancel an admitted request',async(lane)=>{
    const f=await fixture();await f.session.enable();let finish!:(r:Response)=>void;let signal!:AbortSignal;
    fetcher.mockImplementation((url:string,options:RequestInit)=>{
      if(url.endsWith(lane==='transcribing'?'/transcribe':'/speak')){signal=options.signal!;return new Promise(r=>{finish=r;});}
      return Promise.resolve(new Response(JSON.stringify(metadata(url,{sttReady:false,ttsReady:false,reason:'Local lane busy.'}))));
    });
    let stopping:Promise<void>|undefined;
    if(lane==='transcribing'){f.session.record(f.request());await f.session.start();stopping=f.session.stop();}else f.session.speak({...f.request(),text:'Grounded narration'});
    await flush();await vi.advanceTimersByTimeAsync(2000);expect(signal.aborted).toBe(false);expect(f.emit.mock.calls.some(([e])=>e.type==='failed')).toBe(false);
    finish(lane==='transcribing'?new Response(JSON.stringify({...f.request(),text:'Fresh transcript'})):new Response(new Uint8Array([0,0]),{headers:{'Content-Type':'audio/pcm','X-Sample-Rate':'24000'}}));await stopping;await flush();
    expect(f.emit.mock.calls.some(([e])=>e.type===(lane==='transcribing'?'transcript':'finished'))).toBe(true);
  });
  it.each(['phone-busy','service-gone','service-replaced'])('%s during buffered playback stops parent audio and suppresses a late finished callback',async(reason)=>{
    const f=await fixture();await f.session.enable();let finish!:()=>void;let signal!:AbortSignal;
    vi.mocked(f.media.play).mockImplementation((_audio,s)=>{signal=s;return new Promise(r=>{finish=r;});});
    f.session.speak({...f.request(),text:'Grounded narration'});await flush();
    expect(f.media.play).toHaveBeenCalledOnce();
    fetcher.mockImplementation((url:string)=>Promise.resolve(new Response(JSON.stringify(isStatus(url)?metadata(url,{sttReady:false,ttsReady:false,reason},false):{}))));
    await vi.advanceTimersByTimeAsync(2000);
    expect(signal.aborted).toBe(true);expect(f.media.dispose).toHaveBeenCalledOnce();expect(f.view()).toBeNull();
    expect(fetcher.mock.calls.some(([url])=>new URL(url).searchParams.get('sessionId')===f.opened.sessionId)).toBe(true);
    expect(f.emit).toHaveBeenCalledWith({sessionId:f.opened.sessionId,type:'closed'});
    const count=f.emit.mock.calls.length;finish();await flush();
    expect(f.emit).toHaveBeenCalledTimes(count);expect(f.emit.mock.calls.some(([event])=>event.type==='finished')).toBe(false);
  });
  it('busy lane health with a valid original lease does not stop buffered playback or leak internal lease fields',async()=>{
    const f=await fixture();await f.session.enable();let finish!:()=>void;let signal!:AbortSignal;
    vi.mocked(f.media.play).mockImplementation((_audio,s)=>{signal=s;return new Promise(r=>{finish=r;});});
    f.session.speak({...f.request(),text:'Grounded narration'});await flush();
    const busy={sttReady:false,ttsReady:false,reason:'Local lane busy.'};
    fetcher.mockImplementation((url:string)=>Promise.resolve(new Response(JSON.stringify(metadata(url,busy)))));
    await vi.advanceTimersByTimeAsync(2000);expect(signal.aborted).toBe(false);expect(f.media.dispose).not.toHaveBeenCalled();
    expect(await f.session.status()).toEqual(busy);expect(f.view()).not.toHaveProperty('sessionLeaseValid');
    finish();await flush();expect(f.emit).toHaveBeenCalledWith({...f.request(),type:'finished'});
  });
  it('missing internal session-lease metadata fails closed during buffered playback',async()=>{
    const f=await fixture();await f.session.enable();let signal!:AbortSignal;
    vi.mocked(f.media.play).mockImplementation((_audio,s)=>{signal=s;return new Promise(()=>{});});
    f.session.speak({...f.request(),text:'Grounded narration'});await flush();
    fetcher.mockImplementation(()=>Promise.resolve(new Response(JSON.stringify(ready))));
    await vi.advanceTimersByTimeAsync(2000);expect(signal.aborted).toBe(true);expect(f.media.dispose).toHaveBeenCalledOnce();
  });
  it('two global active requests and one per mounted screen are enforced',async()=>{
    const a=await fixture('a'),b=await fixture('b'),c=await fixture('c');for(const f of [a,b,c])await f.session.enable();
    a.session.record(a.request());b.session.record(b.request());expect(()=>c.session.record(c.request())).toThrow(expect.objectContaining({code:'voice_busy'}));expect(()=>a.session.record(a.request('two'))).toThrow();
    await a.session.cancel(a.request());expect(c.session.record(c.request())).toMatchObject({state:'awaiting-start'});
  });
  it('two mounted screens of the same plugin share one parent capture admission',async()=>{
    const a=await fixture(),b=await fixture();await a.session.enable();await b.session.enable();a.session.record(a.request());
    expect(()=>b.session.record(b.request())).toThrow(expect.objectContaining({code:'voice_busy'}));await a.session.cancel(a.request());expect(b.session.record(b.request())).toMatchObject({state:'awaiting-start'});
  });
  it('closing permits a new session with fresh owner consent',async()=>{
    const f=await fixture();await f.session.enable();await f.session.close(f.opened.sessionId);const next=await f.session.open();expect(next.sessionId).not.toBe(f.opened.sessionId);expect(f.view()?.enabled).toBe(false);
  });
});

describe('canonical parent recording format',()=>{
  it('resamples actual48kHz to15 seconds of16k monoPCM16 within480044 bytes',()=>{
    const bytes=voiceWav(new Float32Array(48_000*15).fill(.5),48_000);const data=new DataView(bytes.buffer);
    expect(bytes.length).toBe(480_044);expect(data.getUint32(24,true)).toBe(16_000);expect(data.getUint16(22,true)).toBe(1);expect(data.getUint32(40,true)).toBe(480_000);expect(data.getInt16(44,true)).toBe(16384);
  });
  it('rejects duration overflow and invalid rate before allocating output',()=>{
    expect(()=>voiceWav(new Float32Array(240_001),16000)).toThrow();expect(()=>voiceWav(new Float32Array(1),NaN)).toThrow();expect(()=>voiceWav(new Float32Array(0),16000)).toThrow();
  });
});
