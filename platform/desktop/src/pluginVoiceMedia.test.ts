import {afterEach,beforeEach,describe,expect,it,vi} from 'vitest';
import {BrowserVoiceMedia} from './pluginVoiceMedia';

/** Synthetic browser media only; these tests never acquire a real device. */
let media:BrowserVoiceMedia;
let context:ReturnType<typeof audioContext>;
let tracks:Array<{stop:ReturnType<typeof vi.fn>}>;
function audioContext(){
  const source={connect:vi.fn(),disconnect:vi.fn()};
  const processor={connect:vi.fn(),disconnect:vi.fn(),onaudioprocess:null as ((event:{inputBuffer:{getChannelData(channel:number):Float32Array}})=>void)|null};
  const gain={connect:vi.fn(),disconnect:vi.fn(),gain:{value:1}};
  return {sampleRate:192_000,resume:vi.fn().mockResolvedValue(undefined),close:vi.fn().mockResolvedValue(undefined),destination:{},source,processor,gain,
    createMediaStreamSource:vi.fn(()=>source),createScriptProcessor:vi.fn(()=>processor),createGain:vi.fn(()=>gain)};
}
beforeEach(()=>{
  tracks=[];context=audioContext();
  vi.stubGlobal('navigator',{mediaDevices:{getUserMedia:vi.fn().mockImplementation(()=>{const track={stop:vi.fn()};tracks.push(track);return Promise.resolve({getTracks:()=>[track]});})}});
  vi.stubGlobal('AudioContext',vi.fn().mockImplementation(function(){return context;}));
  media=new BrowserVoiceMedia();
});
afterEach(()=>{media.dispose();vi.unstubAllGlobals();});
describe('trusted parent microphone lifecycle with synthetic media IO',()=>{
  it('native permission opt-in immediately releases its temporary stream',async()=>{
    await media.enable();expect(tracks).toHaveLength(1);expect(tracks[0].stop).toHaveBeenCalledOnce();expect(context.processor.onaudioprocess).toBeNull();
  });
  it.each(['resume','create','connect'])('releases every acquired track when %s fails before capture ownership',async(kind)=>{
    await media.enable();
    if(kind==='resume')context.resume.mockRejectedValue(new Error('resume denied'));
    if(kind==='create')context.createMediaStreamSource.mockImplementation(()=>{throw new Error('graph unavailable');});
    if(kind==='connect')context.processor.connect.mockImplementation(()=>{throw new Error('connect failed');});
    await expect(media.record()).rejects.toThrow();expect(tracks).toHaveLength(2);expect(tracks[1].stop).toHaveBeenCalledOnce();expect(context.processor.onaudioprocess).toBeNull();
  });
  it('192kHz device data is incrementally bounded and stops tracks at15 seconds without any UI timer',async()=>{
    await media.enable();const recording=await media.record();const samples=new Float32Array(4096).fill(.25);
    for(let i=0;i<704;i++)context.processor.onaudioprocess?.({inputBuffer:{getChannelData:()=>samples}});
    expect(tracks[1].stop).toHaveBeenCalledOnce();expect(context.processor.onaudioprocess).toBeNull();
    const wav=await recording.stop();expect(wav.length).toBe(480_044);const view=new DataView(wav.buffer);expect(view.getUint32(24,true)).toBe(16_000);expect(view.getInt16(44,true)).toBe(8192);expect(tracks[1].stop).toHaveBeenCalledOnce();
  });
  it('dispose stops current tracks and clears audio graph callback',async()=>{
    await media.enable();await media.record();media.dispose();expect(tracks[1].stop).toHaveBeenCalledOnce();expect(context.processor.onaudioprocess).toBeNull();expect(context.close).toHaveBeenCalledOnce();
  });
});
