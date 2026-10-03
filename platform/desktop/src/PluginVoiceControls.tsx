import type { PluginVoiceSession, VoiceView } from './pluginVoice';

/** This consent/capture bar is outside the opaque plugin iframe. */
export function PluginVoiceControls({view,session,pluginName}:{view:VoiceView|null;session:PluginVoiceSession|null;pluginName:string}) {
  if (!view || !session) return null;
  const busy = ['awaiting-start','recording','transcribing','speaking'].includes(view.state);
  return <section className="banner" aria-label="OAIY microphone session" style={{display:'flex',flexWrap:'wrap',gap:12,alignItems:'center'}}>
    <div style={{flex:'1 1 280px'}}>
      <strong>Voice with {pluginName}</strong>
      <p style={{margin:'4px 0',fontSize:13}}>OAIY asks for native microphone permission for this session. You control Start and Stop. Recording lasts at most 15 seconds; only its transcript reaches the plugin. Playback uses the local voice service.</p>
      <span role="status" aria-live="polite">{view.state === 'recording' ? `Recording ${(view.elapsedMs/1000).toFixed(1)} / 15 seconds` : view.state.replaceAll('-',' ')}{view.message ? ` — ${view.message}` : ''}</span>
      {view.reason && <p style={{margin:'4px 0',fontSize:13}}>{view.reason}</p>}
    </div>
    {!view.enabled && <><button className="btn" onClick={()=>void session.enable()}>Enable microphone session</button><button className="btn" onClick={()=>session.decline()}>Decline</button></>}
    {view.state === 'awaiting-start' && <button className="btn" disabled={!view.sttReady} onClick={()=>void session.start()}>Start recording</button>}
    {view.state === 'recording' && <button className="btn" onClick={()=>void session.stop()}>Stop recording</button>}
    {busy && <button className="btn" onClick={()=>void session.close(view.sessionId)}>Stop voice session</button>}
    {!busy && view.enabled && <button className="btn" onClick={()=>void session.close(view.sessionId)}>Close voice session</button>}
  </section>;
}
