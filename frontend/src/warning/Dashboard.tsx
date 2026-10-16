import * as React from 'react';
import {Activity,ArrowLeft,ArrowRight,CheckCircle2,CircleAlert,Clock3,Eye,FileJson,Film,Layers3,Pause,Play,RotateCcw,ShieldAlert,SlidersHorizontal,Video,VideoOff,Wifi} from 'lucide-react';
import {BevView} from './BevView';
import {CAMERAS,CAM_NAMES,emptyState,paintLevel,parseFrameState,ZONE_NAME,ZONES,type Detection,type FrameState,type Track,type Zone} from './model';
import {mockFrame,scenarioNames,TOTAL_FRAMES,type Scenario} from './mock';
import './warning.css';
type FeedMode='mock'|'api'|'json'|'webcam';
const zero='—';
const names={mock:'Demo mô phỏng',api:'Backend API',json:'Tệp JSON',webcam:'Webcam'} as const;
function uniqueCameras(state:FrameState,mode:FeedMode):string[]{
 if(mode==='webcam')return ['WEBCAM'];
 const fromState=Object.keys(state.images);
 const detected=state.dets.map(d=>d.cam);
 if(fromState.length===0&&detected.length===0)return [...CAMERAS];
 const provided=[...fromState,...detected].filter(Boolean);
 const preferred=provided.every(c=>c.startsWith('CAM_P_'))?[...CAMERAS]:['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK_RIGHT','CAM_BACK'];
 return [...new Set([...preferred.filter(c=>provided.includes(c)),...provided])].slice(0,6);
}
function CameraTile({name,state,synthetic,webcamRef,webcam}:{name:string;state:FrameState;synthetic:boolean;webcamRef?:React.RefObject<HTMLVideoElement|null>;webcam:boolean}){
 const dets=state.dets.filter(d=>d.cam===name);
 const strongest=dets.some(d=>!d.uncertain&&d.level===2)?'red':dets.some(d=>d.uncertain||d.level===1)?'amber':'neutral';
 const size=state.image_sizes?.[name]??[1920,1536];
 const image=state.images[name];
 return <article className={`vd-camera vd-camera--${strongest}`} aria-label={`${name}, ${dets.length} detections`}>
  <div className="vd-camera-bg">
    {image?<img src={image} alt={`Camera ${name}`} draggable={false}/>:webcam?<video ref={webcamRef} autoPlay muted playsInline/>:<div className="vd-camera-placeholder" aria-label="Chưa có ảnh camera"><div className="vd-perspective"><span/><span/><span/></div><div className="vd-no-frame"><VideoOff size={22}/><strong>NO CAMERA FRAME</strong><small>{synthetic?'SƠ ĐỒ MÔ PHỎNG · KHÔNG PHẢI VIDEO THẬT':'Chờ dữ liệu ảnh từ Backend'}</small></div></div>}
    {!webcam&&<svg className="vd-dets" viewBox={`0 0 ${size[0]} ${size[1]}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      {dets.map((d,i)=>{const x=Math.max(0,Math.min(size[0],d.xyxy[0])),y=Math.max(0,Math.min(size[1],d.xyxy[1]));const w=Math.max(0,Math.min(size[0],d.xyxy[2])-x),h=Math.max(0,Math.min(size[1],d.xyxy[3])-y);const c=paintLevel(d)==='red'?'#ff6273':paintLevel(d)==='amber'?'#ffbe63':'#65cced';
       return <g key={i}><rect x={x} y={y} width={w} height={h} stroke={c} strokeWidth="6" fill="none" strokeDasharray={d.uncertain?'20 14':undefined}/><rect x={x} y={Math.max(0,y-66)} width={Math.min(size[0]-x,Math.max(245,String(d.cls).length*19+130))} height="65" rx="4" fill={c}/><text x={x+14} y={Math.max(50,y-19)} fontSize="35" fill="#081521" fontWeight="700">{d.cls.toUpperCase()} · {(d.conf*100).toFixed(0)}%{d.uncertain?' ?':''}</text></g>;})}
    </svg>}
  </div>
  <div className="vd-cam-head"><span className="vd-cam-indicator"/><strong>{CAM_NAMES[name]??name.replaceAll('_',' ')}</strong><span className="vd-camera-id">{name}</span></div>
  <div className="vd-cam-foot"><span>{webcam?'LIVE PREVIEW':image?'IMAGE STREAM':'NO FRAME'}</span><span className={dets.length?'has-detections':''}>{webcam?'AI chưa kết nối':`${dets.length} OBJECT${dets.length===1?'':'S'}`}</span></div>
  {strongest!=='neutral'&&<div className={`vd-cam-alert vd-cam-alert--${strongest}`}>{strongest==='red'?'NGUY HIỂM':dets.some(d=>d.uncertain)?'KHÔNG CHẮC':'CẢNH BÁO'}</div>}
 </article>;
}
function TrackTable({tracks}:{tracks:Track[]}){
 return <div className="vd-tracks-body" role="table" aria-label="Tracked objects"><div role="row" className="vd-tablehead"><span>ĐỐI TƯỢNG</span><span>VỊ TRÍ</span><span>TRẠNG THÁI</span></div>
 {tracks.length===0?<div className="vd-no-tracks">Chưa có track được gửi tới FE.</div>:tracks.slice(0,7).map(t=><div role="row" className="vd-track-row" key={t.id}>
  <span><span className={`vd-track-dot ${paintLevel(t)}`}/><strong>#{t.id}</strong> <small>{t.cls}</small></span>
  <span className="vd-mono">{Math.hypot(t.x,t.y).toFixed(1)} m</span>
  <span className={`vd-track-status ${paintLevel(t)}`}>{t.uncertain?'KHÔNG CHẮC':t.level===2?'NGUY HIỂM':t.level===1?'CẢNH BÁO':'BÌNH THƯỜNG'}</span>
 </div>)}</div>;
}
export default function Dashboard(){
 const [feed,setFeed]=React.useState<FeedMode>('mock');
 const [scenario,setScenario]=React.useState<Scenario>('side');
 const [frame,setFrame]=React.useState(83);
 const [playing,setPlaying]=React.useState(true);
 const [showGt,setShowGt]=React.useState(false);
 const [showLidar,setShowLidar]=React.useState(true);
 const [model,setModel]=React.useState('yolo11n.pt');
 const [external,setExternal]=React.useState<FrameState|null>(null);
 const [imported,setImported]=React.useState<FrameState[]>([]);
 const [online,setOnline]=React.useState(false);
 const [error,setError]=React.useState('');
 const [notice,setNotice]=React.useState('');
 const [recording,setRecording]=React.useState(false);
 const webcamRef=React.useRef<HTMLVideoElement|null>(null);
 const streamRef=React.useRef<MediaStream|null>(null);
 const captureRef=React.useRef<MediaStream|null>(null);
 const recorderRef=React.useRef<MediaRecorder|null>(null);
 const videoChunks=React.useRef<Blob[]>([]);
 const fileRef=React.useRef<HTMLInputElement|null>(null);
 const state=feed==='mock'?mockFrame(frame,scenario,model):feed==='webcam'?emptyState('webcam'):feed==='json'?(imported[Math.min(frame,imported.length-1)]??emptyState('json')):(external??emptyState('backend'));
 const total=feed==='mock'?TOTAL_FRAMES:feed==='json'?imported.length:0;
 const isSynthetic=feed==='mock';
 const cameras=uniqueCameras(state,feed);
 const hasCritical=state.tracks.some(t=>t.level===2&&!t.uncertain);
 const hasWarn=state.tracks.some(t=>t.level===1&&!t.uncertain);
 const uncertainCount=state.tracks.filter(t=>t.uncertain).length;
 const riskLabel=feed==='webcam'?'CHƯA CÓ AI':feed==='api'&&!online?'MẤT KẾT NỐI':hasCritical?'NGUY HIỂM':hasWarn?'CẢNH BÁO':uncertainCount?'KHÔNG CHẮC':'BÌNH THƯỜNG';
 const riskTone=hasCritical?'danger':feed==='api'&&!online||hasWarn||uncertainCount?'warning':'normal';
 const togglePlayback=React.useCallback(()=>setPlaying(x=>!x),[]);
 const step=React.useCallback((d:number)=>{setPlaying(false);setFrame(n=>Math.max(0,Math.min((feed==='json'?imported.length:TOTAL_FRAMES)-1,n+d)));},[feed,imported.length]);
 React.useEffect(()=>{if(!playing||(feed!=='mock'&&feed!=='json')||total<2)return;const timer=window.setInterval(()=>setFrame(n=>(n+1)%total),100);return ()=>window.clearInterval(timer);},[playing,feed,total]);
 React.useEffect(()=>{
  if(feed!=='api')return;
  let active=true,controller:AbortController|null=null;
  const request=async()=>{controller?.abort();controller=new AbortController();try{
    const response=await fetch('/warning-api/state',{cache:'no-store',signal:controller.signal});if(!response.ok)throw new Error(`HTTP ${response.status}`);
    const parsed=parseFrameState(await response.json());if(!active)return;setOnline(true);setError('');if(playing)setExternal(parsed);
  }catch(e){if(!active||(e instanceof DOMException&&e.name==='AbortError'))return;setOnline(false);setError(`Không thể đọc Backend: ${e instanceof Error?e.message:String(e)}`);setExternal(null);}};
  void request();const interval=window.setInterval(()=>void request(),350);
  return ()=>{active=false;window.clearInterval(interval);controller?.abort();};
 },[feed,playing]);
 React.useEffect(()=>{if(feed==='webcam'&&webcamRef.current&&streamRef.current)webcamRef.current.srcObject=streamRef.current;},[feed]);
 const stopWebcam=React.useCallback(()=>{streamRef.current?.getTracks().forEach(t=>t.stop());streamRef.current=null;},[]);
 React.useEffect(()=>()=>{stopWebcam();captureRef.current?.getTracks().forEach(t=>t.stop());},[stopWebcam]);
 const chooseFeed=(next:FeedMode)=>{if(feed==='webcam'&&next!=='webcam')stopWebcam();setFeed(next);setOnline(false);setError('');setPlaying(next==='mock'||next==='json');setFrame(0);setExternal(null);};
 const openWebcam=async()=>{if(!navigator.mediaDevices?.getUserMedia){setNotice('Trình duyệt yêu cầu HTTPS hoặc localhost để truy cập camera.');return;}
   try{const stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});stopWebcam();streamRef.current=stream;chooseFeed('webcam');streamRef.current=stream;setNotice('Webcam preview: chưa có AI detection / TTC.');}
   catch(e){setNotice(`Không mở được webcam: ${e instanceof Error?e.message:String(e)}`);}
 };
 const openFile=async(file?:File)=>{if(!file)return;try{if(file.size>30*1024*1024)throw new Error('JSON lớn hơn 30MB; gửi ảnh bằng URL hoặc API.');const parsed:unknown=JSON.parse(await file.text());
   const sequence=Array.isArray(parsed)?parsed:(typeof parsed==='object'&&parsed!==null&&'frames' in parsed)?(parsed as {frames:unknown}).frames:[parsed];
   if(!Array.isArray(sequence)||sequence.length===0)throw new Error('Không có frame nào');
   const states=sequence.slice(0,3000).map(parseFrameState);setImported(states);chooseFeed('json');setNotice(`Đã nạp ${states.length} frame từ JSON (không chạy model).`);
 }catch(e){setNotice(`JSON không hợp lệ: ${e instanceof Error?e.message:String(e)}`);}};
 const stopRecord=React.useCallback(()=>{if(recorderRef.current?.state==='recording')recorderRef.current.stop();setRecording(false);},[]);
 const toggleRecord=async()=>{
   if(recording){stopRecord();return;}
   if(!navigator.mediaDevices?.getDisplayMedia||typeof MediaRecorder==='undefined'){setNotice('Trình duyệt không hỗ trợ ghi tab. Dùng --record của OpenCV để xuất MP4.');return;}
   try{const capture=await navigator.mediaDevices.getDisplayMedia({video:{frameRate:20},audio:false});captureRef.current=capture;
     const type=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(m=>MediaRecorder.isTypeSupported(m))||'';
     const rec=new MediaRecorder(capture,type?{mimeType:type}:undefined);videoChunks.current=[];
     rec.ondataavailable=e=>{if(e.data.size>0)videoChunks.current.push(e.data);};
     rec.onstop=()=>{const blob=new Blob(videoChunks.current,{type:'video/webm'});if(blob.size){const url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`vindamage-${Date.now()}.webm`;link.click();window.setTimeout(()=>URL.revokeObjectURL(url),3000);}capture.getTracks().forEach(t=>t.stop());captureRef.current=null;setRecording(false);};
     capture.getVideoTracks()[0]?.addEventListener('ended',()=>stopRecord(),{once:true});recorderRef.current=rec;rec.start(1000);setRecording(true);setNotice('Đang ghi tab dưới dạng WebM (không phải MP4).');
   }catch(e){setNotice(`Không thể ghi màn hình: ${e instanceof Error?e.message:String(e)}`);}
 };
 React.useEffect(()=>{const handler=(e:KeyboardEvent)=>{
   if((e.target as HTMLElement|null)?.closest('input,select,textarea,[contenteditable=true]')||e.ctrlKey||e.metaKey||e.altKey)return;
   if(e.code==='Space'){e.preventDefault();if(feed!=='webcam')togglePlayback();}
   if(e.key==='ArrowLeft'){e.preventDefault();if(total)step(-1);}
   if(e.key==='ArrowRight'){e.preventDefault();if(total)step(1);}
   if(e.key.toLowerCase()==='g')setShowGt(v=>!v);
   if(e.key.toLowerCase()==='m'&&feed==='mock')setModel(v=>v==='yolo11n.pt'?'yolo11m.pt':'yolo11n.pt');
   if(e.key.toLowerCase()==='r')void toggleRecord();
   if(e.key.toLowerCase()==='q')setNotice('Trình duyệt không cho đóng tab bằng phím Q. Hãy đóng tab thủ công.');
  };window.addEventListener('keydown',handler);return ()=>window.removeEventListener('keydown',handler);
 // Keyboard listener refreshed as callbacks change so actions always use current feed and record status.
 });
 const sec=(frame*.1).toFixed(1).padStart(4,'0');
 return <div className="vd-shell">
  <header className="vd-header">
   <div className="vd-brand"><div className="vd-brandmark"><span>V</span><i/></div><div><strong>VinDamage<span className="vd-dot">.</span></strong><small>SURROUND RIDER WARNING</small></div></div>
   <div className="vd-header-right"><span className="vd-version">SURROUND SAFETY · FEATURE ONLY</span><span className={`vd-source-pill ${feed==='api'?(online?'on':'off'):feed==='mock'?'mock':''}`}><span/>{feed==='api'?(online?'BACKEND CONNECTED':'BACKEND OFFLINE'):feed==='mock'?'MOCK DATA':feed==='webcam'?'LOCAL CAMERA':'IMPORTED JSON'}</span></div>
  </header>
  <div className="vd-body vd-body--single">
   <div className="vd-feature-toolbar" aria-label="Điều khiển VinDamage">
     <div className="vd-feature-feeds">
       <span className="vd-feature-label">Nguồn dữ liệu</span>
       <div className="vd-feed-options" role="group" aria-label="Chọn nguồn dữ liệu">
        <button type="button" className={feed==='mock'?'selected':''} aria-pressed={feed==='mock'} onClick={()=>chooseFeed('mock')}><Film size={16}/><span>Demo</span></button>
        <button type="button" className={feed==='api'?'selected':''} aria-pressed={feed==='api'} onClick={()=>chooseFeed('api')}><Wifi size={16}/><span>Backend</span></button>
        <button type="button" className={feed==='json'?'selected':''} aria-pressed={feed==='json'} onClick={()=>fileRef.current?.click()}><FileJson size={16}/><span>Nhập JSON</span></button>
        <button type="button" className={feed==='webcam'?'selected':''} aria-pressed={feed==='webcam'} onClick={()=>void openWebcam()}><Video size={16}/><span>Webcam</span></button>
        <input ref={fileRef} aria-label="Chọn JSON state" type="file" accept=".json,application/json" hidden onChange={e=>{void openFile(e.target.files?.[0]);e.target.value='';}}/>
       </div>
     </div>
     <div className="vd-feature-scenarios">
       <label className="vd-feature-label" htmlFor="vd-scenario-picker">Kịch bản demo</label>
       <select id="vd-scenario-picker" value={scenario} onChange={e=>{const s=e.target.value as Scenario;chooseFeed('mock');setScenario(s);setFrame(s==='side'?83:s==='cutin'?130:s==='uncertain'?80:15);setPlaying(true);}}>
        {(Object.keys(scenarioNames) as Scenario[]).map(name=><option value={name} key={name}>{scenarioNames[name]}</option>)}
       </select>
     </div>
     <p className="vd-feature-disclaimer"><ShieldAlert size={16}/> Minh họa giao diện; không dùng làm hệ thống an toàn thực tế.</p>
   </div>
  <main className="vd-main">
   <div className="vd-page-title"><div><p className="vd-eyebrow"><span className="vd-eyebrow-line"/>PERCEPTION / SURROUND VIEW</p><h1>Surround Safety <span>Monitor</span></h1><p className="vd-subtitle">Giám sát đa camera · định vị BEV · cảnh báo phương tiện áp sát</p></div><div className="vd-title-actions"><span className="vd-mode-tag"><span className={feed==='api'&&online?'live':''}/>{names[feed]}</span><button type="button" className="vd-iconbtn" title="Quay lại" onClick={()=>{setFrame(0);setPlaying(false);}}><RotateCcw size={18}/></button></div></div>
   <section className={`vd-alert-banner ${riskTone}`} aria-live="polite"><div className="vd-alert-icon">{hasCritical?<ShieldAlert size={24}/>:hasWarn||uncertainCount?<CircleAlert size={24}/>:<CheckCircle2 size={24}/>}</div>
     <div className="vd-alert-text"><span>SAFETY STATUS <span className="vd-banner-sep">/</span> {feed==='mock'?'SIMULATED SCENARIO':(feed==='api'&&!online)?'WAITING FOR BACKEND':(feed==='webcam')?'CAMERA PREVIEW':'BACKEND REPORTED'}</span><strong>{riskLabel}</strong></div>
     <div className="vd-alert-extra"><span>ACTIVE ALERTS</span><strong>{ZONES.reduce((n,z)=>n+state.stats.alerts[z],0).toString().padStart(2,'0')}</strong></div><span className="vd-alert-flag">{isSynthetic?'DEMO ONLY':'FE VISUALIZATION'}</span>
   </section>
   <div className="vd-primary-grid"><section className="vd-panel vd-camera-panel"><div className="vd-panel-header"><div><div className="vd-panel-title"><span className="vd-title-icon"><Video size={17}/></span><h2>Multi-camera feed</h2><span className="vd-panel-counter">{cameras.length.toString().padStart(2,'0')} CAMERAS</span></div><p>Camera surveillance · Detection overlay</p></div><div className="vd-panel-end"><span className="vd-minor-dot"/> {feed==='webcam'?'1 camera':cameras.length+' camera channels'}</div></div>
    <div className={`vd-camera-grid ${feed==='webcam'?'single':''}`}>{cameras.map(name=><CameraTile key={name} name={name} state={state} synthetic={isSynthetic} webcam={feed==='webcam'} webcamRef={webcamRef}/>)}</div>
    <div className="vd-camera-legend"><span><i className="vd-legend-blue"/>Detection</span><span><i className="vd-legend-amber"/>Cảnh báo / Không chắc</span><span><i className="vd-legend-red"/>Nguy hiểm</span><span className="vd-legend-note">{isSynthetic?'BOX & TRACK: GENERATED MOCK':'Boxes: from Backend / JSON'}</span></div>
   </section>
   <section className="vd-panel vd-bev-panel"><div className="vd-panel-header"><div><div className="vd-panel-title"><span className="vd-title-icon"><Layers3 size={18}/></span><h2>Bird's eye view</h2><span className="vd-panel-counter">BEV 360°</span></div><p>Hệ tọa độ xe · x trước, y trái · đơn vị mét</p></div><button className="vd-iconbtn vd-small" title="Bật tắt LiDAR" aria-pressed={showLidar} onClick={()=>setShowLidar(v=>!v)}><SlidersHorizontal size={16}/></button></div>
    <div className="vd-bev-canvas"><BevView state={state} showGt={showGt} showLidar={showLidar} live={feed==='webcam'}/><div className="vd-bev-toplabel">TOP-DOWN <span>·</span> EGO FRAME</div><div className="vd-bev-bottomlabel">{state.lidar_points?.length?'LiDAR points · P-115-compatible overlay':'NO LIDAR POINTS'}<span>± 15m</span></div></div>
    <div className="vd-bev-controls"><label><input type="checkbox" checked={showGt} onChange={e=>setShowGt(e.target.checked)}/> Ground truth <small>(G)</small></label><label><input type="checkbox" checked={showLidar} onChange={e=>setShowLidar(e.target.checked)}/> LiDAR layer</label></div>
    <div className="vd-zone-legend"><span><i className="vd-zone-side"/>Sát sườn</span><span><i className="vd-zone-front"/>Tạt đầu</span><span><i className="vd-zone-line"/>Phía trước</span></div>
   </section></div>
   <div className="vd-bottom-grid"><section className="vd-panel vd-timeline-panel"><div className="vd-panel-header slim"><div className="vd-panel-title"><span className="vd-title-icon"><Film size={17}/></span><h2>Playback control</h2></div><span className="vd-panel-counter">{feed==='api'?'POLL 350MS':feed==='webcam'?'CAMERA LIVE':`${(frame+1).toString().padStart(3,'0')} / ${total.toString().padStart(3,'0')}`}</span></div>
    <div className="vd-playbar"><div className="vd-play-controls"><button aria-label="Lùi một frame" disabled={!total} onClick={()=>step(-1)}><ArrowLeft size={18}/></button><button className="vd-play-main" aria-label={playing?'Pause':'Play'} disabled={feed==='webcam'} onClick={togglePlayback}>{playing?<Pause size={18} fill="currentColor"/>:<Play size={18} fill="currentColor"/>}</button><button aria-label="Tiến một frame" disabled={!total} onClick={()=>step(1)}><ArrowRight size={18}/></button></div><span className="vd-play-time">{total?`00:${sec}`:'--:--'}</span><div className="vd-scrub">{total?<input type="range" aria-label="Frame timeline" min="0" max={Math.max(total-1,0)} value={frame} onChange={e=>{setFrame(Number(e.target.value));setPlaying(false);}}/>:<div className="vd-scrub-disabled"/>}</div><span className="vd-play-time">{total?`${(total*.1).toFixed(1)}s`:'LIVE'}</span><button className={`vd-record ${recording?'active':''}`} onClick={()=>void toggleRecord()}><span/>{recording?'STOP':'RECORD'}</button></div>
    <div className="vd-shortcuts"><span><kbd>Space</kbd> Pause</span><span><kbd>←</kbd><kbd>→</kbd> Frame</span><span><kbd>G</kbd> Ground truth</span><span><kbd>M</kbd> Mock model</span><span><kbd>R</kbd> Record WebM</span></div>
   </section>
   <section className="vd-panel vd-objects-panel"><div className="vd-panel-header slim"><div className="vd-panel-title"><span className="vd-title-icon"><Eye size={17}/></span><h2>Tracked objects</h2></div><span className="vd-panel-counter">{state.stats.n_tracks} ACTIVE</span></div><TrackTable tracks={state.tracks}/></section></div>
   <div className="vd-metrics-grid"><div className="vd-metric"><div><Activity size={17}/><span>FRAME RATE</span></div><strong>{feed==='mock'?'—':state.stats.fps===null?zero:state.stats.fps.toFixed(1)} <small>FPS</small></strong><span>{feed==='mock'?'Not measured · mock':'Reported by backend'}</span></div>
   <div className="vd-metric"><div><Clock3 size={17}/><span>PROCESSING LATENCY</span></div><strong>{state.stats.latency_ms===null?zero:state.stats.latency_ms.toFixed(0)} <small>ms</small></strong><span>{feed==='mock'?'No real timing':'Backend measurement'}</span></div>
   <div className="vd-metric"><div><Layers3 size={17}/><span>ACTIVE TRACKS</span></div><strong>{state.stats.n_tracks.toString().padStart(2,'0')} <small>objects</small></strong><span>Across available cameras</span></div>
   <div className="vd-metric"><div><ShieldAlert size={17}/><span>WARNING ZONES</span></div><strong>{ZONES.filter(z=>state.stats.alerts[z]>0).length} <small>/ 3 zones</small></strong><span>{ZONES.map(z=>`${ZONE_NAME[z]} ${state.stats.alerts[z]}`).join(' · ')}</span></div></div>
   {(error||notice)&&<div className="vd-notice" role="status"><CircleAlert size={16}/>{error||notice}<button type="button" onClick={()=>{setNotice('');setError('');}} aria-label="Đóng thông báo">×</button></div>}
   <footer className="vd-footer"><span>VIN DAMAGE · PERCEPTION & SAFETY ENGINEERING</span><span>{isSynthetic?'SYNTHETIC FIXTURE · NOT REAL-WORLD RESULTS':feed==='webcam'?'LOCAL CAMERA PREVIEW · NO MODEL INFERENCE':feed==='api'?'BACKEND STATE CONTRACT V1':'IMPORTED DATA · VERIFY SOURCE'}</span><span>FEATURE PAGE / V1.1</span></footer>
  </main></div>
 </div>;
}
