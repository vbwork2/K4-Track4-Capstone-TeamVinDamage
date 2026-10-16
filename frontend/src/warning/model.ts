/** VinDamage FE contract. Spatial positions in metres in ego coordinates: x forward, y left. */
export const CAMERAS = ['CAM_P_F','CAM_P_FL','CAM_P_FR','CAM_P_L','CAM_P_R','CAM_P_B'] as const;
export type Zone = 'SAT_SUON'|'TAT_DAU'|'PHIA_TRUOC';
export type Mode = 'replay'|'live';
export type Level = 0|1|2;
export interface Detection {cam:string;cls:string;conf:number;xyxy:[number,number,number,number];track_id?:number|string;level:Level;uncertain:boolean}
export interface Track {id:number|string;cls:string;x:number;y:number;vx:number;vy:number;trail:number[][];zone:Zone|null;level:Level;ttc:number|null;uncertain:boolean}
export interface GroundTruth {cls:string;x:number;y:number;l?:number;w?:number;yaw?:number}
export interface FrameState {
  t:number;source:string;scene:string;model:string;images:Record<string,string>;
  image_sizes?:Record<string,[number,number]>;
  dets:Detection[];tracks:Track[];gt:GroundTruth[];
  lidar_points?:number[][]; // ego frame x forward, y left, z up: optional P-115 LiDAR BEV layer
  ego_box?:{length:number;width:number;x_offset:number};
  zone_cfg?:Record<string,unknown>;
  stats:{fps:number|null;latency_ms:number|null;n_tracks:number;alerts:Record<Zone,number>};
}
export const ZONES:Zone[]=['SAT_SUON','TAT_DAU','PHIA_TRUOC'];
export const ZONE_NAME:Record<Zone,string>={SAT_SUON:'SÁT SƯỜN',TAT_DAU:'TẠT ĐẦU',PHIA_TRUOC:'PHÍA TRƯỚC'};
export const CAM_NAMES:Record<string,string>={CAM_P_F:'FRONT',CAM_P_FL:'FRONT LEFT',CAM_P_FR:'FRONT RIGHT',CAM_P_L:'LEFT SIDE',CAM_P_R:'RIGHT SIDE',CAM_P_B:'REAR',CAM_FRONT:'FRONT',CAM_FRONT_LEFT:'FRONT LEFT',CAM_FRONT_RIGHT:'FRONT RIGHT',CAM_BACK:'REAR',CAM_BACK_LEFT:'REAR LEFT',CAM_BACK_RIGHT:'REAR RIGHT',WEBCAM:'WEBCAM'};
export const emptyState=(source='phenikaa'):FrameState=>({t:0,source,scene:'',model:'—',images:{},dets:[],tracks:[],gt:[],stats:{fps:null,latency_ms:null,n_tracks:0,alerts:{SAT_SUON:0,TAT_DAU:0,PHIA_TRUOC:0}}});
const finite=(v:unknown,defaultValue=0)=>typeof v==='number'&&Number.isFinite(v)?v:defaultValue;
const zone=(v:unknown):Zone|null=>ZONES.includes(v as Zone)?v as Zone:null;
const level=(v:unknown):Level=>v===2?2:v===1?1:0;
const isRecord=(v:unknown):v is Record<string,unknown>=>typeof v==='object'&&v!==null&&!Array.isArray(v);
/** Validate untrusted API/JSON input. Never infer a red alert from an uncertain track. */
export function parseFrameState(value:unknown):FrameState {
  if(!isRecord(value))throw new Error('Expected a state object');
  const images:Record<string,string>={};
  if(isRecord(value.images))for(const [cam,raw] of Object.entries(value.images)){
    // Allow relative URLs and generated JPEG data URLs, not javascript: or arbitrary HTML payloads.
    if(typeof raw==='string'&&(raw.startsWith('data:image/jpeg;base64,')||raw.startsWith('data:image/png;base64,')||raw.startsWith('/')||raw.startsWith('blob:')||raw.startsWith('http://')||raw.startsWith('https://')))images[cam]=raw;
  }
  const dets:Detection[]=Array.isArray(value.dets)?value.dets.slice(0,500).filter(isRecord).map(d=>({
    cam:String(d.cam??''),cls:String(d.cls??'object'),conf:Math.max(0,Math.min(1,finite(d.conf))),
    xyxy:Array.isArray(d.xyxy)&&d.xyxy.length===4?d.xyxy.map(x=>finite(x)) as Detection['xyxy']:[0,0,0,0],
    track_id:typeof d.track_id==='string'||typeof d.track_id==='number'?d.track_id:undefined,
    uncertain:Boolean(d.uncertain),level:Boolean(d.uncertain)?0:level(d.level)
  })):[];
  const tracks:Track[]=Array.isArray(value.tracks)?value.tracks.slice(0,500).filter(isRecord).map((tr,i)=>({
    id:typeof tr.id==='number'||typeof tr.id==='string'?tr.id:i,cls:String(tr.cls??'object'),x:finite(tr.x),y:finite(tr.y),vx:finite(tr.vx),vy:finite(tr.vy),
    trail:Array.isArray(tr.trail)?tr.trail.slice(-30).filter((p):p is number[]=>Array.isArray(p)&&p.length>=2&&p.slice(0,2).every(n=>typeof n==='number'&&Number.isFinite(n))).map(p=>p.slice(0,2)):[],
    zone:zone(tr.zone),level:Boolean(tr.uncertain)?0:level(tr.level),uncertain:Boolean(tr.uncertain),ttc:typeof tr.ttc==='number'&&Number.isFinite(tr.ttc)?tr.ttc:null
  })):[];
  const gt:GroundTruth[]=Array.isArray(value.gt)?value.gt.slice(0,500).filter(isRecord).map(g=>({cls:String(g.cls??'object'),x:finite(g.x),y:finite(g.y),l:finite(g.l,4),w:finite(g.w,2),yaw:finite(g.yaw)})):[];
  const sizes:Record<string,[number,number]>={};
  if(isRecord(value.image_sizes))for(const [cam,size] of Object.entries(value.image_sizes))if(Array.isArray(size)&&size.length===2&&size.every(n=>typeof n==='number'&&Number.isFinite(n)&&n>0))sizes[cam]=[size[0],size[1]];
  const rawPoints=Array.isArray(value.lidar_points)?value.lidar_points:[];
  const lidar_points=rawPoints.slice(0,15000).filter((p):p is number[]=>Array.isArray(p)&&p.length>=2&&p.slice(0,3).every(n=>typeof n==='number'&&Number.isFinite(n))).map(p=>p.slice(0,3));
  const stats=isRecord(value.stats)?value.stats:{};const alerts:Record<Zone,number>={SAT_SUON:0,TAT_DAU:0,PHIA_TRUOC:0};
  // Treat the backend's reported alert count as authoritative (no manufactured predictions).
  if(isRecord(stats.alerts))for(const k of ZONES)alerts[k]=Math.max(0,Math.floor(finite(stats.alerts[k])));
  const ego=isRecord(value.ego_box)?value.ego_box:{};
  return {t:finite(value.t),source:String(value.source??'unknown'),scene:String(value.scene??''),model:String(value.model??'—'),images,image_sizes:sizes,
    dets,tracks,gt,lidar_points,
    ego_box:{length:Math.max(.5,finite(ego.length,4.3)),width:Math.max(.5,finite(ego.width,1.9)),x_offset:finite(ego.x_offset)},
    stats:{fps:typeof stats.fps==='number'&&Number.isFinite(stats.fps)?stats.fps:null,latency_ms:typeof stats.latency_ms==='number'&&Number.isFinite(stats.latency_ms)?stats.latency_ms:null,
      n_tracks:Math.max(0,Math.floor(finite(stats.n_tracks,tracks.length))),alerts}};
}
export function paintLevel(item:Pick<Track,'level'|'uncertain'>|Pick<Detection,'level'|'uncertain'>):'neutral'|'amber'|'red'{return item.uncertain?'amber':item.level===2?'red':item.level===1?'amber':'neutral';}
