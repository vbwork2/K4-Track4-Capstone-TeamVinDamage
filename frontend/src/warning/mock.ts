import {CAMERAS,emptyState,type FrameState,type Track,type Zone} from './model.ts';
export type Scenario='side'|'cutin'|'uncertain'|'clear';
export const scenarioNames:Record<Scenario,string>={side:'Xe máy vượt sát sườn',cutin:'Xe máy tạt đầu',uncertain:'Vật bị cắt mép ảnh',clear:'Lưu thông bình thường'};
const interpolate=(a:number,b:number,t:number)=>a+(b-a)*t;
export const TOTAL_FRAMES=240;
/** Deterministic, entirely synthetic mock fixtures: never report these numbers as measurements. */
export function mockFrame(index:number,scenario:Scenario,model='yolo11n.pt'):FrameState {
 const i=Math.max(0,Math.min(TOTAL_FRAMES-1,Math.floor(index))),p=i/(TOTAL_FRAMES-1);
 const tracks:Track[]=[];
 const add=(id:number,cls:string,x:number,y:number,vx:number,vy:number,zone:Zone|null,level:0|1|2,uncertain=false)=>{
   tracks.push({id,cls,x,y,vx,vy,zone,level:uncertain?0:level,uncertain,ttc:!uncertain&&level>0?Number((1.1+p*.3).toFixed(1)):null,
     trail:Array.from({length:11},(_,n)=>[x-vx*(10-n)*.1,y-vy*(10-n)*.1])});
 };
 if(scenario==='side')add(21,'motorcycle',interpolate(-4,4,p),-2.45,3.8,0,'SAT_SUON',p>.17&&p<.84?2:1);
 if(scenario==='cutin')add(12,'motorcycle',interpolate(10,4,p),interpolate(-4,-.75,p),-3.2,1.7,'TAT_DAU',p>.42?2:1);
 if(scenario==='uncertain')add(31,'motorcycle',interpolate(-2,3,p),2.35,1.6,0,'SAT_SUON',0,true);
 add(44,'car',interpolate(11,8,p),3.7,-.6,0,null,0);
 if(scenario==='clear')add(72,'bicycle',interpolate(9,7,p),-5,0,0,null,0);
 const state=emptyState('phenikaa');
 state.t=1700000000+i*.1;state.scene='COMPLEX CITY / SYNTHETIC';state.model=model;state.tracks=tracks;
 state.dets=tracks.filter(t=>t.cls!=='bicycle').map(t=>{
   const cam=t.id===21?'CAM_P_R':t.id===31?'CAM_P_L':t.id===12?'CAM_P_F':'CAM_P_FL';
   const cx=t.id===31?1810:t.id===21?interpolate(1550,970,p):t.id===12?interpolate(1250,700,p):560;
   const cy=t.id===44?770:915;
   return {cam,cls:t.cls,conf:t.uncertain?.51:t.level===2?.91:.84,xyxy:[cx-120,cy-240,cx+125,cy+80] as [number,number,number,number],track_id:t.id,level:t.level,uncertain:t.uncertain};
 });
 state.gt=tracks.slice(0,2).map(t=>({cls:t.cls,x:t.x+.3,y:t.y-.2,l:2.2,w:1.0,yaw:0}));
 state.images={};state.image_sizes=Object.fromEntries(CAMERAS.map(c=>[c,[1920,1536]]));
 state.stats={fps:0,latency_ms:null,n_tracks:tracks.length,alerts:{SAT_SUON:tracks.filter(t=>t.zone==='SAT_SUON'&&t.level>0&&!t.uncertain).length,TAT_DAU:tracks.filter(t=>t.zone==='TAT_DAU'&&t.level>0&&!t.uncertain).length,PHIA_TRUOC:0}};
 return state;
}
