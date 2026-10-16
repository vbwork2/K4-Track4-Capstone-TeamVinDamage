import * as React from 'react';
import {paintLevel,type FrameState} from './model';
const PX=16, CX=260,CY=283;
const coord=(x:number,y:number):[number,number]=>[CX-y*PX,CY-x*PX];
const grid=Array.from({length:31},(_,i)=>i-15);
const colors={neutral:'#42b4f7',amber:'#fbbf57',red:'#ff6679'};
export function BevView({state,showGt,showLidar,live}:{state:FrameState;showGt:boolean;showLidar:boolean;live:boolean}){
 const {tracks,gt}=state;
 return <svg className="vd-bev-svg" viewBox="0 0 520 568" role="img" aria-label="Bird-eye view: vị trí các vật theo hệ tọa độ xe, x tiến và y trái">
  <defs><pattern id="bevdots" width="16" height="16" patternUnits="userSpaceOnUse"><circle cx="0" cy="0" r=".7" fill="#33506b"/></pattern><radialGradient id="bevglow"><stop stopColor="#203d59" stopOpacity=".45"/><stop offset="1" stopColor="#091526" stopOpacity="0"/></radialGradient><filter id="vdlight"><feGaussianBlur stdDeviation="4"/></filter></defs>
  <rect width="520" height="568" fill="#091524"/><rect width="520" height="568" fill="url(#bevdots)"/>
  <circle cx={CX} cy={CY} r="235" fill="url(#bevglow)"/>
  {grid.map(n=><React.Fragment key={n}><line x1={CX+n*PX} y1={30} x2={CX+n*PX} y2={536} stroke={n%5===0?'#29435a':'#1c3247'} strokeWidth={n%5===0?1:.5} opacity={n%5===0?.9:.55}/><line x1={22} y1={CY+n*PX} x2={498} y2={CY+n*PX} stroke={n%5===0?'#29435a':'#1c3247'} strokeWidth={n%5===0?1:.5} opacity={n%5===0?.9:.55}/></React.Fragment>)}
  {[-10,-5,5,10].map(n=><g key={n}><text x="12" y={CY-n*PX-4} fill="#68809b" fontSize="10">{n}m</text></g>)}
  {/* All filled polygons are VISUAL guides only; warning decisions come from backend track.level. */}
  <rect x={CX-65} y={CY-49} width="30" height="98" rx="6" fill="#edb44c" fillOpacity=".12" stroke="#c89b4a" strokeOpacity=".6" strokeDasharray="4 4"/>
  <rect x={CX+35} y={CY-49} width="30" height="98" rx="6" fill="#edb44c" fillOpacity=".12" stroke="#c89b4a" strokeOpacity=".6" strokeDasharray="4 4"/>
  <path d={`M ${CX-35} ${CY-42} H ${CX+35} L ${CX+92} ${CY-164} H ${CX-92} Z`} fill="#59b8df" fillOpacity=".05" stroke="#4388a9" strokeDasharray="6 6" strokeOpacity=".55"/>
  <path d={`M ${CX-40} ${CY-50} H ${CX+40} L ${CX+95} ${CY-122} H ${CX-95} Z`} fill="#f59e5e" fillOpacity=".07"/>
  {live&&<path d={`M0 ${CY} H520 V568 H0Z`} fill="#081321" opacity=".86"/>}
  {showLidar&&state.lidar_points?.slice(0,10000).map((p,i)=>{const [xx,yy]=coord(p[0],p[1]);return xx>4&&xx<516&&yy>4&&yy<564?<circle key={i} cx={xx} cy={yy} r=".9" fill={p[2]>.8?'#f5bf63':p[2]>.1?'#5dd4c9':'#447b9b'} opacity=".72"/>:null;})}
  {showGt&&gt.map((g,i)=>{const [x,y]=coord(g.x,g.y);return <g key={i} transform={`translate(${x} ${y}) rotate(${-(g.yaw??0)*180/Math.PI})`}><rect x={-(g.w??1)/2*PX} y={-(g.l??2)/2*PX} width={(g.w??1)*PX} height={(g.l??2)*PX} fill="none" stroke="#a68cff" strokeWidth="1.7" strokeDasharray="5 3"/></g>;})}
  {tracks.map(t=>{const [x,y]=coord(t.x,t.y),c=colors[paintLevel(t)];const history=t.trail.map(p=>coord(p[0],p[1]).join(',')).join(' ');
   return <g key={t.id}>
    {t.trail.length>1&&<polyline points={history} stroke={c} strokeWidth="2" opacity=".7" fill="none" strokeDasharray={t.uncertain?'4 4':undefined}/>}<circle cx={x} cy={y} r="13" fill={c} opacity=".17" filter="url(#vdlight)"/>
    <circle cx={x} cy={y} r={t.level===2&&!t.uncertain?8:6.5} fill="#0b1726" stroke={c} strokeWidth={t.uncertain?2.7:2}/><circle cx={x} cy={y} r="2.6" fill={c}/>
    {(Math.abs(t.vx)+Math.abs(t.vy)>.1)&&<path d={`M${x} ${y} L${x-t.vy*5} ${y-t.vx*5}`} stroke={c} strokeWidth="2" markerEnd="none"/>}
    <rect x={x+12} y={y-18} width={Math.max(52,String(t.id).length*9+31)} height="22" rx="5" fill="#101f31" stroke="#2b465a"/><text x={x+19} y={y-3} fill={c} fontSize="11" fontWeight="700">ID {t.id}</text>
   </g>;
  })}
  <g transform={`translate(${CX},${CY-(state.ego_box?.x_offset??0)*PX})`}><rect x={-(state.ego_box?.width??1.9)*PX/2} y={-(state.ego_box?.length??4.3)*PX/2} width={(state.ego_box?.width??1.9)*PX} height={(state.ego_box?.length??4.3)*PX} rx="6" fill="#dce9f3" stroke="#8bd7f9" strokeWidth="2"/><rect x="-11" y="-20" width="22" height="27" rx="4" fill="#1d526f"/><path d="M-9 -28 L0 -39 L9 -28" fill="none" stroke="#4fcdff" strokeWidth="3"/><rect x="-11" y="16" width="22" height="8" rx="3" fill="#6c90a9"/></g>
  <text x={CX} y={CY+57} fill="#cfebff" textAnchor="middle" fontSize="11" letterSpacing="2">EGO</text>
  <path d="M485 45 l0 -22 m-6 7 l6 -7 l6 7" stroke="#92bfd6" fill="none" strokeWidth="2"/><text x="485" y="61" fontSize="10" fill="#92bfd6" textAnchor="middle">FRONT</text>
  <rect x="12" y="535" width="53" height="20" rx="4" fill="#162c41"/><text x="38" y="549" textAnchor="middle" fill="#aec6d9" fontSize="10">1 m/grid</text>
 </svg>;
}
