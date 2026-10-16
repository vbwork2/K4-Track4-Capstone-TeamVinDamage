import assert from 'node:assert/strict';
import {test} from 'node:test';
import {parseFrameState,paintLevel} from '../src/warning/model.ts';
import {mockFrame} from '../src/warning/mock.ts';
test('mock scenarios are not real timing metrics',()=>{
 const frame=mockFrame(83,'side');assert.equal(frame.stats.fps,0);assert.equal(frame.stats.latency_ms,null);assert.equal(Object.keys(frame.images).length,0);
 assert.equal(frame.source,'phenikaa');assert.equal(frame.dets.length>0,true);
});
test('cut edge objects stay uncertain regardless of backend level',()=>{
 const input={tracks:[{id:1,x:1,y:2,level:2,uncertain:true}],dets:[{cam:'CAM_P_L',xyxy:[0,1,40,99],level:2,uncertain:true}]};
 const parsed=parseFrameState(input);assert.equal(parsed.tracks[0].level,0);assert.equal(parsed.dets[0].level,0);assert.equal(paintLevel(parsed.tracks[0]),'amber');
});
test('invalid camera image links blocked and no invented results',()=>{
 const parsed=parseFrameState({images:{A:'javascript:alert(1)',B:'data:text/html,<h1>x</h1>',C:'data:image/jpeg;base64,YWJj'},tracks:[],dets:[]});
 assert.deepEqual(Object.keys(parsed.images),['C']);assert.equal(parsed.stats.fps,null);assert.equal(parsed.tracks.length,0);
});
