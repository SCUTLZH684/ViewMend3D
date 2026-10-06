import assert from 'node:assert/strict';
import { replayStep, sequenceStep } from '../web/replay-state.js';
const frames=Array.from({length:8},(_,i)=>({event:i+1,decision:{selected_index:i},checkpoint_index:i}));
for(let i=0;i<8;i++) {
  const observed=replayStep(frames,i);
  assert.equal(observed.frame,frames[i]); assert.equal(observed.mapFrame,frames[i]); assert.equal(observed.decision,null);
  const planned=replayStep(frames,i,true);
  assert.equal(planned.frame,frames[i]); assert.equal(planned.mapFrame,frames[i]);
  assert.equal(planned.decision,frames[i+1]?.decision || null);
  assert.equal(replayStep(frames,i,false,true).mapFrame,frames[Math.max(0,i-1)]);
}
assert.throws(()=>replayStep(frames,8)); assert.throws(()=>replayStep(frames,-1));
assert.deepEqual(sequenceStep(8,0),{phase:'reference',index:-1});
for(let i=0;i<8;i++) {
  assert.deepEqual(sequenceStep(8,2*i+1),{phase:'observed',index:i});
  if(i<7) assert.deepEqual(sequenceStep(8,2*i+2),{phase:'plan',index:i});
}
assert.throws(()=>sequenceStep(8,16)); assert.throws(()=>sequenceStep(0,0));
console.log('Replay decision/acquisition and before/after alignment passed (8 frames).');
