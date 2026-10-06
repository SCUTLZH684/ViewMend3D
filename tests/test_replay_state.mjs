import assert from 'node:assert/strict';
import { replayStep } from '../web/replay-state.js';
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
console.log('Replay decision/acquisition and before/after alignment passed (8 frames).');
