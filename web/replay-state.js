// Decision i selects observation i. Preparing after frame i uses decision i+1.
export function replayStep(frames, index, preparing=false, before=false) {
  if (!Number.isInteger(index) || index<0 || index>=frames.length) throw new Error('无效回放帧');
  return { frame:frames[index], next:frames[index+1] || null,
    decision:preparing ? frames[index+1]?.decision || null : null,
    mapFrame:frames[before && index>0 ? index-1 : index] };
}

// 0: reference, 1: first acquisition, 2: plan second, 3: acquire second…
export function sequenceStep(frameCount, step) {
  if(!Number.isInteger(frameCount) || frameCount<1 || !Number.isInteger(step) || step<0 || step>=frameCount*2)
    throw new Error('无效过程步骤');
  return step===0 ? {phase:'reference',index:-1} : {
    phase:step%2===0?'plan':'observed', index:Math.floor((step-1)/2)
  };
}
