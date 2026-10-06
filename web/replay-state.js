// Decision i selects observation i. Preparing after frame i uses decision i+1.
export function replayStep(frames, index, preparing=false, before=false) {
  if (!Number.isInteger(index) || index<0 || index>=frames.length) throw new Error('无效回放帧');
  return { frame:frames[index], next:frames[index+1] || null,
    decision:preparing ? frames[index+1]?.decision || null : null,
    mapFrame:frames[before && index>0 ? index-1 : index] };
}
