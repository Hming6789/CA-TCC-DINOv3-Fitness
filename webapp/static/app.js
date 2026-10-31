'use strict';
const $ = id => document.getElementById(id);
const state = {file:null, url:null, valid:false, ready:false, busy:false, xhr:null, revision:0, model:null};
const number = new Intl.NumberFormat('en', {maximumFractionDigits:1});
const show = (id, value) => { $(id).hidden = !value; };
const announce = text => { $('live-status').textContent = text; };
function error(message='') { $('error-message').textContent = message; show('error-message', Boolean(message)); }
function buttons() { $('analyze-button').disabled = !state.ready || !state.valid || state.busy; $('upload-zone').disabled=state.busy; $('remove-video').disabled=state.busy; }
function clearResult() {
  show('results-idle',true); show('prediction',false); show('result-meta',false);
  $('result-state').textContent='Awaiting analysis'; $('result-state').classList.remove('complete'); $('probability-unit').textContent='—';
  renderBars((state.model?.class_names || ['Squat','Push-up','Lunge']).map(label=>({label,probability:null})));
}
function renderBars(items, winner) {
  const root=$('probability-bars'); root.replaceChildren();
  for (const item of items) {
    const row=document.createElement('div'); row.className='probability-row'+(item.label===winner?' winner':'');
    const heading=document.createElement('div'), label=document.createElement('span'), value=document.createElement('strong');
    label.textContent=item.label; value.textContent=item.probability===null?'—':`${number.format(item.probability*100)}%`;
    heading.append(label,value); const track=document.createElement('div'), bar=document.createElement('i');
    track.className='bar-track'; track.setAttribute('aria-hidden','true'); bar.style.width=`${item.probability===null?0:Math.max(0,Math.min(100,item.probability*100))}%`;
    track.append(bar); row.append(heading,track); root.append(row);
  }
}
function clearVideo() {
  state.revision++; state.file=null; state.valid=false; error(); clearResult();
  const video=$('video-preview'); video.pause(); video.removeAttribute('src'); video.load();
  if(state.url) URL.revokeObjectURL(state.url); state.url=null;
  $('video-file').value=''; show('video-preview',false); show('empty-stage',true); show('video-details',false);
  $('upload-title').replaceChildren(document.createTextNode('Drop your video here, or '));
  const em=document.createElement('em'); em.textContent='browse files'; $('upload-title').append(em);
  $('upload-subtitle').textContent='One exercise per clip · Up to 30 seconds · 32 MB max'; buttons();
}
async function selectVideo(file) {
  if(state.busy || !file) return;
  clearVideo();
  if(!/\.(mp4|webm|mov|avi|mkv)$/i.test(file.name)) return error('Choose an MP4, WebM, MOV, AVI or MKV video.');
  if(file.size===0) return error('This video file is empty.');
  if(file.size>32*1024*1024) return error('Choose a video smaller than 32 MB.');
  state.file=file; state.url=URL.createObjectURL(file); const revision=state.revision;
  $('upload-title').textContent=file.name; $('upload-subtitle').textContent=`${number.format(file.size/1024/1024)} MB · Click to choose another video`;
  const video=$('video-preview');
  video.onloadedmetadata=()=>{
    if(revision!==state.revision) return;
    if(!Number.isFinite(video.duration) || video.duration<=0 || video.duration>30) { state.valid=false; error('Use a readable clip of 30 seconds or less.'); buttons(); return; }
    state.valid=true; $('file-meta').textContent=`${number.format(video.duration)} sec · ${video.videoWidth} × ${video.videoHeight}`;
    show('video-details',true); error(); buttons(); announce('Video ready for analysis.');
  };
  video.onerror=()=>{
    if(revision!==state.revision) return;
    // AVI/MKV/MOV codecs may be decodable on the server but not in the browser.
    state.valid=true; $('file-meta').textContent='Browser playback unavailable · Server will validate this clip';
    show('video-details',true); buttons();
  };
  video.src=state.url; show('empty-stage',false); show('video-preview',true); clearResult(); buttons();
}
async function loadStatus() {
  try {
    const response=await fetch('/api/status',{signal:AbortSignal.timeout(15000)});
    if(!response.ok) throw new Error('Status unavailable');
    state.model=await response.json(); state.ready=Boolean(state.model.ready);
    $('header-status').textContent=state.ready?'Model connected':'Preview mode';
    $('header-dot').className='status-dot'+(state.ready?' ready':'');
    show('preview-note',!state.ready); clearResult(); buttons();
  } catch {
    state.ready=false; $('header-status').textContent='Server unavailable'; $('header-dot').className='status-dot offline';
    show('preview-note',true); buttons();
  }
}
function analyze() {
  if(!state.file || !state.valid || !state.ready || state.busy) return;
  state.busy=true; error(); clearResult(); show('preview-note',false); show('progress-state',true); show('analysis-overlay',true);
  $('result-state').textContent='Analyzing'; $('progress-label').textContent='Uploading video'; buttons(); $('video-preview').pause();
  const xhr=new XMLHttpRequest(); state.xhr=xhr;
  xhr.open('POST','/api/analyze'); xhr.timeout=240000;
  xhr.upload.onprogress=e=>{ $('progress-label').textContent=e.lengthComputable?`Uploading · ${Math.round(e.loaded/e.total*100)}%`:'Uploading video'; };
  xhr.upload.onload=()=>{ $('progress-label').textContent='Analyzing sampled frames'; };
  xhr.onload=()=>{
    let result; try{result=JSON.parse(xhr.responseText);}catch{error('The server returned an unreadable response. Try again.');return;}
    if(xhr.status<200 || xhr.status>=300){ error(typeof result.detail==='string'?result.detail:'Analysis could not finish. Try a shorter MP4 clip.'); return; }
    if(!Array.isArray(result.probabilities) || result.probabilities.length!==3 || !Number.isFinite(result.confidence)) {error('The server returned an incomplete analysis.');return;}
    show('results-idle',false); show('prediction',true); show('result-meta',true);
    $('predicted-class').textContent=result.predicted_class; $('confidence-value').textContent=`${number.format(result.confidence*100)}%`;
    $('prediction-note').textContent=result.confidence<.6?'The model is uncertain. Try a clearer, single-exercise clip.':'Exercise identity recognized from the sampled video frames.';
    $('frames-analyzed').textContent=String(result.video.sampled_frames); $('processing-time').textContent=`${number.format(result.inference_seconds)} sec`;
    $('probability-unit').textContent='%'; renderBars(result.probabilities,result.predicted_class);
    $('result-state').textContent='Analysis complete'; $('result-state').classList.add('complete');
    announce(`Analysis complete. Predicted movement: ${result.predicted_class}.`);
  };
  xhr.onerror=()=>error('The connection was interrupted. Please try again.');
  xhr.ontimeout=()=>error('The analysis took too long. Try a shorter clip or try again later.');
  xhr.onabort=()=>{ $('result-state').textContent='Analysis cancelled'; announce('Analysis cancelled.'); };
  xhr.onloadend=()=>{
    state.busy=false; state.xhr=null; show('progress-state',false); show('analysis-overlay',false); buttons();
    if($('prediction').hidden && $('result-state').textContent==='Analyzing') $('result-state').textContent='Ready to retry';
  };
  const body=new FormData(); body.append('file',state.file); xhr.send(body);
}
function openDialog(title, build) {
  $('dialog-title').textContent=title; $('dialog-body').replaceChildren(); build($('dialog-body')); $('info-dialog').showModal();
}
function addParagraph(root, text, className='') { const p=document.createElement('p'); p.textContent=text; p.className=className; root.append(p); }
$('guide-button').addEventListener('click',()=>openDialog('A clearer clip. A better result.',root=>{
  addParagraph(root,'Choose a short video of one exercise: squat, push-up or lunge.');
  const list=document.createElement('ol');
  for(const text of ['Keep your full body visible, including hands and feet.','Use a steady camera and good lighting. Avoid cuts between movements.','Upload a clip up to 30 seconds and 32 MB. MP4 is recommended.','Run the analysis to see the predicted exercise and its class probabilities.']) { const li=document.createElement('li'); li.textContent=text; list.append(li); }
  root.append(list); addParagraph(root,'This workspace classifies exercise identity. It does not assess technique, count repetitions or provide a clinical form score.','dialog-note');
}));
$('model-button').addEventListener('click',()=>openDialog('Your analysis model',root=>{
  const dl=document.createElement('dl');
  for(const [key,value] of [['Model','DINOv3 + CA-TCC'],['Status',state.ready?'Connected':'Not connected'],['Input','18 RGB frames · 224 × 224'],['Runtime','ONNX Runtime · CPU']]) {const row=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=value;row.append(dt,dd);dl.append(row);}
  root.append(dl); addParagraph(root,state.ready?'Probabilities reflect the model’s exercise classification; they are not calibrated quality scores.':'A trained classifier has not been connected to this workspace yet. Video preview is available; model predictions will become available after setup.','dialog-note');
}));
$('close-dialog').addEventListener('click',()=>$('info-dialog').close());
$('info-dialog').addEventListener('click',event=>{if(event.target===$('info-dialog')) { const rect=event.target.getBoundingClientRect(); if(event.clientX<rect.left||event.clientX>rect.right||event.clientY<rect.top||event.clientY>rect.bottom)event.target.close(); }});
$('upload-zone').addEventListener('click',()=>$('video-file').click());
$('video-file').addEventListener('change',event=>selectVideo(event.target.files[0]));
for(const type of ['dragenter','dragover']) $('upload-zone').addEventListener(type,event=>{event.preventDefault(); if(!state.busy)$('upload-zone').classList.add('dragging');});
for(const type of ['dragleave','drop']) $('upload-zone').addEventListener(type,event=>{event.preventDefault();$('upload-zone').classList.remove('dragging');});
$('upload-zone').addEventListener('drop',event=>selectVideo(event.dataTransfer.files[0]));
$('remove-video').addEventListener('click',clearVideo);
$('analyze-button').addEventListener('click',analyze);
$('cancel-analysis').addEventListener('click',()=>state.xhr?.abort());
window.addEventListener('beforeunload',()=>{if(state.url) URL.revokeObjectURL(state.url);});
loadStatus();
