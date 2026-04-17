// utils.js — Shared utilities
const DF={};

// Cosine similarity between two arrays
DF.cosineSim=function(a,b){
  let dot=0,na=0,nb=0;
  for(let i=0;i<a.length;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}
  return na&&nb?dot/(Math.sqrt(na)*Math.sqrt(nb)):0;
};

// Euclidean distance between two flat arrays
DF.euclidean=function(a,b){
  let s=0;for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d}
  return Math.sqrt(s);
};

// Flatten landmarks [{x,y,z},...] → [x0,y0,z0,x1,y1,z1,...]
DF.flattenLandmarks=function(lms){
  const r=[];for(const l of lms){r.push(l.x,l.y,l.z)}return r;
};

// Clamp
DF.clamp=function(v,lo,hi){return Math.max(lo,Math.min(hi,v))};

// Score color
DF.scoreColor=function(s){
  if(s>=80)return'#3ABEFF';if(s>=60)return'#6C3BAA';if(s>=40)return'#FFD76E';return'#FF6B9D';
};

// Save/load analysis result to sessionStorage
DF.saveResult=function(data){sessionStorage.setItem('df_result',JSON.stringify(data))};
DF.loadResult=function(){try{return JSON.parse(sessionStorage.getItem('df_result'))}catch{return null}};

// Save/load videos as object URLs (just paths)
DF.saveVideos=function(teacher,student){
  // We store blobs in memory via global
  window._dfTeacherBlob=teacher;
  window._dfStudentBlob=student;
};

// Shared IndexedDB helper
function openDB(){
  return new Promise((res,rej)=>{
    const req=indexedDB.open('danceflow',1);
    req.onupgradeneeded=e=>e.target.result.createObjectStore('videos');
    req.onsuccess=e=>res(e.target.result);
    req.onerror=e=>rej(e);
  });
}
