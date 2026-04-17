// pose.js — MediaPipe Pose extraction from video
const PoseExtractor={
  landmarker:null,

  async init(onProgress){
    if(this.landmarker)return;
    if(onProgress)onProgress('加载姿态识别模型...');
    const vision=await self.FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm'
    );
    this.landmarker=await self.PoseLandmarker.createFromOptions(vision,{
      baseOptions:{
        modelAssetPath:'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate:'GPU'
      },
      runningMode:'VIDEO',
      numPoses:1
    });
  },

  // Extract landmarks from video file (Blob/File)
  // Returns: [{landmarks:[{x,y,z,visibility},...33], ts:number}, ...]
  async extractFromVideo(videoBlob, fps, onProgress){
    if(!this.landmarker)await this.init(onProgress);
    
    const url=URL.createObjectURL(videoBlob);
    const video=document.createElement('video');
    video.muted=true;video.playsInline=true;
    video.src=url;
    
    await new Promise((res,rej)=>{
      video.onloadeddata=res;video.onerror=rej;video.load();
    });
    
    const duration=video.duration;
    const sampleFPS=fps||10; // sample at 10fps for speed
    const totalFrames=Math.floor(duration*sampleFPS);
    const frames=[];
    
    for(let i=0;i<totalFrames;i++){
      const t=i/sampleFPS;
      video.currentTime=t;
      await new Promise(r=>{video.onseeked=r});
      
      const ts=performance.now();
      const result=this.landmarker.detectForVideo(video,ts);
      
      if(result.landmarks&&result.landmarks.length>0){
        frames.push({
          landmarks:result.landmarks[0].map(l=>({x:l.x,y:l.y,z:l.z,visibility:l.visibility||1})),
          ts:t
        });
      }
      if(onProgress)onProgress(`提取关键帧 ${i+1}/${totalFrames}`,i/totalFrames);
    }
    
    URL.revokeObjectURL(url);
    return frames;
  }
};
