// visualizer.js — Skeleton overlay visualization on Canvas
const Visualizer={
  // MediaPipe Pose connections (33 landmarks)
  CONNECTIONS:[
    [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],
    [9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],
    [12,14],[14,16],[16,18],[16,20],[16,22],
    [11,23],[12,24],[23,24],[23,25],[25,27],[27,29],[27,31],
    [24,26],[26,28],[28,30],[28,32]
  ],

  TEACHER_COLOR:'rgba(108,59,170,0.9)',   // purple
  STUDENT_COLOR:'rgba(58,190,255,0.9)',    // blue
  TEACHER_JOINT:'rgba(180,130,255,1)',
  STUDENT_JOINT:'rgba(100,210,255,1)',

  drawSkeleton(ctx,landmarks,color,jointColor,w,h){
    if(!landmarks||!landmarks.length)return;
    // Draw connections
    ctx.strokeStyle=color;
    ctx.lineWidth=2;
    for(const[a,b] of this.CONNECTIONS){
      if(a>=landmarks.length||b>=landmarks.length)continue;
      const la=landmarks[a],lb=landmarks[b];
      if(la.visibility<0.3||lb.visibility<0.3)continue;
      ctx.beginPath();
      ctx.moveTo(la.x*w,la.y*h);
      ctx.lineTo(lb.x*w,lb.y*h);
      ctx.stroke();
    }
    // Draw joints
    ctx.fillStyle=jointColor;
    for(const l of landmarks){
      if(l.visibility<0.3)continue;
      ctx.beginPath();
      ctx.arc(l.x*w,l.y*h,4,0,Math.PI*2);
      ctx.fill();
    }
  },

  // Draw both skeletons overlaid
  drawFrame(canvas,teacherLandmarks,studentLandmarks){
    const ctx=canvas.getContext('2d');
    const w=canvas.width,h=canvas.height;
    ctx.clearRect(0,0,w,h);
    
    // Background
    ctx.fillStyle='rgba(10,14,26,0.8)';
    ctx.fillRect(0,0,w,h);
    
    // Grid lines for reference
    ctx.strokeStyle='rgba(255,255,255,0.03)';
    ctx.lineWidth=1;
    for(let x=0;x<w;x+=w/10){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}
    for(let y=0;y<h;y+=h/10){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
    
    // Draw teacher first (background), then student
    if(teacherLandmarks)this.drawSkeleton(ctx,teacherLandmarks,this.TEACHER_COLOR,this.TEACHER_JOINT,w,h);
    if(studentLandmarks)this.drawSkeleton(ctx,studentLandmarks,this.STUDENT_COLOR,this.STUDENT_JOINT,w,h);
    
    // Legend
    ctx.font='12px "Noto Sans SC"';
    ctx.fillStyle=this.TEACHER_COLOR;ctx.fillRect(10,h-30,12,12);
    ctx.fillStyle='#E8E6F0';ctx.fillText('老师',28,h-20);
    ctx.fillStyle=this.STUDENT_COLOR;ctx.fillRect(70,h-30,12,12);
    ctx.fillStyle='#E8E6F0';ctx.fillText('你',88,h-20);
  }
};
