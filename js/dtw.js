// dtw.js — FastDTW simplified implementation
const FastDTW={
  // Distance between two frames (flat arrays)
  dist(a,b){
    let s=0;
    for(let i=0;i<a.length;i++){const d=a[i]-b[i];s+=d*d}
    return Math.sqrt(s);
  },

  // Standard DTW with window constraint (simplified FastDTW)
  compute(seq1,seq2,radius){
    const n=seq1.length,m=seq2.length;
    radius=radius||Math.max(10,Math.ceil(Math.max(n,m)*0.1));
    
    // Cost matrix (sparse via window)
    const INF=Infinity;
    const cost=new Array(n);
    for(let i=0;i<n;i++){
      cost[i]=new Float64Array(m);
      cost[i].fill(INF);
    }
    
    cost[0][0]=this.dist(seq1[0],seq2[0]);
    
    // Initialize boundaries FIRST
    // Handle i=0 row
    for(let j=1;j<m;j++){
      if(Math.abs(j*n/m)<=radius){
        const d=this.dist(seq1[0],seq2[j]);
        cost[0][j]=d+cost[0][j-1];
      }
    }
    // Handle j=0 column
    for(let i=1;i<n;i++){
      if(Math.abs(i*m/n)<=radius){
        const d=this.dist(seq1[i],seq2[0]);
        cost[i][0]=d+cost[i-1][0];
      }
    }
    
    // Main DP loop
    for(let i=1;i<n;i++){
      const jMin=Math.max(1,Math.round(i*m/n)-radius);
      const jMax=Math.min(m-1,Math.round(i*m/n)+radius);
      for(let j=jMin;j<=jMax;j++){
        const d=this.dist(seq1[i],seq2[j]);
        cost[i][j]=d+Math.min(cost[i-1][j],cost[i][j-1],cost[i-1][j-1]);
      }
    }
    
    // Backtrack to get alignment path
    const path=[];
    let i=n-1,j=m-1;
    path.push([i,j]);
    while(i>0||j>0){
      if(i===0){j--;path.push([i,j]);continue}
      if(j===0){i--;path.push([i,j]);continue}
      const candidates=[
        [i-1,j-1,cost[i-1][j-1]],
        [i-1,j,cost[i-1][j]],
        [i,j-1,cost[i][j-1]]
      ];
      candidates.sort((a,b)=>a[2]-b[2]);
      i=candidates[0][0];j=candidates[0][1];
      path.push([i,j]);
    }
    path.reverse();
    
    // Per-pair distances
    const pairDist=path.map(([a,b])=>this.dist(seq1[a],seq2[b]));
    
    return{path,pairDist,totalCost:cost[n-1][m-1]};
  }
};
