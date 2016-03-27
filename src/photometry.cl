__kernel void photometry(__global float* stamp, __constant float* dark,
                        __constant float* flat, __global float* output)
                        //__local float* local_data)
{
    const int x = (int)get_global_id(0);
    const int s = n * n;

   //for(int l = 0; l < s; l++){
   //     local_data[l] = stamp[x*s + l];
   //}

   //barrier(CLK_LOCAL_MEM_FENCE);

   float2 center = (float2)(centerX, centerY);
   float sum = 0;
   int px_count = 0;
   float sky_sum = 0;
   int sky_count = 0;

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            float2 curr_px = (float2)(i, j);
            int dist = (int)fast_distance(center, curr_px);
            //output[x] = dist;
            //printf("dist: %f, aperture: %f\n", dist, aperture);
            if(dist < aperture){
                sum += (stamp[i*n + j] - dark[i*n + j])/flat[i*n + j];
                px_count++;
            }
            else if(dist > sky_inner && dist < sky_outer){
                sky_sum += (stamp[i*n + j]-dark[i*n + j])/flat[i*n + j];
                sky_count++;
            }
        }
    }

    //printf("o = %f\n", sum);
    //output[x] = sum - (sky_sum / sky_count)*px_count;
    output[x] = sum;

}