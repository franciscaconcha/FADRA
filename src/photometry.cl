__kernel void photometry(__global float* stamp, __constant float* dark,
                        __constant float* flat, __global float* output)
                        //__local float* local_stamp)
{
    int x = get_global_id(0);
    //int y = get_global_id(1);
    int s = n * n;
    int px = x / n;
    int py = x % n;

    float2 curr_px = (float2)(px, py);
    float2 center = (float2)(centerX, centerY);
    int dist = (int)distance(center, curr_px);

    //if(px == 1)
      //  printf("%f, %f, %f, %f\n", px, py, centerX, centerY);

    if(dist < aperture){
      output[x] = 0.0;
      //output[0] += stamp[x];
      //output[1]++;
        }else if (dist > sky_inner && dist < sky_outer){
            output[x] = 0.0;
            //output[2] += stamp[x];
            //output[3]++;
            }
            else{
            output[x] = stamp[x];
            }
            barrier(CLK_GLOBAL_MEM_FENCE);

    /*int px = x / n;
    int py = x % n;

    int s = n * n;

    int lid = get_local_id(0);

    if(x < s){
        //local_stamp[lid] = stamp[x];
        float2 curr_px = (float2)(px, py);
        float2 center = (float2)(centerX, centerY);
        int dist = (int)fast_distance(center, curr_px);
        if(dist < aperture){
            output[0] += (stamp[x] - dark[x]);///flat[x];
            //output[0] = flat[x];
            //atomic_add(&output[0], stamp[x]);
            output[1]++;
            //atomic_add(&output[1], 1);
        }else if (dist > sky_inner && dist < sky_outer){
            output[2] += (stamp[x] - dark[x]);
            //atomic_add(&output[2], stamp[x]);
            output[3]++;
            //atomic_add(&output[3], 1);
            }
        barrier(CLK_GLOBAL_MEM_FENCE);
     }*/
}