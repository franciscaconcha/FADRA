__kernel void photometry2(__global float* stamp, __constant float* dark,
                        __constant float* flat, __global float* output,
                        __local float* local_stamp)
{
    int x = get_global_id(0);
    int s = n * n;
    int curr_im = x / s;

    output[curr_im] = 10;

    barrier(CLK_GLOBAL_MEM_FENCE);

/*
    int px = x / n;
    int py = x % n;


    int lid = get_local_id(0);

    if(x < s){
        //local_stamp[lid] = stamp[x];
        float2 curr_px = (float2)((x / n), (x % n));
        float2 center = (float2)(centerX, centerY);
        int dist = (int)fast_distance(center, curr_px);
        if(dist < aperture){
            output[0] += (stamp[x] - dark[x]);
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