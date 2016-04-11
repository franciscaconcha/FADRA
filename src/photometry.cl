#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable

__kernel void photometry(__global float* stamp, __global float* dark,
                        __global float* flat, __global int* output)
                        //__local float* local_stamp)
{
    int x = get_global_id(0);

    int s = n * n;
    int px = x / n;
    int py = x % n;

    float2 curr_px = (float2)(px, py);
    float2 center = (float2)(centerX, centerY);
    int dist = (int)distance(center, curr_px);

    if(dist < aperture){
        //output[x] = 0.0;
        //output[0] += (int)stamp[x];
        //output[1]++;
        atomic_add(&output[0], (stamp[x]-dark[x]));
        atomic_add(&output[1], 1);
        //output[x] = 1;
    }else if (dist > sky_inner && dist < sky_outer){
        //output[x] = 0.0;
        //output[2] += (int)stamp[x];
        //output[3]++;
        atomic_add(&output[2], (stamp[x]-dark[x]));
        atomic_add(&output[3], 1);
        //output[x] = 2;
    }
    //else{
        //output[x] = stamp[x];
        //}
}
