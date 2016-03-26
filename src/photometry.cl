__kernel void photometry(__global float* data, __constant float* dark,
                        __constant float* flat, __global float* output,
                        __local float* local_data)
{
    const int x = (int)get_global_id(0);
    const int s = n * n;

   for(int j = 0; j < s; j++){
        local_data[j] = data[i*s + j];

   barrier(CLK_LOCAL_MEM_FENCE);

   float2 center = (float2)(centerX, centerY);
   float sum = 0;
   int px_count = 0;
   float sky_sum = 0;
   float sky_count = 0;

    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            float2 curr_px = (float2)(i, j);
            float dist = fast_distance(center, curr_px);
            if(dist < aperture){
                sum += (a(i, j) - dark(i, j))/flat(i,j);
                px_count++;
            }
            else if(dist > inner_sky && dist < outer_sky){
                sky_sum += (a(i, j)-dark(i, j))/flat(i, j);
                sky_count++:
            }
        }
    }

    output[x] = sum - (sky_sum / sky_count)*px_count;

}