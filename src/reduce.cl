__kernel void reduce(__constant float* dark, __constant float* flat,
                    __global float* images, __global float* res)
{
    const int i = (int)get_global_id(0);
    const int s = 100 * 100;

    /**__local float loc_im[100][100];
    int x = (int)get_local_id(0);
    int y = (int)get_local_id(1);
    loc_im[y][x] = images[i*s + x];**/

   for(int j = 0; j < s; j++){
        res[i*s + j] = (images[i*s + j] - dark[j])/flat[j];
   }
}