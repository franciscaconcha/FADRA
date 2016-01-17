__kernel void reduce(__global float* dark, __global float* flat, __global float* images, __global float* res)
{
    const int i = (int)get_global_id(0);
    const int s = 1000 * 1000;

   for(int j = 0; j < s; j++){
        res[i*s + j] = (images[i*s + j] - dark[j])/flat[j];
   }
}