__kernel void add(__global float* a, __global float* b, __global float* d, __global float* c)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    int i = y*SIZE + x;

    c[i] = (a[i] - b[i])/d[i];

}