__kernel void convolve(__global float* input, __constant float* filter, __global float* output)
{
    int IMAGE_W = 100;
    int IMAGE_H = 100;
    int FILTER_SIZE = 5;
    int HALF_FILTER_SIZE = 2;
    int TWICE_HALF_FILTER_SIZE = 4;
    int HALF_FILTER_SIZE_IMAGE_W = 200;

	int rowOffset = get_global_id(1) * IMAGE_W * 4;
	int my = 4 * get_global_id(0) + rowOffset;

	int fIndex = 0;
	float sumR = 0.0;
	float sumG = 0.0;
	float sumB = 0.0;
	float sumA = 0.0;


	for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
	{
		int curRow = my + r * (IMAGE_W * 4);
		for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++, fIndex += 4)
		{
			int offset = c * 4;

			sumR += input[ curRow + offset   ] * filter[ fIndex   ];
			sumG += input[ curRow + offset+1 ] * filter[ fIndex+1 ];
			sumB += input[ curRow + offset+2 ] * filter[ fIndex+2 ];
			sumA += input[ curRow + offset+3 ] * filter[ fIndex+3 ];
		}
	}

	output[ my     ] = sumR;
	output[ my + 1 ] = sumG;
	output[ my + 2 ] = sumB;
	output[ my + 3 ] = sumA;

}