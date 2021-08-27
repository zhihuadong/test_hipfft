#include <iostream>
#include <vector>
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "hipfft.h"
#include  <assert.h>

int main()
{
        // hipFFT gpu compute
        // ========================================

        size_t N = 32;
        size_t Nbytes = N * sizeof(float2);

        // Create HIP device buffer
        float *x;
	float2 *xo ;
        hipMalloc(&x, Nbytes/2);
        hipMalloc(&xo, Nbytes);

        // Initialize data
        std::vector<float> hx(N);
	std::vector<float2> hxo(N) ;
        for (size_t i = 0; i < N; i++)
        {
                hx[i]=i ;
		hxo[i].x=10000.0; // just for easy look for printout
		hxo[i].y=10000.0; // just for easy look for printout
        }

        //  Copy data to device
        hipMemcpy(x, hx.data(), Nbytes/2, hipMemcpyHostToDevice);
        hipMemcpy(xo, hxo.data(), Nbytes, hipMemcpyHostToDevice);

        // Create hipFFT plan
        hipfftHandle  plan = NULL;
	int n[] = { 4 };
	int inembed[] = { 4 } ;
	int onembed[] = { 4 } ;
	hipfftResult status ;
        size_t worksize = 0;
	status = hipfftCreate(&plan);
	assert(status == HIPFFT_SUCCESS) ;

	// set to autoAllocate
	status = hipfftSetAutoAllocation( plan, 1);
	assert(status == HIPFFT_SUCCESS) ;

	//MakePlan
	status = hipfftMakePlanMany( plan, 1, n, inembed, 4, 1, onembed, 1, 4, HIPFFT_R2C, 4, &worksize); 
	std::cout<<"worksize= "<<worksize << std::endl ;
	assert(status == HIPFFT_SUCCESS) ;

	//Execute  
        status = hipfftExecR2C(  plan,  x,  xo) ;
	assert(status == HIPFFT_SUCCESS) ;
	

        // Wait for execution to finish
        hipDeviceSynchronize();


        // Destroy plan
        hipfftDestroy(plan);

        // Copy result back to host
        std::vector<float2> y(N);
        hipMemcpy(y.data(), xo, Nbytes, hipMemcpyDeviceToHost);

        // Print results
        for (size_t i = 0; i < 16; i++)
        {
                std::cout << y[i].x << "+ " << y[i].y << "i "<< std::endl;
        }

        // Free device buffer
        hipFree(x);
        hipFree(xo);


        return 0;
}
