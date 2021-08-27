#include <iostream>
#include <vector>
#include "hip/hip_runtime_api.h"
#include "hip/hip_vector_types.h"
#include "rocfft.h"
#include  <assert.h>

int main()
{
        // rocFFT gpu compute
        // ========================================

        rocfft_setup();

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
		hxo[i].x=10000.0;
		hxo[i].y=10000.0;
        }

        //  Copy data to device
        hipMemcpy(x, hx.data(), Nbytes/2, hipMemcpyHostToDevice);
        hipMemcpy(xo, hxo.data(), Nbytes, hipMemcpyHostToDevice);

        // Create rocFFT plan
        rocfft_plan plan = nullptr;
	rocfft_plan_description description = nullptr  ;
	size_t n[] = { 4 };
        size_t istrips[] = { 4 } ;
        size_t ostrips[] = { 1 } ;
	rocfft_status status ;

        status = rocfft_plan_description_create( &description);
        status = rocfft_plan_description_set_data_layout( description, rocfft_array_type_real, rocfft_array_type_hermitian_interleaved , nullptr, nullptr, 1, istrips, 1, 1, ostrips, 4) ;
        status = rocfft_plan_create(&plan, rocfft_placement_notinplace, rocfft_transform_type_real_forward, rocfft_precision_single, 1, n, 4, description);
	assert(status == rocfft_status_success) ;
        rocfft_plan_get_print( plan); 
        size_t work_buf_size = 0;
        rocfft_plan_get_work_buffer_size(plan, &work_buf_size);
        void* work_buf = nullptr;
        rocfft_execution_info info = nullptr;
	std::cout<<"buf_size: "<<work_buf_size<<" Status: "<< status << std::endl ;
        if(work_buf_size)
        {
                rocfft_execution_info_create(&info);
                hipMalloc(&work_buf, work_buf_size);
                rocfft_execution_info_set_work_buffer(info, work_buf, work_buf_size);
        }


        // Execute plan
        status = rocfft_execute(plan, (void**) &x, (void**) &xo, info);
	assert(status == rocfft_status_success) ;

        // Wait for execution to finish
        hipDeviceSynchronize();

        // Clean up work buffer
        if(work_buf_size)
        {
                hipFree(work_buf);
                rocfft_execution_info_destroy(info);
        }

        // Destroy plan
        rocfft_plan_destroy(plan);

        // Copy result back to host
        std::vector<float2> y(N);
        hipMemcpy(y.data(), xo, Nbytes, hipMemcpyDeviceToHost);
        //hipMemcpy(y.data(), x, Nbytes, hipMemcpyDeviceToHost);

        // Print results
        for (size_t i = 0; i < 16; i++)
        {
                std::cout << y[i].x << "+ " << y[i].y << "i "<< std::endl;
        }

        // Free device buffer
        hipFree(x);

        rocfft_cleanup();

        return 0;
}
