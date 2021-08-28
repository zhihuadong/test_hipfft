I have some issue with hipfft AMD backend with striping.

The inlcude hipfft.cpp file just calculate 4 forward fft's of 4 numbers with striping input. Output is not set to do striping for clearly showing result.

Input array is [0,1,2,3....14,15], trying to do striping 4, distance 1. (transpose of 4x4)

When built with hipfft HIP_PLATFORM=nvidia and run on NVIDIA GPU it generate correct answer .  
But when build with HIP_PLATFORM=amd and run on AMD GPU, it generat wrong answer (not actually doing striping as wanted)

``` bash
testfft$ HIP_PLATFORM=amd hipcc -o hipfft-amd  hipfft.cpp -L /opt/rocm-4.3.0/hipfft/lib/ -lhipfft -I /opt/rocm-4.3.0/hipfft/include
testfft$ ./hipfft-amd
worksize= 128
6+ 0i 
-2+ 2i 
-2+ 0i 
10000+ 10000i 
10+ 0i 
-2+ 2i 
-2+ 0i 
10000+ 10000i 
14+ 0i 
-2+ 2i 
-2+ 0i 
10000+ 10000i 
18+ 0i 
-2+ 2i 
-2+ 0i 
10000+ 10000i 
testfft$ HIP_PLATFORM=nvidia hipcc -o hipfft-nv hipfft.cpp -L /work/install/hipfft-nv/lib/ -lhipfft -I /work/hipfft-nv/include/ -I /opt/rocm/hip/include
hipfft.cpp: In function \u2018int main()\u2019:
hipfft.cpp:37:28: warning: converting to non-pointer type \u2018hipfftHandle\u2019 {aka \u2018int\u2019} from NULL [-Wconversion-null]
   37 |         hipfftHandle  plan = NULL;
      |                            ^
testfft$ LD_LIBRARY_PATH=/work/install/hipfft-nv/lib ./hipfft-nv 
worksize= 1088
24+ 0i 
-8+ 8i 
-8+ 0i 
10000+ 10000i 
28+ 0i 
-8+ 8i 
-8+ 0i 
10000+ 10000i 
32+ 0i 
-8+ 8i 
-8+ 0i 
10000+ 10000i 
36+ 0i 
-8+ 8i 
-8+ 0i 
10000+ 10000i 
testfft$ 
```

The rocfft.cpp is doing samething which give same wrong result as hipfft-amd-- did not do striping even the printed plan said so.

It's doing [0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,6] instead of [0,4,8,12],[1,5,9,13],[2,6,10,14],[3,7,11,15]

That is it took  the distance( elements between batch , 1) correct, but not striping (should be 4).

``` bash
testfft$ hipcc -o rocfft rocfft.cpp -L /opt/rocm-4.3.0/hipfft/lib/ -lhipfft -L /opt/rocm-4.3.0/rocfft/lib/ -lrocfft 
testfft$ ./rocfft

precision: single
transform type: real forward
result placement: not in-place

input array type: real
output array type: hermitian interleaved

dimensions: 1
lengths: 4
batch size: 4

input offset: 0
output offset: 0

input strides: 4
output strides: 1
input distance: 1
output distance: 4

scale: 1.0

buf_size: 128 Status: 0
6+ 0i 
-2+ 2i 
-2+ 0i 
10000+ 10000i 
10+ 0i 
-2+ 2i 
-2+ 0i 
10000+ 10000i 
14+ 0i 
-2+ 2i 
-2+ 0i 
10000+ 10000i 
18+ 0i 
-2+ 2i 
-2+ 0i 
10000+ 10000i 
testfft$
```

Output striping is also not working if I set it in the code. 
