# /usr/local/cuda-12.1/targets/x86_64-linux/include/cuda.h


ignore_types = ["OptixLogCallback"]

optix_ptr_types = {
    "OptixDeviceContext_t": "OptixDeviceContext",
    "OptixModule_t": "OptixModule",
    "OptixProgramGroup_t": "OptixProgramGroup",
    "OptixPipeline_t": "OptixPipeline",
    "OptixDenoiser_t": "OptixDenoiser",
    "OptixTask_t": "OptixTask",
}
cuda_ptr_types = {
    "CUctx_st": "CUcontext",
    "CUmod_st": "CUmodule",
    "CUfunc_st": "CUfunction",
    "CUlib_st": "CUlibrary",
    "CUkern_st": "CUkernel",
    "CUarray_st": "CUarray",
    "CUmipmappedArray_st": "CUmipmappedArray",
    "CUtexref_st": "CUtexref",
    "CUsurfref_st": "CUsurfref",
    "CUevent_st": "CUevent",
    "CUstream_st": "CUstream",
    "CUgraphicsResource_st": "CUgraphicsResource",
    "CUextMemory_st": "CUexternalMemory",
    "CUextSemaphore_st": "CUexternalSemaphore",
    "CUgraph_st": "CUgraph",
    "CUgraphNode_st": "CUgraphNode",
    "CUgraphExec_st": "CUgraphExec",
    "CUmemPoolHandle_st": "CUmemoryPool",
    "CUuserObject_st": "CUuserObject",
}
ptr_types = {**optix_ptr_types, **cuda_ptr_types}
ptr_types_inv = {v: k for k, v in ptr_types.items()}

pointer_types = [v for v in ptr_types.values()]
