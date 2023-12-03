# /usr/local/cuda-12.1/targets/x86_64-linux/include/cuda.h


ignore_types = ["OptixLogCallback"]

optix_ptr_types = {
    "OptixDeviceContext": "OptixDeviceContext_t *",
    "OptixModule": "OptixModule_t *",
    "OptixProgramGroup": "OptixProgramGroup_t *",
    "OptixPipeline": "OptixPipeline_t *",
    "OptixDenoiser": "OptixDenoiser_t *",
    "OptixTask": "OptixTask_t *",
}
cuda_ptr_types = {
    "CUcontext": "CUctx_st *",
    "CUmodule": "CUmod_st *",
    "CUfunction": "CUfunc_st *",
    "CUlibrary": "CUlib_st *",
    "CUkernel": "CUkern_st *",
    "CUarray": "CUarray_st *",
    "CUmipmappedArray": "CUmipmappedArray_st *",
    "CUtexref": "CUtexref_st *",
    "CUsurfref": "CUsurfref_st *",
    "CUevent": "CUevent_st *",
    "CUstream": "CUstream_st *",
    "CUgraphicsResource": "CUgraphicsResource_st *",
    "CUexternalMemory": "CUextMemory_st *",
    "CUexternalSemaphore": "CUextSemaphore_st *",
    "CUgraph": "CUgraph_st *",
    "CUgraphNode": "CUgraphNode_st *",
    "CUgraphExec": "CUgraphExec_st *",
    "CUmemoryPool": "CUmemPoolHandle_st *",
    "CUuserObject": "CUuserObject_st *",
}

ptr_typedefs = optix_ptr_types | cuda_ptr_types
for k, v in list(ptr_typedefs.items()):
    ptr_typedefs[f"const {k}"] = f"{v} const"

pointer_types = list(ptr_typedefs.keys())
