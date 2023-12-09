import torch
from mylib import structs, enums
from mylib.lib_loader import get_optixir
from mylib.c_bindings import c_optix


# context = c_optix.create_context()
ir = get_optixir("/workspaces/doritosplat/insp/optixTriangle/optixTriangle.cu")

vertices = torch.tensor(
    [
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0.0, 0.5, 0],
    ],
    device="cuda",
)
assert vertices.data_ptr() % 64 == 0
context = c_optix.Context()

scene = c_optix.Scene(context)
sizes: structs.OptixAccelBufferSizes = scene.set_vertex_pointer(vertices)
tmp = torch.empty(sizes.tempSizeInBytes, dtype=torch.uint8, device="cuda")
bvh = torch.empty(sizes.outputSizeInBytes, dtype=torch.uint8, device="cuda")
scene.build(tmp, bvh)

module = c_optix.Module(context)
module.load(ir)
module.configure()
headers = module.get_sbt_headers()


raygen_record = torch.zeros(headers[0], dtype=torch.uint8, device="cuda")
hit_record = torch.tensor([headers[1]], dtype=torch.uint8, device="cuda")
miss_record = torch.tensor([headers[2]], dtype=torch.uint8, device="cuda")

module.generate_sbt(raygen_record, hit_record, miss_record)
stream = torch.cuda.Stream()

image = torch.empty(256, 256, dtype=torch.float32, device="cuda")


params = torch.tensor([image.data_ptr()], dtype=torch.int64, device="cuda")

module.launch(params, stream.cuda_stream)
