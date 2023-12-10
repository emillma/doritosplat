import torch
from mylib import structs, enums
from mylib.lib_loader import get_optixir
from mylib.clib import _clib
from torchvision.utils import save_image

torch.set_grad_enabled(False)

ir = get_optixir("/workspaces/doritosplat/mylib/optix/optixTriangle.cu")

vertices = torch.tensor(
    [
        [-0.5, -0.5, 1],
        [0.5, -0.5, 1],
        [0.0, 0.5, 1],
    ],
    device="cuda",
)
assert vertices.data_ptr() % 64 == 0
context = _clib.Context()

scene = _clib.Scene(context)
sizes: structs.OptixAccelBufferSizes = scene.set_vertex_pointer(vertices)
tmp = torch.empty(sizes.tempSizeInBytes, dtype=torch.uint8, device="cuda")
bvh = torch.empty(sizes.outputSizeInBytes, dtype=torch.uint8, device="cuda")
scene.build(tmp, bvh)

module = _clib.Module(context)
module.load(ir)
module.configure()
headers = module.get_sbt_headers()


raygen_record = torch.tensor(headers[0], dtype=torch.uint8, device="cuda")
hit_record = torch.tensor([headers[1]], dtype=torch.uint8, device="cuda")
miss_record = torch.tensor([headers[2]], dtype=torch.uint8, device="cuda")

module.generate_sbt(raygen_record, hit_record, miss_record)
stream = torch.cuda.Stream()

image = torch.zeros(1920, 1080, 3, dtype=torch.float32, device="cuda")

# params = structs.Params(han

params = torch.tensor(
    [
        scene.gas_handle,
        image.data_ptr(),
    ],
    dtype=torch.int64,
    device="cuda",
)

module.launch(params, stream.cuda_stream)
print(image.min(), image.max())
image -= image.min()
image /= image.max()
image = image.permute(2, 1, 0)
save_image(image, "test.png")
