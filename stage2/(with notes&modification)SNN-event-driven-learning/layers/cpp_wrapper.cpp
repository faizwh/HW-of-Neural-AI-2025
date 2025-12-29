#include <torch/extension.h>

// 通过封装高效的 C++ 函数供 Python 调用，为 SNN 中的卷积层反向传播提供性能加速。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cudnn_convolution_backward", &at::cudnn_convolution_backward, "CUDNN convolution backward");
    m.def("cudnn_convolution_backward_input", &at::cudnn_convolution_backward_input, "CUDNN convolution backward for input");
    m.def("cudnn_convolution_backward_weight", &at::cudnn_convolution_backward_weight, "CUDNN convolution backward for weight");
}