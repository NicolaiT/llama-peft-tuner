import numpy as np
import torch
import torch.nn as nn
import math

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
        ):
        
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


try:
    import quant_cuda
    is_cuda = True
except:
    print('CUDA extension not installed.')
    is_cuda = False
    
try:
    import triton
    import triton.language as tl

    # code based https://github.com/fpgaminer/GPTQ-triton
    @triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
            triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
            triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        ],
        key=['M', 'N', 'K'],
    )

    @triton.jit
    def matmul_248_kernel(a_ptr, b_ptr, c_ptr,
                          scales_ptr, zeros_ptr, g_ptr,
                          M, N, K, bits, maxq,
                          stride_am, stride_ak,
                          stride_bk, stride_bn,
                          stride_cm, stride_cn,
                          stride_scales, stride_zeros,
                          BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                          GROUP_SIZE_M: tl.constexpr):
        """
        Compute the matrix multiplication C = A x B.
        A is of shape (M, K) float16
        B is of shape (K//8, N) int32
        C is of shape (M, N) float16
        scales is of shape (G, N) float16
        zeros is of shape (G, N) float16
        g_ptr is of shape (K) int32 
        """
        infearure_per_bits = 32 // bits

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_gk = pid_n * BLOCK_SIZE_N + offs_k
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_mask = (offs_am[:, None] < M)
        # b_ptrs is set up such that it repeats elements along the K axis 8 times
        b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)

        # shifter is used to extract the N bits of each element in the 32-bit word from B
        shifter = (offs_k % infearure_per_bits) * bits
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        for k in range(0, num_pid_k):
            g_ptrs = g_ptr + offs_k + (BLOCK_SIZE_K * k)
            g_idx = tl.load(g_ptrs)
            
            scales_ptrs = scales_ptr + (g_idx[:, None] * stride_scales) + offs_bn[None, :]
            zeros_ptrs = zeros_ptr + (g_idx[:, None] *  stride_zeros) + offs_bn[None, :]

            # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
            scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
        
            a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
            b = tl.load(b_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

            # Now we need to unpack b (which is N-bit values) into 32-bit values
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
            b = b * scales - zeros  # Scale and shift

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk

        c = accumulator.to(tl.float16)
        
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)
    is_trioton = True
except:
    print('trioton not installed.')
    is_trioton = False
        
def make_quant(module, names, bits, groupsize, name=''):
    if isinstance(module, QuantLinear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            delattr(module, attr)
            setattr(module, attr, QuantLinear(bits, groupsize, tmp.in_features, tmp.out_features, tmp.bias is not None))
    for name1, child in module.named_children():
        make_quant(child, names, bits, groupsize, name + '.' + name1 if name != '' else name1)

class QuantLinear(nn.Module): 
    def __init__(self, bits, groupsize, infeatures, outfeatures, bias, kernel_switch_threshold=128, is_trioton=is_trioton, is_cuda=is_cuda):
        super().__init__()
        if bits not in [2,3,4,8]:
            raise NotImplementedError("Only 2,3,4,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.groupsize = groupsize if groupsize != -1 else infeatures
        self.maxq = 2 ** self.bits - 1

        self.register_buffer('qweight', torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32))
        self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures // 32 * self.bits), dtype=torch.int32))
        self.register_buffer('scales', torch.zeros((math.ceil(infeatures / self.groupsize), outfeatures), dtype=torch.float16))
        self.register_buffer('g_idx', torch.tensor([i // self.groupsize  for i in range(infeatures)], dtype = torch.int32))
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures),dtype=torch.float16))
        else:
            self.bias = None
        
        # is performed by unpacking the weights and using torch.matmul
        if self.bits in [2,4,8]: 
            self.register_buffer('wf',torch.tensor(list(range(0,32,self.bits)), dtype=torch.int32).unsqueeze(0),persistent=False)
        elif self.bits == 3:
            self.register_buffer('wf', torch.tensor([[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0],
                                                     [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31],
                                                     [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 0],], dtype=torch.int32).reshape(1,3,12), persistent=False)
            
        self.kernel_switch_threshold = kernel_switch_threshold
        self.is_trioton = is_trioton
        self.is_cuda = is_cuda

    def pack(self, linear, scales, zeros, g_idx = None):
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()
            
        intweight = []
        for idx in range(self.infeatures):
            intweight.append(torch.round((linear.weight.data[:,idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]).to(torch.int)[:,None])
        intweight = torch.cat(intweight,dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2,4,8]:
                for j in range(i, i + (32//self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32//self.bits
                row += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i))
                i += 10
                qweight[row] |= intweight[i] << 30
                row += 1
                qweight[row] |= (intweight[i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 1)
                i += 10
                qweight[row] |= intweight[i] << 31
                row += 1
                qweight[row] |= (intweight[i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qweight[row] |= intweight[j] << (3 * (j - i) + 2)
                i += 10
                row += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
                
        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight) 
        
        zeros -= 1;
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2,4,8]:
                for j in range(i, i + (32//self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32//self.bits
                col += 1
            elif self.bits == 3:
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i))
                i += 10
                qzeros[:, col] |= zeros[:, i] << 30
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 2) & 1
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 1)
                i += 10
                qzeros[:, col] |= zeros[:, i] << 31
                col += 1
                qzeros[:, col] |= (zeros[:, i] >> 1) & 0x3
                i += 1
                for j in range(i, i + 10):
                    qzeros[:, col] |= zeros[:, j] << (3 * (j - i) + 2)
                i += 10
                col += 1
            else:
                raise NotImplementedError("Only 2,3,4,8 bits are supported.")
                
        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros) 
        
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures, )
        x = x.reshape(-1,x.shape[-1])
        if self.bits in [2,4,8]:
            zeros = torch.bitwise_right_shift(torch.unsqueeze(self.qzeros, 2).expand(-1, -1, 32 // self.bits), self.wf.unsqueeze(0)).to(torch.int16 if self.bits == 8 else torch.int8)
            torch.bitwise_and(zeros, (2 ** self.bits) - 1, out=zeros)
        elif self.bits == 3:
            zeros = self.qzeros.reshape(self.qzeros.shape[0], self.qzeros.shape[1]//3, 3, 1).expand(-1, -1, -1, 12)
            zeros = (zeros >> self.wf.unsqueeze(0))
            zeros[:,:,0,10] = (zeros[:,:,0,10]&0x3) | ((zeros[:,:,1,0] << 2)&0x4)
            zeros[:,:,1,11] = (zeros[:,:,1,11]&0x1) | ((zeros[:,:,2,0] << 1)&0x6)
            zeros = zeros & 0x7
            zeros = torch.cat([zeros[:,:,0,:11], zeros[:,:,1,1:12], zeros[:,:,2,1:11]], dim=2)
        zeros = zeros + 1
        zeros = zeros.reshape(self.scales.shape)        
        if self.is_trioton:
            if self.bits in [2,4,8]:
                zeros = self.scales * zeros
                out = torch.empty((x.shape[0], self.outfeatures), device='cuda', dtype=torch.float16)
                grid = lambda META: (triton.cdiv(x.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(self.outfeatures, META['BLOCK_SIZE_N']),)
                matmul_248_kernel[grid](x, self.qweight, out,
                                        self.scales, zeros, self.g_idx,
                                        x.shape[0], self.outfeatures, self.infeatures, self.bits, self.maxq,
                                        x.stride(0), x.stride(1),
                                        self.qweight.stride(0), self.qweight.stride(1),
                                        out.stride(0), out.stride(1),
                                        self.scales.stride(0), zeros.stride(0),) 
            elif self.bits == 3:
                if self.is_cuda is True and (self.kernel_switch_threshold is False or x.shape[0] < self.kernel_switch_threshold):
                    zeros = self.scales * zeros
                    out = torch.zeros((x.shape[0], self.outfeatures), device='cuda', dtype=torch.float32)
                    quant_cuda.vecquant3matmul(x.float(), self.qweight, out, self.scales.float(), zeros.float(), self.g_idx)
                    out = out.half()
                else:
                    weight = self.qweight.reshape(self.qweight.shape[0]//3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
                    weight = (weight >> self.wf.unsqueeze(-1))&0x7
                    weight[:,0,10] = (weight[:,0,10]&0x3) | ((weight[:,1,0] << 2)&0x4)
                    weight[:,1,11] = (weight[:,1,11]&0x1) | ((weight[:,2,0] << 1)&0x6)
                    weight = weight & 0x7
                    weight = torch.cat([weight[:,0,:11], weight[:,1,1:12], weight[:,2,1:11]], dim=1)
                    
                    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
                    
                    weights = (self.scales[self.g_idx] * (weight - zeros[self.g_idx]))
                    out = torch.matmul(x.half(), weights)
        else:
            if  self.is_cuda is True and (self.kernel_switch_threshold is False or x.shape[0] < self.kernel_switch_threshold):
                zeros = self.scales * zeros
                out = torch.zeros((x.shape[0], self.outfeatures), device='cuda', dtype=torch.float32)
                if self.bits == 2:
                    quant_cuda.vecquant2matmul(x.float(), self.qweight, out, self.scales.float(), zeros.float(), self.g_idx)
                elif self.bits == 3:
                    quant_cuda.vecquant3matmul(x.float(), self.qweight, out, self.scales.float(), zeros.float(), self.g_idx)
                elif self.bits == 4:
                    quant_cuda.vecquant4matmul(x.float(), self.qweight, out, self.scales.float(), zeros.float(), self.g_idx)
                elif self.bits == 8:
                    quant_cuda.vecquant8matmul(x.float(), self.qweight, out, self.scales.float(), zeros.float(), self.g_idx)
                out = out.half()
            else:
                if self.bits in [2,4,8]:
                    weight = torch.bitwise_right_shift(torch.unsqueeze(self.qweight, 1).expand(-1, 32 // self.bits, -1), self.wf.unsqueeze(-1)).to(torch.int16 if self.bits == 8 else torch.int8)
                    torch.bitwise_and(weight,(2 ** self.bits) - 1, out=weight)
                elif self.bits == 3:
                    weight = self.qweight.reshape(self.qweight.shape[0]//3, 3, 1, self.qweight.shape[1]).expand(-1, -1, 12, -1)
                    weight = (weight >> self.wf.unsqueeze(-1))&0x7
                    weight[:,0,10] = (weight[:,0,10]&0x3) | ((weight[:,1,0] << 2)&0x4)
                    weight[:,1,11] = (weight[:,1,11]&0x1) | ((weight[:,2,0] << 1)&0x6)
                    weight = weight & 0x7
                    weight = torch.cat([weight[:,0,:11], weight[:,1,1:12], weight[:,2,1:11]], dim=1)
                    
                weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
                    
                weights = (self.scales[self.g_idx] * (weight - zeros[self.g_idx]))
                out = torch.matmul(x.half(), weights)
                
        out = out.reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out
