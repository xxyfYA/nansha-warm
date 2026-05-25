import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


################################################################
# Fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2

        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, 2)
        )

        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1
        k_x1 = torch.cat(
            (
                torch.arange(start=0, end=self.modes1, step=1),
                torch.arange(start=-(self.modes1), end=0, step=1),
            ),
            0,
        ).reshape(m1, 1).repeat(1, m2)
        k_x2 = torch.cat(
            (
                torch.arange(start=0, end=self.modes2, step=1),
                torch.arange(start=-(self.modes2 - 1), end=0, step=1),
            ),
            0,
        ).reshape(1, m2).repeat(m1, 1)
        self.register_buffer("k_x1", k_x1)
        self.register_buffer("k_x2", k_x2)

    @staticmethod
    def _to_complex_weight(w: torch.Tensor) -> torch.Tensor:
        return torch.view_as_complex(w.contiguous())

    @staticmethod
    def _has_batch_shared_coords(x: torch.Tensor, code=None) -> bool:
        return (
            code is None
            and x.dim() == 3
            and x.size(0) > 0
            and (x.size(0) == 1 or x.stride(0) == 0)
        )

    def _fourier_basis(self, x: torch.Tensor, sign: float) -> torch.Tensor:
        # x: (N, 2), output: (N, 2*modes1, 2*modes2-1)
        # Computed in chunks to limit peak GPU memory on large meshes.
        num_nodes = x.shape[0]
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1
        k1_flat = self.k_x1.reshape(-1)
        k2_flat = self.k_x2.reshape(-1)

        chunk_size = 4096
        chunks = []
        for start in range(0, num_nodes, chunk_size):
            end = min(start + chunk_size, num_nodes)
            xc = x[start:end]
            K1 = torch.outer(xc[:, 0], k1_flat).reshape(-1, m1, m2)
            K2 = torch.outer(xc[:, 1], k2_flat).reshape(-1, m1, m2)
            K = K1 + K2
            chunks.append(torch.exp((sign * 1j) * 2 * np.pi * K))
        return torch.cat(chunks, dim=0)

    def _batched_fourier_basis(self, x: torch.Tensor, sign: float) -> torch.Tensor:
        # x: (B, N, 2), output: (B, N, 2*modes1, 2*modes2-1)
        # Computed in chunks to limit peak GPU memory on large meshes.
        batchsize = x.shape[0]
        num_nodes = x.shape[1]
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1
        k1_flat = self.k_x1.reshape(-1)
        k2_flat = self.k_x2.reshape(-1)

        chunk_size = 4096
        chunks = []
        for start in range(0, num_nodes, chunk_size):
            end = min(start + chunk_size, num_nodes)
            xc = x[:, start:end]
            K1 = torch.outer(xc[..., 0].reshape(-1), k1_flat).reshape(batchsize, -1, m1, m2)
            K2 = torch.outer(xc[..., 1].reshape(-1), k2_flat).reshape(batchsize, -1, m1, m2)
            K = K1 + K2
            chunks.append(torch.exp((sign * 1j) * 2 * np.pi * K))
        return torch.cat(chunks, dim=1)

    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None, sm=None):
        batchsize = u.shape[0]

        if x_in is None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        blk_low = u_ft[:, :, :self.modes1, :self.modes2]
        blk_high = u_ft[:, :, -self.modes1:, :self.modes2]
        if sm is not None:
            m_low, m_high = sm
            blk_low = blk_low * (1.0 + m_low[:, None])
            blk_high = blk_high * (1.0 + m_high[:, None])

        w1 = self._to_complex_weight(self.weights1)
        w2 = self._to_complex_weight(self.weights2)
        factor1 = self.compl_mul2d(blk_low, w1)
        factor2 = self.compl_mul2d(blk_high, w2)

        if x_out is None:
            out_ft = torch.zeros(
                batchsize,
                self.out_channels,
                s1,
                s2 // 2 + 1,
                dtype=torch.cfloat,
                device=u.device,
            )
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        u = u + 0j

        if self._has_batch_shared_coords(x_in, code):
            x = x_in[:1]
            if iphi is not None:
                x = iphi(x, code)
            basis = self._fourier_basis(x[0], sign=-1.0)
            Y = torch.einsum("bcn,nxy->bcxy", u, basis)
            return Y

        if iphi is None:
            x = x_in
        else:
            x = iphi(x_in, code)

        basis = self._batched_fourier_basis(x, sign=-1.0)
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        if self._has_batch_shared_coords(x_out, code):
            x = x_out[:1]
            if iphi is not None:
                x = iphi(x, code)
            basis = self._fourier_basis(x[0], sign=1.0)
            Y = torch.einsum("bcxy,nxy->bcn", u_ft, basis)
            return Y.real

        if iphi is None:
            x = x_out
        else:
            x = iphi(x_out, code)

        basis = self._batched_fourier_basis(x, sign=1.0)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        return Y.real


################################################################
# IPHI: learnable mapping from irregular mesh to regular domain
################################################################
class IPHI(nn.Module):
    def __init__(self, width=32):
        super(IPHI, self).__init__()
        """
        Inverse phi: maps irregular 2D coordinates to a computational [0,1]^2 domain.
        Adapted for 2D river/channel mesh (no code conditioning needed).
        x -> xi  where xi in [0,1]^2
        """
        self.width = width
        # Input: (x, y, angle, radius) = 4 features
        self.fc0 = nn.Linear(4, self.width)
        self.fc_no_code = nn.Linear(3 * self.width, 4 * self.width)
        self.fc1 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc2 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc3 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc4 = nn.Linear(4 * self.width, 2)
        self.activation = torch.tanh
        B_freq = np.pi * torch.pow(2, torch.arange(0, self.width // 4, dtype=torch.float))
        self.register_buffer("B_freq", B_freq.reshape(1, 1, 1, self.width // 4))

    def forward(self, x, code=None):
        # x (batch, N_grid, 2)
        # Returns mapped coordinates in [0,1]^2

        # Compute center dynamically from batch mean
        center = x.mean(dim=1, keepdim=True)  # (batch, 1, 2)

        angle = torch.atan2(x[:, :, 1] - center[:, :, 1], x[:, :, 0] - center[:, :, 0])
        radius = torch.norm(x - center, dim=-1, p=2)
        xd = torch.stack([x[:, :, 0], x[:, :, 1], angle, radius], dim=-1)

        # Positional encoding (sin/cos features from NeRF)
        B_freq = self.B_freq
        b, n, d = xd.shape[0], xd.shape[1], xd.shape[2]
        x_sin = torch.sin(B_freq * xd.view(b, n, d, 1)).view(b, n, d * self.width // 4)
        x_cos = torch.cos(B_freq * xd.view(b, n, d, 1)).view(b, n, d * self.width // 4)
        xd = self.fc0(xd)
        xd = torch.cat([xd, x_sin, x_cos], dim=-1).reshape(b, n, 3 * self.width)

        xd = self.fc_no_code(xd)

        xd = self.fc1(xd)
        xd = self.activation(xd)
        xd = self.fc2(xd)
        xd = self.activation(xd)
        xd = self.fc3(xd)
        xd = self.activation(xd)
        xd = self.fc4(xd)
        return x + x * xd


################################################################
# Geo-FNO2d: adapted for 2D hydrodynamic simulation
################################################################
class GeoFNO2d(nn.Module):
    def __init__(
        self,
        modes1,
        modes2,
        width,
        in_channels,
        out_channels,
        s1=40,
        s2=40,
        num_fno_layers: int = 3,
        fc1_hidden: int = 256,
    ):
        super(GeoFNO2d, self).__init__()
        """
        Geo-FNO for 2D irregular mesh hydrodynamic prediction (warm-up model).

        Input: (batch, N_nodes, in_channels) -- 24h forcing features on irregular mesh
        Output: (batch, N_nodes, 1) -- predicted h at t+24 on same irregular mesh

        The network:
        1. Lifts input to width-dimensional channel space via fc0.
        2. Maps from irregular mesh to regular grid via IPHI + spectral conv (conv0).
        3. num_fno_layers layers of spectral conv + 1x1 conv on the regular grid.
        4. Maps back from regular grid to irregular mesh via IPHI + spectral conv (conv4).
        5. Projects from channel space to output (single h value) via fc1, fc2.
        """
        if num_fno_layers < 1:
            raise ValueError(f"num_fno_layers must be >= 1, got {num_fno_layers}")
        self.num_fno_layers = num_fno_layers
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.s1 = s1
        self.s2 = s2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc1_hidden = fc1_hidden

        # Lifting layer
        self.fc0 = nn.Linear(in_channels, self.width)

        # Boundary spectral convs (irregular <-> regular)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2)

        # Middle FNO blocks on the regular grid — count = num_fno_layers
        self.middle_convs = nn.ModuleList([
            SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
            for _ in range(num_fno_layers)
        ])
        self.middle_ws = nn.ModuleList([
            nn.Conv2d(self.width, self.width, 1) for _ in range(num_fno_layers)
        ])
        self.middle_bs = nn.ModuleList([
            nn.Conv2d(2, self.width, 1) for _ in range(num_fno_layers)
        ])

        # Boundary biases (fixed)
        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv1d(2, self.width, 1)

        # Projection layers
        self.fc1 = nn.Linear(self.width, self.fc1_hidden)
        self.fc2 = nn.Linear(self.fc1_hidden, out_channels)

        # IPHI: learnable irregular-to-regular mapping
        self.iphi = IPHI(width=32)
        grid_x = torch.linspace(0, 1, self.s1, dtype=torch.float32)
        grid_y = torch.linspace(0, 1, self.s2, dtype=torch.float32)
        grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing='ij'), dim=-1).unsqueeze(0)
        self.register_buffer("grid", grid)

    def forward(self, u, x_in, x_out=None):
        """
        Args:
            u: (batch, N_nodes, in_channels) -- 24h forcing features
            x_in: (batch, N_nodes, 2) -- 2D coordinates of input nodes
            x_out: (batch, N_nodes, 2) -- 2D coordinates of output nodes (default: same as x_in)
        Returns:
            (batch, N_nodes, 1) -- predicted h at t+24 (direct, no residual)
        """
        if x_out is None:
            x_out = x_in

        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2)

        # Lift to high-dimensional channel space
        u = self.fc0(u)
        u = u.permute(0, 2, 1)  # (batch, width, N)

        # Layer 0: irregular mesh -> regular grid via IPHI
        uc1 = self.conv0(u, x_in=x_in, iphi=self.iphi)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        # Middle FNO blocks (num_fno_layers of them)
        for conv, w, b in zip(self.middle_convs, self.middle_ws, self.middle_bs):
            uc = F.gelu(conv(uc) + w(uc) + b(grid))

        # Layer 4: regular grid -> irregular mesh via IPHI
        u = self.conv4(uc, x_out=x_out, iphi=self.iphi)
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3

        # Project back to output space
        u = u.permute(0, 2, 1)  # (batch, N, width)
        u = self.fc1(u)
        u = F.gelu(u)
        h_pred = self.fc2(u)  # (batch, N, out_channels) = (batch, N, 1)
        return h_pred

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        return self.grid.expand(batchsize, -1, -1, -1)
