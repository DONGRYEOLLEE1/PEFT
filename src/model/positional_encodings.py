import math
import torch
import torch.nn as nn



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]   # x: (batch, seq_len, d_model)



class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len=None):
        # x: (batch, seq_len, dim)
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device = x.device).type_as(self.inv_freq)  # (seq_len,)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, dim//2)
        
        cos = freqs.cos()[None, :, :]  # (1, seq_len, dim//2)
        sin = freqs.sin()[None, :, :]  # (1, seq_len, dim//2)
        
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        
        x_rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        
        return x_rotated



class NTKScalingRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000, scaling_factor=1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scaling_factor = scaling_factor
        # NTK scaling: inv_freq 계산 시 scaling_factor 적용
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim * scaling_factor))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_len=None):
        # x: (batch, seq_len, dim)
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)  # (seq_len,)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, dim//2)
        cos = freqs.cos()[None, :, :]  # (1, seq_len, dim//2)
        sin = freqs.sin()[None, :, :]  # (1, seq_len, dim//2)
        x1 = x[..., :self.dim // 2]
        x2 = x[..., self.dim // 2:]
        x_rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return x_rotated



class YaRNRotaryEmbedding(nn.Module):
    def __init__(
        self, 
        dim,
        base = 10000,
        scaling_factor = 1.0,
        original_max_position_embeddings = 2048,
        extrapolation_factor = 1.0,
        attn_factor = 1.0,
        beta_fast = 32,
        beta_slow = 1
    ):
        """
        Args:
            dim: 임베딩 차원
            base: RoPE의 기본 주파수 
            scaling_factor: Positional Information Scaling
            original_max_position_embeddings: 현재 모델의 위치 임베딩
            extrapolation_factor: 외삽 벡터
            attn_factor: Attention-score Scaling
            beta_fast: 빠른 주파수 임계값
            beta_slow: 느린 주파수 임계값
        """
        super().__init__()
        self.dim = dim
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def _get_mscale(self, scale =  1.0):
        """
        YaRN의 scaling factor 계산.
        긴 시퀀스에서 attention 가중치 크기를 조정하여 안정성 향상
        """
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0
    
    def _get_yarn_linear_ramp_mask_from_wavelens(self, wavelens, low_freq_wavelen, high_freq_wavelen):
        """
        파장 기반 YaRN 선형 램프 마스크 생성
        실제 계산된 파장을 사용하여 각 주파수 빈에 대한 마스크 생성
        
        Args:
            wavelens: 각 주파수 빈의 파장 (dim//2,)
            low_freq_wavelen: 저주파수 경계 파장
            high_freq_wavelen: 고주파수 경계 파장
        
        Returns:
            마스크 텐서 (dim//2,): 0(외삽) ~ 1(보간)
        """
        # 파장 기반으로 마스크 계산
        # 짧은 파장(고주파수) → 0에 가까움 (외삽 사용)
        # 긴 파장(저주파수) → 1에 가까움 (보간 사용)
        mask = torch.where(
            wavelens < low_freq_wavelen,
            torch.zeros_like(wavelens),  # 고주파수: 외삽
            torch.where(
                wavelens > high_freq_wavelen,
                torch.ones_like(wavelens),   # 저주파수: 보간
                (wavelens - low_freq_wavelen) / (high_freq_wavelen - low_freq_wavelen)  # 선형 전환
            )
        )
        return mask
    
    def _get_yanr_linear_rmap_mask(self, min_val, max_val, dim):
        """
        YaRN의 선형 램프 마스크 생성
        주파수 범위에 따라 점진적으로 스케일링 적용
        """
        if min_val == max_val:
            return torch.ones(dim // 2)
        
        # 0부터 1까지의 선형 램프 생성
        ramp = torch.linspace(0,1, dim // 2)
        
        # min_val과 max_val 사이에서 선형 보간
        mask = torch.where(
            ramp < min_val,
            torch.zeros_like(ramp),
            torch.where(
                ramp > max_val,
                torch.ones_like(ramp),
                (ramp - min_val) / (max_val - min_val)
            )
        )
        
        return mask
    
    def forward(self, x, seq_len = None):
        """
        Args:
            x: input tensor (batch_size, seq_len, dim)
            seq_len: 시퀀스 길이 (None인 경우, x.shape[1] 사용)
            
        Returns:
            Rotated Tensor (batch_size, seq_len, dim)
        """
        if seq_len is None:
            seq_len = x.shape[1]
            
        # scaling_factor 계산
        scale = (seq_len / self.original_max_position_embeddings) * self.scaling_factor
        
        # YaRN 주파수 스케일링 적용
        if scale > 1.0:
            # 고주파수와 저주파수 영역 정의
            # 고주파수: 빠르게 변하는 패턴 (지역적 정보)
            # 저주파수: 천천히 변하는 패턴 (전역적 정보)
            low_freq_wavelen = self.original_max_position_embeddings / self.beta_fast
            high_freq_wavelen = self.original_max_position_embeddings / self.beta_slow
            
            # 각 주파수 빈의 파장 계산
            wavelens = 2 * math.pi / self.inv_freq
            
            # YaRN 마스크 생성: 주파수에 따라 다른 스케일링 적용
            # 고주파수 영역: 외삽, 중간 주파수영역: 점진적전환, 저주파수 영역: 내삽
            # yarn_ramp_mask = self._get_yanr_linear_rmap_mask(
            #     low_freq_wavelen, high_freq_wavelen, self.dim
            # ).to(x.device)
            
            yarn_ramp_mask = self._get_yarn_linear_ramp_mask_from_wavelens(
                wavelens, low_freq_wavelen, high_freq_wavelen
            )
            
            # 스케일링된 역주파수 계산
            inv_freq_extrapolation = self.inv_freq / self.extrapolation_factor
            inv_freq_intrapolation = self.inv_freq / scale
            
            # 마스크를 사용하여 주파수별로 다른 스케일링 적용
            inv_freq_scaled = (
                (1 - yarn_ramp_mask) * inv_freq_extrapolation + yarn_ramp_mask * inv_freq_intrapolation
            )
        else:
            inv_freq_scaled = self.inv_freq
            
        # 위치 벡터 생성
        t = torch.arange(seq_len, device = x.device).type_as(inv_freq_scaled)
        
        # 주파수 행렬 계산
        freqs = torch.einsum("i,j->ij", t, inv_freq_scaled)
        
        # cos, sin 값 계산
        cos = freqs.cos()[None, :, :]   # (1, seq_len, dim//2)
        sin = freqs.sin()[None, :, :]   # (1, seq_len, dim//2)
        
        x1 = x[..., :self.dim//2]
        x2 = x[..., self.dim//2:]
        
        x_rotated= torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin],  dim = -1)
        
        # yarn attn scaling 적용
        if scale > 1.0:
            mscale = self._get_mscale(scale * self.attn_factor)
            x_rotated = x_rotated * mscale
            
        return x_rotated