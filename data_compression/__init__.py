from .k_medians_compression import k_medians_compression
from .orthonormal_compression import orthonormal_compression
from .dct_compression import dct_compression, normalize_image
from .k_means_compression import k_means_compression

__all__ = ["dct_compression", "k_means_compression", "k_medians_compression", "normalize_image", "orthonormal_compression"]
