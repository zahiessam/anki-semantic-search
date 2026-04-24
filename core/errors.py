"""Shared error-classification helpers."""

# ============================================================================
# Embedding Error Helpers
# ============================================================================

def is_embedding_dimension_mismatch(exc):
    s = str(exc)
    return "not aligned" in s or ("shapes" in s and "dim" in s)


_is_embedding_dimension_mismatch = is_embedding_dimension_mismatch
