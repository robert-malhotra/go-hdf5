# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

go-hdf5 is a pure Go read-only implementation of the HDF5 file format, targeting compatibility with HDF5 library versions 1.8 through 1.14. The project has no CGO dependencies.

**Current Status:** Design phase - only `design.md` exists with complete specification.

## Build Commands

Once implemented, standard Go commands apply:
```bash
go build ./...              # Build all packages
go test ./...               # Run all tests
go test -v -race ./...      # Tests with race detection
go test -run TestName ./... # Run specific test
```

## Architecture

The implementation follows a layered architecture (bottom to top):

1. **Binary I/O Layer** (`internal/binary/`) - Variable-width field reading (2/4/8-byte offsets/lengths), checksum validation (Jenkins lookup3, Fletcher32)

2. **Storage Layer** (`internal/btree/`, `internal/heap/`) - B-tree implementations (V1 for groups/chunks, V2), heap structures (local, global, fractal)

3. **Filter Layer** (`internal/filter/`) - Decompression pipeline: DEFLATE, shuffle, Fletcher32, applied in reverse order during decoding

4. **Message Layer** (`internal/message/`) - Header message parsing by type ID (0x0001=Dataspace, 0x0003=Datatype, 0x0008=Layout, 0x000B=Filter, 0x000C=Attribute, 0x0006=Link, 0x0011=SymbolTable)

5. **Object Layer** (`internal/object/`) - Object headers V1 (8-byte aligned) and V2 (chunked with checksums)

6. **Public API Layer** (`hdf5/`) - File, Group, Dataset, Attribute, Datatype

## Key Design Patterns

- **Reader-based architecture**: Pass `*binary.Reader` through layers, create new readers at offsets with `reader.At()`
- **Layout abstraction**: Three storage strategies (Compact, Contiguous, Chunked) implementing common interface
- **Filter pipeline**: Filters applied in reverse order; per-chunk filter masks allow selective skipping
- **Caching**: Cache parsed heaps and optionally chunks to avoid repeated reads

## HDF5 Format Reference

- **Superblock signature**: `0x89 H D F \r \n 0x1a \n` at offsets 0, 512, 1024, or 2048
- **Superblock versions**: 0, 1, 2, 3 (V2/V3 have same structure)
- **Object header versions**: 1 and 2
- **Layout classes**: 0=Compact (in header), 1=Contiguous (single block), 2=Chunked (B-tree indexed)
- **Filter IDs**: 1=DEFLATE, 2=Shuffle, 3=Fletcher32, 4=SZIP, 5=N-bit, 6=Scale-offset

## Testing

Test files should be generated with Python/h5py to ensure compatibility:
```python
import h5py
import numpy as np

with h5py.File('test.h5', 'w') as f:
    f.create_dataset('data', data=np.array([1,2,3]), chunks=(3,), compression='gzip')
```

## External References

- [HDF5 File Format Specification](https://docs.hdfgroup.org/hdf5/develop/_f_m_t3.html)
- [jhdf - Pure Java HDF5](https://github.com/jamesmudd/jhdf) - Reference implementation
