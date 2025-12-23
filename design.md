# Software Design Document: Pure Go HDF5 Read-Only Implementation

**Project Name:** go-hdf5  
**Version:** 1.0  
**Date:** December 2024  
**Status:** Draft  
**Scope:** Complete Read-Only Implementation  

---

## Executive Summary

This document specifies a complete read-only implementation of the HDF5 file format in pure Go. The implementation targets compatibility with files produced by HDF5 library versions 1.8 through 1.14.

**Key Metrics:**

| Metric | Target |
|--------|--------|
| Estimated Lines of Code | 15,000 - 20,000 |
| Development Time | 6-8 months (1 developer) |
| File Compatibility | 95%+ of real-world files |
| Test Coverage | 80%+ |

---

## Table of Contents

1. [Goals and Requirements](#1-goals-and-requirements)
2. [Architecture Overview](#2-architecture-overview)
3. [Package Structure](#3-package-structure)
4. [Core Infrastructure](#4-core-infrastructure)
5. [Superblock Implementation](#5-superblock-implementation)
6. [Object Headers](#6-object-headers)
7. [Header Messages](#7-header-messages)
8. [B-tree Implementation](#8-b-tree-implementation)
9. [Heap Implementation](#9-heap-implementation)
10. [Datatype System](#10-datatype-system)
11. [Storage Layouts](#11-storage-layouts)
12. [Filter Pipeline](#12-filter-pipeline)
13. [Public API](#13-public-api)
14. [Testing Strategy](#14-testing-strategy)
15. [Implementation Phases](#15-implementation-phases)

---

## 1. Goals and Requirements

### 1.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-01 | Open and validate HDF5 files | Must |
| FR-02 | Navigate group hierarchy | Must |
| FR-03 | Read dataset metadata | Must |
| FR-04 | Read contiguous datasets | Must |
| FR-05 | Read chunked datasets | Must |
| FR-06 | Read compact datasets | Must |
| FR-07 | Decompress DEFLATE data | Must |
| FR-08 | Apply shuffle filter | Must |
| FR-09 | Validate Fletcher32 checksums | Must |
| FR-10 | Read all numeric types | Must |
| FR-11 | Read fixed-length strings | Must |
| FR-12 | Read variable-length strings | Must |
| FR-13 | Read compound types | Must |
| FR-14 | Read array types | Must |
| FR-15 | Read attributes | Must |
| FR-16 | Read enum types | Should |
| FR-17 | Follow soft links | Should |
| FR-18 | Follow external links | Could |

### 1.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | No CGO dependencies | Required |
| NFR-02 | Memory efficiency | < 100MB overhead |
| NFR-03 | Thread-safe reads | Required |
| NFR-04 | Descriptive errors | Required |

---

## 2. Architecture Overview

### 2.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PUBLIC API LAYER                             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐ │
│  │  File   │  │  Group  │  │ Dataset │  │  Attr   │  │ Datatype  │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └───────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                         OBJECT LAYER                                 │
│  ┌───────────────────┐  ┌───────────────┐  ┌──────────────────────┐│
│  │   Object Header   │  │    Links      │  │     Iterators        ││
│  │   (v1 and v2)     │  │               │  │                      ││
│  └───────────────────┘  └───────────────┘  └──────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                        MESSAGE LAYER                                 │
│  ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐┌─────────────┐ │
│  │Dspace  ││Dtype   ││Layout  ││FillVal ││Filter  ││ Attribute   │ │
│  └────────┘└────────┘└────────┘└────────┘└────────┘└─────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                        STORAGE LAYER                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │   B-trees   │  │    Heaps    │  │ Chunk Cache │  │  Raw I/O   │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                       FILTER LAYER                                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ DEFLATE  │  │ Shuffle  │  │Fletcher32│  │  N-bit   │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
├─────────────────────────────────────────────────────────────────────┤
│                     BINARY I/O LAYER                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  Binary Reader  │  │    Checksum     │  │  Address Manager    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Package Structure

```
go-hdf5/
├── hdf5/                           # Public API
│   ├── file.go                     # File operations
│   ├── group.go                    # Group operations
│   ├── dataset.go                  # Dataset operations
│   ├── attribute.go                # Attribute operations
│   ├── datatype.go                 # Public datatype API
│   ├── dataspace.go                # Public dataspace API
│   └── errors.go                   # Error types
│
├── internal/
│   ├── binary/                     # Binary I/O
│   │   ├── reader.go               # Variable-width reader
│   │   └── checksum.go             # Checksums
│   │
│   ├── superblock/                 # Superblock parsing
│   │   ├── superblock.go           # Common interface
│   │   ├── v0.go                   # Version 0
│   │   ├── v2.go                   # Version 2
│   │   └── v3.go                   # Version 3
│   │
│   ├── object/                     # Object headers
│   │   ├── header.go               # Common interface
│   │   ├── header_v1.go            # Version 1
│   │   └── header_v2.go            # Version 2
│   │
│   ├── message/                    # Header messages
│   │   ├── message.go              # Interface
│   │   ├── dataspace.go            # 0x0001
│   │   ├── datatype.go             # 0x0003
│   │   ├── layout.go               # 0x0008
│   │   ├── filter.go               # 0x000B
│   │   ├── attribute.go            # 0x000C
│   │   ├── link.go                 # 0x0006
│   │   └── symboltable.go          # 0x0011
│   │
│   ├── btree/                      # B-tree implementations
│   │   ├── v1_group.go             # V1 for groups
│   │   ├── v1_chunk.go             # V1 for chunks
│   │   └── v2.go                   # V2 B-tree
│   │
│   ├── heap/                       # Heap implementations
│   │   ├── local.go                # Local heap
│   │   ├── global.go               # Global heap
│   │   └── fractal.go              # Fractal heap
│   │
│   ├── dtype/                      # Datatype handling
│   │   ├── dtype.go                # Structure
│   │   ├── parse.go                # Parsing
│   │   └── convert.go              # Go conversion
│   │
│   ├── layout/                     # Storage layouts
│   │   ├── contiguous.go
│   │   ├── chunked.go
│   │   └── compact.go
│   │
│   └── filter/                     # Filters
│       ├── pipeline.go
│       ├── deflate.go
│       ├── shuffle.go
│       └── fletcher32.go
│
└── testdata/                       # Test files
```

---

## 4. Core Infrastructure

### 4.1 Binary Reader

```go
// internal/binary/reader.go

package binary

import (
    "encoding/binary"
    "io"
)

// Reader provides methods for reading HDF5 binary data
type Reader struct {
    r           io.ReaderAt
    order       binary.ByteOrder
    offsetSize  int  // 2, 4, or 8 bytes
    lengthSize  int  // 2, 4, or 8 bytes
    pos         int64
}

// Config holds reader configuration from superblock
type Config struct {
    ByteOrder   binary.ByteOrder
    OffsetSize  int
    LengthSize  int
}

// NewReader creates a binary reader
func NewReader(r io.ReaderAt, cfg Config) *Reader {
    return &Reader{
        r:          r,
        order:      cfg.ByteOrder,
        offsetSize: cfg.OffsetSize,
        lengthSize: cfg.LengthSize,
    }
}

// At returns a new reader at the given offset
func (r *Reader) At(offset int64) *Reader {
    return &Reader{
        r:          r.r,
        order:      r.order,
        offsetSize: r.offsetSize,
        lengthSize: r.lengthSize,
        pos:        offset,
    }
}

// ReadBytes reads exactly n bytes
func (r *Reader) ReadBytes(n int) ([]byte, error) {
    buf := make([]byte, n)
    _, err := r.r.ReadAt(buf, r.pos)
    r.pos += int64(n)
    return buf, err
}

// ReadUint8 reads an unsigned 8-bit integer
func (r *Reader) ReadUint8() (uint8, error) {
    buf, err := r.ReadBytes(1)
    if err != nil {
        return 0, err
    }
    return buf[0], nil
}

// ReadUint16 reads an unsigned 16-bit integer
func (r *Reader) ReadUint16() (uint16, error) {
    buf, err := r.ReadBytes(2)
    if err != nil {
        return 0, err
    }
    return r.order.Uint16(buf), nil
}

// ReadUint32 reads an unsigned 32-bit integer
func (r *Reader) ReadUint32() (uint32, error) {
    buf, err := r.ReadBytes(4)
    if err != nil {
        return 0, err
    }
    return r.order.Uint32(buf), nil
}

// ReadUint64 reads an unsigned 64-bit integer
func (r *Reader) ReadUint64() (uint64, error) {
    buf, err := r.ReadBytes(8)
    if err != nil {
        return 0, err
    }
    return r.order.Uint64(buf), nil
}

// ReadOffset reads a file offset (variable width)
func (r *Reader) ReadOffset() (uint64, error) {
    buf, err := r.ReadBytes(r.offsetSize)
    if err != nil {
        return 0, err
    }
    return r.decodeUint(buf, r.offsetSize), nil
}

// ReadLength reads a length value (variable width)
func (r *Reader) ReadLength() (uint64, error) {
    buf, err := r.ReadBytes(r.lengthSize)
    if err != nil {
        return 0, err
    }
    return r.decodeUint(buf, r.lengthSize), nil
}

func (r *Reader) decodeUint(buf []byte, size int) uint64 {
    switch size {
    case 2:
        return uint64(r.order.Uint16(buf))
    case 4:
        return uint64(r.order.Uint32(buf))
    case 8:
        return r.order.Uint64(buf)
    default:
        var val uint64
        for i := size - 1; i >= 0; i-- {
            val = (val << 8) | uint64(buf[i])
        }
        return val
    }
}

// IsUndefinedOffset checks for undefined sentinel
func (r *Reader) IsUndefinedOffset(offset uint64) bool {
    switch r.offsetSize {
    case 2:
        return offset == 0xFFFF
    case 4:
        return offset == 0xFFFFFFFF
    case 8:
        return offset == 0xFFFFFFFFFFFFFFFF
    }
    return false
}

// Skip advances position by n bytes
func (r *Reader) Skip(n int64) {
    r.pos += n
}

// Align advances to next multiple of alignment
func (r *Reader) Align(alignment int64) {
    if remainder := r.pos % alignment; remainder != 0 {
        r.pos += alignment - remainder
    }
}

// Pos returns current position
func (r *Reader) Pos() int64 {
    return r.pos
}
```

### 4.2 Checksum Implementations

```go
// internal/binary/checksum.go

package binary

// Lookup3Checksum computes Jenkins lookup3 hash
func Lookup3Checksum(data []byte) uint32 {
    var a, b, c uint32 = 0xdeadbeef, 0xdeadbeef, 0xdeadbeef
    
    i := 0
    for ; i+12 <= len(data); i += 12 {
        a += uint32(data[i]) | uint32(data[i+1])<<8 | 
             uint32(data[i+2])<<16 | uint32(data[i+3])<<24
        b += uint32(data[i+4]) | uint32(data[i+5])<<8 | 
             uint32(data[i+6])<<16 | uint32(data[i+7])<<24
        c += uint32(data[i+8]) | uint32(data[i+9])<<8 | 
             uint32(data[i+10])<<16 | uint32(data[i+11])<<24
        a, b, c = lookup3Mix(a, b, c)
    }
    
    // Handle remaining bytes...
    a, b, c = lookup3Final(a, b, c)
    return c
}

func lookup3Mix(a, b, c uint32) (uint32, uint32, uint32) {
    a -= c; a ^= rotl32(c, 4);  c += b
    b -= a; b ^= rotl32(a, 6);  a += c
    c -= b; c ^= rotl32(b, 8);  b += a
    a -= c; a ^= rotl32(c, 16); c += b
    b -= a; b ^= rotl32(a, 19); a += c
    c -= b; c ^= rotl32(b, 4);  b += a
    return a, b, c
}

func lookup3Final(a, b, c uint32) (uint32, uint32, uint32) {
    c ^= b; c -= rotl32(b, 14)
    a ^= c; a -= rotl32(c, 11)
    b ^= a; b -= rotl32(a, 25)
    c ^= b; c -= rotl32(b, 16)
    a ^= c; a -= rotl32(c, 4)
    b ^= a; b -= rotl32(a, 14)
    c ^= b; c -= rotl32(b, 24)
    return a, b, c
}

func rotl32(x uint32, k uint) uint32 {
    return (x << k) | (x >> (32 - k))
}

// Fletcher32 computes Fletcher-32 checksum
func Fletcher32(data []byte) uint32 {
    var sum1, sum2 uint32
    
    // Pad if odd length
    if len(data)%2 != 0 {
        data = append(data, 0)
    }
    
    for i := 0; i < len(data); i += 2 {
        word := uint32(data[i]) | uint32(data[i+1])<<8
        sum1 = (sum1 + word) % 65535
        sum2 = (sum2 + sum1) % 65535
    }
    
    return (sum2 << 16) | sum1
}
```

---

## 5. Superblock Implementation

```go
// internal/superblock/superblock.go

package superblock

import (
    "encoding/binary"
    "errors"
    "io"
)

var Signature = []byte{0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'}

var (
    ErrNotHDF5            = errors.New("not an HDF5 file")
    ErrUnsupportedVersion = errors.New("unsupported superblock version")
)

// Superblock contains HDF5 file metadata
type Superblock struct {
    Version          uint8
    OffsetSize       uint8
    LengthSize       uint8
    BaseAddress      uint64
    EOFAddress       uint64
    RootGroupAddress uint64
    ByteOrder        binary.ByteOrder
    
    // V0/V1 specific
    GroupLeafNodeK     uint16
    GroupInternalNodeK uint16
}

// Read locates and parses the superblock
func Read(r io.ReaderAt) (*Superblock, error) {
    offsets := []int64{0, 512, 1024, 2048}
    
    for _, offset := range offsets {
        sig := make([]byte, 8)
        if _, err := r.ReadAt(sig, offset); err != nil {
            continue
        }
        
        if !bytesEqual(sig, Signature) {
            continue
        }
        
        // Read version
        verBuf := make([]byte, 1)
        r.ReadAt(verBuf, offset+8)
        version := verBuf[0]
        
        switch version {
        case 0, 1:
            return readV0(r, offset)
        case 2:
            return readV2(r, offset)
        case 3:
            return readV3(r, offset)
        default:
            return nil, ErrUnsupportedVersion
        }
    }
    
    return nil, ErrNotHDF5
}

func readV0(r io.ReaderAt, offset int64) (*Superblock, error) {
    buf := make([]byte, 24)
    if _, err := r.ReadAt(buf, offset+8); err != nil {
        return nil, err
    }
    
    sb := &Superblock{
        Version:            buf[0],
        OffsetSize:         buf[5],
        LengthSize:         buf[6],
        GroupLeafNodeK:     binary.LittleEndian.Uint16(buf[8:10]),
        GroupInternalNodeK: binary.LittleEndian.Uint16(buf[10:12]),
        ByteOrder:          binary.LittleEndian,
    }
    
    pos := offset + 24
    if sb.Version == 1 {
        pos += 4 // Skip indexed storage K
    }
    
    osize := int(sb.OffsetSize)
    addrBuf := make([]byte, osize)
    
    // Base address
    r.ReadAt(addrBuf, pos)
    sb.BaseAddress = decodeUint(addrBuf, osize)
    pos += int64(osize)
    
    pos += int64(osize) // Skip free space
    
    // EOF address
    r.ReadAt(addrBuf, pos)
    sb.EOFAddress = decodeUint(addrBuf, osize)
    pos += int64(osize)
    
    pos += int64(osize) // Skip driver info
    pos += int64(osize) // Skip link name offset
    
    // Root group address
    r.ReadAt(addrBuf, pos)
    sb.RootGroupAddress = decodeUint(addrBuf, osize)
    
    return sb, nil
}

func readV2(r io.ReaderAt, offset int64) (*Superblock, error) {
    buf := make([]byte, 4)
    r.ReadAt(buf, offset+8)
    
    sb := &Superblock{
        Version:    buf[0],
        OffsetSize: buf[1],
        LengthSize: buf[2],
        ByteOrder:  binary.LittleEndian,
    }
    
    pos := offset + 12
    osize := int(sb.OffsetSize)
    addrBuf := make([]byte, osize)
    
    // Base address
    r.ReadAt(addrBuf, pos)
    sb.BaseAddress = decodeUint(addrBuf, osize)
    pos += int64(osize)
    
    pos += int64(osize) // Skip extension address
    
    // EOF address
    r.ReadAt(addrBuf, pos)
    sb.EOFAddress = decodeUint(addrBuf, osize)
    pos += int64(osize)
    
    // Root group address
    r.ReadAt(addrBuf, pos)
    sb.RootGroupAddress = decodeUint(addrBuf, osize)
    
    return sb, nil
}

func readV3(r io.ReaderAt, offset int64) (*Superblock, error) {
    return readV2(r, offset) // Same structure
}

func decodeUint(buf []byte, size int) uint64 {
    switch size {
    case 2:
        return uint64(binary.LittleEndian.Uint16(buf))
    case 4:
        return uint64(binary.LittleEndian.Uint32(buf))
    case 8:
        return binary.LittleEndian.Uint64(buf)
    }
    return 0
}

func bytesEqual(a, b []byte) bool {
    if len(a) != len(b) {
        return false
    }
    for i := range a {
        if a[i] != b[i] {
            return false
        }
    }
    return true
}
```

---

## 6. Object Headers

### 6.1 Version 1 Object Header

```go
// internal/object/header_v1.go

package object

/*
Version 1 Object Header:
Offset  Size  Description
0       1     Version (1)
1       1     Reserved
2       2     Number of messages
4       4     Reference count
8       4     Header size
12      var   Messages (8-byte aligned)
*/

func readV1(r *binary.Reader, address uint64) (*Header, error) {
    version, _ := r.ReadUint8()
    r.Skip(1) // Reserved
    
    numMessages, _ := r.ReadUint16()
    refCount, _ := r.ReadUint32()
    headerSize, _ := r.ReadUint32()
    
    hdr := &Header{
        Version:  1,
        Address:  address,
        RefCount: refCount,
        Messages: make([]message.Message, 0, numMessages),
    }
    
    r.Align(8)
    
    messagesEnd := r.Pos() + int64(headerSize)
    
    for r.Pos() < messagesEnd {
        msgType, _ := r.ReadUint16()
        dataSize, _ := r.ReadUint16()
        flags, _ := r.ReadUint8()
        r.Skip(3) // Reserved
        
        data, _ := r.ReadBytes(int(dataSize))
        r.Align(8)
        
        if msgType == 0 {
            continue // NIL message
        }
        
        msg, err := message.Parse(message.Type(msgType), data, flags, r)
        if err != nil {
            continue // Skip unknown messages
        }
        
        // Handle continuation
        if cont, ok := msg.(*message.Continuation); ok {
            contMsgs, _ := readContinuation(r, cont.Offset, cont.Length, 1)
            hdr.Messages = append(hdr.Messages, contMsgs...)
        } else {
            hdr.Messages = append(hdr.Messages, msg)
        }
    }
    
    return hdr, nil
}
```

### 6.2 Version 2 Object Header

```go
// internal/object/header_v2.go

package object

/*
Version 2 Object Header:
Offset  Size  Description
0       4     Signature ("OHDR")
4       1     Version (2)
5       1     Flags
var     var   Optional timestamps (if flags & 0x04)
var     var   Optional attr thresholds (if flags & 0x08)
var     1-8   Chunk 0 size (size = 1 << (flags & 0x03))
var     var   Messages
var     4     Checksum
*/

func readV2(r *binary.Reader, address uint64) (*Header, error) {
    r.Skip(4) // Signature already verified
    
    version, _ := r.ReadUint8()
    flags, _ := r.ReadUint8()
    
    hdr := &Header{
        Version: 2,
        Address: address,
        Flags:   flags,
    }
    
    // Optional timestamps
    if flags&0x04 != 0 {
        hdr.AccessTime, _ = r.ReadUint32()
        hdr.ModTime, _ = r.ReadUint32()
        hdr.ChangeTime, _ = r.ReadUint32()
        hdr.BirthTime, _ = r.ReadUint32()
    }
    
    // Optional attribute thresholds
    if flags&0x08 != 0 {
        r.Skip(4)
    }
    
    // Chunk 0 size
    sizeFieldSize := 1 << (flags & 0x03)
    chunk0Size, _ := r.ReadUintN(sizeFieldSize)
    
    chunkEnd := r.Pos() + int64(chunk0Size) - 4 // -4 for checksum
    
    for r.Pos() < chunkEnd {
        msgType, _ := r.ReadUint8()
        
        var dataSize uint32
        if msgType == 0xFF {
            msgType, _ = r.ReadUint8()
            dataSize, _ = r.ReadUint32()
        } else {
            s, _ := r.ReadUint16()
            dataSize = uint32(s)
        }
        
        msgFlags, _ := r.ReadUint8()
        
        if hdr.Flags&0x10 != 0 {
            r.Skip(2) // Creation order
        }
        
        data, _ := r.ReadBytes(int(dataSize))
        
        if msgType == 0 {
            continue
        }
        
        msg, _ := message.Parse(message.Type(msgType), data, msgFlags, r)
        if msg != nil {
            hdr.Messages = append(hdr.Messages, msg)
        }
    }
    
    return hdr, nil
}
```

---

## 7. Header Messages

### 7.1 Dataspace Message (0x0001)

```go
// internal/message/dataspace.go

package message

type Dataspace struct {
    Version    uint8
    Rank       int
    Type       DataspaceType
    Dimensions []uint64
    MaxDims    []uint64
}

type DataspaceType uint8

const (
    DataspaceScalar DataspaceType = 0
    DataspaceSimple DataspaceType = 1
    DataspaceNull   DataspaceType = 2
)

func (m *Dataspace) Type() Type { return TypeDataspace }

func (m *Dataspace) NumElements() uint64 {
    if m.Type != DataspaceSimple {
        return 1
    }
    n := uint64(1)
    for _, d := range m.Dimensions {
        n *= d
    }
    return n
}

func parseDataspace(data []byte) (*Dataspace, error) {
    ds := &Dataspace{
        Version: data[0],
        Rank:    int(data[1]),
    }
    
    flags := data[2]
    
    if ds.Version == 2 {
        ds.Type = DataspaceType(data[3])
    } else if ds.Rank == 0 {
        ds.Type = DataspaceScalar
    } else {
        ds.Type = DataspaceSimple
    }
    
    if ds.Type != DataspaceSimple {
        return ds, nil
    }
    
    offset := 4
    if ds.Version == 1 {
        offset = 8
    }
    
    lengthSize := 8 // Assume 8-byte lengths
    
    ds.Dimensions = make([]uint64, ds.Rank)
    for i := 0; i < ds.Rank; i++ {
        ds.Dimensions[i] = binary.LittleEndian.Uint64(data[offset:])
        offset += lengthSize
    }
    
    if flags&0x01 != 0 {
        ds.MaxDims = make([]uint64, ds.Rank)
        for i := 0; i < ds.Rank; i++ {
            ds.MaxDims[i] = binary.LittleEndian.Uint64(data[offset:])
            offset += lengthSize
        }
    }
    
    return ds, nil
}
```

### 7.2 Data Layout Message (0x0008)

```go
// internal/message/layout.go

package message

type DataLayout struct {
    Version     uint8
    Class       LayoutClass
    CompactData []byte
    Address     uint64
    Size        uint64
    ChunkDims   []uint32
    IndexAddr   uint64
}

type LayoutClass uint8

const (
    LayoutCompact    LayoutClass = 0
    LayoutContiguous LayoutClass = 1
    LayoutChunked    LayoutClass = 2
)

func (m *DataLayout) Type() Type { return TypeDataLayout }

func parseDataLayout(data []byte, r *binary.Reader) (*DataLayout, error) {
    layout := &DataLayout{
        Version: data[0],
        Class:   LayoutClass(data[1]),
    }
    
    offset := 2
    
    switch layout.Class {
    case LayoutCompact:
        size := binary.LittleEndian.Uint16(data[offset:])
        offset += 2
        layout.CompactData = data[offset : offset+int(size)]
        
    case LayoutContiguous:
        layout.Address = binary.LittleEndian.Uint64(data[offset:])
        offset += 8
        layout.Size = binary.LittleEndian.Uint64(data[offset:])
        
    case LayoutChunked:
        if layout.Version >= 3 {
            flags := data[offset]
            offset++
            ndims := int(data[offset])
            offset++
            dimSize := int(data[offset])
            offset++
            
            layout.ChunkDims = make([]uint32, ndims)
            for i := 0; i < ndims; i++ {
                layout.ChunkDims[i] = uint32(decodeUint(data[offset:], dimSize))
                offset += dimSize
            }
            
            layout.IndexAddr = binary.LittleEndian.Uint64(data[offset:])
        } else {
            ndims := int(data[2])
            offset = 4
            layout.IndexAddr = binary.LittleEndian.Uint64(data[offset:])
            offset += 8
            
            layout.ChunkDims = make([]uint32, ndims)
            for i := 0; i < ndims; i++ {
                layout.ChunkDims[i] = binary.LittleEndian.Uint32(data[offset:])
                offset += 4
            }
        }
    }
    
    return layout, nil
}
```

### 7.3 Filter Pipeline Message (0x000B)

```go
// internal/message/filter.go

package message

type FilterPipeline struct {
    Version uint8
    Filters []FilterInfo
}

type FilterInfo struct {
    ID         uint16
    Flags      uint16
    ClientData []uint32
}

const (
    FilterDeflate    uint16 = 1
    FilterShuffle    uint16 = 2
    FilterFletcher32 uint16 = 3
    FilterSZIP       uint16 = 4
    FilterNBit       uint16 = 5
    FilterScaleOffset uint16 = 6
)

func (m *FilterPipeline) Type() Type { return TypeFilterPipeline }

func parseFilterPipeline(data []byte) (*FilterPipeline, error) {
    fp := &FilterPipeline{
        Version: data[0],
        Filters: make([]FilterInfo, data[1]),
    }
    
    offset := 2
    if fp.Version == 1 {
        offset = 8
    }
    
    for i := range fp.Filters {
        f := &fp.Filters[i]
        f.ID = binary.LittleEndian.Uint16(data[offset:])
        offset += 2
        
        var nameLen uint16
        if fp.Version == 1 {
            nameLen = binary.LittleEndian.Uint16(data[offset:])
        }
        offset += 2
        
        f.Flags = binary.LittleEndian.Uint16(data[offset:])
        offset += 2
        
        numCD := binary.LittleEndian.Uint16(data[offset:])
        offset += 2
        
        if fp.Version == 1 && nameLen > 0 {
            offset += int(nameLen)
            if nameLen%8 != 0 {
                offset += 8 - int(nameLen%8)
            }
        }
        
        f.ClientData = make([]uint32, numCD)
        for j := range f.ClientData {
            f.ClientData[j] = binary.LittleEndian.Uint32(data[offset:])
            offset += 4
        }
        
        if fp.Version == 1 && numCD%2 != 0 {
            offset += 4
        }
    }
    
    return fp, nil
}
```

---

## 8. B-tree Implementation

### 8.1 V1 B-tree for Chunks

```go
// internal/btree/v1_chunk.go

package btree

type ChunkLocation struct {
    Offset     []uint64
    Address    uint64
    Size       uint32
    FilterMask uint32
}

type V1ChunkBTree struct {
    reader   *binary.Reader
    rootAddr uint64
    ndims    int
}

func NewV1ChunkBTree(r *binary.Reader, rootAddr uint64, ndims int) *V1ChunkBTree {
    return &V1ChunkBTree{reader: r, rootAddr: rootAddr, ndims: ndims}
}

func (bt *V1ChunkBTree) Iterate(fn func(*ChunkLocation) error) error {
    return bt.iterateNode(bt.rootAddr, fn)
}

func (bt *V1ChunkBTree) iterateNode(addr uint64, fn func(*ChunkLocation) error) error {
    r := bt.reader.At(int64(addr))
    
    sig, _ := r.ReadBytes(4)
    if string(sig) != "TREE" {
        return fmt.Errorf("invalid B-tree signature")
    }
    
    nodeType, _ := r.ReadUint8()
    level, _ := r.ReadUint8()
    entriesUsed, _ := r.ReadUint16()
    
    r.Skip(16) // Siblings
    
    if level > 0 {
        // Internal node
        for i := 0; i <= int(entriesUsed); i++ {
            if i < int(entriesUsed) {
                r.Skip(8 + bt.ndims*8) // Key
            }
            childAddr, _ := r.ReadOffset()
            bt.iterateNode(childAddr, fn)
        }
    } else {
        // Leaf node
        for i := 0; i < int(entriesUsed); i++ {
            size, _ := r.ReadUint32()
            filterMask, _ := r.ReadUint32()
            
            offset := make([]uint64, bt.ndims)
            for j := 0; j < bt.ndims; j++ {
                offset[j], _ = r.ReadUint64()
            }
            
            chunkAddr, _ := r.ReadOffset()
            
            fn(&ChunkLocation{
                Offset:     offset,
                Address:    chunkAddr,
                Size:       size,
                FilterMask: filterMask,
            })
        }
    }
    
    return nil
}
```

---

## 9. Heap Implementation

### 9.1 Local Heap

```go
// internal/heap/local.go

package heap

type LocalHeap struct {
    reader   *binary.Reader
    dataAddr uint64
    dataSize uint64
    cache    []byte
}

func NewLocalHeap(r *binary.Reader, headerAddr uint64) (*LocalHeap, error) {
    hr := r.At(int64(headerAddr))
    
    sig, _ := hr.ReadBytes(4)
    if string(sig) != "HEAP" {
        return nil, fmt.Errorf("invalid local heap signature")
    }
    
    hr.Skip(4) // Version + reserved
    
    lh := &LocalHeap{reader: r}
    lh.dataSize, _ = hr.ReadLength()
    hr.ReadLength() // Free list offset
    lh.dataAddr, _ = hr.ReadOffset()
    
    return lh, nil
}

func (lh *LocalHeap) GetString(offset uint64) (string, error) {
    if lh.cache == nil {
        r := lh.reader.At(int64(lh.dataAddr))
        lh.cache, _ = r.ReadBytes(int(lh.dataSize))
    }
    
    end := int(offset)
    for end < len(lh.cache) && lh.cache[end] != 0 {
        end++
    }
    
    return string(lh.cache[offset:end]), nil
}
```

### 9.2 Global Heap

```go
// internal/heap/global.go

package heap

type GlobalHeap struct {
    reader *binary.Reader
    cache  map[uint64]map[uint16][]byte
}

func NewGlobalHeap(r *binary.Reader) *GlobalHeap {
    return &GlobalHeap{
        reader: r,
        cache:  make(map[uint64]map[uint16][]byte),
    }
}

func (gh *GlobalHeap) Get(collAddr uint64, objIndex uint16) ([]byte, error) {
    if coll, ok := gh.cache[collAddr]; ok {
        if data, ok := coll[objIndex]; ok {
            return data, nil
        }
    }
    
    // Read collection
    r := gh.reader.At(int64(collAddr))
    sig, _ := r.ReadBytes(4)
    if string(sig) != "GCOL" {
        return nil, fmt.Errorf("invalid global heap signature")
    }
    
    r.Skip(4) // Version + reserved
    collSize, _ := r.ReadLength()
    
    coll := make(map[uint16][]byte)
    endPos := int64(collAddr + collSize)
    
    for r.Pos() < endPos {
        idx, _ := r.ReadUint16()
        if idx == 0 {
            break
        }
        
        r.Skip(6) // Ref count + reserved
        objSize, _ := r.ReadLength()
        data, _ := r.ReadBytes(int(objSize))
        r.Align(8)
        
        coll[idx] = data
    }
    
    gh.cache[collAddr] = coll
    return coll[objIndex], nil
}
```

---

## 10. Datatype System

```go
// internal/dtype/dtype.go

package dtype

type Class uint8

const (
    ClassFixedPoint Class = 0
    ClassFloatPoint Class = 1
    ClassString     Class = 3
    ClassCompound   Class = 6
    ClassEnum       Class = 8
    ClassVarLen     Class = 9
    ClassArray      Class = 10
)

type ByteOrder uint8

const (
    OrderLE ByteOrder = 0
    OrderBE ByteOrder = 1
)

type Datatype struct {
    Class      Class
    Size       uint32
    ByteOrder  ByteOrder
    Properties interface{}
}

type FixedPointProps struct {
    BitOffset    uint16
    BitPrecision uint16
    Signed       bool
}

type StringProps struct {
    Padding StringPadding
    CharSet CharacterSet
}

type StringPadding uint8
type CharacterSet uint8

type CompoundProps struct {
    Members []CompoundMember
}

type CompoundMember struct {
    Name       string
    ByteOffset uint32
    Type       *Datatype
}

type ArrayProps struct {
    Dimensions []uint32
    BaseType   *Datatype
}

type VarLenProps struct {
    BaseType *Datatype
    IsString bool
}

// Parse parses a datatype from binary
func Parse(data []byte) (*Datatype, error) {
    classAndVersion := data[0]
    class := Class(classAndVersion & 0x0F)
    
    classBits := uint32(data[1]) | uint32(data[2])<<8 | uint32(data[3])<<16
    size := binary.LittleEndian.Uint32(data[4:8])
    
    dt := &Datatype{
        Class: class,
        Size:  size,
    }
    
    props := data[8:]
    
    switch class {
    case ClassFixedPoint:
        dt.ByteOrder = ByteOrder(classBits & 0x01)
        dt.Properties = &FixedPointProps{
            Signed:       (classBits>>3)&0x01 != 0,
            BitOffset:    binary.LittleEndian.Uint16(props[0:2]),
            BitPrecision: binary.LittleEndian.Uint16(props[2:4]),
        }
        
    case ClassFloatPoint:
        dt.ByteOrder = ByteOrder(classBits & 0x01)
        
    case ClassString:
        dt.Properties = &StringProps{
            Padding: StringPadding(classBits & 0x0F),
            CharSet: CharacterSet((classBits >> 4) & 0x0F),
        }
        
    case ClassCompound:
        numMembers := int(classBits & 0xFFFF)
        cp := &CompoundProps{Members: make([]CompoundMember, numMembers)}
        // Parse members...
        dt.Properties = cp
        
    case ClassArray:
        ndims := int(props[0])
        ap := &ArrayProps{Dimensions: make([]uint32, ndims)}
        offset := 4
        for i := 0; i < ndims; i++ {
            ap.Dimensions[i] = binary.LittleEndian.Uint32(props[offset:])
            offset += 4
        }
        baseType, _ := Parse(props[offset:])
        ap.BaseType = baseType
        dt.Properties = ap
        
    case ClassVarLen:
        vp := &VarLenProps{IsString: (classBits & 0x0F) == 1}
        vp.BaseType, _ = Parse(props)
        dt.Properties = vp
    }
    
    return dt, nil
}
```

---

## 11. Storage Layouts

### 11.1 Chunked Layout

```go
// internal/layout/chunked.go

package layout

type ChunkedLayout struct {
    reader    *binary.Reader
    index     *btree.V1ChunkBTree
    chunkDims []uint64
    dtype     *dtype.Datatype
    filters   *filter.Pipeline
}

func (l *ChunkedLayout) Read() ([]byte, error) {
    // Calculate total size from dataspace
    var chunks []*btree.ChunkLocation
    l.index.Iterate(func(c *btree.ChunkLocation) error {
        chunks = append(chunks, c)
        return nil
    })
    
    // Allocate output buffer
    // For each chunk: read, decompress, copy to output
    
    for _, chunk := range chunks {
        data, err := l.readChunk(chunk)
        if err != nil {
            return nil, err
        }
        // Copy to output at correct position
        _ = data
    }
    
    return nil, nil
}

func (l *ChunkedLayout) readChunk(chunk *btree.ChunkLocation) ([]byte, error) {
    r := l.reader.At(int64(chunk.Address))
    raw, _ := r.ReadBytes(int(chunk.Size))
    
    return l.filters.Decode(raw, chunk.FilterMask)
}
```

---

## 12. Filter Pipeline

```go
// internal/filter/pipeline.go

package filter

type Filter interface {
    ID() uint16
    Decode(input []byte) ([]byte, error)
}

type Pipeline struct {
    filters []Filter
}

func NewPipeline(infos []message.FilterInfo) *Pipeline {
    p := &Pipeline{filters: make([]Filter, len(infos))}
    
    for i, info := range infos {
        switch info.ID {
        case 1:
            p.filters[i] = NewDeflateFilter()
        case 2:
            p.filters[i] = NewShuffleFilter(info.ClientData)
        case 3:
            p.filters[i] = NewFletcher32Filter()
        }
    }
    
    return p
}

func (p *Pipeline) Decode(input []byte, filterMask uint32) ([]byte, error) {
    data := input
    
    for i := len(p.filters) - 1; i >= 0; i-- {
        if filterMask&(1<<i) != 0 {
            continue // Filter not applied
        }
        
        var err error
        data, err = p.filters[i].Decode(data)
        if err != nil {
            return nil, err
        }
    }
    
    return data, nil
}

// internal/filter/deflate.go

type DeflateFilter struct{}

func (f *DeflateFilter) ID() uint16 { return 1 }

func (f *DeflateFilter) Decode(input []byte) ([]byte, error) {
    r, err := zlib.NewReader(bytes.NewReader(input))
    if err != nil {
        return nil, err
    }
    defer r.Close()
    return io.ReadAll(r)
}

// internal/filter/shuffle.go

type ShuffleFilter struct {
    elemSize int
}

func (f *ShuffleFilter) ID() uint16 { return 2 }

func (f *ShuffleFilter) Decode(input []byte) ([]byte, error) {
    numElems := len(input) / f.elemSize
    output := make([]byte, len(input))
    
    for i := 0; i < numElems; i++ {
        for j := 0; j < f.elemSize; j++ {
            output[i*f.elemSize+j] = input[j*numElems+i]
        }
    }
    
    return output, nil
}
```

---

## 13. Public API

### 13.1 File

```go
// hdf5/file.go

package hdf5

type File struct {
    path       string
    file       *os.File
    reader     *binary.Reader
    superblock *superblock.Superblock
    root       *Group
}

func Open(path string) (*File, error) {
    f, err := os.Open(path)
    if err != nil {
        return nil, err
    }
    
    sb, err := superblock.Read(f)
    if err != nil {
        f.Close()
        return nil, err
    }
    
    cfg := binary.Config{
        ByteOrder:  sb.ByteOrder,
        OffsetSize: int(sb.OffsetSize),
        LengthSize: int(sb.LengthSize),
    }
    
    hdf := &File{
        path:       path,
        file:       f,
        reader:     binary.NewReader(f, cfg),
        superblock: sb,
    }
    
    root, err := hdf.openGroup(sb.RootGroupAddress, "/")
    if err != nil {
        f.Close()
        return nil, err
    }
    hdf.root = root
    
    return hdf, nil
}

func (f *File) Close() error {
    return f.file.Close()
}

func (f *File) Root() *Group {
    return f.root
}
```

### 13.2 Group

```go
// hdf5/group.go

package hdf5

type Group struct {
    file   *File
    path   string
    header *object.Header
}

func (g *Group) Members() ([]string, error) {
    // Extract from symbol table or links
}

func (g *Group) Open(path string) (Object, error) {
    // Navigate path
}

func (g *Group) Dataset(name string) (*Dataset, error) {
    obj, err := g.Open(name)
    if err != nil {
        return nil, err
    }
    ds, ok := obj.(*Dataset)
    if !ok {
        return nil, fmt.Errorf("not a dataset")
    }
    return ds, nil
}
```

### 13.3 Dataset

```go
// hdf5/dataset.go

package hdf5

type Dataset struct {
    file      *File
    path      string
    dtype     *dtype.Datatype
    dspace    *dspace.Dataspace
    layout    layout.Layout
    pipeline  *filter.Pipeline
}

func (d *Dataset) Shape() []uint64 {
    return d.dspace.Dimensions
}

func (d *Dataset) Read(dest interface{}) error {
    raw, err := d.layout.Read()
    if err != nil {
        return err
    }
    
    return d.convert(raw, dest)
}

func (d *Dataset) ReadSlice(start, count []uint64, dest interface{}) error {
    // Implement hyperslab selection
}
```

---

## 14. Testing Strategy

### Test Categories

1. **Unit Tests** - Each component in isolation
2. **Integration Tests** - Full read paths
3. **Compatibility Tests** - Files from h5py, MATLAB, etc.
4. **Fuzz Tests** - Random/malformed inputs

### Test File Generation Script

```python
import h5py
import numpy as np

def generate_tests():
    # Integers
    with h5py.File('integers.h5', 'w') as f:
        for dtype in ['int8', 'int16', 'int32', 'int64',
                      'uint8', 'uint16', 'uint32', 'uint64']:
            f.create_dataset(dtype, data=np.array([1,2,3], dtype=dtype))
    
    # Floats
    with h5py.File('floats.h5', 'w') as f:
        f.create_dataset('float32', data=np.array([1.5, 2.5], dtype='float32'))
        f.create_dataset('float64', data=np.array([1.5, 2.5], dtype='float64'))
    
    # Chunked + compressed
    with h5py.File('chunked.h5', 'w') as f:
        data = np.random.rand(100, 100)
        f.create_dataset('gzip', data=data, chunks=(10,10), compression='gzip')
        f.create_dataset('shuffle', data=data, chunks=(10,10), shuffle=True)
```

---

## 15. Implementation Phases

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| 1. Core I/O | 1-3 | Superblock parsing, binary reader |
| 2. Object Headers | 4-6 | V1/V2 headers, core messages |
| 3. Groups | 7-9 | Symbol tables, B-trees, navigation |
| 4. Contiguous | 10-12 | Read simple datasets |
| 5. Types | 13-15 | Strings, compounds, arrays |
| 6. Chunked | 16-18 | Chunked storage, B-tree index |
| 7. Filters | 19-21 | DEFLATE, shuffle, Fletcher32 |
| 8. VLen | 22-24 | Global heap, vlen strings |
| 9. New Groups | 25-27 | Fractal heap, V2 B-trees |
| 10. Polish | 28-30 | Caching, docs, benchmarks |

---

## References

1. [HDF5 File Format Specification](https://docs.hdfgroup.org/hdf5/develop/_f_m_t3.html)
2. [jhdf - Pure Java HDF5](https://github.com/jamesmudd/jhdf)
3. [h5py Documentation](https://docs.h5py.org/)

---

*Document End*

