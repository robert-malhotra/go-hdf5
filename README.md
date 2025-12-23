# go-hdf5

A pure Go library for reading HDF5 files. No CGO or external dependencies required.

## Installation

```bash
go get github.com/robert-malhotra/go-hdf5
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    "github.com/robert-malhotra/go-hdf5/hdf5"
)

func main() {
    // Open an HDF5 file
    f, err := hdf5.Open("data.h5")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()

    // Read a dataset
    ds, err := f.OpenDataset("/measurements/temperature")
    if err != nil {
        log.Fatal(err)
    }

    // Get dataset info
    fmt.Printf("Shape: %v\n", ds.Shape())
    fmt.Printf("Elements: %d\n", ds.NumElements())

    // Read data as float64
    data, err := ds.ReadFloat64()
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Data: %v\n", data)
}
```

## Features

### Supported

- **Data types**: All integer types (int8-64, uint8-64), float32, float64, strings (fixed and variable-length)
- **Storage layouts**: Contiguous, chunked (B-tree v1 and v2), compact
- **Compression**: Gzip/deflate, shuffle filter
- **Structure**: Groups, nested groups, soft links, external links
- **Attributes**: On groups and datasets, scalar and array, compound types
- **File formats**: Superblock versions 0-3

### Not Yet Supported

- SZIP compression
- Partial reads (hyperslabs)
- Virtual datasets
- Object/region references
- Writing files

## Usage Examples

### Reading Datasets

```go
// Open a dataset by path
ds, err := f.OpenDataset("/group/subgroup/data")

// Check dataset properties
fmt.Printf("Dimensions: %v\n", ds.Shape())   // e.g., [100, 200]
fmt.Printf("Rank: %d\n", ds.Rank())          // e.g., 2
fmt.Printf("Total elements: %d\n", ds.NumElements())

// Read with type-specific methods
floats, _ := ds.ReadFloat64()
ints, _ := ds.ReadInt32()
strings, _ := ds.ReadString()

// Or read into a typed slice
var data []float64
err := ds.Read(&data)
```

### Navigating Groups

```go
// Get root group
root := f.Root()

// Open a subgroup
grp, err := root.OpenGroup("sensors")

// List all members (groups and datasets)
members, err := grp.Members()
for _, name := range members {
    fmt.Println(name)
}

// Open nested paths directly
ds, err := f.OpenDataset("/sensors/temperature/readings")
```

### Reading Attributes

```go
// On datasets
ds, _ := f.OpenDataset("/data")
attr := ds.Attr("units")
if attr != nil {
    units, _ := attr.ReadScalarString()
    fmt.Printf("Units: %s\n", units)
}

// On groups
grp, _ := f.OpenGroup("/experiment")
attr := grp.Attr("description")
value, _ := attr.Value()  // Auto-typed based on HDF5 datatype

// List all attributes
for _, name := range ds.Attrs() {
    fmt.Println(name)
}

// Read attribute by full path
val, err := f.ReadAttr("/data@units")
```

### Compound Type Attributes

```go
// Read compound attribute (e.g., a point with x, y, z fields)
attr := ds.Attr("origin")
point, err := attr.ReadScalarCompound()
// point is map[string]interface{}{"x": 1.0, "y": 2.0, "z": 3.0}

x := point["x"].(float64)
y := point["y"].(float64)
```

### Walking All Attributes

```go
// Iterate over all attributes in the file
err := f.WalkAttrs(func(info hdf5.AttrInfo) error {
    fmt.Printf("%s = %v\n", info.Path, info.Value)
    // info.Path format: "/group/dataset@attribute"
    // info.ObjectPath: "/group/dataset"
    // info.Name: "attribute"
    return nil
})
```

### Following Links

```go
// Soft links are followed automatically
ds, err := f.OpenDataset("/link_to_data")  // -> resolves to actual dataset

// External links work too (opens the external file automatically)
ds, err := f.OpenDataset("/external_link")  // -> opens external_file.h5:/path
```

### Error Handling

```go
import "errors"

ds, err := f.OpenDataset("/nonexistent")
if errors.Is(err, hdf5.ErrNotFound) {
    fmt.Println("Dataset not found")
}

// Common errors:
// - hdf5.ErrNotFound: Object doesn't exist
// - hdf5.ErrNotDataset: Tried to open a group as a dataset
// - hdf5.ErrNotGroup: Tried to open a dataset as a group
// - hdf5.ErrClosed: File was already closed
// - hdf5.ErrLinkDepth: Too many nested soft/external links (circular reference protection)
```

## API Reference

### File

| Method | Description |
|--------|-------------|
| `Open(path string) (*File, error)` | Open an HDF5 file for reading |
| `Close() error` | Close the file |
| `Root() *Group` | Get the root group |
| `OpenGroup(path string) (*Group, error)` | Open a group by absolute path |
| `OpenDataset(path string) (*Dataset, error)` | Open a dataset by absolute path |
| `GetAttr(path string) (*Attribute, error)` | Get an attribute by path (`/obj@attr`) |
| `ReadAttr(path string) (interface{}, error)` | Read an attribute value by path |
| `WalkAttrs(fn WalkAttrsFunc) error` | Walk all attributes in the file |
| `Version() int` | Get the superblock version |
| `Path() string` | Get the file path |

### Group

| Method | Description |
|--------|-------------|
| `Name() string` | Group name (last path component) |
| `Path() string` | Full path to this group |
| `OpenGroup(path string) (*Group, error)` | Open a subgroup by relative path |
| `OpenDataset(path string) (*Dataset, error)` | Open a dataset by relative path |
| `Members() ([]string, error)` | List all member names |
| `NumObjects() (int, error)` | Count of members |
| `Attrs() []string` | List attribute names |
| `Attr(name string) *Attribute` | Get an attribute by name |
| `HasAttr(name string) bool` | Check if attribute exists |

### Dataset

| Method | Description |
|--------|-------------|
| `Name() string` | Dataset name |
| `Path() string` | Full path to this dataset |
| `Shape() []uint64` | Dimensions (nil for scalar) |
| `Dims() []uint64` | Alias for Shape() |
| `Rank() int` | Number of dimensions |
| `NumElements() uint64` | Total element count |
| `IsScalar() bool` | True if scalar (single value) |
| `DtypeSize() int` | Element size in bytes |
| `Read(dest interface{}) error` | Read into typed slice |
| `ReadFloat64() ([]float64, error)` | Read as float64 |
| `ReadFloat32() ([]float32, error)` | Read as float32 |
| `ReadInt64() ([]int64, error)` | Read as int64 |
| `ReadInt32() ([]int32, error)` | Read as int32 |
| `ReadInt16() ([]int16, error)` | Read as int16 |
| `ReadInt8() ([]int8, error)` | Read as int8 |
| `ReadUint64() ([]uint64, error)` | Read as uint64 |
| `ReadUint32() ([]uint32, error)` | Read as uint32 |
| `ReadUint16() ([]uint16, error)` | Read as uint16 |
| `ReadUint8() ([]uint8, error)` | Read as uint8 |
| `ReadString() ([]string, error)` | Read as strings |
| `ReadRaw() ([]byte, error)` | Read raw bytes |
| `Attrs() []string` | List attribute names |
| `Attr(name string) *Attribute` | Get an attribute |

### Attribute

| Method | Description |
|--------|-------------|
| `Name() string` | Attribute name |
| `Shape() []uint64` | Dimensions |
| `NumElements() uint64` | Element count |
| `IsScalar() bool` | True if scalar |
| `Value() (interface{}, error)` | Auto-typed value |
| `Read(dest interface{}) error` | Read into typed variable |
| `ReadFloat64() ([]float64, error)` | Read as float64 |
| `ReadInt64() ([]int64, error)` | Read as int64 |
| `ReadString() ([]string, error)` | Read as strings |
| `ReadScalarFloat64() (float64, error)` | Read scalar float64 |
| `ReadScalarInt64() (int64, error)` | Read scalar int64 |
| `ReadScalarString() (string, error)` | Read scalar string |
| `ReadCompound() ([]map[string]interface{}, error)` | Read compound type |
| `ReadScalarCompound() (map[string]interface{}, error)` | Read scalar compound |

## Testing

```bash
# Generate test files (requires Python with h5py and numpy)
cd testdata && python3 generate.py

# Run tests
go test ./...

# Run with coverage
go test ./... -cover
```

## License

MIT
