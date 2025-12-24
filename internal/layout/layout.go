// Package layout provides storage layout handlers for reading HDF5 dataset data.
package layout

import (
	"fmt"

	"github.com/rkm/go-hdf5/internal/binary"
	"github.com/rkm/go-hdf5/internal/btree"
	"github.com/rkm/go-hdf5/internal/filter"
	"github.com/rkm/go-hdf5/internal/message"
)

// Layout is the interface for reading dataset data from various storage layouts.
type Layout interface {
	// Read reads all data from the layout.
	Read() ([]byte, error)

	// Class returns the layout class.
	Class() message.LayoutClass
}

// New creates a Layout from a DataLayout message.
func New(
	layout *message.DataLayout,
	dataspace *message.Dataspace,
	datatype *message.Datatype,
	filterPipeline *message.FilterPipeline,
	reader *binary.Reader,
) (Layout, error) {
	if layout == nil {
		return nil, fmt.Errorf("nil layout message")
	}

	switch layout.Class {
	case message.LayoutCompact:
		return NewCompact(layout), nil

	case message.LayoutContiguous:
		return NewContiguous(layout, dataspace, datatype, reader), nil

	case message.LayoutChunked:
		return NewChunked(layout, dataspace, datatype, filterPipeline, reader)

	default:
		return nil, fmt.Errorf("unsupported layout class: %d", layout.Class)
	}
}

// calculateDataSize calculates the total size of data in bytes.
func calculateDataSize(dataspace *message.Dataspace, datatype *message.Datatype) uint64 {
	if dataspace == nil || datatype == nil {
		return 0
	}
	return dataspace.NumElements() * uint64(datatype.Size)
}

// Chunked represents chunked storage layout.
type Chunked struct {
	layout    *message.DataLayout
	dataspace *message.Dataspace
	datatype  *message.Datatype
	pipeline  *filter.Pipeline
	reader    *binary.Reader
}

// NewChunked creates a new chunked layout handler.
func NewChunked(
	layout *message.DataLayout,
	dataspace *message.Dataspace,
	datatype *message.Datatype,
	filterPipeline *message.FilterPipeline,
	reader *binary.Reader,
) (*Chunked, error) {
	var pipeline *filter.Pipeline
	var err error
	if filterPipeline != nil {
		pipeline, err = filter.NewPipeline(filterPipeline)
		if err != nil {
			return nil, fmt.Errorf("creating filter pipeline: %w", err)
		}
	}

	return &Chunked{
		layout:    layout,
		dataspace: dataspace,
		datatype:  datatype,
		pipeline:  pipeline,
		reader:    reader,
	}, nil
}

func (c *Chunked) Class() message.LayoutClass {
	return message.LayoutChunked
}

func (c *Chunked) Read() ([]byte, error) {
	// Get dataset dimensions
	dims := c.dataspace.Dimensions
	if len(dims) == 0 {
		// Scalar dataset - shouldn't be chunked, but handle gracefully
		dims = []uint64{1}
	}

	// Get chunk dimensions from layout
	chunkDims := c.layout.ChunkDims
	if len(chunkDims) == 0 {
		return nil, fmt.Errorf("chunked layout has no chunk dimensions")
	}

	// The chunk dims array may have an extra dimension (element size)
	// Trim to match dataset dimensions
	if len(chunkDims) > len(dims) {
		chunkDims = chunkDims[:len(dims)]
	}

	// Calculate total data size
	elementSize := uint64(c.datatype.Size)
	totalSize := calculateDataSize(c.dataspace, c.datatype)
	if totalSize == 0 {
		return nil, nil
	}

	// Allocate output buffer
	output := make([]byte, totalSize)

	// Calculate chunk size in bytes (uncompressed)
	chunkElements := uint64(1)
	for _, d := range chunkDims {
		chunkElements *= uint64(d)
	}
	chunkSizeBytes := chunkElements * elementSize

	// Detect chunk index type by reading the signature at the index address
	indexType, err := c.detectChunkIndexType()
	if err != nil {
		return nil, fmt.Errorf("detecting chunk index type: %w", err)
	}

	switch indexType {
	case "single":
		return c.readSingleChunk(totalSize)

	case "btree_v1":
		return c.readBTreeV1Chunks(dims, chunkDims, elementSize, chunkSizeBytes, output)

	case "fixed_array":
		return c.readFixedArrayChunks(dims, chunkDims, elementSize, chunkSizeBytes, output)

	case "extensible_array":
		return c.readExtensibleArrayChunks(dims, chunkDims, elementSize, chunkSizeBytes, output)

	case "btree_v2":
		return c.readBTreeV2Chunks(dims, chunkDims, elementSize, chunkSizeBytes, output)

	default:
		return nil, fmt.Errorf("unsupported chunk index type: %s", indexType)
	}
}

// detectChunkIndexType reads the signature at ChunkIndexAddr to determine the index type.
func (c *Chunked) detectChunkIndexType() (string, error) {
	if c.layout.ChunkIndexAddr == 0 || c.layout.ChunkIndexAddr == 0xFFFFFFFFFFFFFFFF {
		return "single", nil // Assume single chunk if no valid address
	}

	nr := c.reader.At(int64(c.layout.ChunkIndexAddr))
	sig, err := nr.ReadBytes(4)
	if err != nil {
		// If we can't read, assume single chunk
		return "single", nil
	}

	sigStr := string(sig)
	switch sigStr {
	case "TREE":
		return "btree_v1", nil
	case "FARY", "FAHD":
		return "fixed_array", nil
	case "EAHD":
		return "extensible_array", nil
	case "BTHD":
		return "btree_v2", nil
	default:
		// Not a known index signature - might be raw chunk data (single chunk)
		return "single", nil
	}
}

// readSingleChunk reads a dataset stored as a single chunk.
func (c *Chunked) readSingleChunk(totalSize uint64) ([]byte, error) {
	nr := c.reader.At(int64(c.layout.ChunkIndexAddr))
	data, err := nr.ReadBytes(int(totalSize))
	if err != nil {
		return nil, fmt.Errorf("reading single chunk: %w", err)
	}

	// Apply filter pipeline if present
	if c.pipeline != nil && !c.pipeline.Empty() {
		data, err = c.pipeline.Decode(data, 0)
		if err != nil {
			return nil, fmt.Errorf("decoding single chunk: %w", err)
		}
	}

	return data, nil
}

// readImplicitChunks reads chunks stored contiguously without an explicit index.
func (c *Chunked) readImplicitChunks(dims []uint64, chunkDims []uint32, elementSize uint64) ([]byte, error) {
	totalSize := calculateDataSize(c.dataspace, c.datatype)
	output := make([]byte, totalSize)

	// Calculate number of chunks in each dimension
	ndims := len(dims)
	numChunks := make([]uint64, ndims)
	totalChunks := uint64(1)
	for d := 0; d < ndims; d++ {
		numChunks[d] = (dims[d] + uint64(chunkDims[d]) - 1) / uint64(chunkDims[d])
		totalChunks *= numChunks[d]
	}

	// Calculate uncompressed chunk size
	chunkBytes := elementSize
	for _, cd := range chunkDims {
		chunkBytes *= uint64(cd)
	}

	// Read chunks in row-major order
	nr := c.reader.At(int64(c.layout.ChunkIndexAddr))
	chunkOffset := make([]uint64, ndims)

	for chunkIdx := uint64(0); chunkIdx < totalChunks; chunkIdx++ {
		// Calculate chunk coordinates
		remaining := chunkIdx
		for d := ndims - 1; d >= 0; d-- {
			chunkOffset[d] = (remaining % numChunks[d]) * uint64(chunkDims[d])
			remaining /= numChunks[d]
		}

		// Read chunk data
		chunkData, err := nr.ReadBytes(int(chunkBytes))
		if err != nil {
			return nil, fmt.Errorf("reading implicit chunk %d: %w", chunkIdx, err)
		}

		// Apply filter pipeline if present
		if c.pipeline != nil && !c.pipeline.Empty() {
			chunkData, err = c.pipeline.Decode(chunkData, 0)
			if err != nil {
				return nil, fmt.Errorf("decoding implicit chunk %d: %w", chunkIdx, err)
			}
		}

		// Copy to output
		err = c.copyChunkToOutput(output, chunkData, chunkOffset, dims, chunkDims, elementSize, chunkBytes)
		if err != nil {
			return nil, fmt.Errorf("copying implicit chunk %d: %w", chunkIdx, err)
		}
	}

	return output, nil
}

// readFixedArrayChunks reads chunks indexed by a fixed array.
func (c *Chunked) readFixedArrayChunks(dims []uint64, chunkDims []uint32, elementSize, chunkSizeBytes uint64, output []byte) ([]byte, error) {
	// Read fixed array header
	entries, err := c.readFixedArrayIndex(dims, chunkDims)
	if err != nil {
		return nil, fmt.Errorf("reading fixed array index: %w", err)
	}

	// Process each chunk
	for _, entry := range entries {
		if entry.Address == 0 || entry.Address == 0xFFFFFFFFFFFFFFFF {
			continue // Skip empty/undefined chunks
		}

		// Read chunk data
		chunkData, err := c.readChunkData(entry)
		if err != nil {
			return nil, fmt.Errorf("reading chunk at offset %v: %w", entry.Offset, err)
		}

		// Apply filter pipeline
		if c.pipeline != nil && !c.pipeline.Empty() {
			chunkData, err = c.pipeline.Decode(chunkData, entry.FilterMask)
			if err != nil {
				return nil, fmt.Errorf("decoding chunk at offset %v: %w", entry.Offset, err)
			}
		}

		// Copy to output
		err = c.copyChunkToOutput(output, chunkData, entry.Offset, dims, chunkDims, elementSize, chunkSizeBytes)
		if err != nil {
			return nil, fmt.Errorf("copying chunk at offset %v: %w", entry.Offset, err)
		}
	}

	return output, nil
}

// readExtensibleArrayChunks reads chunks indexed by an extensible array.
func (c *Chunked) readExtensibleArrayChunks(dims []uint64, chunkDims []uint32, elementSize, chunkSizeBytes uint64, output []byte) ([]byte, error) {
	// Read extensible array header
	entries, err := c.readExtensibleArrayIndex(dims, chunkDims)
	if err != nil {
		return nil, fmt.Errorf("reading extensible array index: %w", err)
	}

	// Process each chunk
	for _, entry := range entries {
		if entry.Address == 0 || entry.Address == 0xFFFFFFFFFFFFFFFF {
			continue // Skip empty/undefined chunks
		}

		// Read chunk data
		chunkData, err := c.readChunkData(entry)
		if err != nil {
			return nil, fmt.Errorf("reading chunk at offset %v: %w", entry.Offset, err)
		}

		// Apply filter pipeline
		if c.pipeline != nil && !c.pipeline.Empty() {
			chunkData, err = c.pipeline.Decode(chunkData, entry.FilterMask)
			if err != nil {
				return nil, fmt.Errorf("decoding chunk at offset %v: %w", entry.Offset, err)
			}
		}

		// Copy to output
		err = c.copyChunkToOutput(output, chunkData, entry.Offset, dims, chunkDims, elementSize, chunkSizeBytes)
		if err != nil {
			return nil, fmt.Errorf("copying chunk at offset %v: %w", entry.Offset, err)
		}
	}

	return output, nil
}

// readBTreeV1Chunks reads chunks indexed by a v1 B-tree.
func (c *Chunked) readBTreeV1Chunks(dims []uint64, chunkDims []uint32, elementSize, chunkSizeBytes uint64, output []byte) ([]byte, error) {
	ndims := len(dims)
	chunkIndex, err := btree.ReadChunkIndex(c.reader, c.layout.ChunkIndexAddr, ndims)
	if err != nil {
		return nil, fmt.Errorf("reading chunk index: %w", err)
	}

	// Process each chunk
	for _, entry := range chunkIndex.Entries {
		// Read raw chunk data from disk
		chunkData, err := c.readChunkData(entry)
		if err != nil {
			return nil, fmt.Errorf("reading chunk at offset %v: %w", entry.Offset, err)
		}

		// Apply filter pipeline (decompress)
		if c.pipeline != nil && !c.pipeline.Empty() {
			chunkData, err = c.pipeline.Decode(chunkData, entry.FilterMask)
			if err != nil {
				return nil, fmt.Errorf("decoding chunk at offset %v: %w", entry.Offset, err)
			}
		}

		// Copy chunk data to the correct position in output buffer
		err = c.copyChunkToOutput(output, chunkData, entry.Offset, dims, chunkDims, elementSize, chunkSizeBytes)
		if err != nil {
			return nil, fmt.Errorf("copying chunk at offset %v: %w", entry.Offset, err)
		}
	}

	return output, nil
}

// readBTreeV2Chunks reads chunks indexed by a v2 B-tree.
func (c *Chunked) readBTreeV2Chunks(dims []uint64, chunkDims []uint32, elementSize, chunkSizeBytes uint64, output []byte) ([]byte, error) {
	ndims := len(dims)
	chunkIndex, err := btree.ReadChunkIndexV2(c.reader, c.layout.ChunkIndexAddr, ndims)
	if err != nil {
		return nil, fmt.Errorf("reading B-tree v2 chunk index: %w", err)
	}

	// Process each chunk
	for _, entry := range chunkIndex.Entries {
		// For B-tree v2 type 10 (no filter), Size may be 0 - calculate from chunk dims
		chunkEntry := entry
		if chunkEntry.Size == 0 {
			chunkEntry.Size = uint32(chunkSizeBytes)
		}

		// Read raw chunk data from disk
		chunkData, err := c.readChunkData(chunkEntry)
		if err != nil {
			return nil, fmt.Errorf("reading chunk at offset %v: %w", entry.Offset, err)
		}

		// Apply filter pipeline (decompress)
		if c.pipeline != nil && !c.pipeline.Empty() {
			chunkData, err = c.pipeline.Decode(chunkData, entry.FilterMask)
			if err != nil {
				return nil, fmt.Errorf("decoding chunk at offset %v: %w", entry.Offset, err)
			}
		}

		// Copy chunk data to the correct position in output buffer
		err = c.copyChunkToOutput(output, chunkData, entry.Offset, dims, chunkDims, elementSize, chunkSizeBytes)
		if err != nil {
			return nil, fmt.Errorf("copying chunk at offset %v: %w", entry.Offset, err)
		}
	}

	return output, nil
}

// readChunkData reads the raw (possibly compressed) chunk data from disk.
func (c *Chunked) readChunkData(entry btree.ChunkEntry) ([]byte, error) {
	if entry.Address == 0 || entry.Address == 0xFFFFFFFFFFFFFFFF {
		return nil, fmt.Errorf("invalid chunk address")
	}

	nr := c.reader.At(int64(entry.Address))
	return nr.ReadBytes(int(entry.Size))
}

// copyChunkToOutput copies decompressed chunk data to the correct position in the output buffer.
func (c *Chunked) copyChunkToOutput(
	output []byte,
	chunkData []byte,
	chunkOffset []uint64,
	dims []uint64,
	chunkDims []uint32,
	elementSize uint64,
	chunkSizeBytes uint64,
) error {
	ndims := len(dims)

	// Handle simple 1D case
	if ndims == 1 {
		startIdx := chunkOffset[0] * elementSize
		copyLen := uint64(len(chunkData))
		if startIdx+copyLen > uint64(len(output)) {
			copyLen = uint64(len(output)) - startIdx
		}
		if copyLen > 0 {
			copy(output[startIdx:startIdx+copyLen], chunkData[:copyLen])
		}
		return nil
	}

	// Handle multi-dimensional case
	// We need to copy row by row (or the innermost dimension)
	return c.copyChunkMultiDim(output, chunkData, chunkOffset, dims, chunkDims, elementSize)
}

// copyChunkMultiDim handles copying multi-dimensional chunk data.
func (c *Chunked) copyChunkMultiDim(
	output []byte,
	chunkData []byte,
	chunkOffset []uint64,
	dims []uint64,
	chunkDims []uint32,
	elementSize uint64,
) error {
	ndims := len(dims)

	// Calculate actual chunk dimensions (may be clipped at edges)
	actualChunkDims := make([]uint64, ndims)
	for d := 0; d < ndims; d++ {
		actualChunkDims[d] = uint64(chunkDims[d])
		// Clip to dataset boundary
		if chunkOffset[d]+actualChunkDims[d] > dims[d] {
			actualChunkDims[d] = dims[d] - chunkOffset[d]
		}
	}

	// Calculate strides for output array (row-major order)
	outputStrides := make([]uint64, ndims)
	outputStrides[ndims-1] = elementSize
	for d := ndims - 2; d >= 0; d-- {
		outputStrides[d] = outputStrides[d+1] * dims[d+1]
	}

	// Calculate strides for chunk (row-major order)
	chunkStrides := make([]uint64, ndims)
	chunkStrides[ndims-1] = elementSize
	for d := ndims - 2; d >= 0; d-- {
		chunkStrides[d] = chunkStrides[d+1] * uint64(chunkDims[d+1])
	}

	// Copy data element by element or row by row
	// For efficiency, copy the innermost dimension as a contiguous block
	return c.copyChunkRecursive(
		output, chunkData,
		chunkOffset, dims, actualChunkDims,
		outputStrides, chunkStrides,
		0, 0, 0, ndims,
	)
}

// copyChunkRecursive recursively copies chunk data for each dimension.
func (c *Chunked) copyChunkRecursive(
	output []byte,
	chunkData []byte,
	chunkOffset []uint64,
	dims []uint64,
	actualChunkDims []uint64,
	outputStrides []uint64,
	chunkStrides []uint64,
	outputIdx uint64,
	chunkIdx uint64,
	dim int,
	ndims int,
) error {
	if dim == ndims-1 {
		// Innermost dimension - copy contiguously
		rowBytes := actualChunkDims[dim] * outputStrides[dim]
		// Add the chunk offset for the innermost dimension
		startIdx := outputIdx + chunkOffset[dim]*outputStrides[dim]
		if startIdx+rowBytes <= uint64(len(output)) && chunkIdx+rowBytes <= uint64(len(chunkData)) {
			copy(output[startIdx:startIdx+rowBytes], chunkData[chunkIdx:chunkIdx+rowBytes])
		}
		return nil
	}

	// Recurse for each position in this dimension
	for i := uint64(0); i < actualChunkDims[dim]; i++ {
		newOutputIdx := outputIdx + (chunkOffset[dim]+i)*outputStrides[dim]
		newChunkIdx := chunkIdx + i*chunkStrides[dim]

		err := c.copyChunkRecursive(
			output, chunkData,
			chunkOffset, dims, actualChunkDims,
			outputStrides, chunkStrides,
			newOutputIdx, newChunkIdx,
			dim+1, ndims,
		)
		if err != nil {
			return err
		}
	}

	return nil
}

// readFixedArrayIndex reads chunk entries from a fixed array index.
func (c *Chunked) readFixedArrayIndex(dims []uint64, chunkDims []uint32) ([]btree.ChunkEntry, error) {
	nr := c.reader.At(int64(c.layout.ChunkIndexAddr))

	// Read fixed array header signature
	sig, err := nr.ReadBytes(4)
	if err != nil {
		return nil, fmt.Errorf("reading fixed array signature: %w", err)
	}
	if string(sig) != "FAHD" {
		return nil, fmt.Errorf("invalid fixed array signature: got %q, expected \"FAHD\"", string(sig))
	}

	// Version (1 byte)
	version, err := nr.ReadUint8()
	if err != nil {
		return nil, err
	}
	if version != 0 {
		return nil, fmt.Errorf("unsupported fixed array version: %d", version)
	}

	// Client ID (1 byte) - 0 for chunked data
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Entry size (1 byte)
	entrySize, err := nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Page bits (1 byte)
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Number of entries (length-sized)
	numEntries, err := nr.ReadLength()
	if err != nil {
		return nil, err
	}

	// Data block address (offset-sized)
	dataBlockAddr, err := nr.ReadOffset()
	if err != nil {
		return nil, err
	}

	// Now read the data block
	return c.readFixedArrayDataBlock(dataBlockAddr, int(numEntries), int(entrySize), dims, chunkDims)
}

// readFixedArrayDataBlock reads chunk entries from a fixed array data block.
func (c *Chunked) readFixedArrayDataBlock(addr uint64, numEntries, entrySize int, dims []uint64, chunkDims []uint32) ([]btree.ChunkEntry, error) {
	nr := c.reader.At(int64(addr))

	// Read data block signature
	sig, err := nr.ReadBytes(4)
	if err != nil {
		return nil, fmt.Errorf("reading fixed array data block signature: %w", err)
	}
	if string(sig) != "FADB" {
		return nil, fmt.Errorf("invalid fixed array data block signature: got %q, expected \"FADB\"", string(sig))
	}

	// Version (1 byte)
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Client ID (1 byte)
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Header address (offset-sized)
	_, err = nr.ReadOffset()
	if err != nil {
		return nil, err
	}

	// Page bitmap (optional, not always present for small arrays)
	// For now, assume no page bitmap and read entries directly

	ndims := len(dims)
	numChunksPerDim := make([]uint64, ndims)
	for d := 0; d < ndims; d++ {
		numChunksPerDim[d] = (dims[d] + uint64(chunkDims[d]) - 1) / uint64(chunkDims[d])
	}

	var entries []btree.ChunkEntry

	for i := 0; i < numEntries; i++ {
		// Calculate chunk offset from linear index
		offset := make([]uint64, ndims)
		remaining := uint64(i)
		for d := ndims - 1; d >= 0; d-- {
			offset[d] = (remaining % numChunksPerDim[d]) * uint64(chunkDims[d])
			remaining /= numChunksPerDim[d]
		}

		// Read entry based on entry size
		// Entry format depends on whether filters are used
		var chunkAddr uint64
		var chunkSize uint32
		var filterMask uint32

		if entrySize <= 8 {
			// Just address (no filters)
			chunkAddr, err = nr.ReadOffset()
			if err != nil {
				return nil, fmt.Errorf("reading chunk address: %w", err)
			}
			// Use full chunk size
			chunkSize = 1
			for _, cd := range chunkDims {
				chunkSize *= cd
			}
			chunkSize *= uint32(c.datatype.Size)
		} else {
			// Address + filter info
			// Entry format for filtered chunks: addr (offsetSize) + size (variable) + mask (4 bytes)
			chunkAddr, err = nr.ReadOffset()
			if err != nil {
				return nil, fmt.Errorf("reading chunk address: %w", err)
			}

			// Size bytes = entrySize - offsetSize - 4 (mask is always 4 bytes)
			sizeBytes := entrySize - c.reader.OffsetSize() - 4
			if sizeBytes > 0 {
				sizeData, err := nr.ReadBytes(sizeBytes)
				if err != nil {
					return nil, fmt.Errorf("reading chunk size: %w", err)
				}
				// Read as little-endian variable-length integer
				for j := 0; j < sizeBytes; j++ {
					chunkSize |= uint32(sizeData[j]) << (8 * j)
				}
			}

			// Filter mask is always 4 bytes
			filterMask, err = nr.ReadUint32()
			if err != nil {
				return nil, fmt.Errorf("reading filter mask: %w", err)
			}
		}

		if chunkAddr != 0 && chunkAddr != 0xFFFFFFFFFFFFFFFF {
			entries = append(entries, btree.ChunkEntry{
				Offset:     offset,
				FilterMask: filterMask,
				Size:       chunkSize,
				Address:    chunkAddr,
			})
		}
	}

	return entries, nil
}

// readExtensibleArrayIndex reads chunk entries from an extensible array index.
func (c *Chunked) readExtensibleArrayIndex(dims []uint64, chunkDims []uint32) ([]btree.ChunkEntry, error) {
	nr := c.reader.At(int64(c.layout.ChunkIndexAddr))

	// Read extensible array header signature
	sig, err := nr.ReadBytes(4)
	if err != nil {
		return nil, fmt.Errorf("reading extensible array signature: %w", err)
	}
	if string(sig) != "EAHD" {
		return nil, fmt.Errorf("invalid extensible array signature: got %q, expected \"EAHD\"", string(sig))
	}

	// Version (1 byte)
	version, err := nr.ReadUint8()
	if err != nil {
		return nil, err
	}
	if version != 0 {
		return nil, fmt.Errorf("unsupported extensible array version: %d", version)
	}

	// Client ID (1 byte)
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Element size (1 byte)
	elemSize, err := nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Max number of elements bits (1 byte)
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Index block element count bits (1 byte)
	idxBlkElmts, err := nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Data block min element count bits (1 byte)
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Super block min element count bits (1 byte)
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Data block page max element count bits (1 byte)
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Number of secondary blocks (length-sized)
	_, err = nr.ReadLength()
	if err != nil {
		return nil, err
	}

	// Secondary block size (length-sized)
	_, err = nr.ReadLength()
	if err != nil {
		return nil, err
	}

	// Number of data blocks (length-sized)
	_, err = nr.ReadLength()
	if err != nil {
		return nil, err
	}

	// Data block size (length-sized)
	_, err = nr.ReadLength()
	if err != nil {
		return nil, err
	}

	// Max index set (length-sized)
	maxIdx, err := nr.ReadLength()
	if err != nil {
		return nil, err
	}

	// Number of elements (length-sized)
	numElements, err := nr.ReadLength()
	if err != nil {
		return nil, err
	}

	// Index block address (offset-sized)
	idxBlockAddr, err := nr.ReadOffset()
	if err != nil {
		return nil, err
	}

	// Read from index block
	return c.readExtensibleArrayIndexBlock(idxBlockAddr, int(idxBlkElmts), int(elemSize), int(numElements), int(maxIdx), dims, chunkDims)
}

// readExtensibleArrayIndexBlock reads the index block of an extensible array.
func (c *Chunked) readExtensibleArrayIndexBlock(addr uint64, idxBlkElmts, elemSize, numElements, maxIdx int, dims []uint64, chunkDims []uint32) ([]btree.ChunkEntry, error) {
	nr := c.reader.At(int64(addr))

	// Read index block signature
	sig, err := nr.ReadBytes(4)
	if err != nil {
		return nil, fmt.Errorf("reading extensible array index block signature: %w", err)
	}
	if string(sig) != "EAIB" {
		return nil, fmt.Errorf("invalid extensible array index block signature: got %q, expected \"EAIB\"", string(sig))
	}

	// Version (1 byte)
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Client ID (1 byte)
	_, err = nr.ReadUint8()
	if err != nil {
		return nil, err
	}

	// Header address (offset-sized)
	_, err = nr.ReadOffset()
	if err != nil {
		return nil, err
	}

	ndims := len(dims)
	numChunksPerDim := make([]uint64, ndims)
	for d := 0; d < ndims; d++ {
		numChunksPerDim[d] = (dims[d] + uint64(chunkDims[d]) - 1) / uint64(chunkDims[d])
	}

	var entries []btree.ChunkEntry

	// Read elements directly stored in index block
	numIdxElmts := 1 << idxBlkElmts
	if numIdxElmts > numElements {
		numIdxElmts = numElements
	}

	for i := 0; i < numIdxElmts; i++ {
		// Calculate chunk offset from linear index
		offset := make([]uint64, ndims)
		remaining := uint64(i)
		for d := ndims - 1; d >= 0; d-- {
			offset[d] = (remaining % numChunksPerDim[d]) * uint64(chunkDims[d])
			remaining /= numChunksPerDim[d]
		}

		// Read element
		var chunkAddr uint64
		var chunkSize uint32
		var filterMask uint32

		if elemSize <= 8 {
			chunkAddr, err = nr.ReadOffset()
			if err != nil {
				return nil, err
			}
			chunkSize = 1
			for _, cd := range chunkDims {
				chunkSize *= cd
			}
			chunkSize *= uint32(c.datatype.Size)
		} else {
			chunkAddr, err = nr.ReadOffset()
			if err != nil {
				return nil, err
			}
			remaining := elemSize - c.reader.OffsetSize()
			if remaining >= 4 {
				chunkSize, err = nr.ReadUint32()
				if err != nil {
					return nil, err
				}
				remaining -= 4
			}
			if remaining >= 4 {
				filterMask, err = nr.ReadUint32()
				if err != nil {
					return nil, err
				}
			}
		}

		if chunkAddr != 0 && chunkAddr != 0xFFFFFFFFFFFFFFFF {
			entries = append(entries, btree.ChunkEntry{
				Offset:     offset,
				FilterMask: filterMask,
				Size:       chunkSize,
				Address:    chunkAddr,
			})
		}
	}

	// Check if there are more elements in data blocks
	if numElements > numIdxElmts {
		return nil, fmt.Errorf("extensible array has %d elements but only %d fit in index block; data block reading not yet implemented", numElements, numIdxElmts)
	}

	return entries, nil
}
