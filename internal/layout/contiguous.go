package layout

import (
	"fmt"

	"github.com/robert-malhotra/go-hdf5/internal/binary"
	"github.com/robert-malhotra/go-hdf5/internal/message"
)

// Contiguous represents contiguous storage layout.
// Data is stored in a single contiguous block in the file.
type Contiguous struct {
	address   uint64
	size      uint64
	dataspace *message.Dataspace
	datatype  *message.Datatype
	reader    *binary.Reader
}

// NewContiguous creates a new contiguous layout handler.
func NewContiguous(
	layout *message.DataLayout,
	dataspace *message.Dataspace,
	datatype *message.Datatype,
	reader *binary.Reader,
) *Contiguous {
	size := layout.Size
	if size == 0 {
		// Calculate size from dataspace and datatype
		size = calculateDataSize(dataspace, datatype)
	}

	return &Contiguous{
		address:   layout.Address,
		size:      size,
		dataspace: dataspace,
		datatype:  datatype,
		reader:    reader,
	}
}

func (c *Contiguous) Class() message.LayoutClass {
	return message.LayoutContiguous
}

// Read reads all data from contiguous storage.
func (c *Contiguous) Read() ([]byte, error) {
	// Check for undefined address (no data allocated)
	if c.reader.IsUndefinedOffset(c.address) {
		return nil, fmt.Errorf("contiguous data not allocated")
	}

	if c.size == 0 {
		return []byte{}, nil
	}

	// Read data directly from the file
	r := c.reader.At(int64(c.address))
	data, err := r.ReadBytes(int(c.size))
	if err != nil {
		return nil, fmt.Errorf("reading contiguous data: %w", err)
	}

	return data, nil
}

// Address returns the data address.
func (c *Contiguous) Address() uint64 {
	return c.address
}

// Size returns the data size in bytes.
func (c *Contiguous) Size() uint64 {
	return c.size
}
