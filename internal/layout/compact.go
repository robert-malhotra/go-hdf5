package layout

import (
	"github.com/robert-malhotra/go-hdf5/internal/message"
)

// Compact represents compact storage layout.
// Data is stored directly in the object header.
type Compact struct {
	data []byte
}

// NewCompact creates a new compact layout handler.
func NewCompact(layout *message.DataLayout) *Compact {
	return &Compact{
		data: layout.CompactData,
	}
}

func (c *Compact) Class() message.LayoutClass {
	return message.LayoutCompact
}

// Read returns the compact data stored in the object header.
func (c *Compact) Read() ([]byte, error) {
	// Data is already available - just return a copy
	result := make([]byte, len(c.data))
	copy(result, c.data)
	return result, nil
}

// Size returns the size of the compact data.
func (c *Compact) Size() int {
	return len(c.data)
}
