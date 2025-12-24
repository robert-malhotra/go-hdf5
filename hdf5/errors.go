// Package hdf5 provides a pure Go implementation for reading HDF5 files.
package hdf5

import "errors"

// Common errors
var (
	ErrNotHDF5       = errors.New("not an HDF5 file")
	ErrNotFound      = errors.New("object not found")
	ErrNotDataset    = errors.New("object is not a dataset")
	ErrNotGroup      = errors.New("object is not a group")
	ErrUnsupported   = errors.New("unsupported feature")
	ErrInvalidPath   = errors.New("invalid path")
	ErrClosed        = errors.New("file is closed")
	ErrLinkDepth     = errors.New("maximum link depth exceeded")
)

// MaxLinkDepth is the maximum number of soft/external links that can be followed
// in a single path resolution. This prevents stack overflow from deeply nested links.
const MaxLinkDepth = 100
