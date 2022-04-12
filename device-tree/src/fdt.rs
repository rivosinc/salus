// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Wrapper for basic FDT interaction.

use fdt_rs::base::DevTree;
use fdt_rs::base::parse::ParsedTok;
use fdt_rs::prelude::*;
use fdt_rs::modify::modtoken::*;

// Just re-export the errors used by fdt_rs for now.
pub use fdt_rs::error::DevTreeError as Error;
pub use fdt_rs::error::Result as Result;

/// Represents a flattened device-tree (FDT) as passed to the hypervisor by firmware. Currently
/// this is a lightweight wrapper on top of `fdt_rs::base::DevTree`.
#[derive(Copy, Clone, Debug)]
pub struct Fdt<'a> {
    inner: DevTree<'a>,
}

impl<'a> Fdt<'a> {
    /// Constructs an FDT from the FDT blob located at `fdt_addr`.
    ///
    /// # Safety
    /// Must point to a 32-byte-aligned and valid FDT blob.
    pub unsafe fn new_from_raw_pointer(fdt_addr: *const u8) -> Result<Self> {
	let inner = DevTree::from_raw_pointer(fdt_addr)?;
	Ok(Self { inner })
    }

    /// Returns the total size of the FDT.
    pub fn size(&self) -> usize {
	self.inner.totalsize()
    }

    /// Returns the base physical address and size of the first range memory.
    ///
    /// TODO: Handle multiple memory ranges and reserved ranges.
    pub fn get_mem_info(&self) -> (u64, u64) {
	let mut nodes = self.inner.nodes();
	let mem_node = nodes
            .find(|n| match n.name() {
		Ok(name) => Ok(name.starts_with("memory")),
		Err(_) => Ok(false),
            })
            .expect("No memory node found!")
            .unwrap();

	let mut props = mem_node.props();
	let prop = props
            .find(|p| Ok(p.name().unwrap_or("empty") == "reg"))
            .expect("No mem reg")
            .unwrap();
	let base = prop.u64(0).unwrap();
	let size = prop.u64(0).unwrap();

	(base, size)
    }

    /// Writes out the FDT to `out_slice` with the size of memory set to `new_memory_size`.
    ///
    /// TODO: Handle more advanced transformations.
    /// TODO: Handle errors.
    pub fn write_with_updated_memory_size(&self, out_slice: &mut [u8], new_memory_size: u64) {
	let (address_cells, size_cells) = self.get_address_size();
	if address_cells != 2 || size_cells != 2 {
            panic!("#address-cells and #size-cells were not 2! They were, respectively: {:} and {:}", address_cells, size_cells);
	}

	let mut in_memory = false;
	let mut memory_depth = 0;
	let mut depth = 0;

	let mut test = | token: &mut ModifyParsedTok, _prop_size | {
            match token {
		ModifyParsedTok::BeginNode(node) => {
                    in_memory = node.name.starts_with(b"memory");

                    depth += 1;
                    if in_memory { memory_depth = depth; }
		}
		ModifyParsedTok::Prop(prop, ref mut buf) => {
                    let name = self.inner.string_at_offset(prop.name_offset).unwrap();
                    if in_memory && name[..3].as_bytes() == b"reg" {
			let bytes = u64::to_be_bytes(new_memory_size);
			let prop_buf = &mut *buf;

			// 8..16 because the size is located at offset 8 in the propbuf
			// the first 8 bytes represent the address of property "reg"
			prop_buf[8..16].clone_from_slice(&bytes);
                    }
		}
		ModifyParsedTok::EndNode => {
                    depth -= 1;
                    if memory_depth < depth { in_memory = false; }
		}
		ModifyParsedTok::Nop => {}
            }

            ModifyTokenResponse::Pass
	};

	self.inner.modify(out_slice, &mut test).unwrap();
    }

    /// Returns the top-level #address-cells/#size-cells of the FDT.
    fn get_address_size(&self) -> (u32, u32) {
	let mut nodes = self.inner.parse_iter();
	let mut address_cells = 0;
	let mut size_cells = 0;

	let mut depth = 0;

	while let Ok(Some(token)) = nodes.next() {
            match token {
		ParsedTok::BeginNode(_) => {
                    depth += 1;
		}

		ParsedTok::EndNode => {
                    depth -= 1;

                    if depth == 0 {
			return (address_cells, size_cells)
                    }
		}

		ParsedTok::Prop(prop) => {
                    if depth == 1 {
			// the following unwraps should not cause an error because prop_buf[0..4] encodes 4
			// bytes, which can be turned into a big endian u32.

			if self.inner.string_at_offset(prop.name_offset).unwrap() == "#address-cells" {
                            address_cells = u32::from_be_bytes(prop.prop_buf[0..4].try_into().unwrap());
			}

			if self.inner.string_at_offset(prop.name_offset).unwrap() == "#size-cells" {
                            size_cells = u32::from_be_bytes(prop.prop_buf[0..4].try_into().unwrap());
			}
                    }
		}

		_ => {}
            }
	}

	(0, 0)
    }
}
