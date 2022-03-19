// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Wrapper for basic FDT interaction.
#![no_std]

use core::slice;

use fdt_rs::base::*;
use fdt_rs::base::parse::ParsedTok;
use fdt_rs::prelude::*;
use fdt_rs::modify::modtoken::*;

/// update the fdt to reflect the RAM available to the host
pub fn set_fdt_host_ram_size(hw_dt_slice: &[u8], host_slice: &mut [u8], host_ram_size: u64) {
    // the total length of the fdt blob is the second 32bit word in the header.
    // Safe because source_fdt length is check to be long enough to read the overall length.
    assert!(hw_dt_slice.len() >= 16); // TODO don't assert.
    let devtree = unsafe { DevTree::new(hw_dt_slice).unwrap() };

    update_dt(&devtree, host_ram_size, host_slice)
}

/// # Safety
/// must be called with a fdt_base that points to a valid device tree header.
pub unsafe fn get_dt_len(fdt_base: u64) -> usize {
    let temp_slice = slice::from_raw_parts(fdt_base as *const u8, 16);
    DevTree::read_totalsize(temp_slice).unwrap()
}

/// # Safety
/// must be called with a fdt_base that points to a valid device tree header.
pub unsafe fn get_mem_info(fdt_base: u64) -> (u64, u64) {
    let len = get_dt_len(fdt_base);
    let dt_slice = slice::from_raw_parts(fdt_base as *const u8, len);
    let dt = DevTree::new(dt_slice).unwrap();
    let mut nodes = dt.nodes();
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
    let size = prop.u64(1).unwrap();

    (base, size)
}

/// Get the size of RAM passed in the device tree.
pub fn get_ram_size(fdt: &[u8]) -> u64 {
    // the total length of the fdt blob is the second 32bit word in the header.
    // Safe because source_fdt length is check to be long enough to read the overall length.
    assert!(fdt.len() >= 16); // TODO don't assert.
    let dt = unsafe { DevTree::new(fdt).unwrap() };
    // Get the compatible node iterator
    let mut nodes = dt.nodes();

    // Get the memory nodes
    let mem_node = nodes
        .find(|n| match n.name() {
            Ok(name) => Ok(name.starts_with("memory")),
            Err(_) => Ok(false),
        })
        .expect("No memory node found!")
        .unwrap();

    let mut props = mem_node.props();

    while let Ok(Some(prop)) = props.next() {
        if prop.name().unwrap_or("noname") == "reg" {
            // wild ass hack here to tell the host about the amount of memory it's allotted.
            // TODO - Add a reasonable way to update a device tree.
            let prop_slice = &prop.raw()[8..];
            let ram_size_bytes: [u8; 8] = prop_slice.try_into().unwrap();
            let ram_size = u64::from_be_bytes(ram_size_bytes);
            return ram_size;
        }
    }
    
    0
}

// Could be inlined with `make_fdt`, but better to limit the unsafe block size.
/// updates the given DeviceTree and write it to `write_addr`.
// TODO - specify transformations needed.
// TODO - return result, fail if output slice isn't long enough.
fn update_dt(dt: &DevTree, host_ram_size: u64, host_slice: &mut [u8]) {
    let (address_cells, size_cells) = get_address_size(*dt);
    if address_cells != 2 || size_cells != 2 {
        panic!("#address-cells and #size-cells were not 2! They were, respectively: {:} and {:}", address_cells, size_cells);
    }

    let mut in_memory = false;
    let mut memory_depth = 0;
    let mut depth= 0;

    let mut test = | token: &mut ModifyParsedTok, _prop_size | {
        match token {
            ModifyParsedTok::BeginNode(inner) => {
                in_memory = inner.name.starts_with(b"memory");

                depth += 1;
                if in_memory { memory_depth = depth; }
            }
            ModifyParsedTok::Prop(inner, ref mut buf) => {
                let name = dt.string_at_offset(inner.name_offset).unwrap();
                if in_memory && name[..3].as_bytes() == b"reg" {
                    let bytes = u64::to_be_bytes(host_ram_size);
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

    dt.modify(host_slice, &mut test).unwrap();
}

fn get_address_size(dt: DevTree) -> (u32, u32) {
    let mut nodes = dt.parse_iter();
    let mut address_cells = 0;
    let mut size_cells = 0;

    let mut depth = 0;

    while let Ok(Some(token)) = nodes.next() {
        match token {
            ParsedTok::BeginNode(_inner) => {
                depth += 1;
            }

            ParsedTok::EndNode => {
                depth -= 1;

                if depth == 0 {
                    return (address_cells, size_cells)
                }
            }

            ParsedTok::Prop(inner) => {
                if depth == 1 {
                    // the following unwraps should not cause an error because prop_buf[0..4] encodes 4
                    // bytes, which can be turned into a big endian u32.

                    if dt.string_at_offset(inner.name_offset).unwrap() == "#address-cells" {
                        address_cells = u32::from_be_bytes(inner.prop_buf[0..4].try_into().unwrap());
                    }

                    if dt.string_at_offset(inner.name_offset).unwrap() == "#size-cells" {
                        size_cells = u32::from_be_bytes(inner.prop_buf[0..4].try_into().unwrap());
                    }
                }
            }

            _ => {}
        }
    }

    (0, 0)
}
