// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

//! Wrapper for basic FDT interaction.
#![no_std]

use core::slice;

use fdt_rs::base::*;
use fdt_rs::prelude::*;

/// update the fdt to reflect the RAM available to the host
pub fn set_fdt_host_ram_size(source_fdt: &mut [u8], host_ram_size: u64) {
    // the total length of the fdt blob is the second 32bit word in the header.
    // Safe because source_fdt length is check to be long enough to read the overall length.
    assert!(source_fdt.len() >= 16); // TODO don't assert.
    let devtree = unsafe { DevTree::new(source_fdt).unwrap() };

    update_dt(&devtree, host_ram_size)
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
fn update_dt(dt: &DevTree, host_ram_size: u64) {
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
            unsafe {
                // probably not actually safe.
                let prop_slice_mut =
                    core::slice::from_raw_parts_mut(prop_slice.as_ptr() as *mut u8, 8);
                prop_slice_mut.clone_from_slice(&host_ram_size.to_be_bytes());
            }
            break;
        }
    }
}
