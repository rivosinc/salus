// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::ArrayString;
use core::{alloc::Allocator, fmt};
use device_tree::{DeviceTree, DeviceTreeResult};

/// A builder for the host VM's device-tree. Starting with the hypervisor's device-tree, makes the
/// necessary modifications to create a device-tree that reflects the hardware available to the
/// host VM.
pub struct HostDtBuilder<A: Allocator + Clone> {
    tree: DeviceTree<A>,
}

impl<A: Allocator + Clone> HostDtBuilder<A> {
    /// Creates a new builder from the hypervisor device-tree.
    pub fn new(hyp_dt: &DeviceTree<A>) -> DeviceTreeResult<Self> {
        let mut host_dt = DeviceTree::new(hyp_dt.alloc());
        let hyp_root = hyp_dt.get_node(hyp_dt.root().unwrap()).unwrap();
        let host_root_id = host_dt.add_node("", None)?;
        let host_root = host_dt.get_mut_node(host_root_id).unwrap();

        // Clone the properties of the root node as-is.
        host_root.set_props(hyp_root.props().cloned())?;

        // Selectively clone the sub-nodes.
        for &c in hyp_root.children() {
            let hyp_child = hyp_dt.get_node(c).unwrap();
            if hyp_child.name().starts_with("chosen") || hyp_child.name() == "soc" {
                // For "soc" and "chosen" just clone the properties. We'll add sub-nodes (e.g. for
                // devices we're passing through) later if necessary.
                let host_child_id = host_dt.add_node(hyp_child.name(), Some(host_root_id))?;
                let host_child = host_dt.get_mut_node(host_child_id).unwrap();
                if hyp_child.name().starts_with("chosen") {
                    if let Some(p) = hyp_child.props().find(|p| p.name().starts_with("bootargs")) {
                        host_child.insert_prop(p.clone())?;
                    }
                } else {
                    host_child.set_props(hyp_child.props().cloned())?;
                }
            } else if hyp_child.name() == "cpus" {
                // Clone the CPU node hierarchy as-is.
                let mut iter = hyp_dt.iter_from(c).unwrap();
                let mut pid = host_root_id;
                let mut depth = 0;
                while let Some(src_node) = iter.next() {
                    let dst_id = host_dt.add_node(src_node.name(), Some(pid))?;
                    let dst_node = host_dt.get_mut_node(dst_id).unwrap();
                    dst_node.set_props(src_node.props().cloned())?;

                    // Determine the parent of the next node. We should either be descending by one
                    // level, or ascending to the level of an ancestor.
                    if iter.depth() > depth {
                        assert_eq!(depth + 1, iter.depth());
                        pid = dst_id;
                        depth += 1;
                    } else {
                        while depth != iter.depth() {
                            let parent = host_dt.get_node(pid).unwrap();
                            pid = parent.parent().unwrap();
                            depth -= 1;
                        }
                    }
                }
            }
        }

        Ok(Self { tree: host_dt })
    }

    pub fn add_memory_node(mut self, mem_base: u64, mem_size: u64) -> DeviceTreeResult<Self> {
        let mut mem_name = ArrayString::<32>::new();
        fmt::write(&mut mem_name, format_args!("memory@{:08x}", mem_base)).unwrap();
        let mem_id = self.tree.add_node(mem_name.as_str(), self.tree.root())?;
        let mem_node = self.tree.get_mut_node(mem_id).unwrap();
        mem_node.add_prop("device_type")?.set_value_str("memory")?;
        // TODO: Assumes #address-cells/#size-cells of 2.
        mem_node
            .add_prop("reg")?
            .set_value_u64(&[mem_base, mem_size])?;

        Ok(self)
    }

    pub fn set_initramfs_addr(mut self, start_addr: u64, len: u64) -> DeviceTreeResult<Self> {
        let chosen_id = self
            .tree
            .iter()
            .find(|n| n.name().starts_with("chosen"))
            .unwrap()
            .id();
        let chosen_node = self.tree.get_mut_node(chosen_id).unwrap();

        chosen_node
            .add_prop("linux,initrd-start")?
            .set_value_u64(&[start_addr])?;
        let end_addr = start_addr.checked_add(len).unwrap();
        chosen_node
            .add_prop("linux,initrd-end")?
            .set_value_u64(&[end_addr])?;

        Ok(self)
    }

    pub fn tree(self) -> DeviceTree<A> {
        self.tree
    }
}
