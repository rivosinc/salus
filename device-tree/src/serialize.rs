// SPDX-FileCopyrightText: 2023 Rivos Inc.
//
// SPDX-License-Identifier: Apache-2.0

use crate::DeviceTree;
use core::mem;

const FDT_ALIGN: usize = 4;

/// Aligns the given size to the natrual FDT token alignemnt.
fn fdt_align_size(size: usize) -> usize {
    (size + FDT_ALIGN - 1) & !(FDT_ALIGN - 1)
}

// As specified in v0.3 of the specification.
const FDT_MAGIC: u32 = 0xd00dfeed;
const FDT_VERSION: u32 = 17;
const FDT_LAST_COMP_VERSION: u32 = 16;

/// The FDT header strucutre. Defines the version and layout of the FDT binary.
#[derive(Clone, Debug)]
#[repr(C)]
struct FdtHeader {
    magic: u32,
    totalsize: u32,
    off_dt_struct: u32,
    off_dt_strings: u32,
    off_mem_rsvmap: u32,
    version: u32,
    last_comp_version: u32,
    boot_cpuid_phys: u32,
    size_dt_strings: u32,
    size_dt_struct: u32,
}

/// A single entry in the FDT memory reservation block.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct FdtResvMap {
    address: u64,
    size: u64,
}

/// The structure immediately following an FDT_PROP token in the structure block.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct FdtProp {
    len: u32,
    nameoff: u32,
}

/// Represents a single token in the structure block of an FDT.
enum FdtToken<'a> {
    NodeStart(&'a [u8]),
    NodeEnd,
    Prop(FdtProp, &'a [u8]),
    End,
}

impl<'a> FdtToken<'a> {
    /// Creates a new FDT_BEGIN_NODE token with the node's name as a (NULL-terminated) string.
    pub fn new_node_start(name: &'a [u8]) -> Self {
        FdtToken::NodeStart(name)
    }

    /// Creates a new FDT_END_NODE token.
    pub fn new_node_end() -> Self {
        FdtToken::NodeEnd
    }

    /// Creates a new FDT_PROP token. `name_offset` is the offset of the property's name in the
    /// FDT's strings block.
    pub fn new_prop(name_offset: u32, value: &'a [u8]) -> Self {
        let prop = FdtProp {
            len: value.len().try_into().unwrap(),
            nameoff: name_offset,
        };
        FdtToken::Prop(prop, value)
    }

    /// Creates a new FDT_END token.
    pub fn new_end() -> Self {
        FdtToken::End
    }

    /// Returns the number of bytes this token will consume in the FDT structure block.
    pub fn size(&self) -> usize {
        match self {
            FdtToken::NodeStart(name) => mem::size_of::<u32>() + fdt_align_size(name.len()),
            FdtToken::NodeEnd => mem::size_of::<u32>(),
            FdtToken::Prop(_, value) => {
                mem::size_of::<u32>() + mem::size_of::<FdtProp>() + fdt_align_size(value.len())
            }
            FdtToken::End => mem::size_of::<u32>(),
        }
    }

    /// Returns the token's raw code.
    pub fn code(&self) -> u32 {
        match self {
            FdtToken::NodeStart(_) => 0x1,
            FdtToken::NodeEnd => 0x2,
            FdtToken::Prop(_, _) => 0x3,
            FdtToken::End => 0x9,
        }
    }
}

/// Describes the layout of the FDT binary.
#[derive(Clone, Debug)]
struct FdtLayout {
    header_size: usize,
    resv_map_size: usize,
    struct_size: usize,
    strings_size: usize,
}

/// A writer for an FDT binary. We use the pre-determined layout to slice up the buffer into
/// sub-sections, consuming those slices as we push data into the sub-sections.
struct FdtWriter<'a> {
    header_buf: &'a mut [u8],
    resv_map_buf: &'a mut [u8],
    struct_buf: &'a mut [u8],
    strings_buf: &'a mut [u8],
    string_pos: usize,
}

impl<'a> FdtWriter<'a> {
    /// Creates a new writer with the given buffer and FDT layout. The buffer must be large enough
    /// to hold the FDT described by the layout.
    pub fn new(mut buf: &'a mut [u8], layout: &FdtLayout) -> Self {
        let header_buf = buf.take_mut(..layout.header_size).unwrap();
        let resv_map_buf = buf.take_mut(..layout.resv_map_size).unwrap();
        let struct_buf = buf.take_mut(..layout.struct_size).unwrap();
        let strings_buf = buf.take_mut(..layout.strings_size).unwrap();
        Self {
            header_buf,
            resv_map_buf,
            struct_buf,
            strings_buf,
            string_pos: 0,
        }
    }

    /// Writes the header structure to the FDT binary.
    pub fn push_header(&mut self, header: FdtHeader) {
        let vals = unsafe {
            // Safe since the layout of FdtHeader is the same as an array of u32s.
            mem::transmute::<FdtHeader, [u32; 10]>(header)
        };
        for val in vals {
            let buf = self.header_buf.take_mut(..mem::size_of::<u32>()).unwrap();
            buf.copy_from_slice(&val.to_be_bytes());
        }
    }

    /// Writes a memory reservation map entry to the FDT binary.
    pub fn push_resv_map(&mut self, resv_map: FdtResvMap) {
        let vals = unsafe {
            // Safe since the layout of FdtResvMap is the same as an array of u64s.
            mem::transmute::<FdtResvMap, [u64; 2]>(resv_map)
        };
        for val in vals {
            let buf = self.resv_map_buf.take_mut(..mem::size_of::<u64>()).unwrap();
            buf.copy_from_slice(&val.to_be_bytes());
        }
    }

    /// Writes a token to the structure block in the FDT binary.
    pub fn push_token(&mut self, token: FdtToken) {
        self.push_struct_u32(token.code());
        match token {
            FdtToken::NodeStart(name) => {
                self.push_struct_raw(name);
            }
            FdtToken::Prop(prop, val) => {
                self.push_struct_u32(prop.len);
                self.push_struct_u32(prop.nameoff);
                self.push_struct_raw(val);
            }
            _ => {}
        };
    }

    /// Writes a (NULL-terminated) string to the strings block in the FDT binary, returning the
    /// offset of the string within the block.
    pub fn push_string(&mut self, string: &[u8]) -> u32 {
        let dest = self.strings_buf.take_mut(..string.len()).unwrap();
        dest.copy_from_slice(string);
        let offset = self.string_pos;
        self.string_pos += string.len();
        offset.try_into().unwrap()
    }

    fn push_struct_u32(&mut self, val: u32) {
        let buf = self.struct_buf.take_mut(..mem::size_of::<u32>()).unwrap();
        buf.copy_from_slice(&val.to_be_bytes());
    }

    fn push_struct_raw(&mut self, buf: &[u8]) {
        let aligned = fdt_align_size(buf.len());
        let dest = self.struct_buf.take_mut(..aligned).unwrap();
        dest[..buf.len()].copy_from_slice(buf);
    }
}

/// Helper for serializing a `DeviceTree` object to an FDT binary, as specified in v0.3 of the
/// Devicetree Specification.
pub struct DeviceTreeSerializer<'a> {
    layout: FdtLayout,
    tree: &'a DeviceTree,
}

impl<'a> DeviceTreeSerializer<'a> {
    fn get_layout(tree: &DeviceTree) -> FdtLayout {
        let mut struct_size = 0;
        let mut strings_size = 0;
        for node in tree.iter() {
            struct_size +=
                FdtToken::new_node_start(node.name_raw()).size() + FdtToken::new_node_end().size();
            for p in node.props() {
                strings_size += p.name_raw().len();
                struct_size += FdtToken::new_prop(0, p.value_raw()).size();
            }
        }
        struct_size += FdtToken::new_end().size();

        FdtLayout {
            header_size: mem::size_of::<FdtHeader>(),
            resv_map_size: mem::size_of::<FdtResvMap>(),
            struct_size,
            strings_size,
        }
    }

    /// Creates a new serializer using the given tree.
    pub fn new(tree: &'a DeviceTree) -> Self {
        Self {
            layout: Self::get_layout(tree),
            tree,
        }
    }

    /// Returns the total output buffer size required for the FDT.
    pub fn output_size(&self) -> usize {
        let total_size = self.layout.header_size
            + self.layout.resv_map_size
            + self.layout.struct_size
            + self.layout.strings_size;
        fdt_align_size(total_size)
    }

    /// Writes the device-tree to the given output buffer. The buffer must large enough to hold the
    /// FDT, or at least `output_size()` bytes in length.
    pub fn write_to(&self, buf: &mut [u8]) {
        let mut writer = FdtWriter::new(buf, &self.layout);

        let struct_offset = self.layout.header_size + self.layout.resv_map_size;
        let strings_offset = struct_offset + self.layout.struct_size;
        let header = FdtHeader {
            magic: FDT_MAGIC,
            totalsize: self.output_size().try_into().unwrap(),
            off_dt_struct: struct_offset.try_into().unwrap(),
            off_dt_strings: strings_offset.try_into().unwrap(),
            off_mem_rsvmap: self.layout.header_size.try_into().unwrap(),
            version: FDT_VERSION,
            last_comp_version: FDT_LAST_COMP_VERSION,
            boot_cpuid_phys: 0, // TODO: Need a way to specify this if it can ever be non-0.
            size_dt_strings: self.layout.strings_size.try_into().unwrap(),
            size_dt_struct: self.layout.struct_size.try_into().unwrap(),
        };
        writer.push_header(header);

        // We don't support reserve-map entries presently; push an empty one to terminate the list.
        writer.push_resv_map(FdtResvMap {
            address: 0,
            size: 0,
        });

        // Now push the tree structure.
        let mut iter = self.tree.iter();
        let mut depth = 0;
        while let Some(node) = iter.next() {
            writer.push_token(FdtToken::new_node_start(node.name_raw()));

            for p in node.props() {
                let name_offset = writer.push_string(p.name_raw());
                writer.push_token(FdtToken::new_prop(name_offset, p.value_raw()));
            }

            if iter.depth() > depth {
                assert_eq!(depth + 1, iter.depth());
                depth += 1;
            } else {
                // The next node is a sibling, or sibling of an ancestor. Emit NodeEnds until we
                // reach the appropriate depth.
                writer.push_token(FdtToken::new_node_end());
                while depth != iter.depth() {
                    writer.push_token(FdtToken::new_node_end());
                    depth -= 1;
                }
            }
        }
        writer.push_token(FdtToken::new_end());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Fdt;
    use alloc::vec;

    fn stub_tree() -> DeviceTree {
        // Create a tree with basic 'memory' and 'chosen' nodes.
        let mut tree = DeviceTree::new();
        let root = tree.add_node("", None).unwrap();
        {
            let node = tree.get_mut_node(root).unwrap();
            node.add_prop("#address-cells")
                .unwrap()
                .set_value_u32(&[2])
                .unwrap();
            node.add_prop("#size-cells")
                .unwrap()
                .set_value_u32(&[2])
                .unwrap();
        }
        {
            let id = tree.add_node("memory", Some(root)).unwrap();
            let node = tree.get_mut_node(id).unwrap();
            node.add_prop("device_type")
                .unwrap()
                .set_value_str("memory")
                .unwrap();
            node.add_prop("reg")
                .unwrap()
                .set_value_u64(&[0x8000_0000, 0x4000_0000])
                .unwrap();
        }
        {
            let id = tree.add_node("chosen", Some(root)).unwrap();
            let node = tree.get_mut_node(id).unwrap();
            node.add_prop("bootargs")
                .unwrap()
                .set_value_str("console=ttyS0")
                .unwrap();
        }

        tree
    }

    #[test]
    fn end_to_end() {
        let tree = stub_tree();
        let writer = DeviceTreeSerializer::new(&tree);
        let mut buf = vec![0; writer.output_size()];
        writer.write_to(&mut buf);

        // Now make sure our FDT library can parse it.
        let fdt = unsafe {
            // Not safe, but it's just a test.
            Fdt::new_from_raw_pointer(buf.as_ptr()).unwrap()
        };
        let mut iter = fdt.memory_regions();
        let region = iter.next().unwrap();
        assert_eq!(region.base(), 0x8000_0000);
        assert_eq!(region.size(), 0x4000_0000);
        assert!(iter.next().is_none());
        assert!(fdt.reserved_memory_regions().next().is_none());
    }
}
