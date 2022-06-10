// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use alloc::vec::Vec;
use core::{alloc::Allocator, fmt, mem, result, str};
use fdt_rs::base::parse::ParsedTok;
use fdt_rs::prelude::*;
use hyp_alloc::{Arena, ArenaId};

use crate::{DeviceTreeError, DeviceTreeResult, Fdt};

/// Copies a string from `src` to `dest`, adding null termination if necessary.
///
/// TODO: Consider using `cstr_core` or a similar crate for dealing with C-style strings.
fn copy_string_with_null_termination<A: Allocator>(
    dest: &mut Vec<u8, A>,
    src: &str,
) -> DeviceTreeResult<()> {
    let has_null = src.ends_with('\0');
    dest.truncate(0);
    dest.try_reserve(src.len() + if has_null { 0 } else { 1 })?;
    dest.splice(0.., src.bytes());
    if !has_null {
        dest.push(b'\0');
    }
    Ok(())
}

/// Represents an individual property of a device or bus in the tree. Properties are (name, value)
/// pairs where the name is a NULL-terminated string and the value is an array of 0 or more bytes.
/// How the value is interpreted is determined based on the property name and surrounding context,
/// but is typically either a string or an array of u32s/u64s.
pub struct DeviceTreeProp<A: Allocator + Clone> {
    name: Vec<u8, A>,
    buf: Vec<u8, A>,
}

/// Represents a bus or device in the tree. Nodes in the tree must have a parent unless they are the
/// root node and may have any number of child nodes or properties. Node names are NULL-termianted
/// strings.
pub struct DeviceTreeNode<A: Allocator + Clone> {
    alloc: A,
    id: Option<NodeId<A>>,
    name: Vec<u8, A>,
    parent: Option<NodeId<A>>,
    children: Vec<NodeId<A>, A>,
    props: Vec<DeviceTreeProp<A>, A>,
}

pub type NodeArena<A> = Arena<DeviceTreeNode<A>, A>;
pub type NodeId<A> = ArenaId<DeviceTreeNode<A>>;

/// A tree representation of the hardware in a system based on v0.3 of the Devicetree Specification.
/// This struct supports mutating the device-tree and constructing the device-tree from a flattened
/// device-tree blob (FDT). Implemented internally as an index-tree.
///
/// Note that while this implementation guarantees that the tree structure itself is well-formed,
/// it does not guarantee that individual nodes or properties are semantically correct with respect
/// to the Devicetree Specification (for example, `phandle` references or various `#*-cells`
/// properties). It is up to the user to ensure that these are properly constructed.
pub struct DeviceTree<A: Allocator + Clone> {
    alloc: A,
    node_arena: NodeArena<A>,
    root: Option<NodeId<A>>,
}

impl<A: Allocator + Clone> DeviceTree<A> {
    /// Constructs an empty device-tree using the given allocator.
    pub fn new(alloc: A) -> Self {
        Self {
            alloc: alloc.clone(),
            node_arena: NodeArena::new(alloc),
            root: None,
        }
    }

    /// Constructs a device-tree form the given flattened device-tree (FDT) blob.
    pub fn from(fdt: &Fdt, alloc: A) -> DeviceTreeResult<Self> {
        let fdt = fdt.inner();
        let mut tree = Self::new(alloc);
        let mut iter = fdt.parse_iter();

        let mut parent = None;
        while let Ok(Some(token)) = iter.next() {
            match token {
                ParsedTok::BeginNode(n) => {
                    let id = tree.add_node(str::from_utf8(n.name)?, parent)?;
                    parent = Some(id);
                }
                ParsedTok::EndNode => {
                    // Unwrap ok: parent must be a valid node.
                    let pnode = tree
                        .get_node(parent.ok_or(DeviceTreeError::MalformedFdt)?)
                        .unwrap();
                    parent = pnode.parent();
                }
                ParsedTok::Prop(p) => {
                    // Unwrap ok: parent must be a valid node.
                    let node = tree
                        .get_mut_node(parent.ok_or(DeviceTreeError::MalformedFdt)?)
                        .unwrap();
                    let prop = node.add_prop(fdt.string_at_offset(p.name_offset)?)?;
                    prop.set_value_raw(p.prop_buf)?;
                }
                _ => {}
            }
        }
        Ok(tree)
    }

    /// Returns the ID of the root node.
    pub fn root(&self) -> Option<NodeId<A>> {
        self.root
    }

    /// Returns a reference to the node with the given ID.
    pub fn get_node(&self, node_id: NodeId<A>) -> Option<&DeviceTreeNode<A>> {
        self.node_arena.get(node_id)
    }

    /// Returns a mutable reference to the node with the given ID.
    pub fn get_mut_node(&mut self, node_id: NodeId<A>) -> Option<&mut DeviceTreeNode<A>> {
        self.node_arena.get_mut(node_id)
    }

    /// Creates a new node in the tree with the given name and parent node. If parent is `None`,
    /// then the node is inserted as the root of the tree.
    pub fn add_node(
        &mut self,
        name: &str,
        parent: Option<NodeId<A>>,
    ) -> DeviceTreeResult<NodeId<A>> {
        let node = self.alloc_node(name)?;
        let id = node.id();
        if let Some(pid) = parent {
            node.set_parent(parent);
            let pnode = self
                .get_mut_node(pid)
                .ok_or(DeviceTreeError::InvalidNodeId)?;
            pnode.add_child(id)?;
        } else if self.root.is_none() {
            self.root = Some(id);
        } else {
            // We already have a root; parent can't be None.
            return Err(DeviceTreeError::InvalidNodeId)?;
        }
        Ok(id)
    }

    /// Removes the node and all of its descendents from the tree.
    pub fn remove_node(&mut self, id: NodeId<A>) -> DeviceTreeResult<()> {
        let node = self
            .get_mut_node(id)
            .ok_or(DeviceTreeError::InvalidNodeId)?;
        if let Some(pid) = node.parent() {
            node.set_parent(None);
            let pnode = self
                .get_mut_node(pid)
                .ok_or(DeviceTreeError::InvalidNodeId)?;
            pnode.remove_child(id);
        } else {
            self.root = None
        }

        // TODO: Can we avoid dynamic memory allocation / having to build a list of nodes?
        let mut to_remove = Vec::new_in(self.alloc.clone());
        // Unwrap ok: we already know the ID is valid.
        let num_nodes = self.iter_from(id).unwrap().count();
        to_remove.try_reserve(num_nodes)?;
        for n in self.iter_from(id).unwrap() {
            to_remove.push(n.id());
        }
        for i in to_remove {
            self.node_arena.remove(i);
        }
        Ok(())
    }

    /// Returns an iterator traversing the nodes in the tree in depth-first order, starting at the
    /// root node.
    pub fn iter(&self) -> DeviceTreeIter<A> {
        DeviceTreeIter::new(self, self.root)
    }

    /// Returns an iterator traversing the nodes in the tree in depth-first order, starting at the
    /// given node ID.
    pub fn iter_from(&self, root: NodeId<A>) -> DeviceTreeResult<DeviceTreeIter<A>> {
        let _ = self
            .node_arena
            .get(root)
            .ok_or(DeviceTreeError::InvalidNodeId)?;
        Ok(DeviceTreeIter::new(self, Some(root)))
    }

    /// Returns the allocator used by this device-tree.
    pub fn alloc(&self) -> A {
        self.alloc.clone()
    }

    fn alloc_node(&mut self, name: &str) -> DeviceTreeResult<&mut DeviceTreeNode<A>> {
        let id = self
            .node_arena
            .try_insert(DeviceTreeNode::new(name, self.alloc.clone())?)?;
        let node = self.get_mut_node(id).unwrap();
        node.set_id(id);
        Ok(node)
    }
}

impl<A: Allocator + Clone> DeviceTreeNode<A> {
    fn new(name: &str, alloc: A) -> DeviceTreeResult<Self> {
        let mut node = Self {
            alloc: alloc.clone(),
            id: None,
            name: Vec::new_in(alloc.clone()),
            parent: None,
            children: Vec::new_in(alloc.clone()),
            props: Vec::new_in(alloc),
        };
        node.set_name(name)?;
        Ok(node)
    }

    /// Returns the ID of this ndoe.
    pub fn id(&self) -> NodeId<A> {
        self.id.unwrap()
    }

    /// Returns this node's name.
    pub fn name(&self) -> &str {
        // Unwrap ok since self.name must be a valid, NULL-terminated string by construction.
        str::from_utf8(&self.name)
            .unwrap()
            .strip_suffix('\0')
            .unwrap()
    }

    /// Returns this node's name as a raw byte slice.
    pub(crate) fn name_raw(&self) -> &[u8] {
        &self.name
    }

    /// Sets this node's name.
    pub fn set_name(&mut self, name: &str) -> DeviceTreeResult<()> {
        copy_string_with_null_termination(&mut self.name, name)
    }

    /// Returns the node ID of this node's parent, if it has one.
    pub fn parent(&self) -> Option<NodeId<A>> {
        self.parent
    }

    /// Returns an iterator over this node's child node IDs.
    pub fn children(&self) -> impl ExactSizeIterator<Item = &NodeId<A>> {
        self.children.iter()
    }

    /// Inserts the given property into this node's set of properties.
    pub fn insert_prop(&mut self, prop: DeviceTreeProp<A>) -> DeviceTreeResult<()> {
        self.props.try_reserve(1)?;
        self.props.push(prop);
        Ok(())
    }

    /// Creates a new property for this node with the given name, returning a mutable reference
    /// to the newly-created property.
    pub fn add_prop(&mut self, name: &str) -> DeviceTreeResult<&mut DeviceTreeProp<A>> {
        let index = self.props.len();
        self.props.try_reserve(1)?;
        self.props
            .push(DeviceTreeProp::new(name, self.alloc.clone())?);
        Ok(&mut self.props[index])
    }

    /// Removes a property with the given name from this node
    pub fn remove_prop(&mut self, name: &str) -> DeviceTreeResult<()> {
        let index = self
            .props()
            .position(|p| p.name() == name)
            .ok_or(DeviceTreeError::PropNotFound)?;
        self.props.remove(index);
        Ok(())
    }

    /// Replaces this node's properties with those from the given iterator.
    pub fn set_props<I>(&mut self, props: I) -> DeviceTreeResult<()>
    where
        I: IntoIterator<Item = DeviceTreeProp<A>>,
    {
        self.props.truncate(0);
        for p in props {
            self.insert_prop(p)?;
        }
        Ok(())
    }

    /// Returns an iterator over this node's properties.
    pub fn props(&self) -> impl ExactSizeIterator<Item = &DeviceTreeProp<A>> {
        self.props.iter()
    }

    /// Returns a mutable iterator over this node's properties.
    pub fn props_mut(&mut self) -> impl ExactSizeIterator<Item = &mut DeviceTreeProp<A>> {
        self.props.iter_mut()
    }

    /// Returns true if this node is marked as disabled.
    pub fn disabled(&self) -> bool {
        self.props()
            .any(|p| p.name() == "status" && p.value_str().unwrap_or("") == "disabled")
    }

    /// Returns true if any of the provided compatible strings matches this node.
    pub fn compatible<I>(&self, compat_strings: I) -> bool
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        compat_strings.into_iter().any(|compat_str| {
            self.props()
                .filter(|p| p.name() == "compatible")
                .any(|p| p.value_str().unwrap_or("").contains(compat_str.as_ref()))
        })
    }

    fn set_id(&mut self, id: NodeId<A>) {
        self.id = Some(id);
    }

    fn set_parent(&mut self, parent: Option<NodeId<A>>) {
        self.parent = parent;
    }

    fn add_child(&mut self, child: NodeId<A>) -> DeviceTreeResult<()> {
        self.children.try_reserve(1)?;
        self.children.push(child);
        Ok(())
    }

    fn remove_child(&mut self, child: NodeId<A>) {
        self.children.retain(|&i| i != child);
    }

    fn fmt_in(
        &self,
        tree: &DeviceTree<A>,
        f: &mut fmt::Formatter,
    ) -> result::Result<(), fmt::Error> {
        // TODO: Use identation to make this prettier.
        writeln!(f, "{} {{", self.name())?;
        for p in self.props() {
            writeln!(f, "{}", p)?;
        }
        for &n in self.children() {
            let node = tree.get_node(n).unwrap();
            node.fmt_in(tree, f)?;
        }
        writeln!(f, "}}")
    }
}

impl<A: Allocator + Clone> DeviceTreeProp<A> {
    fn new(name: &str, alloc: A) -> DeviceTreeResult<Self> {
        let mut prop = Self {
            name: Vec::new_in(alloc.clone()),
            buf: Vec::new_in(alloc),
        };
        prop.set_name(name)?;
        Ok(prop)
    }

    /// Returns this property's name.
    pub fn name(&self) -> &str {
        // Unwrap ok since self.name must be a valid, NULL-terminated string by construction.
        str::from_utf8(&self.name)
            .unwrap()
            .strip_suffix('\0')
            .unwrap()
    }

    /// Returns this property's name as a raw byte slice.
    pub(crate) fn name_raw(&self) -> &[u8] {
        &self.name
    }

    /// Sets this property's name.
    pub fn set_name(&mut self, name: &str) -> DeviceTreeResult<()> {
        copy_string_with_null_termination(&mut self.name, name)
    }

    /// Returns this property's value as a raw byte slice.
    pub fn value_raw(&self) -> &[u8] {
        &self.buf
    }

    /// Sets this property's value from a raw byte slice.
    pub fn set_value_raw(&mut self, val: &[u8]) -> DeviceTreeResult<()> {
        self.buf.truncate(0);
        self.buf.try_reserve(val.len())?;
        self.buf.splice(0.., val.iter().cloned());
        Ok(())
    }

    /// Returns this property's value as a string, if it can be read as one.
    pub fn value_str(&self) -> Option<&str> {
        let value = str::from_utf8(&self.buf).ok()?;
        value.strip_suffix('\0')
    }

    /// Sets this property's value from a string.
    pub fn set_value_str(&mut self, val: &str) -> DeviceTreeResult<()> {
        copy_string_with_null_termination(&mut self.buf, val)
    }

    /// Returns this property's value as an iterator over a set of u32s.
    pub fn value_u32(&self) -> DeviceTreePropIter<u32, { mem::size_of::<u32>() }> {
        DeviceTreePropIter::new(self.buf.as_slice(), u32::from_be_bytes)
    }

    /// Sets this property's value from a set of u32s.
    pub fn set_value_u32(&mut self, vals: &[u32]) -> DeviceTreeResult<()> {
        self.set_value::<u32, { mem::size_of::<u32>() }>(vals, u32::to_be_bytes)
    }

    /// Returns this property's value as an iterator over a set of u64s.
    pub fn value_u64(&self) -> DeviceTreePropIter<u64, { mem::size_of::<u64>() }> {
        DeviceTreePropIter::new(self.buf.as_slice(), u64::from_be_bytes)
    }

    /// Sets this property's value from a set of u64s.
    pub fn set_value_u64(&mut self, vals: &[u64]) -> DeviceTreeResult<()> {
        self.set_value::<u64, { mem::size_of::<u64>() }>(vals, u64::to_be_bytes)
    }

    // TODO: The unstable generic_const_exprs would allow us to set N from size_of::<T>.
    fn set_value<T: Copy, const N: usize>(
        &mut self,
        vals: &[T],
        set_func: fn(T) -> [u8; N],
    ) -> DeviceTreeResult<()> {
        self.buf.truncate(0);
        self.buf.try_reserve(vals.len() * N)?;
        self.buf.resize(vals.len() * N, 0);
        let mut remainder = self.buf.as_mut_slice();
        for &v in vals {
            let (left, right) = remainder.split_array_mut::<N>();
            *left = set_func(v);
            remainder = right;
        }
        Ok(())
    }
}

impl<A: Allocator + Clone> Clone for DeviceTreeProp<A> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            buf: self.buf.clone(),
        }
    }
}

/// An iterator over a device-tree in depth-first order.
pub struct DeviceTreeIter<'tree, A: Allocator + Clone> {
    tree: &'tree DeviceTree<A>,
    root: Option<NodeId<A>>,
    current: Option<NodeId<A>>,
    depth: usize,
}

impl<'tree, A: Allocator + Clone> DeviceTreeIter<'tree, A> {
    /// Creates a new iterator starting at the given node.
    pub fn new(tree: &'tree DeviceTree<A>, root: Option<NodeId<A>>) -> Self {
        Self {
            tree,
            root,
            current: root,
            depth: 0,
        }
    }

    /// Returns the depth of the iterator in terms of number of levels removed from the root
    /// of the iterator, e.g. depth is 0 for the root node, 1 for children of the root node, and
    /// so on.
    pub fn depth(&self) -> usize {
        self.depth
    }

    fn set_next_node(&mut self, node: &DeviceTreeNode<A>) {
        if node.children().len() > 0 {
            self.depth += 1;
            self.current = node.children().next().cloned();
            return;
        }

        // Go up the tree until we find and un-visited anscestor, stopping if we've reached the root
        // of the iterator.
        let mut current = node;
        while let Some(parent) = current.parent() {
            // Unwrap ok: root must've been non-empty for us to have reached this point.
            if current.id() == self.root.unwrap() {
                break;
            }

            // Unwrap ok: if starting node was valid, it must be part of a well-formed tree.
            let pnode = self.tree.get_node(parent).unwrap();
            let mut iter = pnode.children();
            let _ = iter.find(|&&id| id == current.id()).unwrap();
            if let Some(&sibling) = iter.next() {
                self.current = Some(sibling);
                return;
            }
            current = pnode;
            self.depth -= 1;
        }
        self.current = None;
    }
}

impl<'tree, A: Allocator + Clone> Iterator for DeviceTreeIter<'tree, A> {
    type Item = &'tree DeviceTreeNode<A>;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.tree.get_node(self.current?)?;
        self.set_next_node(node);
        Some(node)
    }
}

/// An iterator over fixed-sized items in a device-tree property.
pub struct DeviceTreePropIter<'a, T, const N: usize> {
    buf: &'a [u8],
    parse_func: fn([u8; N]) -> T,
}

impl<'a, T, const N: usize> DeviceTreePropIter<'a, T, N> {
    fn new(buf: &'a [u8], parse_func: fn([u8; N]) -> T) -> Self {
        Self { buf, parse_func }
    }
}

impl<'a, T, const N: usize> Iterator for DeviceTreePropIter<'a, T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buf.len() < N {
            return None;
        }
        let (left, right) = self.buf.split_array_ref::<N>();
        let ret = (self.parse_func)(*left);
        self.buf = right;
        Some(ret)
    }
}

impl<A: Allocator + Clone> fmt::Display for DeviceTree<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        match self.root {
            Some(r) => {
                let node = self.get_node(r).unwrap();
                write!(f, "\\")?;
                node.fmt_in(self, f)
            }
            None => write!(f, "empty"),
        }
    }
}

impl<A: Allocator + Clone> fmt::Display for DeviceTreeProp<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        // Consider a property printable as a string if its ASCII and doesn't contain repeated nulls.
        fn printable(s: &str) -> bool {
            s.is_ascii() && !s.contains("\0\0")
        }

        if let Some(s) = self.value_str().filter(|&s| printable(s)) {
            write!(f, "{} = \"{}\";", self.name(), s)?;
        } else if self.value_raw().is_empty() {
            write!(f, "{};", self.name())?;
        } else {
            write!(f, "{} =", self.name())?;
            for v in self.value_u32() {
                write!(f, " 0x{:08x}", v)?;
            }
            write!(f, ";")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::alloc::Global;

    fn stub_tree() -> DeviceTree<Global> {
        let mut tree = DeviceTree::new(Global);
        // Create the following structure:
        //       root
        //       /   \
        //      a     d
        //     / \    /
        //    b   c   e
        let root = tree.add_node("", None).unwrap();
        let a = tree.add_node("a", Some(root)).unwrap();
        let _ = tree.add_node("b", Some(a)).unwrap();
        let _ = tree.add_node("c", Some(a)).unwrap();
        let d = tree.add_node("d", Some(root)).unwrap();
        let _ = tree.add_node("e", Some(d)).unwrap();
        tree
    }

    #[test]
    fn tree_construction() {
        let tree = stub_tree();
        let root_node = tree.get_node(tree.root().unwrap()).unwrap();
        assert_eq!(root_node.name(), "");
        assert_eq!(root_node.children().count(), 2);
        assert!(root_node.parent().is_none());
        let a_node = root_node
            .children()
            .map(|&id| tree.get_node(id).unwrap())
            .find(|node| node.name() == "a")
            .unwrap();
        assert_eq!(a_node.children().count(), 2);
        assert_eq!(a_node.parent().unwrap(), root_node.id());
        let c_node = a_node
            .children()
            .map(|&id| tree.get_node(id).unwrap())
            .find(|node| node.name() == "c")
            .unwrap();
        assert_eq!(c_node.children().count(), 0);
        assert_eq!(c_node.parent().unwrap(), a_node.id());
    }

    #[test]
    fn node_removal() {
        let mut tree = stub_tree();
        let a_id = tree.iter().find(|node| node.name() == "a").unwrap().id();
        assert!(tree.remove_node(a_id).is_ok());
        assert!(tree.get_node(a_id).is_none());
        assert!(tree.iter().find(|node| node.name() == "b").is_none());
        assert!(tree.iter().find(|node| node.name() == "c").is_none());
        let root_node = tree.iter().next().unwrap();
        assert_eq!(root_node.children().count(), 1);
    }

    #[test]
    fn traversal_order() {
        let tree = stub_tree();
        let mut iter = tree.iter();
        assert_eq!(iter.depth(), 0);
        assert_eq!(iter.next().unwrap().name(), "");
        assert_eq!(iter.depth(), 1);
        assert_eq!(iter.next().unwrap().name(), "a");
        assert_eq!(iter.depth(), 2);
        assert_eq!(iter.next().unwrap().name(), "b");
        assert_eq!(iter.depth(), 2);
        assert_eq!(iter.next().unwrap().name(), "c");
        assert_eq!(iter.depth(), 1);
        assert_eq!(iter.next().unwrap().name(), "d");
        assert_eq!(iter.depth(), 2);
        assert_eq!(iter.next().unwrap().name(), "e");
        assert!(iter.next().is_none());
    }

    #[test]
    fn properties() {
        let mut tree = stub_tree();
        let node = tree.get_mut_node(tree.root().unwrap()).unwrap();
        {
            let prop = node.add_prop("hello").unwrap();
            prop.set_value_str("world").unwrap();
        }
        {
            let prop = node.add_prop("foo").unwrap();
            prop.set_value_u32(&[0xdeadbeef]).unwrap();
        }
        assert_eq!(node.props().count(), 2);
        for p in node.props() {
            match p.name() {
                "hello" => {
                    assert_eq!(p.value_str().unwrap(), "world");
                }
                "foo" => {
                    assert_eq!(p.value_u32().next().unwrap(), 0xdeadbeef);
                }
                name => {
                    panic!("bad property name {}", name);
                }
            }
        }
    }
}
