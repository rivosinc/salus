// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use arrayvec::{ArrayString, ArrayVec};
use core::fmt;
use device_tree::{DeviceTree, DeviceTreeNode, DeviceTreeResult};
use spin::Once;

const MAX_ISA_STRING_LEN: usize = 256;

/// The maximum number of CPUs we can support.
pub const MAX_CPUS: usize = 128;

/// Logical CPU number. Not necessarily the same as hart ID; see `CpuInfo` for translating between
/// hart ID and logical CPU ID.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub struct CpuId(usize);

impl CpuId {
    /// Creates a `CpuId` from the raw index.
    pub fn new(raw: usize) -> Self {
        CpuId(raw)
    }

    /// Returns the raw value of the CPU ID.
    pub fn raw(&self) -> usize {
        self.0
    }
}

/// Holds static global information about CPU features and topology.
#[derive(Debug)]
pub struct CpuInfo {
    // True if the Sstc extension is supported.
    has_sstc: bool,
    // True if the Sscofpmf extension is supported.
    has_sscofpmf: bool,
    // CPU timer frequency.
    timer_frequency: u32,
    // ISA string as reprted in the device-tree. All CPUs are expected to have the same ISA.
    isa_string: ArrayString<MAX_ISA_STRING_LEN>,
    // Mapping of logical CPU index to hart IDs.
    hart_ids: ArrayVec<u32, MAX_CPUS>,
    // Mapping of logical CPU index to the CPU's 'interrupt-controller' phandle in the device-tree.
    intc_phandles: ArrayVec<u32, MAX_CPUS>,
}

static CPU_INFO: Once<CpuInfo> = Once::new();

fn hart_id_from_cpu_node(node: &DeviceTreeNode) -> u32 {
    node.props()
        .find(|p| p.name() == "reg")
        .expect("No 'reg' property in CPU node")
        .value_u32()
        .next()
        .unwrap()
}

fn intc_node_from_cpu_node<'a>(dt: &'a DeviceTree, node: &'_ DeviceTreeNode) -> &'a DeviceTreeNode {
    dt.iter_from(node.id())
        .unwrap()
        .find(|n| n.name() == "interrupt-controller")
        .expect("No CPU 'interrupt-controller' sub-node")
}

fn intc_phandle_from_cpu_node(dt: &DeviceTree, node: &DeviceTreeNode) -> u32 {
    let intc_node = intc_node_from_cpu_node(dt, node);
    intc_node
        .props()
        .find(|p| p.name() == "phandle")
        .expect("No 'phandle' property in CPU interrupt controller node")
        .value_u32()
        .next()
        .unwrap()
}

fn isa_string_has_extension(isa_string: &str, extension: &str) -> bool {
    isa_string.split('_').any(|f| f == extension)
}

impl CpuInfo {
    /// Initializes the global `CpuInfo` state from the a device-tree. Must be called first before
    /// get(). Panics if the device-tree is malformed (missing CPU nodes or expected properties).
    pub fn parse_from(dt: &DeviceTree) {
        // Locate the /cpus node in the device-tree.
        let mut iter = dt.iter();
        let cpus_node = iter
            .find(|n| n.name() == "cpus")
            .expect("No /cpus in device-tree");

        // We only support 32-bit hart IDs.
        let ac = cpus_node
            .props()
            .find(|p| p.name() == "#address-cells")
            .expect("No #address-cells in /cpus");
        assert_eq!(ac.value_u32().next().unwrap(), 1);

        // 'timebase-frequency' appears in the top-level /cpus node.
        let timer_frequency = cpus_node
            .props()
            .find(|p| p.name() == "timebase-frequency")
            .expect("No 'timebase-frequency' in /cpus")
            .value_u32()
            .next()
            .unwrap();

        let mut cpus_iter = iter.filter(|n| {
            n.props()
                .any(|p| (p.name() == "device_type") && (p.value_str().unwrap_or("") == "cpu"))
        });

        // Pull ISA details from CPU0.
        let cpu0 = cpus_iter.next().expect("No node for CPU0 in device-tree");
        let isa_string = cpu0
            .props()
            .find(|p| p.name() == "riscv,isa")
            .expect("No 'riscv,isa' property in device-tree")
            .value_str()
            .unwrap();
        let mmu_string = cpu0
            .props()
            .find(|p| p.name() == "mmu-type")
            .expect("No 'mmu-type' property in device-tree")
            .value_str()
            .unwrap();
        // All of our memory management currently assumes SV48 compatibility.
        assert!(mmu_string == "riscv,sv48" || mmu_string == "riscv,sv57");
        // 'interrupt-controller' sub-node indicates if AIA is supported. We rely on AIA support
        // in order to direct interrupts to VS level, so mandate its presence.
        let intc_node = intc_node_from_cpu_node(dt, cpu0);
        intc_node
            .props()
            .find(|p| {
                p.name() == "compatible"
                    && p.value_str().unwrap_or("").contains("riscv,cpu-intc-aia")
            })
            .expect("CPU does not support AIA");

        let mut hart_ids = ArrayVec::new();
        hart_ids.push(hart_id_from_cpu_node(cpu0));
        let mut intc_phandles = ArrayVec::new();
        intc_phandles.push(intc_phandle_from_cpu_node(dt, cpu0));

        // Now parse hart IDs and phandles for the secondary CPUs. We assume the CPUs are homogenous.
        for cpu in cpus_iter {
            hart_ids.push(hart_id_from_cpu_node(cpu));
            intc_phandles.push(intc_phandle_from_cpu_node(dt, cpu));
        }

        let cpu_info = CpuInfo {
            has_sstc: isa_string_has_extension(isa_string, "sstc"),
            has_sscofpmf: isa_string_has_extension(isa_string, "sscofpmf"),
            isa_string: ArrayString::from(isa_string).unwrap(),
            timer_frequency,
            hart_ids,
            intc_phandles,
        };
        CPU_INFO.call_once(|| cpu_info);
    }

    /// Returns a reference to the global CpuInfo structure. Panics if parse_from() hasn't been
    /// called yet.
    pub fn get() -> &'static CpuInfo {
        CPU_INFO.get().unwrap()
    }

    /// Returns true if the Sstc extension is supported.
    pub fn has_sstc(&self) -> bool {
        self.has_sstc
    }

    /// Returns true if the Sscofpmf extension is supported.
    pub fn has_sscofpmf(&self) -> bool {
        self.has_sscofpmf
    }

    /// Returns the total number of CPUs.
    pub fn num_cpus(&self) -> usize {
        self.hart_ids.len()
    }

    /// Returns the hart ID corresponding to the logical CPU ID.
    pub fn cpu_to_hart_id(&self, cpu: CpuId) -> Option<u32> {
        self.hart_ids.get(cpu.raw()).cloned()
    }

    /// Returns the logical CPU ID corresponding to the hart ID.
    pub fn hart_id_to_cpu(&self, hart_id: u32) -> Option<CpuId> {
        self.hart_ids
            .iter()
            .position(|&h| h == hart_id)
            .map(CpuId::new)
    }

    /// Returns the phandle of the interrup-controller node for the given CPU in the device-tree.
    pub fn cpu_to_intc_phandle(&self, cpu: CpuId) -> Option<u32> {
        self.intc_phandles.get(cpu.raw()).cloned()
    }

    /// Returns the logical CPU ID with the interrupt controller referenced by `intc_phandle` in
    /// the device-tree.
    pub fn intc_phandle_to_cpu(&self, intc_phandle: u32) -> Option<CpuId> {
        self.intc_phandles
            .iter()
            .position(|&p| p == intc_phandle)
            .map(CpuId::new)
    }

    /// Populates the host device-tree with CPU nodes.
    pub fn add_host_cpu_nodes(&self, dt: &mut DeviceTree) -> DeviceTreeResult<()> {
        let cpus_id = dt.add_node("cpus", dt.root())?;
        let cpus_node = dt.get_mut_node(cpus_id).unwrap();
        cpus_node.add_prop("#address-cells")?.set_value_u32(&[1])?;
        cpus_node.add_prop("#size-cells")?.set_value_u32(&[0])?;
        cpus_node
            .add_prop("timebase-frequency")?
            .set_value_u32(&[self.timer_frequency])?;

        // Now add the CPUs themselves. We identity map the (virtual) hart IDs exposed to the host
        // VM to ease translation.
        for (i, &phandle) in self.intc_phandles.iter().enumerate() {
            let mut cpu_name = ArrayString::<16>::new();
            fmt::write(&mut cpu_name, format_args!("cpu@{:x}", i)).unwrap();
            let cpu_node_id = dt.add_node(cpu_name.as_str(), Some(cpus_id))?;
            let cpu_node = dt.get_mut_node(cpu_node_id).unwrap();
            cpu_node.add_prop("device_type")?.set_value_str("cpu")?;
            cpu_node.add_prop("compatible")?.set_value_str("riscv")?;
            cpu_node.add_prop("reg")?.set_value_u32(&[i as u32])?;
            cpu_node.add_prop("mmu-type")?.set_value_str("riscv,sv48")?;
            cpu_node
                .add_prop("riscv,isa")?
                .set_value_str(self.isa_string.as_str())?;
            cpu_node.add_prop("status")?.set_value_str("okay")?;

            // Each CPU needs a sub-node for its interrupt controller.
            let intc_id = dt.add_node("interrupt-controller", Some(cpu_node_id))?;
            let intc_node = dt.get_mut_node(intc_id).unwrap();
            intc_node
                .add_prop("#interrupt-cells")?
                .set_value_u32(&[1])?;
            intc_node.add_prop("interrupt-controller")?;
            intc_node.add_prop("phandle")?.set_value_u32(&[phandle])?;
            intc_node
                .add_prop("compatible")?
                .set_value_str("riscv,cpu-intc-aia\0riscv,cpu-intc")?;
        }

        // TODO: Add CPU topology info (socket, package, etc)  in 'cpu-map'.
        Ok(())
    }
}
