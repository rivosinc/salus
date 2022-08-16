// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use riscv_page_tables::{FirstStagePagingMode, GuestStagePageTable, GuestStagePagingMode};
use riscv_pages::Pfn;
use tock_registers::register_bitfields;
use tock_registers::LocalRegisterCopy;

// Supervisor status.
register_bitfields![u64,
    pub sstatus [
        // Enable or disable all interrupts in S-mode.
        sie OFFSET(1) NUMBITS(1) [],
        // Indicates whether supervisor interrupts were enabled prior
        // to trapping into S-mode.
        spie OFFSET(5) NUMBITS(1) [],
        // U-mode big endian enable flag.
        ube OFFSET(6) NUMBITS(1) [],
        // Privilege level hart was executing before entering S-mode.
        spp OFFSET(8) NUMBITS(1) [
            User = 0,
            Supervisor = 1,
        ],
        // Encodes the status of the vector unit.
        vs OFFSET(9) NUMBITS(2) [
            Off = 0,
            Initial = 1,
            Clean = 2,
            Dirty = 3,
        ],
        // Encodes the status of the floating-point unit.
        fs OFFSET(13) NUMBITS(2) [
            Off = 0,
            Initial = 1,
            Clean = 2,
            Dirty = 3,
        ],
        // Encodes the status of additional U-mode extensions
        // and associated state.
        xs OFFSET(15) NUMBITS(2) [
            AllOff = 0,
            SomeOn = 1,
            SomeClean = 2,
            SomeDirty = 3,
        ],
        // Supervisor User Memory - Modifies the privilege with which S-mode
        // loads and stores access virtual memory.
        sum OFFSET(18) NUMBITS(1) [],
        // Make eXecutable Readable - modifies the privilege with which loads
        // access virtual memory.
        mxr OFFSET(19) NUMBITS(1) [],
        // Native base integer ISA width for U-mode.
        uxl OFFSET(32) NUMBITS(2) [
            Xlen32 = 1,
            Xlen64 = 2,
        ],
        // Summarizes whether either the FS field or XS field signals the
        // presence of some dirty state that will require saving extended
        // user context to memory.
        sd OFFSET(63) NUMBITS(1) [],
    ]
];

// Supervisor interrupt enable register.
register_bitfields![u64,
    pub sie [
        ssoft OFFSET(1) NUMBITS(1) [],
        stimer OFFSET(5) NUMBITS(1) [],
        sext OFFSET(9) NUMBITS(1) [],
    ]
];

// Trap handler base address.
register_bitfields![u64,
    pub stvec [
        trap_addr OFFSET(2) NUMBITS(60) [],
        mode OFFSET(0) NUMBITS(2) [
            Direct = 0,
            Vectored = 1
        ]
    ]
];

pub trait StvecHelpers {
    fn get_trap_address(&self) -> u64;
}

impl StvecHelpers for LocalRegisterCopy<u64, stvec::Register> {
    fn get_trap_address(&self) -> u64 {
        self.read(stvec::trap_addr) << 2
    }
}

// U-mode counter availability control.
register_bitfields![u64,
    pub scounteren [
        cycle OFFSET(0) NUMBITS(1) [],
        time OFFSET(1) NUMBITS(1) [],
        instret OFFSET(2) NUMBITS(1) [],
        hpm OFFSET(3) NUMBITS(29) [],
    ]
];

// Scratch register for supervisor use.
register_bitfields![u64,
    pub sscratch [
        val OFFSET(0) NUMBITS(64) []
    ]
];

// Address of where a trap was taken in HS mode.
register_bitfields![u64,
    pub sepc [
        trap_addr OFFSET(0) NUMBITS(64) []
    ]
];

// Trap cause.
register_bitfields![u64,
    pub scause [
        is_interrupt OFFSET(63) NUMBITS(1) [],
        reason OFFSET(0) NUMBITS(63) []
    ],
    // Per the spec, implementations are allowed to use the higher bits of the
    // interrupt/exception reason for their own purposes.  For regular parsing,
    // we only concern ourselves with the "standard" values.
    pub(crate) reason [
        reserved OFFSET(5) NUMBITS(58) [],
        std OFFSET(0) NUMBITS(5) []
    ]
];

// Supervisor trap bad address or instruction.
register_bitfields![u64,
    pub stval [
        // The interpretation of this field is determined by the type of trap.
        // In general, this holds the faulting address for page or alignment faults,
        // and the faulting instruction for illegal instruction faults.
        val OFFSET(0) NUMBITS(64) [],
    ]
];

// Supervisor nterrupt pending register.
register_bitfields![u64,
    pub sip [
        ssoft OFFSET(1) NUMBITS(1) [],
        stimer OFFSET(5) NUMBITS(1) [],
        sext OFFSET(9) NUMBITS(1) [],
    ]
];

// Supervisor timer compare register.
register_bitfields![u64,
    pub stimecmp [
        cmp_val OFFSET(0) NUMBITS(64) [],
    ]
];

// IMSIC indirect CSR address register.
register_bitfields![u64,
    pub siselect [
        reg_addr OFFSET(0) NUMBITS(64) [],
    ]
];

// IMSIC indirect CSR value register.
register_bitfields![u64,
    pub sireg [
        reg_val OFFSET(0) NUMBITS(64) [],
    ]
];

// External interrupt claim reigster.
register_bitfields![u64,
    pub stopei [
        interrupt_id OFFSET(16) NUMBITS(11) [],
        interrupt_prio OFFSET(0) NUMBITS(11) [],
    ]
];

// Supervisor address translation register.
register_bitfields![u64,
    pub satp [
        // Physical page number of the root translation table.
        ppn OFFSET(0) NUMBITS(44) [],
        // Address-space ID.
        asid OFFSET(44) NUMBITS(16) [],
        // Translation mode.
        mode OFFSET(60) NUMBITS(4) [
            Bare = 0,
            Sv39 = 8,
            Sv48 = 9,
            Sv57 = 10,
            Sv64 = 11,
        ],
    ]
];

pub trait SatpHelpers {
    fn set_from<T: FirstStagePagingMode>(&mut self, pt_root: &GuestStagePageTable<T>, asid: u64);
}

impl SatpHelpers for LocalRegisterCopy<u64, satp::Register> {
    fn set_from<T: FirstStagePagingMode>(&mut self, pt: &GuestStagePageTable<T>, asid: u64) {
        self.modify(satp::asid.val(asid));
        self.modify(satp::ppn.val(Pfn::from(pt.get_root_address()).bits()));
        self.modify(satp::mode.val(T::SATP_VALUE));
    }
}

// Top-level interrupt claim reigster.
register_bitfields![u64,
    pub stopi [
        interrupt_id OFFSET(16) NUMBITS(8) [],
        interrupt_prio OFFSET(0) NUMBITS(8) [],
    ]
];

// Hypervisor status register.
register_bitfields![u64,
    pub hstatus [
        // VS mode endianness control.
        vsbe OFFSET(6) NUMBITS(1) [],
        // A guest virtual address was written to stval as a result of the trap.
        gva OFFSET(6) NUMBITS(1) [],
        // Virtualization mode at time of trap.
        spv OFFSET(7) NUMBITS(1) [],
        // Privilege level the virtual hart was executing before entering HS-mode.
        spvp OFFSET(8) NUMBITS(1) [
            User = 0,
            Supervisor = 1,
        ],
        // Allow hypervisor instructions in U-mode.
        hu OFFSET(9) NUMBITS(1) [],
        // Selects the guest external interrupt source for VS external interrupts.
        vgein OFFSET(12) NUMBITS(6) [],
        // Trap on SFENCE, SINVAL, or changes to vsatp.
        vtvm OFFSET(20) NUMBITS(1) [],
        // Trap on WFI timeout.
        vtw OFFSET(21) NUMBITS(1) [],
        // Trap SRET instruction.
        vtsr OFFSET(22) NUMBITS(1) [],
        // Native base integer ISA width for VS-mode.
        vsxl OFFSET(32) NUMBITS(2) [
            Xlen32 = 1,
            Xlen64 = 2,
        ],
    ]
];

// Hypervisor exception delegation register.
register_bitfields![u64,
    pub hedeleg [
        instr_misaligned OFFSET(0) NUMBITS(1) [],
        instr_fault OFFSET(1) NUMBITS(1) [],
        illegal_instr OFFSET(2) NUMBITS(1) [],
        breakpoint OFFSET(3) NUMBITS(1) [],
        load_misaligned OFFSET(4) NUMBITS(1) [],
        load_fault OFFSET(5) NUMBITS(1) [],
        store_misaligned OFFSET(6) NUMBITS(1) [],
        store_fault OFFSET(7) NUMBITS(1) [],
        u_ecall OFFSET(8) NUMBITS(1) [],
        instr_page_fault OFFSET(12) NUMBITS(1) [],
        load_page_fault OFFSET(13) NUMBITS(1) [],
        store_page_fault OFFSET(15) NUMBITS(1) [],
    ]
];

// Hypervisor interrupt delegation register.
register_bitfields![u64,
    pub hideleg [
        vssoft OFFSET(2) NUMBITS(1) [],
        vstimer OFFSET(6) NUMBITS(1) [],
        vsext OFFSET(10) NUMBITS(1) [],
    ]
];

// Hypervisor interrupt enable register.
register_bitfields![u64,
    pub hie [
        vssoft OFFSET(2) NUMBITS(1) [],
        vstimer OFFSET(6) NUMBITS(1) [],
        vsext OFFSET(10) NUMBITS(1) [],
        sgext OFFSET(12) NUMBITS(1) [],
    ]
];

// VS-mode counter availability control.
register_bitfields![u64,
    pub hcounteren [
        cycle OFFSET(0) NUMBITS(1) [],
        time OFFSET(1) NUMBITS(1) [],
        instret OFFSET(2) NUMBITS(1) [],
        hpm OFFSET(3) NUMBITS(29) [],
    ]
];

// Hypervisor guest external interrupt enable.
register_bitfields![u64,
    pub hgeie [
        // Number of implemented bits depends on GEILEN for the platform.
        interrrupts OFFSET(1) NUMBITS(63) [],
    ]
];

// Faulting guest phsyical address.
register_bitfields![u64,
    pub htval [
        gpa_div4 OFFSET(0) NUMBITS(64) [],
    ]
];

pub trait HtvalHelpers {
    fn get_gpa(&self) -> u64;
}

impl HtvalHelpers for LocalRegisterCopy<u64, htval::Register> {
    fn get_gpa(&self) -> u64 {
        self.read(htval::gpa_div4) << 2
    }
}

// Hypervisor interrupt pending register.
register_bitfields![u64,
    pub hip [
        vssoft OFFSET(2) NUMBITS(1) [],
        vstimer OFFSET(6) NUMBITS(1) [],
        vsext OFFSET(10) NUMBITS(1) [],
        sgext OFFSET(12) NUMBITS(1) [],
    ]
];

// Hypervisor virtual interrupt pending.
register_bitfields![u64,
    pub hvip [
        vssoft OFFSET(2) NUMBITS(1) [],
        vstimer OFFSET(6) NUMBITS(1) [],
        vsext OFFSET(10) NUMBITS(1) [],
    ]
];

// Hypervisor trap instruction.
register_bitfields![u64,
    pub htinst [
        // May have been transformed; see priv spec.
        instruction OFFSET(0) NUMBITS(64) [],
    ]
];

// Hypervisor guest external interrupt pending.
register_bitfields![u64,
    pub hgeip [
        // Number of implemented bits depends on GEILEN for the platform.
        interrrupts OFFSET(1) NUMBITS(63) [],
    ]
];

// Hypervisor environment config register.
register_bitfields![u64,
    pub henvcfg [
        // Fence of I/O implies memory.
        fiom OFFSET(0) NUMBITS(1) [],
        // Enable stimecmp in VS.
        stce OFFSET(63) NUMBITS(1) [],
        // TODO: Bits for other extensions we don't care about yet.
    ]
];

// Hypervisor (2nd-stage) address translation register.
register_bitfields![u64,
    pub hgatp [
        // Physical page number of the root translation table.
        ppn OFFSET(0) NUMBITS(44) [],
        // Virtual machine ID.
        vmid OFFSET(44) NUMBITS(14) [],
        // Translation mode.
        mode OFFSET(60) NUMBITS(4) [
            Bare = 0,
            Sv39x4 = 8,
            Sv48x4 = 9,
            Sv57x4 = 10,
        ],
    ]
];

pub trait HgatpHelpers {
    fn set_from<T: GuestStagePagingMode>(&mut self, pt_root: &GuestStagePageTable<T>, vmid: u64);
}

impl HgatpHelpers for LocalRegisterCopy<u64, hgatp::Register> {
    fn set_from<T: GuestStagePagingMode>(&mut self, pt: &GuestStagePageTable<T>, vmid: u64) {
        self.modify(hgatp::vmid.val(vmid));
        self.modify(hgatp::ppn.val(Pfn::from(pt.get_root_address()).bits()));
        self.modify(hgatp::mode.val(T::HGATP_VALUE));
    }
}

// Hypervisor time offset register.
register_bitfields![u64,
    pub htimedelta [
        // Delta applied to reads from 'time' from VS/VU modes.
        delta OFFSET(0) NUMBITS(64) [],
    ]
];

// Performance counter registers.
register_bitfields![u64,
    pub hpmcounter [
        value OFFSET(0) NUMBITS(64) [],
    ]
];
