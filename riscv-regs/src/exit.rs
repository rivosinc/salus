// Copyright (c) 2021 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/// Valid interrupt casues for S mode to handle.
#[derive(Debug, PartialEq, Eq)]
pub enum SupervisorInterruptCause {
    SupervisorSoftware,
    VirtualSupervisorSoftware,
    SupervisorExternal,
    VirtualSupervisoeExternal,
    SupervisorGuestExternal,
}

/// Valid exception causes for S mode to handle.
#[derive(Debug, PartialEq, Eq)]
pub enum SupervisorExceptionCause {
    InstructionAddressMisaligned,
    InstructionAccessFault,
    IllegalInstruction,
    Breakpoint,
    LoadAddressMisaligned,
    LoadAccessFault,
    StoreAmoAddressMisaligned,
    SoreAmoAccessFault,
    EcallUMode,
    EcallVsMode,
    InstructionPageFault,
    LoadPageFault,
    StoreAmoPageFault,
    GuestInstructionPageFault,
    GuestLoadPageFault,
    VirtualInstruction,
    GuestStoreAmoPageFault,
}

/// Reasons for guest exits
#[derive(Debug, PartialEq, Eq)]
pub enum GuestExit {
    Interrupt(SupervisorInterruptCause),
    Exception(SupervisorExceptionCause),
}
