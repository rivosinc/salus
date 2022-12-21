// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use assertions::const_assert;
use core::{marker::PhantomData, ptr};

use crate::TeeHostFunction::*;
use crate::{ecall_send, Error, Result, SbiMessage};
use crate::{
    NaclShmem, TsmInfo, TsmPageType, TsmShmemScratch, TvmCreateParams, NACL_SCRATCH_BYTES,
};

/// Provides volatile accessors to a TEE `NaclShmem` area.
pub struct TsmShmemAreaRef<'a> {
    ptr: *mut NaclShmem,
    _lifetime: PhantomData<&'a NaclShmem>,
}

impl<'a> TsmShmemAreaRef<'a> {
    /// Creates a new `TsmShmemAreaRef` from a raw pointer to a `NaclShmem`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that `ptr` is suitably aligned and points to a `NaclShmem`
    /// structure that is valid for the lifetime `'a`.
    pub unsafe fn new(ptr: *mut NaclShmem) -> Self {
        Self {
            ptr,
            _lifetime: PhantomData,
        }
    }

    fn shmem_scratch_ptr(&self) -> *mut TsmShmemScratch {
        // Safety: We're only dereferencing here to get the address of the `scratch` field. Further,
        // it is safe to cast the `scratch field to a `TsmShmemScratch` struct since it has the same
        // size & alignemnt of the `scratch` field and both are POD structs.
        unsafe { ptr::addr_of_mut!((*self.ptr).scratch) as *mut _ }
    }

    /// Reads the HS or VS CSR at `csr_num`.
    pub fn csr(&self, csr_num: u16) -> u64 {
        let index = NaclShmem::csr_index(csr_num);
        // Safety: `index` is guaranteed to be a valid index into `csrs` and the caller guaranteed
        // at construction that `ptr` points to a valid `TsmShmemArea`.
        unsafe { ptr::addr_of!((*self.ptr).csrs[index]).read_volatile() }
    }

    /// Writes the HS or VS CSR at `csr_num`.
    pub fn set_csr(&self, csr_num: u16, val: u64) {
        let index = NaclShmem::csr_index(csr_num);
        // Safety: `index` is guaranteed to be a valid index into `csrs` and the caller guaranteed
        // at construction that `ptr` points to a valid `TsmShmemArea`.
        unsafe { ptr::addr_of_mut!((*self.ptr).csrs[index]).write_volatile(val) }
    }

    /// Reads the general purpose register at `index`, which must be a valid GPR number.
    pub fn gpr(&self, index: usize) -> u64 {
        assert!(index < 32);
        // Safety: `index` is guaranteed to be a valid GPR index and the caller guaranteed at
        // construction that `ptr` points to a valid `TsmShmemArea`.
        unsafe { ptr::addr_of!((*self.shmem_scratch_ptr()).guest_gprs[index]).read_volatile() }
    }

    /// Writes the general purpose register at `index`, which must be a valid GPR number.
    pub fn set_gpr(&self, index: usize, val: u64) {
        assert!(index < 32);
        // Safety: `index` is guaranteed to be a valid GPR index and the caller guaranteed at
        // construction that `ptr` points to a valid `TsmShmemArea`.
        unsafe {
            ptr::addr_of_mut!((*self.shmem_scratch_ptr()).guest_gprs[index]).write_volatile(val)
        };
    }
}

fn _assert_scratch_size() {
    const_assert!(core::mem::size_of::<TsmShmemScratch>() == NACL_SCRATCH_BYTES);
}

/// Initiates a TSM fence on this CPU.
pub fn tsm_initiate_fence() -> Result<()> {
    let msg = SbiMessage::TeeHost(TsmInitiateFence);
    // Safety: TsmInitiateFence doesn't read or write any memory we have access to.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Initiates a fence for the given TVM.
pub fn tvm_initiate_fence(vmid: u64) -> Result<()> {
    let msg = SbiMessage::TeeHost(TvmInitiateFence { guest_id: vmid });
    // Safety: TvmInitiateFence doesn't read or write any memory we have access to.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Returns information about the TSM that booted this host context.
pub fn get_info() -> Result<TsmInfo> {
    let mut tsm_info = TsmInfo::default();
    let tsm_info_size = core::mem::size_of::<TsmInfo>() as u64;
    let msg = SbiMessage::TeeHost(TsmGetInfo {
        dest_addr: &mut tsm_info as *mut _ as u64,
        len: tsm_info_size,
    });
    // Safety: The passed info pointer is uniquely owned so it's safe to modify in SBI.
    let tsm_info_len = unsafe { ecall_send(&msg)? };

    if tsm_info_len != tsm_info_size {
        return Err(Error::Failed);
    }

    Ok(tsm_info)
}

/// Converts the given page range to confidential memory for use in creating or filling pages of a
/// TVM.
///
/// # Safety
///
/// The address provided must point to memory that won't be accessed again by the calling program
/// until it is reclaimed from confidential memory.
pub unsafe fn convert_pages(addr: u64, num_pages: u64) -> Result<()> {
    let msg = SbiMessage::TeeHost(TsmConvertPages {
        page_addr: addr,
        num_pages,
    });
    // Safety: The passed-in pages are unmapped and we do not access them again until they're
    // reclaimed.
    ecall_send(&msg)?;
    Ok(())
}

/// Reclaims pages that were previously converted to confidential memory with `convert_pages`.
pub fn reclaim_pages(addr: u64, num_pages: u64) -> Result<()> {
    let msg = SbiMessage::TeeHost(TsmReclaimPages {
        page_addr: addr,
        num_pages,
    });
    // Safety: The referenced pages are made accessible again, which is safe since we haven't
    // done anything with them since they were converted.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Creates a new TVM.
///
/// # Params:
///
/// - tvm_page_directory_addr: The base physical address of the 16kB confidential memory region that
/// should be used for the TVM's page directory. Must be 16kB-aligned.
///
/// - tvm_state_addr: The base physical address of the confidential memory region to be used to hold
/// the TVM's global state. Must be page-aligned and `TsmInfo::tvm_state_pages` pages in length.
pub fn tvm_create(tvm_page_directory_addr: u64, tvm_state_addr: u64) -> Result<u64> {
    let tvm_create_params = TvmCreateParams {
        tvm_page_directory_addr,
        tvm_state_addr,
    };
    let msg = SbiMessage::TeeHost(TvmCreate {
        params_addr: (&tvm_create_params as *const TvmCreateParams) as u64,
        len: core::mem::size_of::<TvmCreateParams>() as u64,
    });
    // Safety: creating a TVM will only touch pages that have already been converted to confidential
    // memory, so it can't affect memory safety as the host doesn't have access to those pages.
    let vmid = unsafe { ecall_send(&msg)? };
    Ok(vmid)
}

/// Finalizes the given TVM, setting the initial entry point for the TVM's boot vCPU.
pub fn tvm_finalize(vmid: u64, entry_sepc: u64, entry_arg: u64) -> Result<()> {
    let msg = SbiMessage::TeeHost(Finalize {
        guest_id: vmid,
        entry_sepc,
        entry_arg,
    });
    // Safety: `Finalize` doesn't touch memory.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Destroys a TVM created with `tvm_create`.
pub fn tvm_destroy(vmid: u64) -> Result<()> {
    let msg = SbiMessage::TeeHost(TvmDestroy { guest_id: vmid });
    // Safety: destroying a VM doesn't write to memory that's accessible from the host.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Runs the given vcpu of the specified TVM.
pub fn tvm_run(vmid: u64, vcpu_id: u64) -> Result<u64> {
    let msg = SbiMessage::TeeHost(TvmCpuRun {
        guest_id: vmid,
        vcpu_id,
    });
    // Safety: running a VM will only write to the shared-memory area registered in add_vcpu().
    unsafe { ecall_send(&msg) }
}

/// Adds pages to be used for page table entries of the given vmid.
pub fn add_page_table_pages(vmid: u64, page_addr: u64, num_pages: u64) -> Result<()> {
    let msg = SbiMessage::TeeHost(AddPageTablePages {
        guest_id: vmid,
        page_addr,
        num_pages,
    });
    // Safety: `AddPageTablePages` only accesses pages that have been previously converted. Passing
    // non-converted memory will result in a failure and not touch the memory.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Adds a vCPU with ID `vcpu_id` to the guest `vmid`, using 'state_page_addr' to hold the vCPU's
/// internal state.
///
/// The address `state_page_addr` must reference confidential memory pages. The caller must provide
/// `TsmInfo::tvm_vcpu_state_pages` pages.
pub fn add_vcpu(vmid: u64, vcpu_id: u64, state_page_addr: u64) -> Result<()> {
    let msg = SbiMessage::TeeHost(TvmCpuCreate {
        guest_id: vmid,
        vcpu_id,
        state_page_addr,
    });
    // Safety: TvmCpuCreate only accesses pages that have been converted and thus must already be
    // inaccessible to the calling program.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Declares a memory region in the guest's physical address space.
pub fn add_memory_region(vmid: u64, guest_addr: u64, len: u64) -> Result<()> {
    let msg = SbiMessage::TeeHost(TvmAddMemoryRegion {
        guest_id: vmid,
        guest_addr,
        len,
    });
    // Safety: `TvmAddMemoryRegion` doesn't access our memory at all.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Copies the data from the pages backing `src_data` to the guest and records their measurement for
/// attestation.  src_data must be aligned to the given page size.
pub fn add_measured_pages(
    vmid: u64,
    src_data: &[u8],
    dest_addr: u64,
    page_type: TsmPageType,
    guest_addr: u64,
) -> Result<()> {
    if src_data
        .as_ptr()
        .align_offset(page_type.size_bytes() as usize)
        != 0
    {
        return Err(Error::InvalidParam);
    }

    let msg = SbiMessage::TeeHost(TvmAddMeasuredPages {
        guest_id: vmid,
        src_addr: src_data.as_ptr() as u64,
        dest_addr,
        page_type,
        num_pages: src_data.len() as u64 / page_type.size_bytes(),
        guest_addr,
    });
    // Safety: `TvmAddMeasuredPages` only writes pages that have already been converted, and only
    // reads the pages pointed to by `src_addr`. This is safe because those pages are owned by the
    // borrowed slice and safe to read from.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Adds previously converted pages to the guest at the given address. The page will be left cleared
/// and read zeros to the guest.
pub fn add_zero_pages(
    vmid: u64,
    page_addr: u64,
    page_type: TsmPageType,
    num_pages: u64,
    guest_addr: u64,
) -> Result<()> {
    let msg = SbiMessage::TeeHost(TvmAddZeroPages {
        guest_id: vmid,
        page_addr,
        page_type,
        num_pages,
        guest_addr,
    });
    // Safety: `TvmAddZeroPages` only touches pages that we've already converted.
    unsafe { ecall_send(&msg) }?;
    Ok(())
}

/// Adds pages shared between the host and the given TVM.
///
/// # Safety
///
/// The pages to be shared must be owned by the caller and treated as volatile for the entire time
/// the page is shared.
pub unsafe fn add_shared_pages(
    vmid: u64,
    page_addr: u64,
    page_type: TsmPageType,
    num_pages: u64,
    guest_addr: u64,
) -> Result<()> {
    let msg = SbiMessage::TeeHost(TvmAddSharedPages {
        guest_id: vmid,
        page_addr,
        page_type,
        num_pages,
        guest_addr,
    });
    ecall_send(&msg)?;
    Ok(())
}
