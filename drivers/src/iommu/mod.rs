// Copyright (c) 2022 by Rivos Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

mod core;
mod device_directory;
mod error;
mod msi_page_table;
mod queue;
mod registers;

pub use self::core::Iommu;
pub use device_directory::{DeviceId, GscId};
pub use error::Error as IommuError;
pub use error::Result as IommuResult;
pub use msi_page_table::MsiPageTable;

#[cfg(test)]
mod tests {
    use super::device_directory::*;
    use super::queue::*;
    use super::*;
    use crate::imsic::*;
    use page_tracking::{HwMemMapBuilder, HypPageAlloc, PageList, PageTracker};
    use riscv_page_tables::{GuestStagePageTable, PagingMode, Sv48x4};
    use riscv_pages::*;

    const IMSIC_START: u64 = 0x2800_0000;
    const IMSIC_SIZE: u64 = 0x0010_0000;
    const GUEST_IMSIC_START: u64 = 0x3800_0000;

    fn stub_mem() -> (PageTracker, PageList<Page<ConvertedClean>>) {
        const ONE_MEG: usize = 1024 * 1024;
        const MEM_ALIGN: usize = 2 * ONE_MEG;
        const MEM_SIZE: usize = 256 * ONE_MEG;
        let backing_mem = vec![0u8; MEM_SIZE + MEM_ALIGN];
        let aligned_pointer = unsafe {
            // Not safe - just a test
            backing_mem
                .as_ptr()
                .add(backing_mem.as_ptr().align_offset(MEM_ALIGN))
        };
        let start_pa = RawAddr::supervisor(aligned_pointer as u64);
        let hw_map = unsafe {
            // Not safe - just a test
            HwMemMapBuilder::new(PageSize::Size4k as u64)
                .add_memory_region(start_pa, MEM_SIZE.try_into().unwrap())
                .unwrap()
                .add_mmio_region(
                    DeviceMemType::Imsic,
                    RawAddr::supervisor(IMSIC_START),
                    IMSIC_SIZE,
                )
                .unwrap()
                .build()
        };
        let hyp_mem = HypPageAlloc::new(hw_map);
        let (page_tracker, host_pages) = PageTracker::from(hyp_mem, PageSize::Size4k as u64);
        // Leak the backing ram so it doesn't get freed
        std::mem::forget(backing_mem);
        (page_tracker, host_pages)
    }

    fn stub_msi_page_table(
        page_tracker: PageTracker,
        pages: &mut PageList<Page<ConvertedClean>>,
        owner: PageOwnerId,
    ) -> (MsiPageTable, SupervisorImsicGeometry) {
        let src_base = PageAddr::new(RawAddr::guest(GUEST_IMSIC_START, owner)).unwrap();
        let src_geometry = ImsicGeometry::new(src_base, 0, 24, 3, 3, 5).unwrap();
        let dest_base = PageAddr::new(RawAddr::supervisor(IMSIC_START)).unwrap();
        let dest_geometry = ImsicGeometry::new(dest_base, 0, 24, 4, 4, 15).unwrap();
        assert_eq!(
            MsiPageTable::required_table_size(&src_geometry),
            PageSize::Size4k as u64
        );
        let msi_pt_page = page_tracker
            .assign_page_for_internal_state(pages.pop().unwrap(), owner)
            .unwrap();
        let msi_pt = MsiPageTable::new(
            msi_pt_page.into(),
            src_geometry,
            dest_geometry.clone(),
            page_tracker,
            owner,
        )
        .unwrap();
        (msi_pt, dest_geometry)
    }

    fn stub_guest_page_table(
        page_tracker: PageTracker,
        pages: &mut PageList<Page<ConvertedClean>>,
        owner: PageOwnerId,
    ) -> GuestStagePageTable<Sv48x4> {
        // Find 4 properly aligned pages from `pages`.
        let root_pages = SequentialPages::from_pages(
            pages
                .skip_while(|p| p.addr().bits() % Sv48x4::TOP_LEVEL_ALIGN != 0)
                .take(4)
                .map(|p| {
                    page_tracker
                        .assign_page_for_internal_state(p, owner)
                        .unwrap()
                }),
        )
        .unwrap();
        GuestStagePageTable::new(root_pages, owner, page_tracker).unwrap()
    }

    #[test]
    fn msi_page_table() {
        let (page_tracker, mut pages) = stub_mem();
        let (msi_pt, dest_geometry) =
            stub_msi_page_table(page_tracker.clone(), &mut pages, PageOwnerId::host());
        let (addr, mask) = msi_pt.msi_address_pattern();
        assert_eq!(addr.bits(), 0x3800_0000);
        assert_eq!(mask, 0x0003_f000);

        let dest_loc = ImsicLocation::new(
            ImsicGroupId::new(0),
            ImsicHartId::new(3),
            ImsicFileId::guest(0),
        );
        let dest_addr = dest_geometry.location_to_addr(dest_loc).unwrap();
        // Not safe, just a test.
        let imsic_page = unsafe { ImsicGuestPage::<ConvertedClean>::new(dest_addr) };
        page_tracker
            .assign_page_for_mapping(imsic_page, PageOwnerId::host())
            .unwrap();

        let bad_src = ImsicLocation::new(
            ImsicGroupId::new(1),
            ImsicHartId::new(1),
            ImsicFileId::supervisor(),
        );
        assert!(msi_pt.map(bad_src, dest_loc).is_err());

        let good_src = ImsicLocation::new(
            ImsicGroupId::new(0),
            ImsicHartId::new(4),
            ImsicFileId::supervisor(),
        );
        assert!(msi_pt.map(good_src, dest_loc).is_ok());
        assert!(msi_pt.map(good_src, dest_loc).is_err());
        assert!(msi_pt.unmap(good_src).is_ok());

        let src_loc = ImsicLocation::new(
            ImsicGroupId::new(0),
            ImsicHartId::new(2),
            ImsicFileId::supervisor(),
        );
        let unowned_dest = ImsicLocation::new(
            ImsicGroupId::new(0),
            ImsicHartId::new(1),
            ImsicFileId::guest(1),
        );
        assert!(msi_pt.unmap(src_loc).is_err());
        assert!(msi_pt.map(src_loc, unowned_dest).is_err());
    }

    #[test]
    fn device_directory() {
        let (page_tracker, mut pages) = stub_mem();
        let (msi_pt, _) =
            stub_msi_page_table(page_tracker.clone(), &mut pages, PageOwnerId::host());
        let pt = stub_guest_page_table(page_tracker.clone(), &mut pages, PageOwnerId::host());

        let ddt_page = page_tracker
            .assign_page_for_internal_state(pages.pop().unwrap(), PageOwnerId::host())
            .unwrap();
        let ddt = DeviceDirectory::<Ddt3Level>::new(ddt_page);
        for i in 0..16 {
            let id = DeviceId::new(i).unwrap();
            ddt.add_device(id, &mut || {
                page_tracker
                    .assign_page_for_internal_state(pages.pop().unwrap(), PageOwnerId::host())
                    .ok()
            })
            .unwrap();
        }

        let gscid = GscId::new(0);
        let dev = DeviceId::new(2).unwrap();
        assert!(ddt.enable_device(dev, &pt, &msi_pt, gscid).is_ok());
        assert!(ddt.disable_device(dev).is_ok());
        let bad_dev = DeviceId::new(1 << 16).unwrap();
        assert!(ddt.enable_device(bad_dev, &pt, &msi_pt, gscid).is_err());

        let (bad_msi_pt, _) = stub_msi_page_table(
            page_tracker.clone(),
            &mut pages,
            PageOwnerId::new(5).unwrap(),
        );
        assert!(ddt.enable_device(dev, &pt, &bad_msi_pt, gscid).is_err());
    }

    #[test]
    fn command_queue() {
        let (page_tracker, mut pages) = stub_mem();
        let queue_page = page_tracker
            .assign_page_for_internal_state(pages.pop().unwrap(), PageOwnerId::host())
            .unwrap();
        let mut cq = CommandQueue::new(queue_page);
        assert!(cq.is_empty());
        assert!(cq.push(Command::iotinval_gvma(None, None)).is_ok());
        assert!(cq.push(Command::iodir_inval_ddt(DeviceId::new(2))).is_ok());
        assert!(cq.push(Command::iofence()).is_ok());
        assert!(!cq.is_empty());
        assert_eq!(cq.tail(), 3);
        assert!(cq.update_head(7).is_err());
        assert!(cq.update_head(3).is_ok());
        assert!(cq.push(Command::iodir_inval_ddt(None)).is_ok());
        assert!(cq.push(Command::iofence()).is_ok());
        assert!(cq.update_head(1).is_err());
        assert!(cq.update_head(4).is_ok());
    }
}
